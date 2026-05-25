"""Curriculum-DPO training entry point (Stage 2 of GraphDancer post-training).

Loads the pair parquet produced by ``scripts/dpo/build_preference_pairs.py``,
wires the E2H biased-mixture curriculum sampler over difficulty buckets, and
trains via the ``CurriculumDPOTrainer`` subclass (which masks ``<information>``
tokens from the loss).

Example:
    python -m verl.trainer.dpo.train \\
        --pair_parquet ./data/dpo_pairs.parquet \\
        --base_model <path-to-curriculum-PPO-checkpoint> \\
        --output_dir ./checkpoints/dpo \\
        ... (see argparse for the full surface)

This script is **not** wired into Hydra — it uses argparse to stay decoupled
from the PPO config tree. All knobs needed for a single DPO run are positional.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

HERE = os.path.dirname(os.path.abspath(__file__))
GD_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # GraphDancer/
sys.path.insert(0, GD_ROOT)

from verl.trainer.dpo.curriculum_dpo_trainer import CurriculumDPOTrainer  # noqa: E402
from verl.trainer.dpo.data import DPOCollator, DPOPairDataset  # noqa: E402

# We borrow the curriculum sampler from the PPO experimental dataset; it works
# directly over (bucket -> indices) and `extra_info.difficulty` per row, both
# of which our DPO pair parquet provides.
from verl.experimental.dataset.crl_e2h import E2HGaussianSampler  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_parquet", required=True, help="DPO training pair parquet")
    ap.add_argument("--base_model", required=True, help="HF id or local path of π_init/π_ref")
    ap.add_argument("--output_dir", required=True)

    # DPO hparams (defaults from project_curriculum_dpo_plan.md §5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-7)
    ap.add_argument("--total_steps", type=int, default=100)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--global_batch_size", type=int, default=64)
    ap.add_argument("--per_device_batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_length", type=int, default=24576)
    ap.add_argument("--max_completion_length", type=int, default=4096)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)

    # Curriculum sampler (defaults match paper)
    ap.add_argument("--e2h_q", type=str, default="0.5,0.5,0",
                    help="Comma-separated q bias prior over [easy,medium,hard]")
    ap.add_argument("--e2h_eta_start", type=float, default=0.2)
    ap.add_argument("--e2h_eta_end", type=float, default=0.8)
    ap.add_argument("--e2h_beta", type=float, default=3.0)
    ap.add_argument("--e2h_sigma", type=float, default=0.75)
    ap.add_argument("--disable_curriculum", action="store_true",
                    help="Ablation: do NOT use E2H curriculum sampler. Falls back to "
                         "HF Trainer's default (random with seed) shuffling order.")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"loading tokenizer + ref model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load π_init (will be trained) and π_ref (frozen; provided to DPOTrainer)
    print("loading π_init (trainable)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print("loading π_ref (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # ------------------------------------------------------------------ data
    print(f"loading DPO pair parquet: {args.pair_parquet}")
    ds = DPOPairDataset(args.pair_parquet)
    collator = DPOCollator(
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )
    print(f"  loaded {len(ds)} pairs")

    # ------------------------------------------------------------------ DPOConfig
    grad_accum = max(1, args.global_batch_size // (args.per_device_batch_size * max(1, torch.cuda.device_count())))
    cfg = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.total_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        logging_steps=1,
        bf16=True,
        beta=args.beta,
        max_length=args.max_prompt_length + args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        # We pre-tokenize via collator; tell TRL not to retokenize
        remove_unused_columns=False,
        # Activation checkpointing — required to fit chosen+rejected with
        # 24K prompt + 2K completion in a 3B model on 8×H200 even after
        # FSDP param-offload.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        seed=args.seed,
    )

    # ------------------------------------------------------------------ curriculum sampler
    # E2HGaussianSampler signature: __init__(data_source, data_config: DictConfig)
    # `data_source` must expose `.dataframe` (our DPOPairDataset does)
    # `data_config` is a DictConfig (Hydra-style) with keys:
    #   train_batch_size, gen_batch_size?, buckets, difficulty_key, seed,
    #   e2h_total_training_steps, e2h_beta, e2h_sigma, e2h_mix_eta_start,
    #   e2h_mix_eta_end, e2h_mix_q, e2h_eps
    from omegaconf import OmegaConf

    sampler = None
    if not args.disable_curriculum:
        q_list = [float(x) for x in args.e2h_q.split(",")]
        data_cfg = OmegaConf.create({
            "train_batch_size": args.global_batch_size,
            "buckets": ["easy", "medium", "hard"],
            # The pair parquet stores difficulty under top-level "difficulty" column
            # AND inside extra_info.difficulty. The sampler can read either.
            "difficulty_key": "extra_info.difficulty",
            "seed": args.seed,
            "e2h_total_training_steps": args.total_steps,
            "e2h_beta": args.e2h_beta,
            "e2h_sigma": args.e2h_sigma,
            "e2h_mix_eta_start": args.e2h_eta_start,
            "e2h_mix_eta_end": args.e2h_eta_end,
            "e2h_mix_q": q_list,
            "e2h_eps": 1e-8,
        })
        sampler = E2HGaussianSampler(data_source=ds, data_config=data_cfg)
        print("[curriculum] ENABLED — E2HGaussianSampler with paper-faithful params")
    else:
        print("[curriculum] DISABLED (--disable_curriculum) — falling back to default RandomSampler")

    # ------------------------------------------------------------------ trainer
    trainer = CurriculumDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,  # TRL >= 0.10
    )

    # Override the trainer's data sampler with our curriculum sampler.
    # E2HGaussianSampler is a regular Sampler (yields one int per iteration —
    # see crl_e2h.py:115), NOT a BatchSampler. We must therefore pass it as
    # `sampler=` with an explicit `batch_size`, not as `batch_sampler=`.
    # `per_device_train_batch_size` is the right batch_size for the DataLoader
    # since accelerate handles cross-device aggregation via gradient
    # accumulation (already wired by DPOConfig.gradient_accumulation_steps).
    if sampler is not None:
        def _get_train_dl():
            return DataLoader(
                ds,
                sampler=sampler,
                batch_size=args.per_device_batch_size,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
        trainer.get_train_dataloader = _get_train_dl  # type: ignore

    print("starting training...")
    trainer.train()
    print(f"done; checkpoints under {args.output_dir}")


if __name__ == "__main__":
    main()
