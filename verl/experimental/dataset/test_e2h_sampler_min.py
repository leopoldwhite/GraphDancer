"""
Minimal test for VERL E2H curriculum samplers (Gaussian/Cosine).

This mirrors the Graph-RL mini test but uses VERL's sampler API:
- Uses torch DataLoader with `sampler=E2H*Sampler` and `batch_size`.
- The sampler builds buckets from a pandas DataFrame (dataframe column) or
  a HF datasets.Dataset if your RLHFDataset exposes that.

Run examples (from repo root):
  python -m verl.experimental.dataset.test_e2h_sampler_min \
    --train-parquet /path/to/train.parquet \
    --difficulty-key extra_info.difficulty \
    --sampler gaussian \
    --batch-size 128 --epochs 100 --show-steps 200

Notes:
- This script is model-free. It only exercises the sampler and shows batch compositions.
- If your `extra_info` column is JSON strings, the sampler can parse it.

Scheme B (Gaussian mix) notes:
- For Gaussian sampler, you can mix the Gaussian distribution p_g with a bias distribution q:
    p(t) = (1 - eta(t)) * p_g(t) + eta(t) * q
- Control knobs exposed here:
  - --mix-eta (constant)
  - --mix-eta-start/--mix-eta-end (linear schedule over t in [0, T-1])
  - --mix-q: "easy" | "hard" | "uniform" | JSON list (e.g. "[1,0,0]") | JSON dict (e.g. "{\"easy\":1,\"hard\":0}")
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from omegaconf import OmegaConf
from verl.experimental.dataset.crl_e2h import E2HGaussianSampler, E2HCosineSampler


class FrameDataset(Dataset):
    """Minimal dataset backed by a pandas DataFrame.

    - Exposes `dataframe` so samplers can read columns to build buckets.
    - __getitem__ returns a small dict with at least `extra_info` for composition.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataframe.iloc[int(idx)]
        # Only keep fields necessary for composition/debugging
        out = {
            "extra_info": row.get("extra_info", {}),
            "data_source": row.get("data_source", None),
        }
        return out


def simple_collate(batch: List[dict]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for d in batch:
        for k, v in d.items():
            out.setdefault(k, []).append(v)
    return out


def _count_from_extras(extras: List[dict]) -> Dict[str, int]:
    cnt: Counter = Counter()
    for e in extras:
        try:
            label = str(dict(e).get("difficulty", "None")).lower()
        except Exception:
            label = "None"
        cnt[label] += 1
    return dict(cnt)


def run_min_test(
    train_parquet: str,
    difficulty_key: str = "extra_info.difficulty",
    sampler_name: str = "gaussian",
    batch_size: int = 128,
    steps_per_epoch: int | None = None,
    epochs: int = 100,
    total_training_steps: int | None = 200,
    show_steps: int = 200,
    sigma: float = 0.75,
    beta: float = 0.25,
    eps: float = 1e-8,
    mix_eta: float = 0.0,
    mix_eta_start: float | None = None,
    mix_eta_end: float | None = None,
    mix_q=None,
    seed: int = 42,
    out_plot: str | None = None,
    plot_comp: bool = True,
):
    if not os.path.exists(train_parquet):
        raise FileNotFoundError(f"Missing train parquet: {train_parquet}")
    df = pd.read_parquet(train_parquet)
    print(f"Loaded dataframe: rows={len(df)} cols={list(df.columns)}")
    dataset = FrameDataset(df)

    drop_last = True
    inferred_steps = len(dataset) // batch_size if drop_last else int(np.ceil(len(dataset) / batch_size))
    steps = inferred_steps if steps_per_epoch is None or steps_per_epoch <= 0 else int(steps_per_epoch)
    # Determine total training steps T for global schedule
    tentative_T = steps * int(max(1, int(epochs)))
    T = int(total_training_steps) if (total_training_steps is not None and int(total_training_steps) > 0) else tentative_T

    print(f"Dataset size: {len(dataset)}")
    sampler_lc = str(sampler_name).lower()
    use_cos = sampler_lc in ["cos", "cosine", "e2h-c", "e2h_c"]
    cfg = {
        "train_batch_size": int(batch_size),
        "buckets": None,
        "difficulty_key": difficulty_key,
        "e2h_sigma": float(sigma),
        "e2h_beta": float(beta),
        "e2h_eps": float(eps),
        # Scheme B (Gaussian + bias mix)
        "e2h_mix_eta": float(mix_eta),
        "e2h_mix_eta_start": (None if mix_eta_start is None else float(mix_eta_start)),
        "e2h_mix_eta_end": (None if mix_eta_end is None else float(mix_eta_end)),
        "e2h_mix_q": mix_q,
        "seed": int(seed),
        "e2h_total_training_steps": int(T),
    }
    data_cfg = OmegaConf.create(cfg)
    if use_cos:
        sampler = E2HCosineSampler(data_source=dataset, data_config=data_cfg)
    else:
        sampler = E2HGaussianSampler(data_source=dataset, data_config=data_cfg)
    # override computed steps_per_epoch for fair comparison across datasets
    sampler.steps_per_epoch = max(1, int(steps))

    print(f"Using T={T}, steps/epoch={steps}, epochs={epochs}")
    print(f"Buckets: {getattr(sampler, 'bucket_names', [])}")
    if (not use_cos) and (mix_eta is not None):
        print(
            "Gaussian mix: "
            f"eta={float(mix_eta):.4f}, eta_start={mix_eta_start}, eta_end={mix_eta_end}, q={mix_q}"
        )
    loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, sampler=sampler, collate_fn=simple_collate)

    printed = 0
    bucket_names: List[str] = list(getattr(sampler, 'bucket_names', []))
    collected_t: list[int] = []
    collected_probs: list[list[float]] = []
    collected_comp: list[list[float]] = []
    global_t_seen = 0
    for epoch in range(max(1, int(epochs))):
        for i, batch in enumerate(loader):
            # sampler updates internal _last_probs each step
            last_probs = getattr(loader.sampler, "_last_probs", {})
            buckets = getattr(loader.sampler, "bucket_names", [])
            comp = _count_from_extras(batch.get("extra_info", []))
            print(f"[epoch {epoch} step {i}] probs={{{', '.join(f'{b}:{last_probs.get(b, 0.0):.3f}' for b in buckets)}}} comp={comp}")
            printed += 1
            # record series
            global_t_seen += 1
            t_used = min(global_t_seen - 1, max(1, T) - 1)
            collected_t.append(t_used)
            if bucket_names:
                collected_probs.append([float(last_probs.get(b, 0.0)) for b in bucket_names])
                total_b = float(batch_size) if batch_size > 0 else 1.0
                collected_comp.append([float(comp.get(b, 0)) / total_b for b in bucket_names])
            if printed >= int(show_steps):
                break
        if printed >= int(show_steps):
            break

    print("OK: VERL E2H mini test completed.")

    # Visualization and CSV dump similar to Graph-RL
    if out_plot is not None and len(collected_t) > 0 and bucket_names:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            nrows = 2 if plot_comp else 1
            fig_h = 6 if plot_comp else 3.2
            fig, axes = plt.subplots(nrows, 1, figsize=(9, fig_h), sharex=True)
            if nrows == 1:
                axes = [axes]

            xs = np.array(collected_t)
            arr_probs = np.array(collected_probs) if collected_probs else None
            if arr_probs is not None:
                ax = axes[0]
                for k, b in enumerate(bucket_names):
                    ax.plot(xs, arr_probs[:, k], label=f"prob:{b}")
                ax.set_ylabel("P(bucket|t)")
                title = f"E2H-{sampler_name} probabilities over steps (T={T})"
                if (not use_cos) and float(mix_eta) > 0.0:
                    eta_note = (
                        f"mix eta={float(mix_eta):.3f}"
                        if (mix_eta_start is None and mix_eta_end is None)
                        else f"mix eta={mix_eta_start}->{mix_eta_end}"
                    )
                    title += f" | {eta_note} | q={mix_q}"
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9)

            if plot_comp and len(collected_comp) > 0:
                ax2 = axes[1]
                arr_comp = np.array(collected_comp)
                for k, b in enumerate(bucket_names):
                    ax2.plot(xs, arr_comp[:, k], label=f"comp%:{b}")
                ax2.set_xlabel("t (step index)")
                ax2.set_ylabel("batch fraction")
                ax2.set_title("Observed batch composition (normalized)")
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='best', fontsize=9)
            else:
                axes[0].set_xlabel("t (step index)")

            plt.tight_layout()
            plt.savefig(out_plot, dpi=150)
            print(f"Saved visualization to: {out_plot}")
        except Exception as e:
            print(f"[WARN] Matplotlib plotting failed: {e}")

        # CSV dump
        try:
            df_out = pd.DataFrame({"t": collected_t})
            for k, b in enumerate(bucket_names):
                if len(collected_probs) > 0:
                    df_out[f"prob_{b}"] = [row[k] for row in collected_probs]
                if len(collected_comp) > 0:
                    df_out[f"comp_{b}"] = [row[k] for row in collected_comp]
            csv_path = out_plot.rsplit('.', 1)[0] + ".csv"
            df_out.to_csv(csv_path, index=False)
            print(f"Saved schedule CSV to: {csv_path}")
        except Exception as e:
            print(f"[WARN] Saving CSV failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Minimal test for VERL E2H samplers")
    ap.add_argument(
        "--dataset-dir",
        type=str,
        default=os.environ.get("DATASET_DIR", "./data/grbench_fewshot_merged_biomedical"),
        help="Directory containing train.parquet",
    )
    ap.add_argument("--train-parquet", type=str, default=None)
    ap.add_argument("--difficulty-key", type=str, default="extra_info.difficulty")
    ap.add_argument("--sampler", type=str, default="gaussian", help="gaussian or cosine")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--total-training-steps", type=int, default=200, help="Override T for global schedule; 0 to use steps_per_epoch*epochs")
    ap.add_argument("--steps-per-epoch", type=int, default=0)
    ap.add_argument("--show-steps", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--beta", type=float, default=3)
    ap.add_argument("--eps", type=float, default=1e-8)
    # Scheme B (Gaussian mix) knobs
    ap.add_argument("--mix-eta", type=float, default=0.0, help="Gaussian-only: constant eta for p=(1-eta)p_g+eta*q")
    ap.add_argument("--mix-eta-start", type=float, default=None, help="Gaussian-only: eta schedule start (linear)")
    ap.add_argument("--mix-eta-end", type=float, default=None, help="Gaussian-only: eta schedule end (linear)")
    ap.add_argument(
        "--mix-q",
        type=str,
        default="easy",
        help='Gaussian-only: bias distribution q. Preset: "easy"|"hard"|"uniform", or JSON list "[1,0,0]" or JSON dict \'{"easy":1,"hard":0}\'',
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", type=str, default="e2h_schedule.png", help="Output PNG path")
    ap.add_argument("--out-dir", type=str, default=None, help="Optional output directory for plot/csv")
    ap.add_argument("--no-plot-comp", action="store_true", help="Disable composition subplot")
    args = ap.parse_args()

    train_parquet = args.train_parquet
    if train_parquet is None:
        if args.dataset_dir is None:
            raise ValueError("Provide --train-parquet or --dataset-dir with train.parquet")
        train_parquet = os.path.join(args.dataset_dir, "train.parquet")

    # resolve output path
    out_plot = args.plot
    if args.out_dir:
        try:
            os.makedirs(args.out_dir, exist_ok=True)
        except Exception:
            pass
        base = os.path.basename(out_plot)
        out_plot = os.path.join(args.out_dir, base)

    # Parse mix_q: allow JSON list/dict; otherwise keep as preset string
    mix_q = args.mix_q
    if isinstance(mix_q, str):
        s = mix_q.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                mix_q = json.loads(s)
            except Exception:
                mix_q = s

    run_min_test(
        train_parquet=train_parquet,
        difficulty_key=args.difficulty_key,
        sampler_name=args.sampler,
        batch_size=int(args.batch_size),
        steps_per_epoch=int(args.steps_per_epoch),
        epochs=int(args.epochs),
        total_training_steps=int(args.total_training_steps),
        show_steps=int(args.show_steps),
        sigma=float(args.sigma),
        beta=float(args.beta),
        eps=float(args.eps),
        mix_eta=float(args.mix_eta),
        mix_eta_start=args.mix_eta_start,
        mix_eta_end=args.mix_eta_end,
        mix_q=mix_q,
        seed=int(args.seed),
        out_plot=out_plot,
        plot_comp=(not bool(args.no_plot_comp)),
    )


if __name__ == "__main__":
    main()
