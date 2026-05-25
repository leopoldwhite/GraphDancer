"""Dataset / collator for curriculum DPO.

Reads the pair parquet produced by ``scripts/dpo/phase0_build_pair_parquet.py``
and yields per-pair samples plus the agent-token loss masks used by
``CurriculumDPOTrainer.concatenated_forward``.

Key plumbing:
- ``prompt`` is stored as the verl chat-format list ``[{role, content}, ...]``.
  We render it into a string via ``tokenizer.apply_chat_template`` (no generation
  prompt — the chosen/rejected completion strings already include the response
  tokens the model would have generated).
- ``chosen_info_spans`` / ``rejected_info_spans`` are character-level spans of
  ``<information>...</information>`` blocks. We tokenize the completion text,
  then back-project the spans onto token boundaries to build the binary
  agent-token mask (1 = agent-generated, 0 = environment observation).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def _char_mask_to_token_mask(
    text: str,
    info_spans: List[Tuple[int, int]],
    offsets: List[Tuple[int, int]],
) -> List[int]:
    """Project char-level info spans onto token offsets.

    Token i is masked-out (mask=0) iff its character span overlaps any info span.
    """
    mask = [1] * len(offsets)
    if not info_spans:
        return mask
    for ti, (ts, te) in enumerate(offsets):
        if te <= ts:
            continue
        for s, e in info_spans:
            if not (te <= s or ts >= e):
                mask[ti] = 0
                break
    return mask


def _render_prompt(tokenizer, prompt_field) -> str:
    """Render parquet ``prompt`` field into a string.

    Verl stores it as a list of {role, content} dicts (chat-template ready).
    Falls back to str(prompt_field) for plain strings.

    `add_generation_prompt=True` is REQUIRED for on-policy DPO correctness:
    during Phase 0 sampling, the model received the prompt followed by
    `<|im_start|>assistant\\n` and then generated the response (the chosen /
    rejected text). For DPO loss to score logp on the same conditional
    distribution, the prompt rendering at training time MUST end with that
    same `<|im_start|>assistant\\n` marker before the chosen / rejected text
    begins. Without it, the very first chosen-token conditional is computed
    against a context the model never saw during rollout, producing biased
    log-probabilities. (Audit 2026-05-04.)
    """
    if isinstance(prompt_field, list):
        return tokenizer.apply_chat_template(
            prompt_field, tokenize=False, add_generation_prompt=True
        )
    if hasattr(prompt_field, "tolist"):
        # numpy array of dicts
        return tokenizer.apply_chat_template(
            list(prompt_field), tokenize=False, add_generation_prompt=True
        )
    return str(prompt_field)


class DPOPairDataset(Dataset):
    """Per-row pair dataset reading the Phase-0 → Phase-A parquet.

    Yields samples as dicts with the keys expected by the curriculum-DPO
    trainer:
        prompt: str
        chosen: str
        rejected: str
        chosen_info_spans: list[(int,int)]   (char level)
        rejected_info_spans: list[(int,int)] (char level)
        difficulty: str  (for curriculum sampler downstream)
        qid: any
    """

    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        # Sanity: required columns
        required = {"prompt", "chosen", "rejected", "chosen_info_spans", "rejected_info_spans", "extra_info"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DPO pair parquet missing columns: {missing}")

    # E2HGaussianSampler reads `.dataframe` directly to extract per-row difficulty
    @property
    def dataframe(self):
        return self.df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        ei = row["extra_info"]
        if hasattr(ei, "tolist"):
            ei = ei.tolist()

        def _to_span_list(field):
            # parquet may load list-of-list/tuple as numpy object array; normalize
            if field is None:
                return []
            try:
                if hasattr(field, "tolist"):
                    field = field.tolist()
                if not field:  # empty list/tuple
                    return []
                return [tuple(s) for s in field]
            except Exception:
                return []

        return {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "chosen_info_spans": _to_span_list(row["chosen_info_spans"]),
            "rejected_info_spans": _to_span_list(row["rejected_info_spans"]),
            "difficulty": ei.get("difficulty") if isinstance(ei, dict) else "unknown",
            "qid": row.get("qid"),
        }


class DPOCollator:
    """Tokenize prompt/chosen/rejected and emit the per-token agent mask.

    The collator's output schema matches what ``trl.DPOTrainer`` expects, plus:
        chosen_loss_mask:   (B, completion_len)  binary
        rejected_loss_mask: (B, completion_len)  binary

    The trainer subclass ANDs these into TRL's standard loss mask.
    """

    def __init__(
        self,
        tokenizer,
        max_prompt_length: int = 24576,
        max_completion_length: int = 4096,
    ):
        self.tok = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        # Ensure pad token (TRL needs it)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _tokenize_completion(self, text: str, info_spans):
        """Tokenize a completion and return (input_ids, attn_mask, agent_mask)."""
        enc = self.tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_completion_length,
            return_tensors=None,
        )
        offsets = enc["offset_mapping"]
        agent_mask = _char_mask_to_token_mask(text, list(info_spans or []), offsets)
        return enc["input_ids"], enc["attention_mask"], agent_mask

    def _tokenize_prompt(self, prompt_field) -> Tuple[List[int], List[int]]:
        text = _render_prompt(self.tok, prompt_field)
        enc = self.tok(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors=None,
        )
        return enc["input_ids"], enc["attention_mask"]

    def __call__(self, batch: List[Dict[str, Any]]):
        out: Dict[str, Any] = {
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "chosen_loss_mask": [],
            "rejected_loss_mask": [],
        }
        for s in batch:
            p_ids, p_attn = self._tokenize_prompt(s["prompt"])
            c_ids, c_attn, c_mask = self._tokenize_completion(s["chosen"], s["chosen_info_spans"])
            r_ids, r_attn, r_mask = self._tokenize_completion(s["rejected"], s["rejected_info_spans"])
            out["prompt_input_ids"].append(p_ids)
            out["prompt_attention_mask"].append(p_attn)
            out["chosen_input_ids"].append(c_ids)
            out["chosen_attention_mask"].append(c_attn)
            out["rejected_input_ids"].append(r_ids)
            out["rejected_attention_mask"].append(r_attn)
            out["chosen_loss_mask"].append(c_mask)
            out["rejected_loss_mask"].append(r_mask)

        # Pad to a fixed length within batch. Chosen and rejected are padded
        # to a JOINT max length (max across both lists) so the per-token agent
        # masks can be safely cat'd along batch dim later in
        # CurriculumDPOTrainer.concatenated_forward — TRL's concatenated_inputs
        # also pads chosen/rejected to a shared completion length.
        def _pad_to(seqs, target_len: int, pad_value: int = 0):
            return torch.tensor(
                [list(x)[:target_len] + [pad_value] * max(0, target_len - len(x)) for x in seqs],
                dtype=torch.long,
            )

        prompt_max = max(len(x) for x in out["prompt_input_ids"])
        completion_max = max(
            max(len(x) for x in out["chosen_input_ids"]),
            max(len(x) for x in out["rejected_input_ids"]),
        )

        return {
            "prompt_input_ids": _pad_to(out["prompt_input_ids"], prompt_max, pad_value=self.tok.pad_token_id),
            "prompt_attention_mask": _pad_to(out["prompt_attention_mask"], prompt_max, pad_value=0),
            "chosen_input_ids": _pad_to(out["chosen_input_ids"], completion_max, pad_value=self.tok.pad_token_id),
            "chosen_attention_mask": _pad_to(out["chosen_attention_mask"], completion_max, pad_value=0),
            "rejected_input_ids": _pad_to(out["rejected_input_ids"], completion_max, pad_value=self.tok.pad_token_id),
            "rejected_attention_mask": _pad_to(out["rejected_attention_mask"], completion_max, pad_value=0),
            "chosen_loss_mask": _pad_to(out["chosen_loss_mask"], completion_max, pad_value=0).float(),
            "rejected_loss_mask": _pad_to(out["rejected_loss_mask"], completion_max, pad_value=0).float(),
        }
