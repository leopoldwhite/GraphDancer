"""Curriculum-DPO trainer: subclass of trl.DPOTrainer that applies an
agent-token mask to the per-token log-prob aggregation.

The mask is needed because graph-agent trajectories interleave model-generated
content (`<think>`, `<graph>`, `<answer>` blocks) with environment-injected
observations (`<information>...</information>`). Only the agent-generated
tokens should contribute to the DPO loss — observations are deterministic
inputs from the graph executor, not policy decisions.

This mirrors the PPO-side masking in ``verl/trainer/ppo/ray_trainer.py:_create_loss_mask``.

Each batch must include `chosen_loss_mask` and `rejected_loss_mask` entries
(0/1 tensors over the chosen/rejected completion tokens). The dataset module
constructs these from char-level info spans stored in the pair parquet.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from trl import DPOTrainer
from trl.trainer.utils import flush_left, flush_right, selective_log_softmax


class CurriculumDPOTrainer(DPOTrainer):
    """DPO trainer that ANDs a per-token agent mask onto TRL's loss_mask.

    The override copies TRL's ``concatenated_forward`` body (TRL 0.19.x) and
    modifies the single line that constructs ``loss_mask`` to additionally
    apply the agent mask carried in the batch.

    If the batch lacks ``chosen_loss_mask`` / ``rejected_loss_mask``, behavior
    falls back to vanilla TRL DPOTrainer (no agent masking).
    """

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        # TRL's default _prepare_dataset assumes a HuggingFace `datasets.Dataset`
        # (calls .map() to tokenize the prompt/chosen/rejected columns). Our
        # `DPOPairDataset` is a plain torch.utils.data.Dataset and our
        # `DPOCollator` handles tokenization + info-mask projection at
        # batch-collation time — so we skip TRL's pre-tokenization entirely.
        return dataset

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[list, torch.LongTensor]],
        is_ref_model: bool = False,
    ):
        # Keep this method synced with TRL 0.19.x DPOTrainer.concatenated_forward;
        # only the lines marked "[CURRICULUM-DPO PATCH]" are changed.

        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Pass-through vision keys if present
        for k in ("pixel_values", "pixel_attention_mask", "image_sizes"):
            if k in concatenated_batch:
                model_kwargs[k] = concatenated_batch[k]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        # [CURRICULUM-DPO PATCH] — concatenate chosen/rejected agent masks if present.
        # Each has shape (B, completion_len); zeros mark `<information>` tokens.
        agent_mask: Optional[torch.Tensor] = None
        if "chosen_loss_mask" in batch and "rejected_loss_mask" in batch:
            agent_mask = torch.cat(
                [batch["chosen_loss_mask"], batch["rejected_loss_mask"]], dim=0
            ).to(completion_attention_mask.device)
            # Pad/truncate to match completion_attention_mask shape if needed
            if agent_mask.shape[1] < completion_attention_mask.shape[1]:
                pad = torch.zeros(
                    agent_mask.shape[0],
                    completion_attention_mask.shape[1] - agent_mask.shape[1],
                    dtype=agent_mask.dtype, device=agent_mask.device,
                )
                agent_mask = torch.cat([agent_mask, pad], dim=1)
            elif agent_mask.shape[1] > completion_attention_mask.shape[1]:
                agent_mask = agent_mask[:, : completion_attention_mask.shape[1]]
        # [/CURRICULUM-DPO PATCH]

        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
            # [CURRICULUM-DPO PATCH]
            if agent_mask is not None:
                loss_mask = loss_mask & agent_mask.bool()
            # [/CURRICULUM-DPO PATCH]
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Standard TRL: prompt zeroed out, completion = completion_attention_mask
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )
            # [CURRICULUM-DPO PATCH] — AND completion side with agent mask
            if agent_mask is not None:
                completion_loss_mask = loss_mask[:, prompt_input_ids.shape[1] :]
                completion_loss_mask = completion_loss_mask * agent_mask.to(loss_mask.dtype)
                loss_mask = torch.cat(
                    (torch.zeros_like(prompt_attention_mask), completion_loss_mask),
                    dim=1,
                )
            # [/CURRICULUM-DPO PATCH]

            # Truncation (same logic as TRL)
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. "
                        "Should be one of ['keep_end', 'keep_start']"
                    )
            else:
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
            logits = outputs.logits
            # Shift labels for next-token loss
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        # Compute per-token logp and aggregate via masked sum (TRL standard from here)
        if self.is_encoder_decoder:
            logits = logits[:, :-1] if not self.is_encoder_decoder else logits
            labels = labels[:, 1:] if not self.is_encoder_decoder else labels
            loss_mask = loss_mask[:, 1:] if not self.is_encoder_decoder else loss_mask
        else:
            logits = logits[:, :-1]
            labels = labels[:, :-1]
            loss_mask = loss_mask[:, :-1]
        labels[~loss_mask] = 0

        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps = per_token_logps * loss_mask  # zero out masked positions

        all_logps = per_token_logps.sum(-1)
        size_completion = loss_mask.sum(-1)

        chosen_logps = all_logps[:num_examples]
        rejected_logps = all_logps[num_examples:]
        chosen_size = size_completion[:num_examples]
        rejected_size = size_completion[num_examples:]

        # Average logp (used by some DPO variants); guard against zero-length
        chosen_avg = chosen_logps / chosen_size.clamp(min=1)
        rejected_avg = rejected_logps / rejected_size.clamp(min=1)

        out: Dict[str, Any] = {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": logits[:num_examples],
            "rejected_logits": logits[num_examples:],
            "mean_chosen_logits": logits[:num_examples].mean(),
            "mean_rejected_logits": logits[num_examples:].mean(),
            "chosen_logps_avg": chosen_avg,
            "rejected_logps_avg": rejected_avg,
        }
        if self.aux_loss_enabled and getattr(outputs, "aux_loss", None) is not None:
            out["aux_loss"] = outputs.aux_loss
        return out
