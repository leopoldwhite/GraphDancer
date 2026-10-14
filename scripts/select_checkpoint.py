#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""Select a checkpoint on the validation partition.

Rule: highest mean validation ROUGE-L across decode seeds (default seeds 1 and 2),
where each seed's score is the mean over the 107 validation questions; exact ties
go to the earlier step. Expects evaluation outputs with run tags
``<experiment>-step<STEP>`` (see README), e.g.

    for s in 150 175 200; do
      MODEL_PATH=outputs/hf/graphdancer-qwen2.5-3b/step_$s RUN_TAG=graphdancer-qwen2.5-3b-step$s \\
        SETS=validation SEEDS="1 2" bash scripts/eval_ppo.sh
    done
    python scripts/select_checkpoint.py --experiment graphdancer-qwen2.5-3b --steps 150 175 200
"""
import argparse
import json
import sys
from statistics import mean

from eval_utils import ACADEMIC_BIOMEDICAL
from score_rouge import score_set


def select(per_step):
    """per_step: {step: [validation R-L per seed]} -> selected step."""
    ranked = sorted(per_step, key=lambda step: (-mean(per_step[step]), step))
    return ranked[0], ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    per_step = {}
    for step in args.steps:
        tag = f"{args.experiment}-step{step}"
        per_step[step] = [mean(r["rouge_l"] for r in score_set(args.results_dir, tag, "validation",
                                                               ACADEMIC_BIOMEDICAL, seed))
                          for seed in args.seeds]
    selected, ranked = select(per_step)
    print(json.dumps({"experiment": args.experiment, "decode_seeds": args.seeds,
                      "rule": "highest mean validation R-L across seeds; ties -> earlier step",
                      "validation_rouge_l": {str(s): {"per_seed": v, "mean": mean(v)} for s, v in per_step.items()},
                      "ranked_steps": ranked, "selected_step": selected}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
