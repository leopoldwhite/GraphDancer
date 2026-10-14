#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""ROUGE-L of one evaluation run (one checkpoint, one decode seed).

R-L = 100 x ROUGE-L F1 between the reference answer (gt_answer) and the extracted
model answer (model_answer), rouge_score.RougeScorer(['rougeL'], use_stemmer=False).
Columns follow the paper: In-domain pools the 214 Academic + Biomedical test
questions (each question weighted equally); E-commerce = amazon, Literature =
goodreads, Legal = legal; AVG is the unweighted mean of the four columns.

    python scripts/score_rouge.py --run-tag graphdancer-qwen2.5-3b-step200 --seed 3
    python scripts/score_rouge.py --run-tag ... --seed 1 --set validation
"""
import argparse
import json
import sys
from statistics import mean

from rouge_score.rouge_scorer import RougeScorer

from eval_utils import ACADEMIC_BIOMEDICAL, COLUMNS, load_records, to_text

_SCORER = RougeScorer(["rougeL"], use_stemmer=False)


def rouge_l(gold, prediction):
    return 100 * _SCORER.score(to_text(gold), to_text(prediction))["rougeL"].fmeasure


def score_set(results_dir, run_tag, set_name, domains, seed):
    rows = []
    for domain in domains:
        for record in load_records(results_dir, run_tag, set_name, domain, seed):
            rows.append({"domain": domain, "index": int(record["extra_info"]["index"]),
                         "rouge_l": rouge_l(record["gt_answer"], record["model_answer"])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--set", choices=["paper", "validation"], default="paper",
                        help="paper = the four test columns; validation = the 107-question validation set")
    parser.add_argument("--per-question", help="optional JSONL output with per-question scores")
    args = parser.parse_args()

    if args.set == "validation":
        rows = score_set(args.results_dir, args.run_tag, "validation", ACADEMIC_BIOMEDICAL, args.seed)
        summary = {"run_tag": args.run_tag, "decode_seed": args.seed, "validation_n": len(rows),
                   "validation_rouge_l": mean(r["rouge_l"] for r in rows)}
    else:
        rows, columns = [], {}
        for name, (set_name, domains) in COLUMNS.items():
            scored = score_set(args.results_dir, args.run_tag, set_name, domains, args.seed)
            columns[name] = {"n": len(scored), "rouge_l": mean(r["rouge_l"] for r in scored)}
            rows += [dict(r, column=name) for r in scored]
        summary = {"run_tag": args.run_tag, "decode_seed": args.seed, "columns": columns,
                   "avg_rouge_l": mean(c["rouge_l"] for c in columns.values())}
    if args.per_question:
        with open(args.per_question, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
