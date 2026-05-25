#!/usr/bin/env python3
"""Compute paper-style metrics (EM / BLEU / Rouge-L) from JSONL dumped by RewardManager.

Each line: {qid, domain, data_source, prediction, golden}
- prediction is the raw model response text (may contain <think>/<graph>/<answer> tags)
- golden is a list of acceptable answers (or string)

Extracts the final <answer>...</answer> from prediction; if none, falls back to
the whole response (so format-broken outputs still get partial Rouge-L credit).
"""
import argparse
import json
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

import evaluate

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def extract_answer(pred: str) -> str:
    m = list(ANSWER_RE.finditer(pred))
    if m:
        return m[-1].group(1).strip()
    return pred.strip()


def em(pred: str, golds) -> int:
    if isinstance(golds, str):
        golds = [golds]
    np_ = normalize(pred)
    return int(any(normalize(g) == np_ for g in golds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="JSONL dump from EVAL_DUMP_JSONL")
    ap.add_argument("--out_json", default=None, help="Optional path to write summary JSON")
    args = ap.parse_args()

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    by_domain: dict = defaultdict(lambda: {"preds": [], "refs": [], "ems": []})

    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            domain = rec.get("domain") or "unknown"
            golds = rec.get("golden")
            if isinstance(golds, list):
                gold_ref = golds[0]
            else:
                gold_ref = str(golds)
            ans = extract_answer(rec.get("prediction", ""))
            by_domain[domain]["preds"].append(ans)
            by_domain[domain]["refs"].append(gold_ref)
            by_domain[domain]["ems"].append(em(ans, golds))

    summary = {}
    print(f"{'domain':<14}{'N':>6}{'EM':>10}{'BLEU':>10}{'R1':>10}{'R2':>10}{'R-L':>10}{'R-LSum':>10}")
    print("-" * 80)
    rouge_l_avg = []
    em_avg = []
    for domain, d in sorted(by_domain.items()):
        preds, refs, ems = d["preds"], d["refs"], d["ems"]
        em_score = sum(ems) / len(ems)
        try:
            r = rouge.compute(predictions=preds, references=refs)
        except Exception as e:
            print(f"rouge failed for {domain}: {e}", file=sys.stderr)
            r = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
        try:
            b = bleu.compute(predictions=preds, references=[[ref] for ref in refs])
            bleu_s = b.get("bleu", 0.0)
        except Exception as e:
            print(f"bleu failed for {domain}: {e}", file=sys.stderr)
            bleu_s = 0.0
        summary[domain] = {
            "n": len(preds),
            "EM": em_score,
            "BLEU": bleu_s,
            "Rouge1": r["rouge1"],
            "Rouge2": r["rouge2"],
            "RougeL": r["rougeL"],
            "RougeLSum": r["rougeLsum"],
        }
        print(
            f"{domain:<14}{len(preds):>6}{em_score:>10.4f}{bleu_s:>10.4f}"
            f"{r['rouge1']:>10.4f}{r['rouge2']:>10.4f}{r['rougeL']:>10.4f}{r['rougeLsum']:>10.4f}"
        )
        rouge_l_avg.append(r["rougeL"])
        em_avg.append(em_score)

    if rouge_l_avg:
        print("-" * 80)
        print(f"{'AVG':<14}{'':>6}{sum(em_avg)/len(em_avg):>10.4f}{'':>10}{'':>10}{'':>10}"
              f"{sum(rouge_l_avg)/len(rouge_l_avg):>10.4f}")
        summary["AVG"] = {
            "EM": sum(em_avg) / len(em_avg),
            "RougeL": sum(rouge_l_avg) / len(rouge_l_avg),
        }

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\nsummary written to {args.out_json}")


if __name__ == "__main__":
    main()
