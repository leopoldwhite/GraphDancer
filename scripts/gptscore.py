#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""GPTScore: percentage of answers an LLM judge marks correct, using the GRBench judge prompt.

The judge sees the question, the extracted model answer and the reference answer and
must reply with exactly one word, Yes or No. Columns and AVG follow score_rouge.py.
Requires an OpenAI-compatible endpoint; nothing is sent unless --judge-model is given.

    export OPENAI_API_KEY=...                  # your key
    export OPENAI_BASE_URL=...                 # optional, for other OpenAI-compatible servers
    python scripts/gptscore.py --run-tag graphdancer-qwen2.5-3b-step200 --seed 3 \\
        --judge-model <model-name> --cache outputs/gptscore/graphdancer-qwen2.5-3b-step200-s3.jsonl

Judgments are cached per question (resume-safe). Replies other than exactly
"Yes"/"No" (after trimming whitespace) are retried up to --max-attempts times and
otherwise counted as incorrect and flagged in the cache.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from eval_utils import COLUMNS, load_records, to_text

SYSTEM_PROMPT = "You are a generative language model evaluator."
EVAL_PROMPT = (
    "Question:{} \nModel prediction: {} \nGround truth: {}. \n"
    "Please help me judge if the model prediction is correct or not "
    "given the question and ground truth answer. "
    "Please use one word (Yes or No) to answer. Do not explain."
)


def judge(client, args, question, prediction, gold):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EVAL_PROMPT.format(question, prediction, gold)}]
    kwargs = {"model": args.judge_model, "messages": messages}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    raw = None
    for attempt in range(args.max_attempts):
        try:
            response = client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
        except Exception as error:  # network / rate-limit errors: back off and retry
            raw = f"<error: {type(error).__name__}>"
            time.sleep(min(60, 2 ** attempt))
            continue
        if raw in ("Yes", "No"):
            return raw, raw, attempt + 1
    return "No", raw, args.max_attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--judge-model", required=True, help="judge model name at your endpoint")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="OpenAI-compatible base URL (default: OPENAI_BASE_URL or the OpenAI API)")
    parser.add_argument("--temperature", type=float, default=None, help="omit to use the endpoint default")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", required=True, help="JSONL file of judgments (created / resumed)")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and count pending calls only")
    args = parser.parse_args()

    items = []
    for column, (set_name, domains) in COLUMNS.items():
        for domain in domains:
            for record in load_records(args.results_dir, args.run_tag, set_name, domain, args.seed):
                items.append({"column": column, "domain": domain, "index": int(record["extra_info"]["index"]),
                              "question": to_text(record["question"]), "prediction": to_text(record["model_answer"]),
                              "gold": to_text(record["gt_answer"])})
    cache_path = Path(args.cache)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["domain"], row["index"])] = row
    pending = [item for item in items if (item["domain"], item["index"]) not in done]
    print(f"{len(items)} questions, {len(done)} cached, {len(pending)} to judge", file=sys.stderr)
    if args.dry_run:
        return 0
    if pending:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a") as cache, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [(item, pool.submit(judge, client, args, item["question"], item["prediction"], item["gold"]))
                       for item in pending]
            for item, future in futures:
                judgment, raw, attempts = future.result()
                row = dict(item, judgment=judgment, raw=raw, attempts=attempts, valid=(raw in ("Yes", "No")),
                           judge_model=args.judge_model)
                cache.write(json.dumps(row) + "\n")
                cache.flush()
                done[(item["domain"], item["index"])] = row
    columns = {}
    for column in COLUMNS:
        rows = [done[(i["domain"], i["index"])] for i in items if i["column"] == column]
        columns[column] = {"n": len(rows), "gptscore": 100 * mean(r["judgment"] == "Yes" for r in rows),
                           "invalid_replies": sum(not r.get("valid", True) for r in rows)}
    print(json.dumps({"run_tag": args.run_tag, "decode_seed": args.seed, "judge_model": args.judge_model,
                      "columns": columns, "avg_gptscore": mean(c["gptscore"] for c in columns.values())}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
