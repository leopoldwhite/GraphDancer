#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""Build the training / evaluation parquet files of the fixed 70:10:20 partition.

Inputs
  * the public GRBench question files (JSON lines, one file per graph):
      healthcare.json  computer_science.json  biology.json  chemistry.json
      materials_science.json  medicine.json  physics.json
      amazon.json  literature.json  legal.json
    e.g. ``huggingface-cli download PeterJinGo/GRBench --repo-type dataset --local-dir data/grbench_raw``
  * ``data/split701020_v1/split.jsonl``           question IDs, partition, frozen level, template id
  * ``data/split701020_v1/prompt_templates.json`` one rollout prompt per graph ({QUESTION} placeholder)

Outputs (under --out-dir, default ``data/grbench``)
  train/train.parquet                       749 Academic + Biomedical training questions
  eval/validation/<graph>.parquet           107 validation questions (7 graphs)
  eval/test/<graph>.parquet                 214 in-domain test questions (7 graphs)
  eval/ood/{amazon,goodreads,legal}.parquet 620 unseen-domain test questions
  manifest.json                             counts and canonical-record SHA-256 per file

Row order follows split.jsonl, which is the order used in our runs (the curriculum
and uniform samplers index rows by position). Each record is checked against the
per-question ``qa_sha256`` and, when present, the canonical file hashes in
``data/split701020_v1/manifest.json``. Parquet bytes depend on the pandas/pyarrow
version, so verification uses canonical JSON of the records instead.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ACADEMIC_BIOMEDICAL = ["biomedical", "dblp", "maple_Biology", "maple_Chemistry",
                       "maple_Materials_Science", "maple_Medicine", "maple_Physics"]
UNSEEN = ["amazon", "goodreads", "legal"]


def normalize_question(question):
    question = str(question)
    return question if question.endswith("?") else question.strip() + "?"


def qa_hash(question, answer):
    payload = json.dumps({"question": question, "answer": answer}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_grbench(grbench_dir, files):
    tables = {}
    for name in sorted(set(files)):
        path = grbench_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing GRBench question file: {path}")
        rows = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["qid"])] = row
        tables[name] = rows
    return tables


def build_record(entry, source, templates):
    question = normalize_question(source["question"])
    answer = str(source["answer"])
    if qa_hash(question, answer) != entry["qa_sha256"]:
        raise ValueError(f"GRBench record differs from the frozen split: {entry['domain']} qid={entry['qid']}")
    template = templates[entry["domain"]]
    return {
        "qid": str(entry["qid"]),
        "question": question,
        "answer": answer,
        "data_source": "grbench",
        "prompt": [{"role": "user", "content": template.replace("{QUESTION}", question)}],
        "ability": "fact-reasoning",
        "reward_model": {"ground_truth": {"target": answer}, "style": "rule"},
        "extra_info": {"domain": entry["domain"], "index": int(entry["index"]), "split": entry["partition"],
                       "level": entry["level"], "template_id": int(entry["template_id"])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grbench-dir", type=Path, required=True,
                        help="directory with the public GRBench question files (*.json, JSON lines)")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "split701020_v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "grbench")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    entries = [json.loads(line) for line in (args.split_dir / "split.jsonl").read_text().splitlines() if line.strip()]
    templates = json.loads((args.split_dir / "prompt_templates.json").read_text())
    expected_manifest = args.split_dir / "manifest.json"
    expected = json.loads(expected_manifest.read_text()).get("files", {}) if expected_manifest.exists() else {}
    sources = load_grbench(args.grbench_dir, [e["grbench_file"] for e in entries])

    outputs = collections.OrderedDict()
    for entry in entries:
        record = build_record(entry, sources[entry["grbench_file"]][str(entry["qid"])], templates)
        part, domain = entry["partition"], entry["domain"]
        if part == "train":
            outputs.setdefault("train/train.parquet", []).append(record)
        else:
            outputs.setdefault(f"eval/{part}/{domain}.parquet", []).append(record)

    counts = {name: len(rows) for name, rows in outputs.items()}
    assert counts["train/train.parquet"] == 749
    assert sum(n for k, n in counts.items() if k.startswith("eval/validation/")) == 107
    assert sum(n for k, n in counts.items() if k.startswith("eval/test/")) == 214
    assert sum(n for k, n in counts.items() if k.startswith("eval/ood/")) == 620

    manifest = {"version": "split701020-v1", "files": {}}
    mismatched = []
    for name, rows in outputs.items():
        digest = canonical_hash(rows)
        manifest["files"][name] = {"n": len(rows), "canonical_sha256": digest}
        if name in expected and expected[name]["canonical_sha256"] != digest:
            mismatched.append(name)
    if mismatched:
        raise SystemExit(f"canonical hash mismatch for: {mismatched}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        path = args.out_dir / name
        if path.exists() and not args.overwrite:
            raise SystemExit(f"{path} exists; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "verified_against_release_manifest": bool(expected),
                      "files": counts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
