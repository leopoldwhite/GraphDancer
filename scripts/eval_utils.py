# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for reading evaluation outputs written by scripts/eval_ppo.sh."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "split701020_v1" / "split.jsonl"

ACADEMIC_BIOMEDICAL = ["biomedical", "dblp", "maple_Biology", "maple_Chemistry",
                       "maple_Materials_Science", "maple_Medicine", "maple_Physics"]
SET_DOMAINS = {"validation": ACADEMIC_BIOMEDICAL, "test": ACADEMIC_BIOMEDICAL,
               "ood": ["amazon", "goodreads", "legal"]}
# Paper columns: pooled in-domain test, then the three unseen domains.
COLUMNS = {"in_domain": ("test", ACADEMIC_BIOMEDICAL), "e_commerce": ("ood", ["amazon"]),
           "literature": ("ood", ["goodreads"]), "legal": ("ood", ["legal"])}


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value)


def expected_ids(set_name, domain):
    """(domain, index) identities of a partition/domain according to the frozen split file."""
    partition = {"validation": "validation", "test": "test", "ood": "ood"}[set_name]
    ids = set()
    for line in SPLIT_FILE.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["partition"] == partition and row["domain"] == domain:
                ids.add((row["domain"], int(row["index"])))
    return ids


def result_path(results_dir, run_tag, set_name, domain, seed):
    exp = f"{run_tag}__{set_name}__{domain}__s{seed}"
    return Path(results_dir) / exp / domain.replace("maple_", "maple-") / "results.jsonl"


def load_records(results_dir, run_tag, set_name, domain, seed):
    """Records of one evaluation run, checked for completeness against the split file."""
    path = result_path(results_dir, run_tag, set_name, domain, seed)
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation output: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [(r["extra_info"]["domain"], int(r["extra_info"]["index"])) for r in records]
    expected = expected_ids(set_name, domain)
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"{path}: {len(set(ids))} unique questions, expected {len(expected)} for {set_name}/{domain}")
    return records
