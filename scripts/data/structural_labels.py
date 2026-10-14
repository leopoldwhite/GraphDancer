#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""Structural difficulty levels (Easy / Medium / Hard) used by the curriculum.

Each Academic / Biomedical question is labelled once, before training, from the
text of its GRBench question template (no model outputs are used):

  hard    the template contains an aggregation, superlative or ranking cue
          ("most", "closest", "top k", "at least", "average", ...); hard cues take
          precedence over the slot count
  medium  two or more entity slots, or one slot plus an extra constraint stated
          in prose, without an aggregation/ranking cue
  easy    one entity slot and no aggregation, ranking or extra constraint
  (unmatched templates default to medium)

The frozen labels of the 70:10:20 partition are shipped in
``data/split701020_v1/split.jsonl`` (field ``level``) and are authoritative.
``--check`` re-applies the rule to ``data/split701020_v1/question_templates.json``
and reports agreement with the shipped labels for every question that matched
a template (template_id > 0). Questions whose wording matched no template carry
template_id 0 and keep their frozen label.

Usage:
    python scripts/data/structural_labels.py --check
    python scripts/data/structural_labels.py --template 'Who is the closest collaborator with author {author_name}?'
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "split701020_v1"

# Aggregation or superlative over a set that must first be constructed.
AGGREGATION = re.compile(
    r"\bmost\b|\bclosest\b|exact same|\btop \d|at least|need to know|"
    r"\bmajority\b|\baverage\b|\bmedian\b|"
    r"\bearliest\b|\blatest\b|\boldest\b|\bnewest\b|"
    r"\bhighest\b|\blowest\b|\blargest\b|\bsmallest\b|\bfewest\b", re.I)

# Ranking / recommendation stated without a superlative word.
RANKING = re.compile(r"should be recommended|which can support|given this query|"
                     r"\bsubstitutive\b|\bcomplementary\b|exact matched", re.I)

# A second constraint stated in prose rather than as a second {slot}.
PROSE_CONSTRAINT = re.compile(r"same \w+ and (?:share|published|same)|and share similar|"
                              r"but the series is different|supported by the given", re.I)

SLOT = re.compile(r"\{[^}]+\}")


def level_of_template(text):
    """Return 'easy' | 'medium' | 'hard' for a question template string."""
    if not text:
        return "medium"
    if AGGREGATION.search(text) or RANKING.search(text):
        return "hard"
    if PROSE_CONSTRAINT.search(text):
        return "medium"
    return "medium" if len(SLOT.findall(text)) >= 2 else "easy"


def template_namespace(domain):
    """Template ids are numbered per graph family."""
    domain = str(domain)
    if domain == "biomedical":
        return "bio"
    if domain == "dblp" or domain.startswith("maple"):
        return "acad"
    return domain


def check(split_dir):
    templates = json.loads((split_dir / "question_templates.json").read_text())
    rows = [json.loads(line) for line in (split_dir / "split.jsonl").read_text().splitlines() if line.strip()]
    agree, disagree, unmatched = 0, [], collections.Counter()
    for row in rows:
        if row["partition"] == "ood":
            continue
        if int(row["template_id"]) == 0:
            unmatched[row["level"]] += 1
            continue
        key = f"{template_namespace(row['domain'])}#{row['template_id']}"
        predicted = level_of_template(templates.get(key, ""))
        if predicted == row["level"]:
            agree += 1
        else:
            disagree.append((row["domain"], row["qid"], key, row["level"], predicted))
    counts = collections.Counter((r["partition"], r["level"]) for r in rows if r["partition"] != "ood")
    print(json.dumps({
        "template_matched_questions": agree + len(disagree),
        "rule_agrees_with_frozen_label": agree,
        "disagreements": disagree[:20],
        "template_id_0_questions_frozen_labels": dict(unmatched),
        "level_counts": {f"{p}/{l}": n for (p, l), n in sorted(counts.items())},
    }, indent=2))
    return 0 if not disagree else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--check", action="store_true", help="re-apply the rule and compare with split.jsonl")
    parser.add_argument("--template", help="label a single template string")
    args = parser.parse_args()
    if args.template is not None:
        print(level_of_template(args.template))
        return 0
    return check(args.split_dir)


if __name__ == "__main__":
    sys.exit(main())
