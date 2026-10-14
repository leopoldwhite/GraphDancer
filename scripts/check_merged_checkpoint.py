#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate a merged HF checkpoint (index, shards, STEP.txt) without loading the model."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

helper = Path(__file__).resolve().parents[1] / "verl/utils/checkpoint/hf_checkpoint.py"
spec = spec_from_file_location("hf_checkpoint", helper)
module = module_from_spec(spec)
spec.loader.exec_module(module)
check = module.check

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_merged_checkpoint.py DIRECTORY EXPECTED_STEP")
    try:
        check(sys.argv[1], sys.argv[2])
    except Exception as error:
        print(f"Checkpoint not ready: {sys.argv[1]}: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {sys.argv[1]} (step {sys.argv[2]})")
