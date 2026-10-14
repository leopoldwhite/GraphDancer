#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate a completed indexed HF checkpoint without loading tensor data."""
import json
from pathlib import Path

from safetensors import safe_open


def check(directory, step):
    directory = Path(directory)
    if (directory / '.in_progress').exists():
        raise ValueError('merge is still in progress')
    if (directory / 'STEP.txt').read_text().strip() != str(step):
        raise ValueError('STEP.txt does not match the requested checkpoint')
    weights = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    if not isinstance(weights, dict) or not weights:
        raise ValueError('empty or invalid HF weight map')
    for name in set(weights.values()):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('invalid shard path in HF index')
        with safe_open(str(directory / name), framework='numpy') as shard:
            if not {key for key, value in weights.items() if value == name} <= set(shard.keys()):
                raise ValueError(f'indexed tensors missing from {name}')

