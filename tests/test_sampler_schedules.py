#!/usr/bin/env python3
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the curriculum sampler and its schedule controls.

Uses the frozen training rows (row order and level labels from
data/split701020_v1/split.jsonl, or a built train parquet via --train-parquet)
and the real sampler factory ``verl.trainer.main_ppo.create_rl_sampler``.

Checks: forward schedule endpoints, Static = mean of the forward schedule,
Reverse = forward reversed, Fast = forward compressed into T//2 updates then held,
pure easy-to-hard (eta = 0), resume equality from state_dict, uniform routing,
and rejection of unknown schedule modes.

    python tests/test_sampler_schedules.py [--train-parquet data/grbench/train/train.parquet]
"""
import argparse
import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verl.experimental.dataset.crl_e2h import E2HGaussianSampler  # noqa: E402,F401

GRAPHDANCER = dict(train_batch_size=128, buckets=["easy", "medium", "hard"],
                   difficulty_key="extra_info.level", e2h_total_training_steps=200,
                   e2h_beta=3, e2h_sigma=0.75, e2h_mix_eta_start=0.2, e2h_mix_eta_end=0.8,
                   e2h_mix_q=[0.34, 0.33, 0.33], use_e2h=True, e2h_sampler="gaussian",
                   dataloader_num_workers=0, sampler=None, shuffle=True)


def load_factory():
    """Execute create_rl_sampler from main_ppo.py without importing Ray/vLLM."""
    tree = ast.parse((ROOT / "verl/trainer/main_ppo.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_rl_sampler")
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "<sampler>", "exec"), scope)
    return scope["create_rl_sampler"]


def load_frame(train_parquet):
    if train_parquet:
        return pd.read_parquet(train_parquet)
    rows = [json.loads(line) for line in (ROOT / "data/split701020_v1/split.jsonl").read_text().splitlines()
            if line.strip()]
    rows = [r for r in rows if r["partition"] == "train"]
    return pd.DataFrame({"extra_info": [{"domain": r["domain"], "index": r["index"], "level": r["level"]}
                                        for r in rows]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-parquet", default=None)
    args = parser.parse_args()
    frame = load_frame(args.train_parquet)

    class Dataset:
        dataframe = frame

        def __len__(self):
            return len(self.dataframe)

    assert len(Dataset()) == 749
    make = load_factory()

    def sampler(**overrides):
        return make(OmegaConf.create(dict(GRAPHDANCER, **overrides)), Dataset())

    samplers = {m: sampler(e2h_schedule_mode=m) for m in ["forward", "static", "reverse", "fast"]}
    probs = {m: np.array([s._probs_for_step(t) for t in range(200)]) for m, s in samplers.items()}
    # forward: p_0 = 0.8 * g_0 + 0.2 * q, p_199 = 0.2 * g_199 + 0.8 * q
    g0 = np.exp(-np.arange(3) ** 2 / (2 * 0.75 ** 2)); g0 /= g0.sum()
    g1 = np.exp(-(2 - np.arange(3)) ** 2 / (2 * 0.75 ** 2)); g1 /= g1.sum()
    q = np.array([0.34, 0.33, 0.33])
    assert np.allclose(probs["forward"][0], 0.8 * g0 + 0.2 * q)
    assert np.allclose(probs["forward"][-1], 0.2 * g1 + 0.8 * q)
    assert np.allclose(probs["reverse"], probs["forward"][::-1])
    assert np.allclose(probs["static"], probs["forward"].mean(axis=0))
    assert np.allclose(probs["fast"][99:], probs["forward"][-1])
    assert np.allclose(probs["fast"][0], probs["forward"][0])
    for mode, s in samplers.items():
        assert np.allclose(probs[mode].sum(axis=1), 1) and (probs[mode] >= 0).all()
        assert len(list(s)) == 640  # 5 updates x 128 questions per epoch
        state = copy.deepcopy(s.state_dict())
        expected = list(s)
        restored = sampler(e2h_schedule_mode=mode)
        restored.load_state_dict(state)
        assert list(restored) == expected, mode
    pure = sampler(e2h_mix_eta_start=0.0, e2h_mix_eta_end=0.0)
    pure_probs = np.array([pure._probs_for_step(t) for t in range(200)])
    assert np.allclose(pure_probs[0], g0) and np.allclose(pure_probs[-1], g1)
    uniform = make(OmegaConf.create(dict(GRAPHDANCER, use_e2h=False)), Dataset())
    assert type(uniform).__name__ == "RandomSampler" and len(set(uniform)) == 749
    try:
        sampler(e2h_schedule_mode="typo")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown schedule mode accepted")
    print(json.dumps({"forward_initial": probs["forward"][0].round(4).tolist(),
                      "forward_final": probs["forward"][-1].round(4).tolist(),
                      "static": probs["static"][0].round(4).tolist()}))
    print("PASS: factory routing, forward endpoints, static/reverse/fast controls, eta=0, resume, rejection")
    return 0


if __name__ == "__main__":
    sys.exit(main())
