"""Curriculum-DPO trainer.

This package implements Stage 2 (Curriculum-DPO) of the GraphDancer two-stage
post-training pipeline. It adds DPO refinement on top of the Stage 1 PPO
checkpoint without modifying any existing PPO code. It reuses:

- the E2H biased-mixture curriculum sampler at
  ``verl/experimental/dataset/crl_e2h.py``
- the rule-based reward decomposition (EM/VF/AP) at
  ``verl/utils/reward_score/qa_em_format.py``
- trace-level diagnostics (evidence hit, loop-limit, invalid_tool) inlined
  in ``scripts/dpo/pair_yield.py``

Entry point: ``verl/trainer/dpo/train.py`` (argparse-driven).
"""
