#!/usr/bin/env bash
#
# GraphDancer Stage 2 launcher: Curriculum-DPO training.
#
# Trains a DPO refinement of the Stage 1 Curriculum-PPO checkpoint over
# self-generated preference pairs, organized by the same graph-aware
# curriculum used in Stage 1.
#
# Expected pipeline:
#   1. Train Stage 1 (Curriculum-PPO) -> ${PPO_CHECKPOINT}
#   2. Sample K=8 trajectories per Academic training question from that
#      checkpoint (verl rollout in val_only mode with val_kwargs.n=8).
#   3. Run scripts/dpo/build_preference_pairs.py to lex-rank the K=8
#      trajectories and emit a pair parquet.
#   4. Run this launcher to perform DPO training.
#
# Configuration via environment variables (override as needed):

set -euo pipefail

export GRAPH_DIR=${GRAPH_DIR:-./data/processed_data}
export PAIR_PARQUET=${PAIR_PARQUET:-./data/dpo/pair_parquet/pairs.parquet}
export PPO_CHECKPOINT=${PPO_CHECKPOINT:-./checkpoints/graphdancer_curriculum_ppo/actor/global_step_200}
export OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/graphdancer_curriculum_dpo}

# DPO hyperparameters (Appendix A.2 of the paper)
BETA=${BETA:-0.1}
LEARNING_RATE=${LEARNING_RATE:-2e-7}
TOTAL_STEPS=${TOTAL_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}

# Curriculum scheduler (matches Stage 1 defaults: beta=3, sigma=0.75, eta 0.2->0.8)
CURR_BETA=${CURR_BETA:-3}
CURR_SIGMA=${CURR_SIGMA:-0.75}
ETA_START=${ETA_START:-0.2}
ETA_END=${ETA_END:-0.8}

cd "$(dirname "$0")/../.."   # repo root

python -m verl.trainer.dpo.train \
    --pair_parquet "${PAIR_PARQUET}" \
    --base_model "${PPO_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --beta "${BETA}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_steps "${TOTAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --curriculum_beta "${CURR_BETA}" \
    --curriculum_sigma "${CURR_SIGMA}" \
    --eta_start "${ETA_START}" \
    --eta_end "${ETA_END}"
