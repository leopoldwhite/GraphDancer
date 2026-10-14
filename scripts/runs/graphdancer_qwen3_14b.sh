#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GraphDancer on Qwen3-14B (thinking mode), 2 nodes x 8 GPUs. Start a Ray cluster and export RAY_ADDRESS first (see README).
# Paths and resources come from scripts/train_ppo.sh (GRAPH_DIR, TRAIN_FILE, OUTPUT_DIR, ...);
# extra Hydra overrides are passed through.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE=qwen3-14b
export SAMPLER=curriculum
export SCHEDULE=forward
export ETA_START=0.2
export ETA_END=0.8
export NNODES=${NNODES:-2}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-graphdancer-qwen3-14b}
exec bash "$HERE/../train_ppo.sh" "$@"
