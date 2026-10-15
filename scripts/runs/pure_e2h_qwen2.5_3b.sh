#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Pure E2H control (eta = 0): the easy-to-hard Gaussian schedule without the biased mixture.
# Paths and resources come from scripts/train_ppo.sh (GRAPH_DIR, TRAIN_FILE, OUTPUT_DIR, ...);
# extra Hydra overrides are passed through.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE=qwen2.5-3b
export SAMPLER=curriculum
export SCHEDULE=forward
export ETA_START=0.0
export ETA_END=0.0
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-pure-e2h-qwen2.5-3b}
exec bash "$HERE/../train_ppo.sh" "$@"
