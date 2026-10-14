#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Static control: fixed level proportions equal to the mean of the GraphDancer schedule over 200 updates.
# Paths and resources come from scripts/train_ppo.sh (GRAPH_DIR, TRAIN_FILE, OUTPUT_DIR, ...);
# extra Hydra overrides are passed through.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE=qwen2.5-3b
export SAMPLER=curriculum
export SCHEDULE=static
export ETA_START=0.2
export ETA_END=0.8
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-static-qwen2.5-3b}
exec bash "$HERE/../train_ppo.sh" "$@"
