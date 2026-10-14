#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reverse control: the GraphDancer schedule presented from hard to easy.
# Paths and resources come from scripts/train_ppo.sh (GRAPH_DIR, TRAIN_FILE, OUTPUT_DIR, ...);
# extra Hydra overrides are passed through.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE=qwen2.5-3b
export SAMPLER=curriculum
export SCHEDULE=reverse
export ETA_START=0.2
export ETA_END=0.8
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-reverse-qwen2.5-3b}
exec bash "$HERE/../train_ppo.sh" "$@"
