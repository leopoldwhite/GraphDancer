#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Vanilla PPO control: uniform question sampling; data, reward, optimizer and 200-update budget as GraphDancer.
# Paths and resources come from scripts/train_ppo.sh (GRAPH_DIR, TRAIN_FILE, OUTPUT_DIR, ...);
# extra Hydra overrides are passed through.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BACKBONE=qwen2.5-3b
export SAMPLER=uniform
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-vanilla-ppo-qwen2.5-3b}
exec bash "$HERE/../train_ppo.sh" "$@"
