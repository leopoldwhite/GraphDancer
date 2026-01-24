#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   scripts/build_graph_embeddings.sh [CUDA_VISIBLE_DEVICES] [BATCH_SIZE] [CONFIG_PATH]
#
# Examples:
#   # Use GPU 0 only, batch size 128
#   scripts/build_graph_embeddings.sh 0 128
#   # Use GPU 0 and 1, batch size 256, custom config
#   scripts/build_graph_embeddings.sh 0,1 256 verl/trainer/config/ppo_trainer.yaml

CUDA_DEVICES="${1:-0,1,2,3,4,5,6}"
BATCH_SIZE="${2:-1024}"
CONFIG_PATH="${3:-verl/trainer/config/ppo_trainer_format.yaml}"

# Resolve project root as the parent of this script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[build-graphs] CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "[build-graphs] batch_size=${BATCH_SIZE}"
echo "[build-graphs] config=${CONFIG_PATH}"

# Parse GPU list (comma-separated)
IFS=',' read -ra GPU_IDS <<< "${CUDA_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

if [[ "${NUM_GPUS}" -eq 0 ]]; then
  echo "[build-graphs] ERROR: No CUDA devices specified."
  exit 1
fi

echo "[build-graphs] Detected ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"

# Ask Python script for all graph aliases
mapfile -t GRAPH_ALIASES < <(python -m scripts.build_graph_embeddings \
  --config "${CONFIG_PATH}" \
  --list-graphs)

NUM_GRAPHS="${#GRAPH_ALIASES[@]}"

if [[ "${NUM_GRAPHS}" -eq 0 ]]; then
  echo "[build-graphs] No graphs found in config; nothing to do."
  exit 0
fi

echo "[build-graphs] Found ${NUM_GRAPHS} graph(s): ${GRAPH_ALIASES[*]}"

# Assign graphs to GPUs in round-robin fashion and launch per-GPU workers
PIDS=()
for ((i = 0; i < NUM_GPUS; i++)); do
  GPU="${GPU_IDS[$i]}"
  GRAPHS_FOR_GPU=()
  for ((j = 0; j < NUM_GRAPHS; j++)); do
    if (( j % NUM_GPUS == i )); then
      GRAPHS_FOR_GPU+=("${GRAPH_ALIASES[$j]}")
    fi
  done

  if [[ "${#GRAPHS_FOR_GPU[@]}" -eq 0 ]]; then
    continue
  fi

  echo "[build-graphs] GPU ${GPU}: graphs ${GRAPHS_FOR_GPU[*]}"

  CUDA_VISIBLE_DEVICES="${GPU}" python -m scripts.build_graph_embeddings \
    --config "${CONFIG_PATH}" \
    --use-gpu 0 \
    --batch-size "${BATCH_SIZE}" \
    --graphs "${GRAPHS_FOR_GPU[@]}" \
    --save-index &

  PIDS+=($!)
done

echo "[build-graphs] Launched ${#PIDS[@]} worker process(es). Waiting for completion..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[build-graphs] All workers finished."
