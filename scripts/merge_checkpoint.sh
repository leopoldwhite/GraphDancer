#!/usr/bin/env bash
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Export FSDP actor checkpoints to Hugging Face format for evaluation.
#
#   scripts/merge_checkpoint.sh EXPERIMENT_NAME STEP [STEP ...]
#   WATCH=1 scripts/merge_checkpoint.sh EXPERIMENT_NAME 100 150 175 200   # run next to training
#
# Reads  $OUTPUT_DIR/checkpoints/<EXPERIMENT_NAME>/global_step_<STEP>/actor
# Writes $OUTPUT_DIR/hf/<EXPERIMENT_NAME>/step_<STEP>  (with STEP.txt; verified)
#
# Training keeps only KEEP_CKPTS FSDP checkpoints (default 2). To screen several
# checkpoints on the validation set, either run this script with WATCH=1 while the
# job trains, or start training with HF_EXPORT_STEPS="..." so that checkpoint
# rotation refuses to delete a listed step before its export is verified.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ $# -ge 2 ] || { echo "usage: $0 EXPERIMENT_NAME STEP [STEP ...]" >&2; exit 2; }
EXP=$1; shift
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
SRC=$OUTPUT_DIR/checkpoints/$EXP
DST=$OUTPUT_DIR/hf/$EXP
POLL=${POLL_SECONDS:-120}

merge_one() {  # returns 0 when step_<s> is verified, 1 when its source is not ready yet
  local s=$1 out=$DST/step_$1 latest
  if python3 scripts/check_merged_checkpoint.py "$out" "$s" >/dev/null 2>&1; then return 0; fi
  latest=$(cat "$SRC/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  [ -d "$SRC/global_step_$s/actor" ] && [ "$latest" -ge "$s" ] || return 1
  mkdir -p "$out"; touch "$out/.in_progress"
  (
    # Shared lock: checkpoint rotation (exclusive) waits until this read completes.
    exec {LOCK}>>"$SRC/.checkpoint_retention.lock"
    flock -s "$LOCK"
    python3 -m verl.model_merger merge --backend fsdp \
      --local_dir "$SRC/global_step_$s/actor" --target_dir "$out" --trust-remote-code
  )
  echo "$s" > "$out/STEP.txt"
  rm -f "$out/.in_progress"
  python3 scripts/check_merged_checkpoint.py "$out" "$s"
}

PENDING=("$@")
while :; do
  REMAINING=()
  for s in "${PENDING[@]}"; do
    [[ "$s" =~ ^[0-9]+$ ]] || { echo "invalid step: $s" >&2; exit 2; }
    merge_one "$s" || REMAINING+=("$s")
  done
  [ ${#REMAINING[@]} -eq 0 ] && { echo "exported: $DST/step_{$(IFS=,; echo "$*")}"; exit 0; }
  [ "${WATCH:-0}" = 1 ] || { echo "not available yet: ${REMAINING[*]} (source $SRC)" >&2; exit 1; }
  PENDING=("${REMAINING[@]}")
  sleep "$POLL"
done
