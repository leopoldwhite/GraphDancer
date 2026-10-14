#!/usr/bin/env bash
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Verl-native evaluation: the same multi-turn rollout loop as training, run in
# validation-only mode, one graph per run and one run per (domain, decode seed).
#
#   MODEL_PATH   HF checkpoint directory (from scripts/merge_checkpoint.sh) or a HF model id   [required]
#   RUN_TAG      name for this evaluation, e.g. graphdancer-qwen2.5-3b-step200                  [required]
#   SETS         any of: validation test ood            (default "test ood")
#                validation = 107 questions (7 graphs), test = 214 in-domain questions (7 graphs),
#                ood = amazon (E-commerce) 200, goodreads (Literature) 240, legal (Legal) 180
#   SEEDS        decode seed(s) (VERL_ROLLOUT_SEED) (default 1)
#   BACKBONE     qwen2.5-3b (default) | qwen3-14b
#   EVAL_DATA_DIR  per-domain parquets (default data/grbench/eval)
#   GRAPH_DIR    GRBench graph environments (default data/graphs)
#   N_GPUS       GPUs on this node (default 4 for 3B, 8 for 14B)
#   RUN_TIMEOUT  optional per-run timeout in seconds (default: none)
#   DRY_RUN=1    print the composed Hydra config of the first run and exit
#
# Output: results/<RUN_TAG>__<set>__<domain>__s<seed>/<domain>/results.jsonl
#         (maple_X is written as maple-X), with question, gt_answer, model_answer and the trace.
# Complete runs are skipped, so the script can be re-run after an interruption.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL_PATH:?set MODEL_PATH to a merged HF checkpoint directory or model id}"
: "${RUN_TAG:?set RUN_TAG, e.g. graphdancer-qwen2.5-3b-step200}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "RUN_TAG may contain only [A-Za-z0-9._-]" >&2; exit 2; }
SETS=${SETS:-"test ood"}
SEEDS=${SEEDS:-"1"}
BACKBONE=${BACKBONE:-qwen2.5-3b}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-data/grbench/eval}
GRAPH_DIR=${GRAPH_DIR:-$ROOT/data/graphs}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-30000}        # rolling context at evaluation
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-500}
MAX_NUM_BATCHED_TOKENS=$(( MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS ))
LOG_DIR=${LOG_DIR:-outputs/logs}
mkdir -p "$LOG_DIR"

case "$BACKBONE" in
  qwen2.5-3b)
    N_GPUS=${N_GPUS:-4}
    MODEL_ARGS=(data.max_start_length=1024 actor_rollout_ref.rollout.tensor_model_parallel_size=1) ;;
  qwen3-14b)
    N_GPUS=${N_GPUS:-8}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    MODEL_ARGS=(
      +data.apply_chat_template_kwargs.enable_thinking=true
      data.max_start_length=8192
      '+graph.continuation_prefix="<think>"'
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
      actor_rollout_ref.ref.log_prob_micro_batch_size=null
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
      actor_rollout_ref.rollout.log_prob_micro_batch_size=null
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
      actor_rollout_ref.rollout.tensor_model_parallel_size=2
      actor_rollout_ref.rollout.max_num_seqs=16
      critic.ppo_micro_batch_size_per_gpu=1 critic.forward_micro_batch_size=null
      critic.forward_micro_batch_size_per_gpu=1
    ) ;;
  *) echo "unknown BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

domains_for() {
  case "$1" in
    validation|test) echo "biomedical dblp maple_Biology maple_Chemistry maple_Materials_Science maple_Medicine maple_Physics" ;;
    ood) echo "amazon goodreads legal" ;;
    *) echo "unknown set: $1" >&2; return 2 ;;
  esac
}

FAILED=""
for SET in $SETS; do
  DOMS=$(domains_for "$SET") || exit 2
  for DOMAIN in $DOMS; do
    case "$DOMAIN" in
      maple_*) G_DIR=$GRAPH_DIR/maple/${DOMAIN#maple_}; G_DATASET=maple ;;
      *)       G_DIR=$GRAPH_DIR/$DOMAIN; G_DATASET=$DOMAIN ;;
    esac
    PARQ=$EVAL_DATA_DIR/$SET/$DOMAIN.parquet
    [ -f "$PARQ" ] || { echo "!! missing $PARQ (run scripts/data/build_parquets.py)"; FAILED="$FAILED $SET/$DOMAIN"; continue; }
    if [ "${DRY_RUN:-0}" != 1 ] && [ ! -f "$G_DIR/graph.json" ]; then
      echo "!! missing $G_DIR/graph.json"; FAILED="$FAILED $SET/$DOMAIN"; continue
    fi
    # One batch per domain: batch size = number of questions in the domain.
    NQ=$(python3 -c "import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))" "$PARQ")
    for SEED in $SEEDS; do
      EXP="${RUN_TAG}__${SET}__${DOMAIN}__s${SEED}"
      RESULT="results/$EXP/${DOMAIN/maple_/maple-}/results.jsonl"
      if [ -f "$RESULT" ] && [ "$(wc -l < "$RESULT")" -ge "$NQ" ]; then
        echo "---- $SET/$DOMAIN seed=$SEED complete, skipping"; continue
      fi
      rm -f "$RESULT"
      echo "---- $SET/$DOMAIN seed=$SEED ($NQ questions) $(date)"
      TIMEOUT=()
      [ -z "${RUN_TIMEOUT:-}" ] || TIMEOUT=(timeout "$RUN_TIMEOUT")
      EXTRA=()
      [ "${DRY_RUN:-0}" != 1 ] || EXTRA=(--cfg job)
      VERL_ROLLOUT_SEED=$SEED PYTHONUNBUFFERED=1 "${TIMEOUT[@]}" python3 -m verl.trainer.main_ppo_format \
        --config-name evaluation_ppo_format \
        data.train_files="$PARQ" \
        data.val_files="$PARQ" \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=$NQ \
        data.val_batch_size=$NQ \
        data.max_prompt_length=$MAX_PROMPT_TOKENS \
        data.max_response_length=$MAX_RESPONSE_TOKENS \
        data.max_obs_length=1024 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=gae \
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.no_think_rl=false \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.temperature=0.7 \
        critic.optim.lr=1e-5 \
        critic.optim.lr_warmup_steps_ratio=0.015 \
        critic.model.path="$MODEL_PATH" \
        critic.model.enable_gradient_checkpointing=true \
        critic.model.use_remove_padding=True \
        critic.ppo_micro_batch_size_per_gpu=16 \
        critic.model.fsdp_config.param_offload=true \
        critic.model.fsdp_config.optimizer_offload=true \
        trainer.critic_warmup=0 \
        trainer.logger="['console']" \
        trainer.val_before_train=true \
        trainer.val_only=true \
        trainer.eval_only=true \
        trainer.trace_include_in_jsonl=true \
        trainer.trace_save_txt=${TRACE_FILES:-false} \
        trainer.trace_save_json=${TRACE_FILES:-false} \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.project_name=graphdancer-eval \
        trainer.experiment_name="$EXP" \
        trainer.total_epochs=1 \
        trainer.resume_mode=disable \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=outputs/eval_tmp/"$EXP" \
        max_turns=10 \
        retriever.url=null \
        retriever.topk=3 \
        graph.graph_dir="$G_DIR/graph.json" \
        graph.dataset="$G_DATASET" \
        graph.embedder_name=sentence-transformers/all-mpnet-base-v2 \
        graph.faiss_gpu=false \
        graph.embed_cache=true \
        graph.embed_cache_dir="$G_DIR" \
        graph.use_threaded=true \
        graph.num_threads=16 \
        "${MODEL_ARGS[@]}" \
        "$@" \
        "${EXTRA[@]}" \
        > "$LOG_DIR/$EXP.eval.log" 2>&1
      rc=$?
      if [ "${DRY_RUN:-0}" = 1 ]; then cat "$LOG_DIR/$EXP.eval.log"; exit $rc; fi
      if [ $rc -ne 0 ] || [ ! -f "$RESULT" ] || [ "$(wc -l < "$RESULT")" -lt "$NQ" ]; then
        echo "!! $SET/$DOMAIN seed=$SEED failed (rc=$rc); see $LOG_DIR/$EXP.eval.log"
        FAILED="$FAILED ${SET}/${DOMAIN}/s${SEED}"
      fi
    done
  done
done
[ -z "$FAILED" ] || { echo "!! failed runs:$FAILED"; exit 1; }
echo "all evaluation runs complete: results/${RUN_TAG}__*"
