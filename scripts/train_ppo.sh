#!/usr/bin/env bash
# Copyright 2025 GraphDancer Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Single-stage PPO training of the GraphDancer graph agent (verl, GAE critic).
#
# All settings are environment variables with repository-relative defaults; any
# additional Hydra override can be appended as an argument. Defaults reproduce the
# paper's GraphDancer run on Qwen2.5-3B-Instruct (4 GPUs, 200 updates).
#
#   BACKBONE        qwen2.5-3b (default) | qwen3-14b
#   MODEL           HF model id or local path (default: per backbone)
#   SAMPLER         curriculum (default) | uniform           (uniform = Vanilla PPO)
#   SCHEDULE        forward (default) | static | reverse | fast   (curriculum controls)
#   ETA_START/ETA_END  biased-mixture weight schedule (default 0.2 / 0.8; 0 / 0 = pure easy-to-hard)
#   Q_PRIOR         level prior q over [easy,medium,hard] (default [0.34,0.33,0.33])
#   TRAIN_FILE      training parquet (default data/grbench/train/train.parquet)
#   GRAPH_DIR       GRBench graph environments (default data/graphs)
#   OUTPUT_DIR      root for checkpoints / rollout dumps / logs (default outputs)
#   EXPERIMENT_NAME run name (default derived from the settings above)
#   TOTAL_STEPS     PPO updates (default 200); SAVE_FREQ (25); KEEP_CKPTS (2)
#   NNODES, N_GPUS_PER_NODE   (default 1 x 4 for 3B, 2 x 8 for 14B; multi-node needs RAY_ADDRESS)
#   ROLLOUT_SEED    vLLM sampling seed for training rollouts (default 2)
#   FREE_CACHE_ENGINE  vLLM sleep mode between rollout and update (default true; see README)
#   HF_EXPORT_STEPS optional, e.g. "100 150 175 200": checkpoint rotation will refuse to delete
#                   these steps until scripts/merge_checkpoint.sh has exported them
#   DRY_RUN=1       print the composed Hydra config and exit
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

BACKBONE=${BACKBONE:-qwen2.5-3b}
SAMPLER=${SAMPLER:-curriculum}
SCHEDULE=${SCHEDULE:-forward}
ETA_START=${ETA_START:-0.2}
ETA_END=${ETA_END:-0.8}
Q_PRIOR=${Q_PRIOR:-[0.34,0.33,0.33]}
TRAIN_FILE=${TRAIN_FILE:-data/grbench/train/train.parquet}
export GRAPH_DIR=${GRAPH_DIR:-$ROOT/data/graphs}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
TOTAL_STEPS=${TOTAL_STEPS:-200}
SAVE_FREQ=${SAVE_FREQ:-25}
KEEP_CKPTS=${KEEP_CKPTS:-2}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
export VERL_ROLLOUT_SEED=${ROLLOUT_SEED:-2}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-24576}      # rolling context during training
export MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-500}    # new tokens per generation
MAX_NUM_BATCHED_TOKENS=$(( MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS ))

case "$SAMPLER" in
  curriculum)
    case "$SCHEDULE" in forward|static|reverse|fast) ;; *) echo "unknown SCHEDULE=$SCHEDULE" >&2; exit 2 ;; esac
    SAMPLER_ARGS=(
      data.use_e2h=true data.e2h_sampler=gaussian
      data.e2h_beta=3 data.e2h_sigma=0.75 data.e2h_eps=1e-8
      "data.e2h_mix_eta_start=$ETA_START" "data.e2h_mix_eta_end=$ETA_END" "data.e2h_mix_q=$Q_PRIOR"
      "data.e2h_schedule_mode=$SCHEDULE"
      data.difficulty_key=extra_info.level "data.buckets=[easy,medium,hard]"
    )
    TAG=curriculum-$SCHEDULE ;;
  uniform)
    SAMPLER_ARGS=(data.use_e2h=false data.shuffle=True)
    TAG=uniform ;;
  *) echo "unknown SAMPLER=$SAMPLER (curriculum|uniform)" >&2; exit 2 ;;
esac

case "$BACKBONE" in
  qwen2.5-3b)
    MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
    NNODES=${NNODES:-1}; N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
    BACKBONE_ARGS=(
      data.max_start_length=1024
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
      critic.ppo_micro_batch_size_per_gpu=8
      actor_rollout_ref.rollout.tensor_model_parallel_size=1
    ) ;;
  qwen3-14b)
    MODEL=${MODEL:-Qwen/Qwen3-14B}
    NNODES=${NNODES:-2}; N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    BACKBONE_ARGS=(
      # Qwen3 thinking mode; "<think>" is supplied after each observation (masked from the loss)
      +data.apply_chat_template_kwargs.enable_thinking=true
      data.max_start_length=8192
      '+graph.continuation_prefix="<think>"'
      # explicit warmup updates (= int(0.285*200) and int(0.015*200))
      actor_rollout_ref.actor.optim.lr_warmup_steps=56
      critic.optim.lr_warmup_steps=3
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
      actor_rollout_ref.ref.log_prob_micro_batch_size=null
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
      actor_rollout_ref.ref.ulysses_sequence_parallel_size=2
      actor_rollout_ref.rollout.log_prob_micro_batch_size=null
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
      actor_rollout_ref.rollout.tensor_model_parallel_size=2
      actor_rollout_ref.rollout.max_num_seqs=16
      critic.ppo_micro_batch_size_per_gpu=1 critic.forward_micro_batch_size=null
      critic.forward_micro_batch_size_per_gpu=1 critic.ulysses_sequence_parallel_size=2
      actor_rollout_ref.rollout.top_p=0.8 actor_rollout_ref.rollout.top_k=20
      ++actor_rollout_ref.rollout.min_p=0.0
      actor_rollout_ref.rollout.val_kwargs.do_sample=true
      actor_rollout_ref.rollout.val_kwargs.temperature=0.7
      actor_rollout_ref.rollout.val_kwargs.top_p=0.8
      actor_rollout_ref.rollout.val_kwargs.top_k=20
    ) ;;
  *) echo "unknown BACKBONE=$BACKBONE (qwen2.5-3b|qwen3-14b)" >&2; exit 2 ;;
esac

EXPERIMENT_NAME=${EXPERIMENT_NAME:-graphdancer-$BACKBONE-$TAG-t$TOTAL_STEPS}
CKPT_DIR=$OUTPUT_DIR/checkpoints/$EXPERIMENT_NAME
ROLLOUT_DIR=$OUTPUT_DIR/rollouts/$EXPERIMENT_NAME
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p "$LOG_DIR"

# Ray: a local cluster by default; set RAY_ADDRESS to join a running multi-node cluster.
RAY_CPUS=${RAY_NUM_CPUS:-null}
if [ -n "${RAY_ADDRESS:-}" ]; then RAY_CPUS=null; fi
if [ "$NNODES" -gt 1 ] && [ -z "${RAY_ADDRESS:-}" ] && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "NNODES=$NNODES requires a running Ray cluster; export RAY_ADDRESS=<head-ip>:<port> (see README)" >&2
  exit 2
fi

# Optional: protect checkpoints that still need an HF export (see scripts/merge_checkpoint.sh).
if [ -n "${HF_EXPORT_STEPS:-}" ] && [ "${DRY_RUN:-0}" != 1 ]; then
  mkdir -p "$CKPT_DIR"
  HF_ROOT=$(cd "$OUTPUT_DIR" && pwd)/hf/$EXPERIMENT_NAME
  python3 - "$CKPT_DIR/.required_hf_exports.json" "$HF_ROOT" $HF_EXPORT_STEPS <<'PY'
import json, sys
path, hf_root, steps = sys.argv[1], sys.argv[2], sorted({int(s) for s in sys.argv[3:]})
json.dump({"steps": steps, "hf_root": hf_root}, open(path, "w"))
print(f"checkpoint rotation will keep steps {steps} until exported to {hf_root}")
PY
fi

EXTRA=()
if [ "${DRY_RUN:-0}" = 1 ]; then EXTRA=(--cfg job); fi

echo "==== $EXPERIMENT_NAME: backbone=$BACKBONE model=$MODEL sampler=$SAMPLER schedule=$SCHEDULE" \
     "eta=$ETA_START->$ETA_END q=$Q_PRIOR steps=$TOTAL_STEPS gpus=${NNODES}x${N_GPUS_PER_NODE}" \
     "rollout_seed=$VERL_ROLLOUT_SEED"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
  --config-name ppo_trainer_format \
  ray_kwargs.ray_init.num_cpus=$RAY_CPUS \
  data.train_files="$TRAIN_FILE" \
  data.val_files=null \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=128 \
  data.val_batch_size=128 \
  data.max_prompt_length=$MAX_PROMPT_TOKENS \
  data.max_response_length=$MAX_RESPONSE_TOKENS \
  data.max_obs_length=1024 \
  data.shuffle_train_dataloader=True \
  data.shuffle=True \
  data.dataloader_num_workers=0 \
  "${SAMPLER_ARGS[@]}" \
  algorithm.adv_estimator=gae \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=3.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path="$MODEL" \
  critic.optim.lr=1e-5 \
  critic.optim.lr_warmup_steps_ratio=0.015 \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=true \
  critic.model.fsdp_config.param_offload=true \
  critic.model.fsdp_config.optimizer_offload=true \
  critic.cliprange_value=0.5 \
  critic.loss_agg_mode=token-mean \
  reward_model.structure_format_score=0.2 \
  reward_model.final_format_score=0.1 \
  reward_model.retrieval_score=0 \
  "${BACKBONE_ARGS[@]}" \
  trainer.critic_warmup=0 \
  trainer.logger="['console']" \
  trainer.val_only=false \
  trainer.eval_only=false \
  trainer.val_before_train=false \
  trainer.test_freq=0 \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$NNODES \
  trainer.save_freq=$SAVE_FREQ \
  trainer.max_actor_ckpt_to_keep=$KEEP_CKPTS \
  trainer.max_critic_ckpt_to_keep=$KEEP_CKPTS \
  trainer.resume_mode=auto \
  trainer.project_name=graphdancer \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.total_epochs=$TOTAL_STEPS \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  max_turns=10 \
  retriever.url=null \
  retriever.topk=3 \
  graph.embedder_name=sentence-transformers/all-mpnet-base-v2 \
  graph.faiss_gpu=false \
  graph.embed_cache=true \
  graph.use_threaded=true \
  graph.num_threads=16 \
  "$@" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG_DIR/$EXPERIMENT_NAME.train.log"
