#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only launcher aligned to VERL's PPO-format entrypoint.
# Ported from refRepos/Graph-RL/eval.sh and adapted to:
# - use verl.trainer.main_ppo_format
# - use ppo_trainer_format.yaml
# - prefer *per_gpu micro-batch knobs consistent with VERL

# Environment (override as needed)
# Optional (never hardcode secrets in repo):
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...   # optional

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export GPUS_PER_NODE=${GPUS_PER_NODE:-1}

export GRAPH_DIR=${GRAPH_DIR:-./data/graphs}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export WAND_PROJECT=${WAND_PROJECT:-GraphDancer}
# VLLM + Qwen-2.5 sometimes prefers xformers attention
# export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# Model and experiment naming
# export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
export BASE_MODEL=${BASE_MODEL:-models/ppo-format-vanilla}
# export BASE_MODEL=${BASE_MODEL:-models/ppo-format-e2h_gaussian_mix_newCurriculum_200steps-v3-t2}
# export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_3b-instruct}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo-format-vanilla}
# export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo-format-e2h_gaussian_mix_newCurriculum_200steps-v3-t2}

export OUTPUT_DIR=${OUTPUT_DIR:-verl_checkpoints}

# Resources
export NNODES=${NNODES:-1}

# Token length settings used by vLLM rollout
# Keep max_num_batched_tokens >= (max_prompt_length + max_response_length)
# to satisfy chunked prefill requirements.
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-8192}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-500}
MAX_NUM_BATCHED_TOKENS=$(( MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS ))

# Domain to evaluate (used for both dataset path and graph selection)
DOMAIN=${DOMAIN:-biomedical}

# Dataset directory (must contain train.parquet/test.parquet)
export DATA_DIR=${DATA_DIR:-data/grbench_fewshot_merged_${DOMAIN}}

  PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    --config-name evaluation_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=${MAX_PROMPT_TOKENS} \
    data.max_response_length=${MAX_RESPONSE_TOKENS} \
    data.max_start_length=1024 \
    data.max_obs_length=1024 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=true \
    trainer.val_only=true \
    trainer.eval_only=true \
    trainer.trace_save_txt=true \
    trainer.trace_save_json=true \
    trainer.trace_include_in_jsonl=true \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=150 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
    max_turns=10 \
    retriever.url=null \
    retriever.topk=3 \
    graph.graph_dir=$GRAPH_DIR/${DOMAIN}/graph.json \
    graph.dataset=${DOMAIN} \
    graph.embedder_name='sentence-transformers/all-mpnet-base-v2' \
    graph.faiss_gpu=false \
    graph.embed_cache=true \
    graph.embed_cache_dir=$GRAPH_DIR/${DOMAIN} \
    graph.use_threaded=true \
    graph.num_threads=16 \
    2>&1 | tee ${EXPERIMENT_NAME}.log
