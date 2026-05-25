#!/usr/bin/env bash
set -euo pipefail

# IMPORTANT:
# - Do NOT hardcode secrets in this repo.
# - If you use Weights & Biases or need gated model downloads, export these in your shell:
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...   # optional

# Environment (override as needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
export DATA_DIR=${DATA_DIR:-data/academicTrain_biomedicalValid}
export GRAPH_DIR=${GRAPH_DIR:-./data/graphs}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export WAND_PROJECT=${WAND_PROJECT:-GraphDancer}

# Optional base model path
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}

# Experiment naming
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo-format-e2h_gaussian_mix_newCurriculum_150steps-v3-t1}

# Resource knobs
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}
export OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints}

MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-8192}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-500}
MAX_NUM_BATCHED_TOKENS=$(( MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS ))

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    --config-name ppo_trainer_format \
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
    data.shuffle=True \
    data.use_e2h=true \
    data.e2h_sampler=gaussian \
    data.e2h_beta=3 \
    data.e2h_sigma=0.75 \
    data.e2h_mix_eta_start=0.2 \
    data.e2h_mix_eta_end=0.8 \
    data.e2h_mix_q=[0.5,0.5,0.0] \
    data.difficulty_key='extra_info.difficulty' \
    data.buckets=[easy,medium,hard] \
    data.e2h_eps=1e-8 \
    data.dataloader_num_workers=0 \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_only=false \
    trainer.eval_only=false \
    trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=200 \
    trainer.total_training_steps=150 \
    trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
    max_turns=10 \
    retriever.url=null \
    retriever.topk=3 \
    graph.graph_dir=$GRAPH_DIR/biomedical/graph.json \
    graph.dataset=biomedical \
    graph.embedder_name='sentence-transformers/all-mpnet-base-v2' \
    graph.faiss_gpu=false \
    graph.embed_cache=true \
    graph.embed_cache_dir=$GRAPH_DIR/biomedical \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    graph.use_threaded=true \
    graph.num_threads=16 \
    2>&1 | tee $EXPERIMENT_NAME.log


