# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main entry for PPO-format training migrated from Graph-RL to latest VERL.

Notes:
- Uses latest RayPPOTrainer from VERL.
- Adapts Role import to `verl.trainer.ppo.utils`.
- RewardManager implements return_dict interface expected by latest trainer.
"""

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.reward_score import qa_em, qa_em_format  # modules provided under verl.utils.reward_score


def _select_rm_score_fn(data_source):
    # Datasets covered in Graph-RL format scoring
    if data_source in [
        'nq',
        'triviaqa',
        'popqa',
        'web_questions',
        'hotpotqa',
        '2wikimultihopqa',
        'musique',
        'bamboogle',
        'strategyqa',
        'grbench',
    ]:
        return qa_em_format.compute_score_em
    else:
        raise NotImplementedError


class RewardManager:
    """Graph-RL style reward manager with format-aware EM scoring.

    Implements both tensor-only return and return_dict API expected by VERL trainer.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        structure_format_score=0.0,
        final_format_score=0.0,
        retrieval_score=0.0,
        format_score=0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score

    def __call__(self, data: DataProto, return_dict: bool = False):
        # passthrough if precomputed rm scores exist
        if 'rm_scores' in data.batch.keys():
            reward_tensor = data.batch['rm_scores']
            return {"reward_tensor": reward_tensor} if return_dict else reward_tensor

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode only responses
            sequences_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                structure_format_score=self.structure_format_score,
                final_format_score=self.final_format_score,
                retrieval_score=self.retrieval_score,
                format_score=self.format_score,
            )

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return {"reward_tensor": reward_tensor} if return_dict else reward_tensor


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    # Initialize Ray (local cluster) if not already initialized
    if not ray.is_initialized():
        # Merge user-provided runtime_env if any
        default_runtime_env = {
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
            }
        }
        ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from pprint import pprint
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    # Resolve and print config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs if needed
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # tokenizer
    tokenizer = hf_tokenizer(local_path)

    # select worker classes and worker group by strategy
    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup as RayWG
        ray_worker_group_cls = RayWG
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup as RayWG
        ray_worker_group_cls = RayWG
    else:
        raise NotImplementedError

    # Import ResourcePoolManager from latest ray_trainer
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    # add reward model role when enabled
    if config.reward_model.enable:
        if config.reward_model.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    global_pool_id = 'global_pool'
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    if Role.RewardModel in role_worker_mapping:
        mapping[Role.RewardModel] = global_pool_id

    # Reward managers
    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        structure_format_score=config.reward_model.structure_format_score,
        final_format_score=config.reward_model.final_format_score,
        retrieval_score=config.reward_model.retrieval_score,
    )
    # For validation, set all format weights to zero to use strict EM
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        structure_format_score=0,
        final_format_score=0,
        retrieval_score=0,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()

