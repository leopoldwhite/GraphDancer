import torch
import re
from collections import defaultdict
import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil

# Tools ported from GraphAgent
from .tools import graph_funcs as graph_funcs_mod


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = None  # unused for graph but kept for compatibility
    topk: int = 3  # unused for graph but kept for compatibility

    # Graph-specific config (single graph backward-compat)
    graph_dir: Optional[str] = None
    dataset: Optional[str] = None
    embedder_name: str = 'sentence-transformers/all-mpnet-base-v2'
    faiss_gpu: bool = False
    embed_cache: bool = True
    embed_cache_dir: Optional[str] = None
    node_text_keys: Optional[Dict[str, List[str]]] = None
    # Multi-graph routing support. If provided, init loads all and uses router.
    # Expected format:
    # {
    #   "biomedical": {"graph_dir": "/path/biomedical/graph.json", "dataset": "biomedical", "embed_cache_dir": "/path/biomedical"},
    #   "dblp": {"graph_dir": "/path/dblp/graph.json", "dataset": "dblp", "embed_cache_dir": "/path/dblp"},
    #   ...
    # }
    domain_graphs: Optional[Dict[str, Dict[str, str]]] = None
    # Trace collection (disabled by default to avoid training overhead).
    # When enabled, per-sample traces (per turn: prediction/action/observation) are stored into DataProto.non_tensor_batch["__trace__"].
    collect_trace: bool = False


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )

        self._init_graph_env()

    def _init_graph_env(self) -> None:
        """Load graph tools.
        - If `domain_graphs` is provided in config, preload all graphs and build a router mapping domain -> tools.
        - Otherwise, load a single graph as before for backward compatibility.
        """
        # Default single-graph fields for backward compatibility
        self.graph = None
        self.graph_funcs = None
        self.node_retriever = None

        # Domain router: domain -> {graph, graph_funcs, node_retriever}
        self._domain_router: Dict[str, Dict[str, Any]] = {}

        # Helper to build a retriever instance for a given domain/graph
        from .tools import retriever as retriever_mod

        def _build_retriever_for(domain: str, graph: Dict[str, Any], graph_dir: str, dataset: Optional[str]) -> Any:
            class _Args:
                pass

            args = _Args()
            args.faiss_gpu = self.config.faiss_gpu
            args.embedder_name = self.config.embedder_name
            args.embed_cache = self.config.embed_cache
            # Prefer a domain-specific cache dir; otherwise fall back to the directory of its graph.json
            domain_cfg = (
                (self.config.domain_graphs or {}).get(domain, {})
                if isinstance(self.config.domain_graphs, dict)
                else {}
            )
            # Prefer domain-specific cache dir; then default to the graph's directory
            # Use global embed_cache_dir only as a last resort
            args.embed_cache_dir = domain_cfg.get('embed_cache_dir') or os.path.dirname(graph_dir) or self.config.embed_cache_dir
            args.dataset = dataset
            if self.config.node_text_keys is not None:
                args.node_text_keys = self.config.node_text_keys
            elif dataset is not None:
                args.node_text_keys = retriever_mod.NODE_TEXT_KEYS[dataset]
            else:
                raise ValueError("node_text_keys or dataset must be provided for graph retriever")
            print(f"[Router] Building retriever for domain '{domain}' with embed_cache_dir: {args.embed_cache_dir}")
            return retriever_mod.Retriever(args, graph)

        # Multi-graph mode
        if self.config.domain_graphs:
            if not isinstance(self.config.domain_graphs, dict):
                raise TypeError("domain_graphs must be a dict: domain -> {graph_dir, dataset, ...}")
            # Debug print: full domain_graphs config for inspection
            try:
                print("[Router] domain_graphs config ->\n" + json.dumps(self.config.domain_graphs, indent=2))
            except Exception:
                print(f"[Router] domain_graphs config (non-serializable): {self.config.domain_graphs}")
            for domain, cfg in self.config.domain_graphs.items():
                gdir = cfg.get('graph_dir')
                if not gdir or not os.path.isfile(gdir):
                    raise FileNotFoundError(f"Graph JSON not found for domain '{domain}': {gdir}")
                with open(gdir, 'r') as f:
                    print(f"[Router] Loading graph for domain '{domain}' from {gdir}")
                    graph = json.load(f)
                gfuncs = graph_funcs_mod.graph_funcs(graph)
                dataset = cfg.get('dataset', domain)
                # Debug print: per-domain config summary
                print(
                    f"[Router] Domain '{domain}' config summary: dataset={dataset}, graph_dir={gdir}, embed_cache_dir={cfg.get('embed_cache_dir')}"
                )
                retr = _build_retriever_for(domain, graph, gdir, dataset)
                print(
                    "[DBG] domain=",
                    domain,
                    "cache_dir=",
                    retr.cache_dir,
                    "cache_file=",
                    os.path.join(retr.cache_dir, f"cache-{retr.model_name.split('/')[-1]}.pkl"),
                )

                self._domain_router[domain] = {
                    'graph': graph,
                    'graph_funcs': gfuncs,
                    'node_retriever': retr,
                }

            # If a default single-graph is also provided, ignore it; router supersedes.
            return

        # Single-graph mode (backward compatible)
        if not self.config.graph_dir:
            return
        if not os.path.isfile(self.config.graph_dir):
            raise FileNotFoundError(f"Graph JSON not found: {self.config.graph_dir}")
        with open(self.config.graph_dir, 'r') as f:
            print(f"Loading graph from {self.config.graph_dir}")
            self.graph = json.load(f)

        self.graph_funcs = graph_funcs_mod.graph_funcs(self.graph)
        self.node_retriever = _build_retriever_for(
            domain=self.config.dataset or 'default',
            graph=self.graph,
            graph_dir=self.config.graph_dir,
            dataset=self.config.dataset,
        )

    def _select_tools_for_domain(self, domain: Optional[str]) -> Tuple[Optional[Any], Optional[Any]]:
        """Return (graph_funcs, node_retriever) for the specified domain.
        Falls back to single-graph tools if router is empty or domain not found.
        """
        if domain and domain in self._domain_router:
            tools = self._domain_router[domain]
            return tools['graph_funcs'], tools['node_retriever']
        # fallback to single-graph mode
        print(f"[Router] No tools found for domain '{domain}', falling back to single-graph mode")
        return self.graph_funcs, self.node_retriever

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(responses, add_special_tokens=False, return_tensors='pt', padding="longest")['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at graph operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        responses_str = [
            resp.split('</graph>')[0] + '</graph>'
            if '</graph>' in resp
            else resp.split('</answer>')[0] + '</answer>'
            if '</answer>' in resp
            else resp
            for resp in responses_str
        ]

        if self.config.no_think_rl:
            raise ValueError('stop')
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""

        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(
                f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}"
            )
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch['input_ids'], cur_responses, next_obs_ids]
        )

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:],
        })
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True,
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)  # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False,
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                pad_to_left=False,
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
        if active_batch size is not divisible by num_gpus, pad with first sequence then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        bsz = gen_batch.batch['input_ids'].shape[0]

        # Determine domains for router from non-tensor batch (extra_info.domain if available)
        domain_list: List[Optional[str]] = []
        if hasattr(gen_batch, 'non_tensor_batch') and isinstance(gen_batch.non_tensor_batch, dict):
            extra_infos = gen_batch.non_tensor_batch.get('extra_info', None)
            if extra_infos is not None:
                try:
                    # numpy object array -> python list
                    extra_infos = list(extra_infos)
                except Exception:
                    print(f"[DBG] Failed to convert extra_infos to list: {extra_infos}")
                    pass
                for info in extra_infos:
                    if isinstance(info, dict):
                        domain_list.append(info.get('domain', self.config.dataset))
                    else:
                        print(f"[DBG] Invalid info type (not dict): {info}")
                        domain_list.append(self.config.dataset)

        # Fallback if not provided
        if not domain_list:
            # Assume uniform default domain across batch
            print(f"[DBG] No domain provided, falling back to default domain: {self.config.dataset}")
            domain_list = [self.config.dataset for _ in range(bsz)]

        # Assertions to ensure multi-graph routing correctness
        if self._domain_router:
            assert hasattr(gen_batch, 'non_tensor_batch') and isinstance(gen_batch.non_tensor_batch, dict), \
                "gen_batch.non_tensor_batch missing; ensure extra_info is passed via DataProto.pop(non_tensor_batch_keys=['extra_info'])."
            assert 'extra_info' in gen_batch.non_tensor_batch, \
                "Missing 'extra_info' in non_tensor_batch for multi-graph routing. Include it when building gen_batch."
            assert len(domain_list) == bsz, f"domain_list length {len(domain_list)} != batch size {bsz}."
            unknown = [d for d in domain_list if d not in self._domain_router]
            assert len(unknown) == 0, (
                f"Found unknown domains not in router: {sorted(set(unknown))[:5]}. "
                f"Router keys: {list(self._domain_router.keys())}"
            )

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Trace collection (eval/debug). We store python objects in a numpy object array for DataProto compatibility.
        collect_trace = bool(getattr(self.config, "collect_trace", False))
        traces: Optional[List[List[Dict[str, Any]]]] = [[] for _ in range(bsz)] if collect_trace else None

        # Ensure meta_info is always defined (e.g., max_turns == 0 edge case)
        meta_info: Dict[str, Any] = {}

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids'],
            )

            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, domains=domain_list
            )

            # Record trace for currently-active samples (before applying dones -> active_mask update).
            if collect_trace and traces is not None:
                try:
                    cur_actions, contents = self.postprocess_predictions(responses_str)
                except Exception:
                    cur_actions, contents = ([""] * bsz, [""] * bsz)
                active_before = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else list(active_mask)
                for i in range(bsz):
                    if not active_before[i]:
                        continue
                    traces[i].append(
                        {
                            "turn": int(step),
                            "phase": "loop",
                            "domain": domain_list[i] if i < len(domain_list) else None,
                            "prediction": responses_str[i],
                            "action": cur_actions[i],
                            "content": contents[i],
                            "observation": next_obs[i],
                            "done": int(dones[i]),
                            "valid_action": int(valid_action[i]),
                            "is_search": int(is_search[i]),
                        }
                    )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            # Update states
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)

        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids'],
            )

            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations (no search)
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False, domains=domain_list
            )

            # Trace final rollout (no tool execution)
            if collect_trace and traces is not None:
                try:
                    cur_actions, contents = self.postprocess_predictions(responses_str)
                except Exception:
                    cur_actions, contents = ([""] * bsz, [""] * bsz)
                active_before = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else list(active_mask)
                # We didn't capture next_obs here (tool execution disabled); keep observation empty.
                for i in range(bsz):
                    if not active_before[i]:
                        continue
                    traces[i].append(
                        {
                            "turn": int(self.config.max_turns),
                            "phase": "final",
                            "domain": domain_list[i] if i < len(domain_list) else None,
                            "prediction": responses_str[i],
                            "action": cur_actions[i],
                            "content": contents[i],
                            "observation": "",
                            "done": int(dones[i]),
                            "valid_action": int(valid_action[i]),
                            "is_search": int(is_search[i]),
                        }
                    )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(original_right_side, responses_ids)

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        dp = self._compose_final_output(original_left_side, original_right_side, meta_info)
        if collect_trace and traces is not None:
            dp.non_tensor_batch["__trace__"] = np.array(traces, dtype=object)
        return dp

    def _compose_final_output(self, left_side: Dict, right_side: Dict, meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([left_side['input_ids'], right_side['responses']], dim=1)

        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat(
            [self.tensor_fn.create_attention_mask(left_side['input_ids']), self.tensor_fn.create_attention_mask(final_output['responses'])],
            dim=1,
        )
        final_output['info_mask'] = torch.cat(
            [self.tensor_fn.create_attention_mask(left_side['input_ids']), self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])],
            dim=1,
        )

        final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    def execute_predictions(
        self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, domains: Optional[List[Optional[str]]] = None
    ) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE: penalty_for_invalid is not included in observation shown to the LLM

        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding

        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []

        # do_search indicates whether to execute the graph function (for the final rollout we set False)
        # Ensure domains aligned to predictions
        if domains is None:
            domains = [self.config.dataset for _ in range(len(cur_actions))]
        for action, content, active, domain in zip(cur_actions, contents, active_mask, domains):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue

            if action == 'answer':
                next_obs.append('')
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
                continue

            if action == 'graph':
                if not do_search:
                    # During final rollout, we do not execute tools
                    next_obs.append('\n\n<information></information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    continue

                func_name, func_arg = parse_graph_call(content)
                if func_name is None:
                    info_text = 'Invalid graph call. Expected Exactly One: FunctionName[args].'
                    next_obs.append(f"\n\n<information>{info_text}</information>\n\n")
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(1)
                    continue

                # Route to domain-specific tools
                graph_funcs, node_retriever = self._select_tools_for_domain(domain)
                info_text = self._execute_graph(func_name, func_arg, graph_funcs, node_retriever)
                next_obs.append(f"\n\n<information>{info_text}</information>\n\n")
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
                continue

            # invalid action
            next_obs.append(
                "\n\n<information>This is a malformed output. Expect <think>...</think><graph>...</graph> or final <think>...</think><answer>...</answer>.</information>\n\n"
            )
            dones.append(0)
            valid_action.append(0)
            is_search.append(0)

        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[str]]:
        """
        Process (text-based) predictions from llm into actions and content.
        Extracts the content inside <graph>...</graph> or <answer>...</answer>.
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):  # for llm output
                pattern = r'<(graph|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)

        return actions, contents

    # Graph execution helpers (ported from GraphAgent)
    def _execute_graph(self, action_type: str, argument: str, graph_funcs, node_retriever) -> str:
        try:
            if graph_funcs is None or node_retriever is None:
                return 'Graph tools are not initialized for this domain. Please configure domain_graphs or graph_dir/dataset.'

            if action_type == 'RetrieveNode':
                try:
                    idd, node = node_retriever.search_single(argument, 1)
                    return f"The ID of this retrieval target node is {idd}."
                except Exception:
                    return 'There is no information that can be matched in the database. Please try another query.'

            elif action_type == 'NeighbourCheck':
                try:
                    node_id, neighbor_type = split_two_args(argument)
                    return f"The {neighbor_type} neighbors of {node_id} are: {graph_funcs.check_neighbours(node_id, neighbor_type)}."
                except KeyError:
                    return 'The node or neighbor type does not exist in the graph. This might be because your given neighbor type is not correct. Please modify it.'
                except Exception:
                    return 'There is something wrong with the arguments for neighbour checking. Expect: NeighbourCheck[node_id, neighbor_type].'

            elif action_type == 'NodeFeature':
                try:
                    node_id, feature_name = split_two_args(argument)
                    return f"The {feature_name} feature of {node_id} are: {graph_funcs.check_nodes(node_id, feature_name)}."
                except KeyError:
                    return 'The node or feature name does not exist in the graph. This might be because your given feature name is not correct. Please modify it.'
                except Exception:
                    return 'There is something wrong with the arguments for node checking. Expect: NodeFeature[node_id, feature_name].'

            elif action_type == 'NodeDegree':
                try:
                    node_id, neighbor_type = split_two_args(argument)
                    return f"The {neighbor_type} neighbor node degree of {node_id} are: {graph_funcs.check_degree(node_id, neighbor_type)}."
                except KeyError:
                    return 'The node or neighbor type does not exist in the graph. This might be because your given neighbor type is not correct. Please modify it.'
                except Exception:
                    return 'There is something wrong with the arguments for degree checking. Expect: NodeDegree[node_id, neighbor_type].'

            else:
                return 'Invalid Action. Valid Actions are RetrieveNode[keyword], NeighbourCheck[node_id, neighbor_type], NodeFeature[node_id, feature_name], NodeDegree[node_id, neighbor_type].'
        except Exception as e:
            return f'Internal error during graph execution: {e}'


# === Helper parsing functions (ported) ===
def parse_graph_call(graph_block: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a graph call block into (FunctionName, arg_str)."""
    gb = (graph_block or '').strip()
    if not gb:
        return None, None
    first_line = gb.splitlines()[0].strip()
    m = re.match(r"^([A-Za-z_]\w*)\[(.*)\]$", first_line)
    if not m:
        return None, None
    func = m.group(1)
    arg = m.group(2).strip()
    return func, arg


def split_two_args(argument: str) -> Tuple[str, str]:
    parts = [p.strip() for p in argument.split(',', 1)]
    if len(parts) != 2:
        raise ValueError("Expect exactly two arguments separated by a comma.")
    a = remove_quotes(parts[0])
    b = remove_quotes(parts[1])
    return a, b


def remove_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith(("'", '"')) and s.endswith(("'", '"'))) and len(s) >= 2:
        return s[1:-1]
    return s

