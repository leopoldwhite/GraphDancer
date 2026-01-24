import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Any

import torch

from .generation_graph import (
    LLMGenerationManager as _BaseLLMGenerationManager,
    parse_graph_call,
)


class LLMGenerationManagerThreaded(_BaseLLMGenerationManager):
    """
    Threaded variant of LLMGenerationManager: executes per-example environment steps
    concurrently using a thread pool. Only execute_predictions is overridden.

    Configure the pool size via `config.num_threads` (optional). Defaults to min(8, batch).
    """

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        do_search: bool = True,
        domains: Optional[List[Optional[str]]] = None,
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        cur_actions, contents = self.postprocess_predictions(predictions)
        n = len(cur_actions)

        # Normalize masks and domains
        if active_mask is None:
            active_list: List[bool] = [True] * n
        elif isinstance(active_mask, torch.Tensor):
            active_list = active_mask.tolist()
        else:
            active_list = list(active_mask)

        if domains is None:
            domains = [getattr(self.config, "dataset", None)] * n

        next_obs: List[str] = [""] * n
        dones: List[int] = [0] * n
        valid_action: List[int] = [0] * n
        is_search: List[int] = [0] * n

        def _worker(i: int, action: Optional[str], content: str, active: bool, domain: Optional[str]):
            _obs = ""
            _done = 0
            _valid = 0
            _search = 0

            if not active:
                return i, _obs, 1, 0, 0

            if action == "answer":
                return i, _obs, 1, 1, 0

            if action == "graph":
                if not do_search:
                    return i, "\n\n<information></information>\n\n", 0, 1, 1

                func_name, func_arg = parse_graph_call(content)
                if func_name is None:
                    info_text = "Invalid graph call. Expected Exactly One: FunctionName[args]."
                    return i, f"\n\n<information>{info_text}</information>\n\n", 0, 0, 1

                graph_funcs, node_retriever = self._select_tools_for_domain(domain)
                info_text = self._execute_graph(func_name, func_arg, graph_funcs, node_retriever)
                return i, f"\n\n<information>{info_text}</information>\n\n", 0, 1, 1

            _obs = (
                "\n\n<information>This is a malformed output. Expect <think>...</think><graph>...</graph> or final <think>...</think><answer>...</answer>.</information>\n\n"
            )
            return i, _obs, 0, 0, 0

        default_workers = min(8, max(1, n))
        max_workers = int(getattr(self.config, "num_threads", default_workers))
        if max_workers < 1:
            max_workers = 1

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gg-exec") as ex:
            futures = [
                ex.submit(_worker, i, a, c, act, d)
                for i, (a, c, act, d) in enumerate(zip(cur_actions, contents, active_list, domains))
            ]
            for fut in as_completed(futures):
                i, _obs, _done, _valid, _search = fut.result()
                next_obs[i] = _obs
                dones[i] = _done
                valid_action[i] = _valid
                is_search[i] = _search
            print(f"[DBG] Threaded generation manager: {len(futures)} futures completed")

        return next_obs, dones, valid_action, is_search


# Backwards-friendly alias
LLMGenerationManager = LLMGenerationManagerThreaded

