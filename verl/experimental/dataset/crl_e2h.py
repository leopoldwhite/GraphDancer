# Copyright 2025
#
# E2H (Easy-to-Hard) curriculum samplers adapted for latest VERL.
# Integrates Graph-RL's Gaussian/Cosine schedulers with VERL's
# AbstractCurriculumSampler interface so they can be configured via
# data.sampler.{class_path,class_name}.

from __future__ import annotations

import math
from typing import Dict, List, Optional
import json

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Sampler

from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class _BaseE2HSampler(AbstractCurriculumSampler):
    def __init__(self, data_source, data_config: DictConfig):
        # dataset and dataframe
        self.dataset = data_source
        try:
            # HF datasets.Dataset
            self.df = self.dataset.dataframe
        except Exception:
            raise ValueError("dataset must expose a 'dataframe' attribute (HF datasets.Dataset)")

        # batch and steps per epoch
        self.batch_size = int(data_config.train_batch_size)
        self.steps_per_epoch = max(1, len(self.dataset) // self.batch_size)
        # total training steps for global scheduling across epochs
        # allow overrides via data_config: e2h_total_training_steps or total_training_steps
        self.total_training_steps = int(
            data_config.get("e2h_total_training_steps", data_config.get("total_training_steps", self.steps_per_epoch))
        )

        # curriculum knobs
        self.buckets_order = None
        if "buckets" in data_config and data_config.buckets is not None:
            self.buckets_order = list(data_config.buckets)
        self.difficulty_key = data_config.get("difficulty_key", None)
        self.drop_last = True

        # RNG and cursors
        self.seed = int(data_config.get("seed", 42))
        self.rng = np.random.RandomState(self.seed)

        # compute buckets
        self.bucket_names, self.bucket_to_indices = self._build_buckets(
            buckets_order=self.buckets_order, difficulty_key=self.difficulty_key
        )
        self.K = len(self.bucket_names)
        assert self.K >= 2, f"E2H requires >=2 buckets, got {self.K}"
        self._cursors = {b: 0 for b in self.bucket_names}
        for b in self.bucket_names:
            self.rng.shuffle(self.bucket_to_indices[b])

        # logging cache
        self._last_probs: Dict[str, float] = {b: 0.0 for b in self.bucket_names}

        # global step across epochs (monotonic, not reset in __iter__)
        self.global_t = 0

    # ------- Abstract API impls -------
    def __iter__(self):
        # Yield exactly steps_per_epoch batches per epoch, while using global steps for scheduling
        steps_this_epoch = self.steps_per_epoch
        for _ in range(steps_this_epoch):
            t_used = min(int(self.global_t), max(1, int(self.total_training_steps)) - 1)
            probs = self._probs_for_step(t_used)
            self._last_probs = {b: float(probs[i]) for i, b in enumerate(self.bucket_names)}
            counts = self.rng.multinomial(self.batch_size, probs)
            batch_indices: List[int] = []
            for i, b in enumerate(self.bucket_names):
                need = int(counts[i])
                if need > 0:
                    batch_indices.extend(self._take_from_bucket(b, need))

            # pad/crop due to rounding
            if len(batch_indices) < self.batch_size:
                deficit = self.batch_size - len(batch_indices)
                order = np.argsort(-probs)
                for j in order:
                    if deficit <= 0:
                        break
                    b = self.bucket_names[j]
                    take = min(deficit, len(self.bucket_to_indices[b]))
                    if take > 0:
                        batch_indices.extend(self._take_from_bucket(b, take))
                        deficit -= take
            elif len(batch_indices) > self.batch_size:
                self.rng.shuffle(batch_indices)
                batch_indices = batch_indices[: self.batch_size]

            # yield one step's flat index stream
            for idx in batch_indices:
                yield idx
            # advance global step
            self.global_t += 1

    def __len__(self) -> int:
        # Return number of samples per epoch; DataLoader(drop_last=True) will form steps_per_epoch batches
        return self.steps_per_epoch * self.batch_size

    def update(self, batch) -> None:  # pragma: no cover - optional hooks for future online signals
        # For this deterministic scheduler we do not need per-batch feedback to update schedule
        return

    # Graph-RL compatibility: expose sampler distribution of last step
    def get_last_probs(self) -> Dict[str, float]:
        """Return the last step's bucket probabilities.

        Matches Graph-RL's sampler API so training loop can log
        metrics like `sampler/p_{bucket}`.
        """
        return dict(self._last_probs)

    # ------- stateful dataloader integration -------
    # Provide explicit state serialization so StatefulDataLoader can checkpoint
    # and restore sampler progress (global step, RNG, bucket order and cursors).
    def state_dict(self) -> dict:  # type: ignore[override]
        try:
            rng_state = self.rng.get_state()
        except Exception:
            rng_state = None
        return {
            "version": 1,
            "global_t": int(self.global_t),
            "rng_state": rng_state,
            "bucket_names": list(self.bucket_names),
            "bucket_to_indices": {b: list(map(int, self.bucket_to_indices[b])) for b in self.bucket_names},
            "_cursors": {b: int(self._cursors[b]) for b in self.bucket_names},
            "_last_probs": dict(self._last_probs),
            # aux (non-critical) fields for sanity
            "batch_size": int(self.batch_size),
            "steps_per_epoch": int(self.steps_per_epoch),
            "total_training_steps": int(self.total_training_steps),
            "seed": int(self.seed),
        }

    def load_state_dict(self, state: dict) -> None:  # type: ignore[override]
        if not isinstance(state, dict):
            return
        # global step and RNG
        gt = state.get("global_t", None)
        if gt is not None:
            try:
                self.global_t = int(gt)
            except Exception:
                pass
        rng_state = state.get("rng_state", None)
        if rng_state is not None:
            try:
                self.rng.set_state(rng_state)
            except Exception:
                pass

        # Restore buckets/cursors if compatible
        saved_bucket_names = state.get("bucket_names", None)
        saved_bucket_to_indices = state.get("bucket_to_indices", None)
        saved_cursors = state.get("_cursors", None)
        if (
            isinstance(saved_bucket_names, list)
            and isinstance(saved_bucket_to_indices, dict)
            and isinstance(saved_cursors, dict)
        ):
            try:
                # Only restore for buckets that exist in current build
                for b in self.bucket_names:
                    if b in saved_bucket_to_indices and b in saved_cursors:
                        # Keep current indices list if dataset changed in size; otherwise restore order
                        saved_idxs = list(saved_bucket_to_indices[b])
                        if len(saved_idxs) == len(self.bucket_to_indices[b]):
                            self.bucket_to_indices[b] = saved_idxs
                        # Restore cursor (bounded)
                        c = int(saved_cursors[b])
                        self._cursors[b] = max(0, min(c, max(0, len(self.bucket_to_indices[b]) - 1)))
            except Exception:
                # On any mismatch, silently fall back to current in-memory structures
                pass

        # Restore last probs when present (for logging continuity)
        last_probs = state.get("_last_probs", None)
        if isinstance(last_probs, dict):
            try:
                self._last_probs = {str(k): float(v) for k, v in last_probs.items() if k in self.bucket_names}
            except Exception:
                pass

    # Utility to align sampler to an external global step (optional fallback)
    def set_global_step(self, t: int) -> None:
        try:
            self.global_t = int(t)
        except Exception:
            pass

    # ------- helpers -------
    def _build_buckets(self, buckets_order: Optional[List[str]], difficulty_key: Optional[str]):
        # Mirror Graph-RL semantics but support both HF datasets.Dataset and pandas.DataFrame.
        df = self.df

        def _extract_nested_value(val, nested_key: str):
            if isinstance(val, dict):
                return val.get(nested_key)
            if isinstance(val, str):
                try:
                    obj = json.loads(val)
                    if isinstance(obj, dict):
                        return obj.get(nested_key)
                except Exception:
                    return None
            return None

        values: Optional[List[Optional[str]]] = None

        # Resolve difficulty by key
        if difficulty_key is not None:
            if "." in difficulty_key:
                first, second = difficulty_key.split(".", 1)
                # HF datasets case
                if hasattr(df, "column_names") and first in getattr(df, "column_names", []):
                    try:
                        col = list(df[first])
                        values = [_extract_nested_value(v, second) for v in col]
                    except Exception:
                        values = None
                # pandas.DataFrame case
                elif hasattr(df, "columns") and first in getattr(df, "columns", []):
                    try:
                        col = df[first].tolist()
                        values = [_extract_nested_value(v, second) for v in col]
                    except Exception:
                        values = None
            else:
                if hasattr(df, "column_names") and difficulty_key in getattr(df, "column_names", []):
                    try:
                        values = list(df[difficulty_key])
                    except Exception:
                        values = None
                elif hasattr(df, "columns") and difficulty_key in getattr(df, "columns", []):
                    try:
                        values = df[difficulty_key].tolist()
                    except Exception:
                        values = None

        # Fallback: infer from data_source keywords if unresolved or all None
        def _infer_from_source_list(src_list: List[object]) -> List[Optional[str]]:
            out: List[Optional[str]] = []
            for x in src_list:
                xx = str(x).lower()
                label = None
                for cand in ["trivial", "easy", "medium", "hard"]:
                    if cand in xx:
                        label = cand
                        break
                out.append(label)
            return out

        if values is None or all(v is None for v in values):
            if hasattr(df, "column_names") and "data_source" in getattr(df, "column_names", []):
                try:
                    values = _infer_from_source_list(list(df["data_source"]))
                except Exception:
                    values = None
            elif hasattr(df, "columns") and "data_source" in getattr(df, "columns", []):
                try:
                    values = _infer_from_source_list(df["data_source"].tolist())
                except Exception:
                    values = None

        if values is None:
            raise ValueError(
                "E2H sampler needs a difficulty signal. Provide data.difficulty_key or embed keywords in 'data_source'."
            )

        # Normalize buckets and compute presence
        if buckets_order is None:
            buckets_order = ["trivial", "easy", "medium", "hard"]
        buckets_order = [str(b).lower() for b in buckets_order]

        present = set([str(v).lower() for v in values if v is not None])
        final_order = [b for b in buckets_order if b in present]
        if len(final_order) < 2:
            raise ValueError(f"Need >=2 difficulty buckets, got {final_order}")

        bucket_to_indices: Dict[str, List[int]] = {b: [] for b in final_order}
        for i, v in enumerate(values):
            vv = str(v).lower() if v is not None else ""
            if vv in bucket_to_indices:
                bucket_to_indices[vv].append(i)

        empty = [b for b, idxs in bucket_to_indices.items() if len(idxs) == 0]
        if empty:
            raise ValueError(
                f"Empty buckets detected for {empty}. Please check difficulty labels or mapping."
            )

        return final_order, bucket_to_indices

    def _take_from_bucket(self, bucket_name: str, n: int) -> List[int]:
        buf = []
        L = len(self.bucket_to_indices[bucket_name])
        if L == 0:
            return buf
        while n > 0:
            cur = self._cursors[bucket_name]
            remain = L - cur
            take = min(remain, n)
            buf.extend(self.bucket_to_indices[bucket_name][cur : cur + take])
            cur += take
            if cur >= L:
                self.rng.shuffle(self.bucket_to_indices[bucket_name])
                cur = 0
            self._cursors[bucket_name] = cur
            n -= take
        return buf

    # To be implemented in subclasses
    def _probs_for_step(self, t: int) -> np.ndarray:  # pragma: no cover - abstract helper
        raise NotImplementedError


class E2HCosineSampler(_BaseE2HSampler):
    def __init__(self, data_source, data_config: DictConfig):
        self.eps = float(data_config.get("e2h_eps", 1.0e-8))
        super().__init__(data_source, data_config)

    def _alpha_t(self, t: int) -> float:
        # Use global schedule length self.total_training_steps
        T = max(1, int(self.total_training_steps))
        if T <= 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * (t / (T - 1))))

    def _probs_for_step(self, t: int) -> np.ndarray:
        alpha = self._alpha_t(t)
        ks = np.arange(self.K, dtype=np.float64)
        scores = alpha * (self.K - ks - 1.0) + (1.0 - alpha) * ks
        scores = np.maximum(scores, 0.0) + float(self.eps)
        probs = np.asarray(scores, dtype=np.float64)
        S = float(np.sum(probs))
        if not np.isfinite(S) or S <= 0.0:
            probs = np.ones(self.K, dtype=np.float64) / float(self.K)
        else:
            probs = probs / S
        # Adjust last bucket to absorb rounding error
        partial = float(np.sum(probs[:-1])) if self.K >= 2 else 0.0
        if self.K >= 2:
            if partial >= 1.0:
                tiny = 1e-12
                denom = partial if partial > 0.0 else 1.0
                probs[:-1] = (probs[:-1] / denom) * (1.0 - tiny)
                probs[-1] = tiny
            else:
                probs[-1] = 1.0 - partial
        else:
            probs[0] = 1.0
        return probs


class E2HGaussianSampler(_BaseE2HSampler):
    def __init__(self, data_source, data_config: DictConfig):
        self.sigma = float(data_config.get("e2h_sigma", 0.75))
        self.beta = float(data_config.get("e2h_beta", 0.25))
        # Scheme B (Gaussian + bias distribution mix) knobs:
        #   p(t) = (1 - eta(t)) * p_gauss(t) + eta(t) * q
        # where q is a user-provided bias distribution over buckets.
        #
        # - Constant: data.e2h_mix_eta
        # - Scheduled: data.e2h_mix_eta_start / data.e2h_mix_eta_end (linear over t in [0, T-1])
        # - Bias distribution: data.e2h_mix_q (list[float] length K, dict bucket->weight, or str)
        self.mix_eta = float(data_config.get("e2h_mix_eta", 0.0))
        self.mix_eta_start = data_config.get("e2h_mix_eta_start", None)
        self.mix_eta_end = data_config.get("e2h_mix_eta_end", None)
        self.mix_q = data_config.get("e2h_mix_q", None)
        super().__init__(data_source, data_config)

    def _eta_t(self, t: int) -> float:
        """Return eta(t) in [0, 1]."""
        # Use global schedule length self.total_training_steps
        T = max(1, int(self.total_training_steps))
        if self.mix_eta_start is not None or self.mix_eta_end is not None:
            try:
                s = float(self.mix_eta_start if self.mix_eta_start is not None else self.mix_eta)
                e = float(self.mix_eta_end if self.mix_eta_end is not None else self.mix_eta)
            except Exception:
                s, e = float(self.mix_eta), float(self.mix_eta)
            frac = 1.0 if T <= 1 else float(t) / float(max(1, T - 1))
            eta = s + (e - s) * frac
        else:
            eta = float(self.mix_eta)
        if not math.isfinite(eta):
            eta = 0.0
        return float(max(0.0, min(1.0, eta)))

    def _q_dist(self) -> np.ndarray:
        """Build and normalize bias distribution q over buckets."""
        K = int(self.K)
        q = np.zeros(K, dtype=np.float64)

        # Defaults: bias toward the first bucket (typically 'easy' given buckets order)
        if self.mix_q is None:
            q[0] = 1.0
            return q

        # OmegaConf containers: convert to plain python types first (ListConfig/DictConfig).
        mix_q = self.mix_q
        try:
            if OmegaConf.is_config(mix_q):
                mix_q = OmegaConf.to_container(mix_q, resolve=True)
        except Exception:
            mix_q = self.mix_q

        # String presets
        if isinstance(mix_q, str):
            mode = mix_q.strip().lower()
            if mode in {"easy", "easiest", "first"}:
                q[0] = 1.0
                return q
            if mode in {"hard", "hardest", "last"}:
                q[-1] = 1.0
                return q
            if mode in {"uniform", "uni"}:
                q[:] = 1.0 / float(K)
                return q
            # Unknown string: fall back to default
            q[0] = 1.0
            return q

        # Dict: bucket_name -> weight
        if isinstance(mix_q, dict):
            for i, b in enumerate(self.bucket_names):
                try:
                    if b in mix_q:
                        q[i] = float(mix_q[b])
                except Exception:
                    continue
        # List/tuple: positional weights
        elif isinstance(mix_q, (list, tuple)):
            try:
                if len(mix_q) == K:
                    q = np.asarray([float(x) for x in mix_q], dtype=np.float64)
                else:
                    # Best-effort: if shorter, pad with zeros; if longer, truncate.
                    arr = [float(x) for x in mix_q]
                    if len(arr) < K:
                        arr = arr + [0.0] * (K - len(arr))
                    q = np.asarray(arr[:K], dtype=np.float64)
            except Exception:
                q = np.zeros(K, dtype=np.float64)
                q[0] = 1.0
        else:
            # Unknown type: fall back to default
            q[0] = 1.0

        q = np.maximum(q, 0.0)
        S = float(np.sum(q))
        if not np.isfinite(S) or S <= 0.0:
            q = np.zeros(K, dtype=np.float64)
            q[0] = 1.0
            return q
        return q / S

    def _probs_for_step(self, t: int) -> np.ndarray:
        # xt = ((t / (T-1))^beta) * (K-1), where T is total_training_steps
        T = max(1, int(self.total_training_steps))
        if T <= 1:
            return np.ones(self.K) / self.K
        xt = ((t / max(1, T - 1)) ** self.beta) * (self.K - 1)
        mu = np.arange(self.K, dtype=np.float64)
        p_g = np.exp(-((xt - mu) ** 2) / (2.0 * (self.sigma ** 2)))
        S = float(np.sum(p_g))
        if not np.isfinite(S) or S <= 0.0:
            p_g = np.ones(self.K, dtype=np.float64) / float(self.K)
        else:
            p_g = p_g / S

        # Scheme B mix: p = (1-eta)*p_g + eta*q
        eta = self._eta_t(t)
        if eta <= 0.0:
            return p_g
        q = self._q_dist()
        probs = (1.0 - eta) * p_g + eta * q
        S2 = float(np.sum(probs))
        if not np.isfinite(S2) or S2 <= 0.0:
            return np.ones(self.K, dtype=np.float64) / float(self.K)
        return probs / S2
