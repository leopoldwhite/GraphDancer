"""Trajectory analyzer for K=8 self-sampled rollouts.

Reads verl-emitted ``results.jsonl`` files from Phase-0 K=8 sampling, computes
per-trajectory features (EM, VF, EH, loop, n_rounds, invalid_tool), applies the
lex-rank pair-selection rule, and reports pair-yield breakdowns per difficulty
bucket.

Inputs:  verl-emitted JSONL files (one per Academic sub-domain). Each line has
         {qid, uid, extra_info{difficulty, domain, ...}, question, model_answer,
          gt_answer, trace?}.
         When val_kwargs.n=8 is set, there are 8 lines per qid (one per rollout).

Output:  JSON summary at --out, plus a stdout-printed table:
           bucket × {L1, L2 (skip-perfect), L3 (resample-fail), L5 (skip-failed)}
         and the chosen-vs-rejected score-gap distribution per bucket.

Imports ``em_check``, ``extract_solution``, ``is_valid_sequence`` from
``verl/utils/reward_score/qa_em_format.py``. Trace-level diagnostics
(``evidence_hit``, ``has_invalid_tool``, ``has_successful_stop``,
``graph_call_valid``) are defined inline below.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Locate repo root so we can import verl.* without modifying sys.path globally
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))           # repo root
sys.path.insert(0, REPO_ROOT)

from verl.utils.reward_score.qa_em_format import (  # noqa: E402
    em_check,
    extract_solution,
    is_valid_sequence,
)

# ---------------------------------------------------------------------------
# Trace-level diagnostics
# ---------------------------------------------------------------------------

VALID_GRAPH_FUNCS = {"RetrieveNode", "NeighbourCheck", "NodeFeature", "NodeDegree"}
CALL_RE = re.compile(r"^([A-Za-z_]\w*)\[(.*)\]$", re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def normalize_answer(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"\b(a|an|the|usd)\b", " ", s)
    s = re.sub(r"[\W_]+", " ", s)
    return " ".join(s.split())


def evidence_hit(trace: Any, gt_answer: str) -> bool:
    """True if any trace observation surfaces a normalized form of the gold answer."""
    gt = normalize_answer(gt_answer or "")
    if not gt:
        return False
    if not isinstance(trace, list):
        return False
    for t in trace:
        if not isinstance(t, dict):
            continue
        obs = t.get("observation", "")
        if isinstance(obs, str) and gt in normalize_answer(obs):
            return True
    return False


def graph_call_valid(content: Any) -> bool:
    """True if `content` is a syntactically valid graph function call."""
    if not isinstance(content, str):
        return False
    m = CALL_RE.match(content.strip())
    if not m:
        return False
    func = m.group(1)
    if func not in VALID_GRAPH_FUNCS:
        return False
    args = (m.group(2) or "").strip()
    if func == "RetrieveNode":
        return len(args) > 0
    if func in {"NeighbourCheck", "NodeFeature", "NodeDegree"}:
        return "," in args
    return True


def has_invalid_tool(trace: Any) -> bool:
    """True if any step is flagged invalid OR contains a malformed graph call."""
    if not isinstance(trace, list) or not trace:
        return True
    for t in trace:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("valid_action"), int) and t["valid_action"] == 0:
            return True
        if t.get("action") == "graph" and not graph_call_valid(t.get("content", "")):
            return True
    return False


def has_successful_stop(trace: Any) -> bool:
    """True if the trajectory terminated by emitting an `<answer>` block."""
    if not isinstance(trace, list) or not trace:
        return False
    last = trace[-1] if isinstance(trace[-1], dict) else {}
    if isinstance(last, dict):
        if last.get("done", 0) in (1, True):
            return True
        if str(last.get("action", "")).lower() == "answer":
            return True
    for t in reversed(trace):
        if not isinstance(t, dict):
            continue
        pred = t.get("prediction", "")
        if isinstance(pred, str) and ANSWER_TAG_RE.search(pred):
            return True
    return False

# Error-observation patterns (verbatim from
# graphdancer/llm_agent/generation_graph.py:600-755). An observation that
# contains ANY of these substrings is a graph-tool failure response, not a
# valid retrieval result. Empty list responses (e.g. "neighbors are: [].")
# are NOT errors — they are valid empty data.
_ERROR_OBS_PATTERNS = (
    "Invalid graph call. Expected Exactly One",        # parse fail
    "This is a malformed output",                       # action neither graph nor answer
    "Graph tools are not initialized",                  # config not loaded
    "There is no information that can be matched",      # RetrieveNode no-match
    "does not exist in the graph",                      # NeighbourCheck/NodeFeature/NodeDegree key error
    "There is something wrong with the arguments",      # arg shape error
    "Invalid Action. Valid Actions are",                # unknown FuncName
    "Internal error during graph execution",            # uncaught exception
)


def _obs_is_error(obs: Any) -> bool:
    if not isinstance(obs, str):
        return False
    return any(p in obs for p in _ERROR_OBS_PATTERNS)


def _canon_call(content: Any) -> Optional[str]:
    """Normalize a graph call for repeat detection: lowercase + strip + collapse
    whitespace inside args. FuncName + args must match exactly to count as repeat."""
    if not isinstance(content, str):
        return None
    s = content.strip().lower()
    m = re.match(r"^([a-z_]\w*)\[(.*)\]$", s, flags=re.DOTALL)
    if not m:
        return None
    func = m.group(1)
    arg = re.sub(r"\s+", "", m.group(2))
    return f"{func}[{arg}]"

# ---------------------------------------------------------------------------
# Per-trajectory feature extraction
# ---------------------------------------------------------------------------


def _norm_diff(s: Any) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip().lower()
    if s in {"trivial", "easy"}:
        return "easy"
    if s == "medium":
        return "medium"
    if s == "hard":
        return "hard"
    return "unknown"


def _diff_of(rec: Dict) -> str:
    ei = rec.get("extra_info") or {}
    if isinstance(ei, str):
        try:
            ei = json.loads(ei)
        except Exception:
            ei = {}
    return _norm_diff(ei.get("difficulty") if isinstance(ei, dict) else None)


def _gt_target(rec: Dict) -> str:
    gt = rec.get("gt_answer", "")
    if isinstance(gt, dict):
        return str(gt.get("target", ""))
    return str(gt) if gt is not None else ""


def _trace_list(rec: Dict) -> List[Dict]:
    tr = rec.get("trace")
    if isinstance(tr, list):
        return tr
    return []


def compute_features(rec: Dict, max_turns: int = 10) -> Dict[str, Any]:
    """Compute per-trajectory features used by the lex-rank pair rule."""
    trace = _trace_list(rec)
    n_turns = len(trace)
    # In verl trace schema, each turn has `action` ∈ {'graph','answer','search',None};
    # graph rounds are the ones with action == 'graph' (information-seeking turns).
    n_graph_rounds = sum(1 for t in trace if isinstance(t, dict) and t.get("action") == "graph")
    gt = _gt_target(rec)

    # The "model_answer" field is already the extracted <answer>...</answer> content;
    # for VF / EM we need the raw concatenated trajectory.
    # Verl trace schema (verified 2026-05-03 against
    # results/eval-gd-x03-s125-legal/legal/results.jsonl): each turn is
    #   {turn, phase, domain, prediction, action, content, observation, done, valid_action, is_search}
    # where `prediction` is the raw agent-emitted text for the turn (already
    # containing the <think>/<graph>/<answer> tags) and `observation` is the
    # env-injected "\n\n<information>...</information>\n\n" block.
    if trace:
        parts = []
        for t in trace:
            if not isinstance(t, dict):
                continue
            pred = t.get("prediction") or ""
            obs = t.get("observation") or ""
            if pred:
                parts.append(str(pred))
            if obs:
                parts.append(str(obs))
        seq_str = "".join(parts)
    else:
        ans = rec.get("model_answer") or ""
        seq_str = f"<answer>{ans}</answer>" if ans else ""

    vf_ok, _msg = is_valid_sequence(seq_str) if seq_str else (False, "empty")
    extracted = extract_solution(seq_str) if seq_str else None
    em_val = int(em_check(extracted, gt)) if extracted else 0

    # Evidence hit + final_answer
    eh_ok = evidence_hit(trace, gt) if trace else False
    stopped = has_successful_stop(trace) if trace else (extracted is not None)
    final_answer = int(bool(stopped))

    # v2-compat features (kept for --lex_version v2 in builder)
    invalid_tool_v2 = has_invalid_tool(trace) if trace else (not vf_ok)
    loop_limit_v2 = (not stopped) and (n_turns >= max_turns)

    # Error rate = (syntax-invalid graph calls + error-observation responses) /
    # total graph calls. Empty-list results count as valid. error_rate=0 when
    # n_graph_rounds=0 (no calls means no failures to attribute).
    n_syntax_err = 0
    n_obs_err = 0
    for t in trace:
        if not isinstance(t, dict) or t.get("action") != "graph":
            continue
        if not graph_call_valid(t.get("content", "")):
            n_syntax_err += 1
        # Inspect observation regardless of syntax — env may still emit error
        # text (e.g. "does not exist") for syntactically-valid but semantically
        # wrong calls. Also count valid_action == 0 cases.
        if _obs_is_error(t.get("observation", "")):
            n_obs_err += 1
        elif isinstance(t.get("valid_action"), int) and t["valid_action"] == 0:
            n_obs_err += 1
    error_rate = (n_syntax_err + n_obs_err) / max(n_graph_rounds, 1) if n_graph_rounds > 0 else 0.0

    # Repeated calls = graph calls whose canon form has appeared earlier in
    # the same trace. Strict FuncName + args match (case- and whitespace-
    # insensitive). Malformed calls (parse fails) are not counted as repeats
    # — they are already penalized via error_rate.
    seen_calls = set()
    n_repeated = 0
    for t in trace:
        if not isinstance(t, dict) or t.get("action") != "graph":
            continue
        key = _canon_call(t.get("content", ""))
        if key is None:
            continue
        if key in seen_calls:
            n_repeated += 1
        else:
            seen_calls.add(key)

    return {
        "em": em_val,
        "vf": int(bool(vf_ok)),
        "eh": int(bool(eh_ok)),
        "final_answer": final_answer,
        "error_rate": float(error_rate),
        "n_repeated": int(n_repeated),
        "n_graph_rounds": int(n_graph_rounds),
        "n_turns": int(n_turns),
        # v2-compat fields
        "loop_limit": int(bool(loop_limit_v2)),
        "invalid_tool": int(bool(invalid_tool_v2)),
    }


def lex_score_v1(feat: Dict[str, Any]) -> Tuple[Any, ...]:
    """ORIGINAL lex tuple (used by all main runs through 2026-05-05, including
    phaseA-fixed-v1 which produced AVG R-L 0.4323): em / eh / vf / -loop_limit
    / -invalid_tool / -n_graph_rounds. Last entry biases toward shorter traces
    (length bias)."""
    return (
        feat["em"],
        feat["eh"],
        feat["vf"],
        -feat["loop_limit"],
        -feat["invalid_tool"],
        -feat["n_graph_rounds"],
    )


def lex_score_v3(feat: Dict[str, Any]) -> Tuple[Any, ...]:
    """v3 lex tuple (2026-05-05): em / eh / vf / final_answer / -error_rate /
    -n_repeated. Replaces v2's loop_limit (binary trigger), invalid_tool
    (syntactic-only), and -n_graph_rounds (length bias) with finer signals.
    Empirically TIED-OR-WORSE than v2 in K=8 setup; pending K=16 retest."""
    return (
        feat["em"],
        feat["eh"],
        feat["vf"],
        feat["final_answer"],
        -feat["error_rate"],
        -feat["n_repeated"],
    )


# Backwards-compat: keep `lex_score` as the v3 alias so existing imports work.
lex_score = lex_score_v3


# ---------------------------------------------------------------------------
# Pair-yield bucketing
# ---------------------------------------------------------------------------


def classify_yield(scores: List[Tuple]) -> str:
    """Given a list of K lex-scores, classify which fallback layer applies.

    L1 = ranks differ → standard pair available
    L2 = all-perfect (1,1,1,…) → skip prompt, no signal
    L3 = all-failed (0,…) → would need to resample; without resample, pair-fail
    L4 = (deferred — only valid in iterative DPO round ≥ 2)
    L5 = same as L3 in round-1 offline mode (skip prompt)
    """
    if not scores:
        return "EMPTY"
    smin = min(scores)
    smax = max(scores)
    if smin != smax:
        return "L1"
    # All scores identical
    em = smax[0]
    eh = smax[1]
    vf = smax[2]
    if em == 1 and eh == 1 and vf == 1:
        return "L2_perfect"
    if em == 0 and eh == 0:
        return "L3_failed"
    return "L1_tie_partial"  # tie but not perfect/failed (e.g. all (0,0,1,*))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_records(jsonl_paths: List[str]) -> List[Dict]:
    out = []
    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def group_by_qid(records: List[Dict]) -> Dict[Any, List[Dict]]:
    """Group rollout records by (domain, qid). Each Academic sub-domain
    (dblp, maple_*) numbers its qids independently from 0; grouping by qid
    alone would silently merge dblp-qid-0 + chemistry-qid-0 + materials-qid-0
    into a single 24-record bucket, producing artifactual lex-rank pairs
    where chosen/rejected come from DIFFERENT graphs (and unrelated ground
    truths). The 2026-05-04 audit found 102/150 pairs in the original
    dpo_pairs.parquet were such cross-domain contaminated pairs."""
    g = defaultdict(list)
    for r in records:
        qid = r.get("qid")
        if qid is None:
            qid = r.get("uid")
        ei = r.get("extra_info") or {}
        if isinstance(ei, str):
            try:
                ei = json.loads(ei)
            except Exception:
                ei = {}
        domain = ei.get("domain", "unknown") if isinstance(ei, dict) else "unknown"
        g[(domain, qid)].append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_glob", required=True,
                    help="Glob pattern over verl-emitted results.jsonl files. Quote it. "
                         "Example: 'results/dpo-phase0/*/*/results.jsonl'")
    ap.add_argument("--out", required=True, help="Output summary JSON path")
    ap.add_argument("--max_turns", type=int, default=10)
    ap.add_argument("--expected_K", type=int, default=8,
                    help="Expected rollouts per qid (n in val_kwargs.n)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.jsonl_glob))
    if not paths:
        print(f"[FATAL] no files match {args.jsonl_glob}", file=sys.stderr)
        sys.exit(1)
    print(f"loading {len(paths)} JSONL files:")
    for p in paths:
        print(f"  {p}")

    records = load_records(paths)
    print(f"total trajectory records: {len(records)}")
    by_qid = group_by_qid(records)
    # Group keys are now (domain, qid) tuples — see group_by_qid docstring for
    # why this matters. The variable name `by_qid` is kept for backwards
    # compatibility but the keys are joint.
    print(f"unique (domain, qid) groups: {len(by_qid)}")

    # Per-prompt analysis
    layer_counter = defaultdict(Counter)            # bucket -> Counter({L1, L2, L3, ...})
    score_gap_per_bucket = defaultdict(list)        # bucket -> [gap, ...]
    feature_records = []                            # for build_pair_parquet downstream

    rollout_count_hist = Counter()                  # to verify K rollouts/qid

    for key, recs in by_qid.items():
        # key is (domain, qid). qid alone (recs[0]['qid']) is recoverable via
        # recs[0] but we keep `key` so feature_records below stores both.
        domain, qid = key if isinstance(key, tuple) else ("unknown", key)
        rollout_count_hist[len(recs)] += 1
        bucket = _diff_of(recs[0])
        feats = [compute_features(r, max_turns=args.max_turns) for r in recs]
        scores = [lex_score(f) for f in feats]
        layer = classify_yield(scores)
        layer_counter[bucket][layer] += 1

        # For L1 / L1_tie_partial we have a viable chosen/rejected pair
        if layer in ("L1", "L1_tie_partial") and scores:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            worst_idx = min(range(len(scores)), key=lambda i: scores[i])
            if best_idx != worst_idx:
                # Score gap: count of components where chosen > rejected (out of 6)
                gap_components = sum(
                    1 for a, b in zip(scores[best_idx], scores[worst_idx]) if a > b
                )
                score_gap_per_bucket[bucket].append(gap_components)
                feature_records.append({
                    "qid": qid,
                    "domain": domain,
                    "bucket": bucket,
                    "best_idx": best_idx,
                    "worst_idx": worst_idx,
                    "best_score": list(scores[best_idx]),
                    "worst_score": list(scores[worst_idx]),
                    "best_features": feats[best_idx],
                    "worst_features": feats[worst_idx],
                    "n_rollouts": len(recs),
                })

    # Build summary
    buckets = ["easy", "medium", "hard", "unknown"]
    summary = {
        "n_qids": len(by_qid),
        "n_records": len(records),
        "rollout_count_hist": dict(rollout_count_hist),
        "expected_K_per_qid": args.expected_K,
        "per_bucket": {},
        "viable_pair_qids": len(feature_records),
    }
    for b in buckets:
        c = layer_counter.get(b, Counter())
        n = sum(c.values())
        if n == 0:
            continue
        gap = score_gap_per_bucket.get(b, [])
        summary["per_bucket"][b] = {
            "n_qids": n,
            "layer_counts": dict(c),
            "layer_fractions": {k: round(v / n, 4) for k, v in c.items()},
            "viable_pair_count": sum(c.get(k, 0) for k in ("L1", "L1_tie_partial")),
            "viable_pair_fraction": round(
                sum(c.get(k, 0) for k in ("L1", "L1_tie_partial")) / n, 4
            ),
            "score_gap_mean": round(sum(gap) / len(gap), 3) if gap else None,
            "score_gap_p10": sorted(gap)[len(gap) // 10] if len(gap) >= 10 else None,
            "score_gap_p90": sorted(gap)[(9 * len(gap)) // 10] if len(gap) >= 10 else None,
        }

    # Write outputs
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nwrote summary to: {args.out}")

    # Stdout table
    print("\n=== Pair yield by bucket ===")
    print(f"{'bucket':<10} {'n_qid':>6} {'L1+tie':>8} {'L2_pf':>6} {'L3_fl':>6} {'gap_mean':>10}")
    for b in buckets:
        d = summary["per_bucket"].get(b)
        if not d:
            continue
        print(
            f"{b:<10} {d['n_qids']:>6} "
            f"{d['viable_pair_fraction']*100:>7.2f}% "
            f"{d['layer_fractions'].get('L2_perfect', 0)*100:>5.2f}% "
            f"{d['layer_fractions'].get('L3_failed', 0)*100:>5.2f}% "
            f"{(d.get('score_gap_mean') or 0):>10.3f}"
        )
    print(f"\ntotal viable pairs: {summary['viable_pair_qids']} / {summary['n_qids']} qids")
    print("\nDecision rule (per project_curriculum_dpo_plan.md):")
    print("  - All buckets viable ≥ 50% AND hard L1+L3≥50%  →  green-light Phase A")
    print("  - Hard 30-50%   →  bump K=16 OR q=[.4,.3,.3] OR raise T to 1.2")
    print("  - Hard < 30%    →  drop hard from training (q=[.6,.4,0]) or iterate-DPO")


if __name__ == "__main__":
    main()
