#!/usr/bin/env python3
"""
Behavioral analysis on evaluation traces.

Primary goal: produce a LaTeX-ready table for:
  - VF (Format Validity): fraction of episodes that follow required structure
  - CV (Call Validity): fraction of tool calls that pass schema validation
  - EH (Evidence Hit): fraction of episodes where normalized gold appears in any <information> block

Inputs:
- One or more models, each provided as --model "NAME=PATH"
  where PATH is either:
    - a directory that contains results.jsonl (e.g., results/<exp>/.../results.jsonl), or
    - a specific results.jsonl path

Outputs (written to --out_dir):
- behavior_table.json: machine-readable per-domain metrics
- behavior_table.tex:  LaTeX table body you can paste into paper

Optional:
- If --plot is set and matplotlib is installed, also saves interaction depth figures.

Assumptions about results.jsonl record:
  - extra_info.domain (e.g., biomedical/goodreads/amazon/legal/academic/...)
  - extra_info.difficulty (easy/medium/hard) if you want hard-only filters
  - gt_answer (string)
  - trace: list[dict] per step with fields like:
      prediction (string), action ("graph"|"answer"|None), content (string), observation (string)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


VALID_GRAPH_FUNCS = {"RetrieveNode", "NeighbourCheck", "NodeFeature", "NodeDegree"}

# Optional mapping from dataset domain to paper domain label.
DEFAULT_DOMAIN_MAP = {
    "amazon": "E-commerce",
    "goodreads": "Literature",
    "biomedical": "Healthcare",
    "legal": "Legal",
}

# Default location of processed question difficulty (Graph-CoT style).
# Keep it anonymous/portable: prefer env var, otherwise use a repo-relative path.
DEFAULT_PROCESSED_DATA_DIR = os.environ.get(
    "PROCESSED_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed_data_newTemp")),
)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def find_results_jsonl(root_or_file: str) -> List[str]:
    """Return a list of results.jsonl paths (possibly multiple domains)."""
    if os.path.isfile(root_or_file):
        return [root_or_file]

    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_or_file):
        if "results.jsonl" in filenames:
            paths.append(os.path.join(dirpath, "results.jsonl"))
    paths.sort()
    return paths


def normalize_answer(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = re.sub(r"\b(a|an|the|usd)\b", " ", s)
    s = re.sub(r"[\W_]+", " ", s)  # remove punctuation-ish
    s = " ".join(s.split())
    return s


def normalize_question(text: str) -> str:
    """Normalize questions to improve matching stability.

    Mirrors `working/Graph-CoT-vllm/eval.py`:
    - Collapse whitespace
    - Remove spaces before punctuation like ?!,.:; and closing quotes/brackets
    - Trim trailing punctuation like ?!,.:;
    """
    if text is None:
        return ""
    t = str(text)
    t = " ".join(t.split())
    t = re.sub(r"\s+([?!,.:;\)\]\}\"\'])", r"\1", t)
    t = t.rstrip(" \t\r\n?!,.:;")
    return t


def canon_difficulty_word(label: Any) -> Optional[str]:
    """Canonicalize difficulty to one of: easy/medium/hard/ood. Returns None if unknown."""
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in {"e", "easy", "simple", "low"}:
        return "easy"
    if s in {"m", "mid", "medium", "normal", "moderate"}:
        return "medium"
    if s in {"h", "hard", "difficult", "high"}:
        return "hard"
    if s in {"ood", "out-of-domain", "out_of_domain", "out of domain"}:
        return "ood"
    return None


def _domain_to_processed_data_jsonl(processed_data_dir: str, domain: str) -> Optional[str]:
    """Map a results.jsonl domain (extra_info.domain) to processed_data_newTemp path."""
    if not domain:
        return None
    d = str(domain)

    # Special handling: maple subdomains can be encoded as "maple_Biology" or "maple-Biology"
    if d.startswith("maple_") or d.startswith("maple-"):
        sep = "_" if "maple_" in d else "-"
        parts = d.split(sep, 1)
        if len(parts) == 2 and parts[1]:
            sub = parts[1]
            p = os.path.join(processed_data_dir, "maple", sub, "data.json")
            return p if os.path.isfile(p) else None

    # Common case: processed_data_newTemp/<domain>/data.json
    p = os.path.join(processed_data_dir, d, "data.json")
    if os.path.isfile(p):
        return p
    return None


def build_processed_difficulty_mapping(processed_data_dir: str, domains: List[str]) -> Dict[str, Dict[str, str]]:
    """Build per-domain mapping: domain -> (question_norm -> difficulty_word) from processed_data_newTemp."""
    out: Dict[str, Dict[str, str]] = {}
    for domain in sorted(set(domains)):
        path = _domain_to_processed_data_jsonl(processed_data_dir, domain)
        if path is None:
            continue
        mapping: Dict[str, str] = {}
        for obj in _iter_jsonl(path):
            q = obj.get("question")
            diff = obj.get("new_level")
            qn = normalize_question(q)
            lab = canon_difficulty_word(diff)
            if not qn or lab is None:
                continue
            # If duplicates exist, keep first; in practice they should be consistent.
            mapping.setdefault(qn, lab)
        if mapping:
            out[domain] = mapping
    return out


TAG_RE = re.compile(r"<(graph|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
CALL_RE = re.compile(r"^([A-Za-z_]\w*)\[(.*)\]$", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
INFO_RE = re.compile(r"<information>.*?</information>", re.DOTALL | re.IGNORECASE)


def format_valid(pred: str) -> bool:
    if not isinstance(pred, str):
        return False
    m = TAG_RE.search(pred)
    return m is not None


def episode_format_valid(trace: Any) -> bool:
    """
    Episode-level format validity (VF):
    - Every recorded step must contain <think>...</think> AND one of <graph>...</graph> or <answer>...</answer>.
    - For graph steps, observation must include <information>...</information>.

    This is stricter than per-step validity; matches the "fraction of episodes" definition.
    """
    if not isinstance(trace, list) or not trace:
        return False
    for t in trace:
        if not isinstance(t, dict):
            return False
        pred = t.get("prediction", "")
        if not isinstance(pred, str) or not pred:
            return False
        if THINK_RE.search(pred) is None:
            return False
        if TAG_RE.search(pred) is None:
            return False
        action = t.get("action", None)
        if action == "graph":
            obs = t.get("observation", "")
            if not isinstance(obs, str) or INFO_RE.search(obs) is None:
                return False
    return True


def graph_call_valid(content: str) -> bool:
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
    # For the 2-arg functions, require at least one comma to separate two args.
    if func in {"NeighbourCheck", "NodeFeature", "NodeDegree"}:
        return "," in args
    return True


def count_graph_rounds(trace: Any) -> int:
    if not isinstance(trace, list):
        return 0
    return sum(1 for t in trace if isinstance(t, dict) and t.get("action") == "graph")


def count_turns(trace: Any) -> int:
    """Number of recorded steps (including answer step)."""
    if not isinstance(trace, list):
        return 0
    return sum(1 for t in trace if isinstance(t, dict))


def tool_validity_metrics(trace: Any) -> Dict[str, float]:
    """Compute per-sample validity metrics based on trace."""
    if not isinstance(trace, list) or not trace:
        return {
            "format_valid_rate": 0.0,
            "graph_call_valid_rate": 0.0,
            "invalid_action_rate": 1.0,
        }

    total = 0
    fmt_ok = 0
    graph_total = 0
    graph_ok = 0
    invalid_action = 0

    for t in trace:
        if not isinstance(t, dict):
            continue
        total += 1
        pred = t.get("prediction", "")
        if format_valid(pred):
            fmt_ok += 1
        action = t.get("action", None)
        valid_action = t.get("valid_action", None)
        if action is None or (isinstance(valid_action, int) and valid_action == 0):
            invalid_action += 1
        if action == "graph":
            graph_total += 1
            if graph_call_valid(t.get("content", "")):
                graph_ok += 1

    if total == 0:
        return {
            "format_valid_rate": 0.0,
            "graph_call_valid_rate": 0.0,
            "invalid_action_rate": 1.0,
        }

    return {
        "format_valid_rate": fmt_ok / total,
        "graph_call_valid_rate": (graph_ok / graph_total) if graph_total > 0 else 0.0,
        "invalid_action_rate": invalid_action / total,
    }


def evidence_hit(trace: Any, gt_answer: str) -> bool:
    """Whether normalized gt_answer appears in any observation text."""
    gt = normalize_answer(gt_answer or "")
    if not gt:
        return False
    if not isinstance(trace, list):
        return False
    for t in trace:
        if not isinstance(t, dict):
            continue
        obs = t.get("observation", "")
        if not isinstance(obs, str) or not obs:
            continue
        if gt in normalize_answer(obs):
            return True
    return False


@dataclass
class SampleRow:
    qid: str
    domain: str
    difficulty: str
    graph_rounds: int
    turns: int
    format_valid_rate: float
    graph_call_valid_rate: float
    invalid_action_rate: float
    evidence_hit: int
    episode_format_valid: int
    # For exact call-level aggregation of CV (schema-valid tool calls)
    graph_calls_total: int
    graph_calls_valid: int


def load_experiment(
    exp: str,
    domain_filter: Optional[str] = None,
    processed_difficulty: Optional[Dict[str, Dict[str, str]]] = None,
    prefer_processed_difficulty: bool = False,
) -> List[SampleRow]:
    rows: List[SampleRow] = []
    for jsonl_path in find_results_jsonl(exp):
        # If exp points to a big results dir (results/<exp>/...), we may see multiple domains.
        for rec in _iter_jsonl(jsonl_path):
            extra = rec.get("extra_info") or {}
            domain = str(extra.get("domain", "unknown"))
            if domain_filter and domain != domain_filter:
                continue
            # Difficulty: by default, use extra_info.difficulty if present; otherwise backfill using processed_data_newTemp mapping.
            difficulty = canon_difficulty_word(extra.get("difficulty", None)) or "unknown"
            if processed_difficulty is not None:
                qn = normalize_question(rec.get("question", ""))
                mapped = processed_difficulty.get(domain, {}).get(qn)
                if prefer_processed_difficulty and mapped is not None:
                    difficulty = mapped
                elif difficulty == "unknown" and mapped is not None:
                    difficulty = mapped
            qid = rec.get("qid")
            if qid is None:
                qid = rec.get("uid")
            if qid is None:
                qid = "unknown"
            qid = str(qid)

            trace = rec.get("trace", None)
            gt = rec.get("gt_answer", "")

            v = tool_validity_metrics(trace)
            # Exact call-level validity counts
            graph_calls_total = 0
            graph_calls_valid = 0
            if isinstance(trace, list):
                for t in trace:
                    if not isinstance(t, dict):
                        continue
                    if t.get("action") == "graph":
                        graph_calls_total += 1
                        if graph_call_valid(t.get("content", "")):
                            graph_calls_valid += 1
            rows.append(
                SampleRow(
                    qid=qid,
                    domain=domain,
                    difficulty=difficulty,
                    graph_rounds=count_graph_rounds(trace),
                    turns=count_turns(trace),
                    format_valid_rate=float(v["format_valid_rate"]),
                    graph_call_valid_rate=float(v["graph_call_valid_rate"]),
                    invalid_action_rate=float(v["invalid_action_rate"]),
                    evidence_hit=1 if evidence_hit(trace, gt) else 0,
                    episode_format_valid=1 if episode_format_valid(trace) else 0,
                    graph_calls_total=graph_calls_total,
                    graph_calls_valid=graph_calls_valid,
                )
            )
    return rows


def _bucket_graph_rounds(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return ">=4"


def _dist_percent(rows: List[SampleRow], hard_only: bool = False) -> Dict[str, float]:
    filt = [r for r in rows if (not hard_only or r.difficulty.lower() == "hard")]
    if not filt:
        return {"1": 0.0, "2": 0.0, "3": 0.0, ">=4": 0.0}
    buckets = {"1": 0, "2": 0, "3": 0, ">=4": 0}
    for r in filt:
        buckets[_bucket_graph_rounds(r.graph_rounds)] += 1
    total = len(filt)
    return {k: (v / total) * 100.0 for k, v in buckets.items()}


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(rows: List[SampleRow]) -> Dict[str, Any]:
    return {
        "n": len(rows),
        "n_hard": sum(1 for r in rows if r.difficulty.lower() == "hard"),
        "avg_graph_rounds": _mean([float(r.graph_rounds) for r in rows]),
        "avg_turns": _mean([float(r.turns) for r in rows]),
        "format_valid_rate_mean": _mean([r.format_valid_rate for r in rows]),
        "graph_call_valid_rate_mean": _mean([r.graph_call_valid_rate for r in rows]),
        "invalid_action_rate_mean": _mean([r.invalid_action_rate for r in rows]),
        "evidence_hit_rate": (_mean([float(r.evidence_hit) for r in rows]) if rows else 0.0),
        # Episode-level VF (this is the one that matches the paper definition)
        "episode_format_valid_rate": (_mean([float(r.episode_format_valid) for r in rows]) if rows else 0.0),
        "dist_graph_rounds_pct": _dist_percent(rows, hard_only=False),
        "dist_graph_rounds_pct_hard": _dist_percent(rows, hard_only=True),
    }

def _latex_escape(s: str) -> str:
    # minimal escaping for table content
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def build_behavior_table(
    model_to_rows: Dict[str, List[SampleRow]],
    domain_map: Dict[str, str],
) -> Tuple[Dict[str, Any], str]:
    """
    Returns:
      - table_json[paper_domain][model] = {VF, CV, EH, n}
      - latex_body (rows only; you can wrap with your own tabular env)
    """
    # Collect domains observed
    domains_seen = set()
    for rows in model_to_rows.values():
        for r in rows:
            domains_seen.add(r.domain)

    # Use mapped labels when available; otherwise keep original key
    paper_domains = sorted({domain_map.get(d, d) for d in domains_seen})

    table: Dict[str, Any] = {}
    lines: List[str] = []

    for pd in paper_domains:
        table[pd] = {}
        # collect model metrics for this pd
        for model, rows in model_to_rows.items():
            sub = [r for r in rows if domain_map.get(r.domain, r.domain) == pd]
            if not sub:
                continue
            # CV is computed over tool calls (exact, call-level)
            graph_calls = sum(int(r.graph_calls_total) for r in sub)
            graph_calls_valid = sum(int(r.graph_calls_valid) for r in sub)
            cv = (graph_calls_valid / graph_calls) if graph_calls > 0 else 0.0

            vf = _mean([float(r.episode_format_valid) for r in sub]) if sub else 0.0
            eh = _mean([float(r.evidence_hit) for r in sub]) if sub else 0.0

            table[pd][model] = {"VF": vf, "CV": cv, "EH": eh, "n": len(sub)}

        # LaTeX rows for this domain
        # Sort models in insertion order of model_to_rows
        first = True
        for model in model_to_rows.keys():
            if model not in table[pd]:
                continue
            m = table[pd][model]
            vf = 100.0 * float(m["VF"])
            cv = 100.0 * float(m["CV"])
            eh = 100.0 * float(m["EH"])
            if first:
                lines.append(
                    f"{_latex_escape(pd)} & {_latex_escape(model)} & {vf:.1f} & {cv:.1f} & {eh:.1f} \\\\"
                )
                first = False
            else:
                lines.append(
                    f" & {_latex_escape(model)} & {vf:.1f} & {cv:.1f} & {eh:.1f} \\\\"
                )
        lines.append("\\midrule")

    latex_body = "\n".join(lines).rstrip()
    return table, latex_body


def build_behavior_table_by_difficulty(
    model_to_rows: Dict[str, List[SampleRow]],
    domain_map: Dict[str, str],
    difficulty_buckets: List[str],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Returns:
      - table_json[difficulty][paper_domain][model] = {VF, CV, EH, n}
      - latex_bodies[difficulty] = latex table body rows (domain sections)
    """
    out_json: Dict[str, Any] = {}
    out_tex: Dict[str, str] = {}

    # For each bucket, filter rows then reuse build_behavior_table
    for bucket in difficulty_buckets:
        bucket_key = bucket
        out_json[bucket_key] = {}

        filtered: Dict[str, List[SampleRow]] = {}
        for model, rows in model_to_rows.items():
            sub = [r for r in rows if str(r.difficulty).lower() == bucket.lower()]
            filtered[model] = sub

        table, tex = build_behavior_table(model_to_rows=filtered, domain_map=domain_map)
        out_json[bucket_key] = table
        out_tex[bucket_key] = tex

    return out_json, out_tex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help='Model spec like "Vanilla PPO=results/<exp>/..." (can be repeated).',
    )
    ap.add_argument("--domain_map_json", default=None, help="Optional JSON file: {raw_domain: paper_domain}.")
    ap.add_argument("--plot", action="store_true", help="If set, also save interaction depth plots (needs matplotlib).")
    ap.add_argument(
        "--difficulty_buckets",
        default="easy,medium,hard",
        help="Comma-separated difficulty buckets to produce per-bucket behavior tables (default: easy,medium,hard).",
    )
    ap.add_argument(
        "--processed_data_dir",
        default=DEFAULT_PROCESSED_DATA_DIR,
        help=(
            "Directory like /data/.../processed_data_newTemp containing <domain>/data.json with {question,new_level}. "
            "Used to backfill difficulty exactly like Graph-CoT."
        ),
    )
    ap.add_argument(
        "--prefer_processed_difficulty",
        action="store_true",
        help="If set, always use processed_data_newTemp new_level as difficulty (overrides extra_info.difficulty).",
    )
    args = ap.parse_args()

    # Parse model specs
    model_to_path: Dict[str, str] = {}
    for spec in args.model:
        if "=" not in spec:
            raise ValueError(f"--model must be in NAME=PATH form, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --model spec: {spec}")
        model_to_path[name] = path

    domain_map = dict(DEFAULT_DOMAIN_MAP)
    if args.domain_map_json is not None:
        with open(args.domain_map_json, "r", encoding="utf-8") as fp:
            user_map = json.load(fp)
        if isinstance(user_map, dict):
            domain_map.update({str(k): str(v) for k, v in user_map.items()})

    model_to_rows: Dict[str, List[SampleRow]] = {}
    model_to_summary: Dict[str, Any] = {}

    # Determine which domains appear in the provided results (by directory name),
    # so we only load processed difficulty for those datasets.
    domains_seen: List[str] = []
    for _, root in model_to_path.items():
        for p in find_results_jsonl(root):
            # results/<exp>/<domain>/results.jsonl -> domain folder name
            try:
                dom = os.path.basename(os.path.dirname(p))
                if dom:
                    domains_seen.append(dom)
            except Exception:
                pass
    processed_map = None
    if args.processed_data_dir and os.path.isdir(args.processed_data_dir) and domains_seen:
        processed_map = build_processed_difficulty_mapping(args.processed_data_dir, domains_seen)

    for model, path in model_to_path.items():
        rows = load_experiment(
            path,
            domain_filter=None,
            processed_difficulty=processed_map,
            prefer_processed_difficulty=bool(args.prefer_processed_difficulty),
        )
        model_to_rows[model] = rows
        model_to_summary[model] = summarize(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    # Write per-model global summaries
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(model_to_summary, fp, ensure_ascii=False, indent=2)

    # Build and write behavioral diagnostics table
    table_json, latex_body = build_behavior_table(model_to_rows=model_to_rows, domain_map=domain_map)
    with open(os.path.join(args.out_dir, "behavior_table.json"), "w", encoding="utf-8") as fp:
        json.dump(table_json, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "behavior_table.tex"), "w", encoding="utf-8") as fp:
        fp.write(latex_body + "\n")

    # Build and write per-difficulty tables
    diff_buckets = [b.strip() for b in str(args.difficulty_buckets).split(",") if b.strip()]
    table_by_diff, tex_by_diff = build_behavior_table_by_difficulty(
        model_to_rows=model_to_rows, domain_map=domain_map, difficulty_buckets=diff_buckets
    )
    with open(os.path.join(args.out_dir, "behavior_table_by_difficulty.json"), "w", encoding="utf-8") as fp:
        json.dump(table_by_diff, fp, ensure_ascii=False, indent=2)
    for b, tex in tex_by_diff.items():
        out_path = os.path.join(args.out_dir, f"behavior_table_{b}.tex")
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write(tex + "\n")

    # Optional plots (avoid hard dependency on matplotlib)
    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "--plot was set but matplotlib is not available. Install matplotlib or rerun without --plot."
            ) from e
        # Reuse old plotting logic by comparing first two models only (if present)
        models = list(model_to_summary.keys())
        if len(models) >= 2:
            label_a, label_b = models[0], models[1]
            summ_a, summ_b = model_to_summary[label_a], model_to_summary[label_b]
            _plot_grouped_bar(
                dist_a=summ_a["dist_graph_rounds_pct"],
                dist_b=summ_b["dist_graph_rounds_pct"],
                label_a=label_a,
                label_b=label_b,
                out_png=os.path.join(args.out_dir, "interaction_depth_overall.png"),
                title="Interaction Depth Distribution (overall)",
            )
            _plot_grouped_bar(
                dist_a=summ_a["dist_graph_rounds_pct_hard"],
                dist_b=summ_b["dist_graph_rounds_pct_hard"],
                label_a=label_a,
                label_b=label_b,
                out_png=os.path.join(args.out_dir, "interaction_depth_hard.png"),
                title="Interaction Depth Distribution (hard)",
            )

    print("Wrote:", os.path.join(args.out_dir, "behavior_table.json"))
    print("Wrote:", os.path.join(args.out_dir, "behavior_table.tex"))
    print("Wrote:", os.path.join(args.out_dir, "behavior_table_by_difficulty.json"))
    print("Wrote:", os.path.join(args.out_dir, "summary.json"))


def _plot_grouped_bar(dist_a: Dict[str, float], dist_b: Dict[str, float], label_a: str, label_b: str, out_png: str, title: str):
    # Local import; matplotlib is optional unless --plot is used.
    import matplotlib.pyplot as plt  # type: ignore

    cats = ["1", "2", "3", ">=4"]
    a = [dist_a.get(c, 0.0) for c in cats]
    b = [dist_b.get(c, 0.0) for c in cats]

    x = list(range(len(cats)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([xi - width / 2 for xi in x], a, width=width, label=label_a)
    ax.bar([xi + width / 2 for xi in x], b, width=width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Percent (%)")
    ax.set_xlabel("Number of <graph> rounds")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


