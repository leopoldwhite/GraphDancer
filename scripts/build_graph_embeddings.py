import argparse
import os
import tempfile
import shutil
from typing import Dict, Optional, List

from omegaconf import OmegaConf, DictConfig

from agents.graph_agent import GraphResourceConfig, GraphRouter


def _to_plain_dict(cfg):
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def build_graph_resource_configs(
    config_path: str,
    use_gpu: Optional[bool] = None,
    batch_size: Optional[int] = None,
    only_graphs: Optional[List[str]] = None,
) -> Dict[str, GraphResourceConfig]:
    """Build GraphResourceConfig objects from a PPO trainer config."""
    cfg = OmegaConf.load(config_path)
    graph_cfg = cfg.get("graph") or {}
    graph_cfg = _to_plain_dict(graph_cfg)

    defaults = {
        "embedder_name": graph_cfg.get("embedder_name", "sentence-transformers/all-mpnet-base-v2"),
        "embed_cache": graph_cfg.get("embed_cache", True),
        "embed_cache_dir": graph_cfg.get("embed_cache_dir"),
        "faiss_gpu": graph_cfg.get("faiss_gpu", False),
        "embed_batch_size": graph_cfg.get("embed_batch_size", 128),
    }

    # Historically, some configs use:
    # - graph.graphs: {alias: {graph_dir, ...}}
    # while others use:
    # - graph.domains: {alias: {graph_dir, ...}}
    # - graph.graph_dir (+ graph.dataset) for a single default graph
    graphs_cfg = graph_cfg.get("graphs", {}) or {}
    if not graphs_cfg:
        graphs_cfg = graph_cfg.get("domains", {}) or {}

    graph_configs: Dict[str, GraphResourceConfig] = {}
    # Also consider a single default graph (graph_dir + dataset).
    default_graph_dir = graph_cfg.get("graph_dir")
    default_graph_name = graph_cfg.get("dataset") or graph_cfg.get("default_graph") or "default"
    if default_graph_dir:
        graphs_cfg = dict(graphs_cfg)
        graphs_cfg.setdefault(
            str(default_graph_name),
            {
                "graph_dir": default_graph_dir,
                "graph_name": default_graph_name,
                "embedder_name": graph_cfg.get("embedder_name", defaults["embedder_name"]),
                "embed_cache": graph_cfg.get("embed_cache", defaults["embed_cache"]),
                "embed_cache_dir": graph_cfg.get("embed_cache_dir", defaults["embed_cache_dir"]),
                "faiss_gpu": graph_cfg.get("faiss_gpu", defaults["faiss_gpu"]),
                "embed_batch_size": graph_cfg.get("embed_batch_size", defaults["embed_batch_size"]),
            },
        )
    for alias, g_cfg in graphs_cfg.items():
        if g_cfg is None:
            continue
        if only_graphs is not None and alias not in only_graphs:
            continue

        graph_dir = g_cfg.get("graph_dir")
        if not graph_dir:
            print(f"[build-graphs] Skip graph '{alias}': missing graph_dir")
            continue

        graph_name = g_cfg.get("graph_name") or alias
        embedder_name = g_cfg.get("embedder_name", defaults["embedder_name"])
        embed_cache = g_cfg.get("embed_cache", defaults["embed_cache"])
        embed_cache_dir = g_cfg.get("embed_cache_dir", defaults["embed_cache_dir"])
        faiss_gpu = g_cfg.get("faiss_gpu", defaults["faiss_gpu"])
        embed_batch_size = g_cfg.get("embed_batch_size", defaults["embed_batch_size"])

        if use_gpu is not None:
            faiss_gpu = use_gpu
        if batch_size is not None:
            embed_batch_size = batch_size

        graph_configs[graph_name] = GraphResourceConfig(
            graph_name=graph_name,
            graph_dir=graph_dir,
            embedder_name=embedder_name,
            embed_cache=embed_cache,
            embed_cache_dir=embed_cache_dir,
            faiss_gpu=faiss_gpu,
            embed_batch_size=embed_batch_size,
        )

    return graph_configs


def list_graph_aliases(config_path: str) -> List[str]:
    """Return the list of graph aliases defined in the PPO trainer config."""
    cfg = OmegaConf.load(config_path)
    graph_cfg = cfg.get("graph") or {}
    graph_cfg = _to_plain_dict(graph_cfg)
    graphs_cfg = graph_cfg.get("graphs", {}) or {}
    if not graphs_cfg:
        graphs_cfg = graph_cfg.get("domains", {}) or {}

    aliases: List[str] = []
    for alias, g_cfg in graphs_cfg.items():
        if not g_cfg:
            continue
        graph_dir = g_cfg.get("graph_dir")
        if not graph_dir:
            continue
        aliases.append(str(alias))
    # single-graph config
    if graph_cfg.get("graph_dir"):
        aliases.append(str(graph_cfg.get("dataset") or graph_cfg.get("default_graph") or "default"))
    return aliases


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute graph embeddings (and build Faiss indexes) for all graphs "
            "defined in a PPO trainer config."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="verl/trainer/config/ppo_trainer.yaml",
        help="Path to PPO trainer config YAML (absolute or relative to project root).",
    )
    parser.add_argument(
        "--use-gpu",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override graph.faiss_gpu (1: True, 0: False). If not set, use config value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override graph.embed_batch_size used for embedding.",
    )
    parser.add_argument(
        "--graphs",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of graph names (aliases) to build. Default: all graphs in config.",
    )
    parser.add_argument(
        "--save-index",
        action="store_true",
        help="If set, also persist the Faiss index to cache_dir for each graph.",
    )
    parser.add_argument(
        "--list-graphs",
        action="store_true",
        help="List available graph aliases in the config and exit.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        # Interpret relative paths as relative to project root (one level above scripts/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, config_path)

    if args.list_graphs:
        for name in list_graph_aliases(config_path):
            print(name)
        return

    use_gpu_override = None
    if args.use_gpu is not None:
        use_gpu_override = bool(args.use_gpu)

    graph_configs = build_graph_resource_configs(
        config_path=config_path,
        use_gpu=use_gpu_override,
        batch_size=args.batch_size,
        only_graphs=args.graphs,
    )

    if not graph_configs:
        print("[build-graphs] No graphs found in config; nothing to do.")
        return

    router = GraphRouter(graph_configs, default_graph=None)

    for graph_name, res_cfg in graph_configs.items():
        if res_cfg.embed_cache:
            save_model_name = res_cfg.embedder_name.split("/")[-1]
            cache_path = os.path.join(res_cfg.embed_cache_dir, f"cache-{save_model_name}.pkl")
            index_path = os.path.join(res_cfg.embed_cache_dir, f"faiss-{save_model_name}.index")

            if args.save_index:
                if os.path.exists(cache_path) and os.path.exists(index_path):
                    print(f"[build-graphs] Skipping '{graph_name}': Cache and Index already exist.")
                    continue
            elif os.path.exists(cache_path):
                print(f"[build-graphs] Skipping '{graph_name}': Cache already exists.")
                continue

        print(
            f"[build-graphs] Building embeddings for graph '{graph_name}' "
            f"from '{res_cfg.graph_dir}' (faiss_gpu={res_cfg.faiss_gpu}, "
            f"batch_size={res_cfg.embed_batch_size})"
        )
        context = router.get(graph_name)
        retriever = context.retriever
        if retriever is None:
            print(f"[build-graphs] WARNING: retriever is None for graph '{graph_name}', skipping.")
            continue

        # At this point, Retriever.__init__ has already:
        #  - processed the graph
        #  - computed / loaded embeddings and saved them to cache (if enabled)
        #  - built a Faiss index in memory

        if args.save_index and getattr(retriever, "index", None) is not None:
            try:
                import faiss
                import numpy as np  # noqa: F401  (may be useful if faiss requires numpy arrays)

                save_model_name = retriever.model_name.split("/")[-1]
                index_path = os.path.join(
                    retriever.cache_dir,
                    f"faiss-{save_model_name}.index",
                )

                index_to_save = retriever.index
                # If index lives on GPU, convert it back to CPU before writing.
                if res_cfg.faiss_gpu:
                    index_to_save = faiss.index_gpu_to_cpu(index_to_save)

                # Write to a temp file first to avoid corruption on failure
                dir_name = os.path.dirname(index_path)
                with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, mode='wb') as tmp_file:
                    # faiss.write_index takes a filename string, not a file object
                    tmp_path = tmp_file.name
                
                # We close the file handle so faiss can write to it by path
                # (NamedTemporaryFile creates it, so it exists)
                faiss.write_index(index_to_save, tmp_path)
                
                # Atomic move
                shutil.move(tmp_path, index_path)
                
                print(f"[build-graphs] Saved Faiss index for '{graph_name}' to '{index_path}'.")
            except Exception as e:
                print(
                    f"[build-graphs] WARNING: failed to save Faiss index for '{graph_name}': {e}"
                )

        print(f"[build-graphs] Done graph '{graph_name}'.")


if __name__ == "__main__":
    main()
