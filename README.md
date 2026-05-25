<p align="center">
  <img src="assets/images/logo_v3.png" alt="GraphDancer" width="35%">
</p>

<h3 align="center">
GraphDancer: Training LLMs to Explore and Reason over Graphs via Two-Stage Curriculum Post-Training
</h3>

<!--- BADGES: START --->
<p align="center">
    <a href="https://arxiv.org/abs/2602.02518"><img src="https://img.shields.io/badge/arXiv-2602.02518-B31B1B.svg?logo=arxiv" alt="arXiv"></a>
    <a href="https://huggingface.co/collections/yuyangbai/graphdancer"><img src="https://img.shields.io/badge/Huggingface-Model-FFD21E.svg?logo=huggingface" alt="Huggingface Model"></a>
    <a href="https://yuyangbai.com/graphdancer/"><img src="https://img.shields.io/badge/Website-GraphDancer-E5426E?logo=googlechrome" alt="Website"></a>
</p>
<!--- BADGES: END --->

**GraphDancer** is a two-stage post-training framework that teaches Large Language Models (LLMs) to interleave natural-language reasoning with graph function execution. The first stage (**Curriculum-PPO**) uses proximal policy optimization with executable rule-based rewards to teach the model how to interact with the graph. The second stage (**Curriculum-DPO**) refines the PPO policy on self-generated preference pairs to make interactions more grounded and efficient. Both stages are organized by a **graph-aware curriculum** that progressively increases task difficulty based on the structural complexity of information-seeking trajectories (S-rounds and E-rounds). With a 3B backbone, GraphDancer outperforms graph-agent baselines built on substantially larger or stronger models.

This repository contains the official implementation of the paper: *GraphDancer: Training LLMs to Explore and Reason over Graphs via Two-Stage Curriculum Post-Training*.

## 🌟 Key Features

* **Two-Stage Post-Training**: Curriculum-PPO (online, rule-based reward over executable graph calls) followed by Curriculum-DPO (offline preference learning over self-sampled trajectories ranked by six trajectory-level keys with fixed tie-break priority).
* **Interleaved Reasoning & Action**: Trains agents to alternate between `<think>` (reasoning), `<graph>` (function execution), `<information>` (environment feedback), and `<answer>` blocks.
* **Graph-Aware Curriculum**: A graph-specific scheduler that decomposes trajectories into Singleton lookup rounds (S-rounds) and Neighborhood expansion rounds (E-rounds), and biases batch composition from Easy to Hard via a time-varying mixture.
* **Rule-Based Reward Shaping**: Format-aware and correctness-based rewards with no learned reward model.
* **Cross-Domain Generalization**: Train on the Academic domain and evaluate on four unseen GRBench domains (E-commerce, Literature, Healthcare, Legal) plus out-of-distribution question types.

## 📂 Repository Structure

```text
├── graphdancer/            # Core logic for multi-turn generation and tool management
│   └── llm_agent/
│       ├── generation*.py  # Rollout drivers for the interleaved think/graph/information loop
│       ├── tensor_helper.py
│       └── tools/          # Graph function executor and retriever
├── verl/
│   ├── trainer/
│   │   ├── ppo/            # Stage 1: Curriculum-PPO (online RL)
│   │   └── dpo/            # Stage 2: Curriculum-DPO (offline preference learning)
│   ├── experimental/
│   │   └── dataset/        # E2H biased-mixture curriculum sampler (crl_e2h.py)
│   └── ...                 # upstream verl utilities
├── scripts/
│   ├── launchers/          # Entry points for Stage 1 / Stage 2 training and evaluation
│   ├── curriculum/         # Curriculum prompt templates (graph-aware and Graph-CoT)
│   ├── dpo/                # Stage 2 preference-pair construction + analysis
│   ├── graph_data/         # Utilities for converting raw graph data to parquet
│   └── build_*.sh          # Dataset construction scripts
├── data/                   # Placeholder for processed datasets and graph files
├── eval.py                 # Evaluation script (Exact Match, BLEU, ROUGE, GPT4-Score)
└── LICENSE                 # CC BY 4.0
```

## 🛠️ Installation

Create a virtual environment and install dependencies. We recommend Python 3.10+.

```bash
conda create -n graphdancer python=3.10 -y
conda activate graphdancer

# Install PyTorch (adjust cuda version as needed)
pip install torch torchvision torchaudio

# Install repository dependencies and the package in editable mode
pip install -r requirements.txt
pip install -e .
```

## 📊 Data Preparation

GraphDancer uses the graph environments and QA formats from **GRBench** (Jin et al., 2024). We provide pre-processed datasets on Hugging Face for a quick start.

### 1. Download Graph Data

```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/yuyangbai/GRBench-copy graphs

# For the legal graph, concatenate the chunks
cd graphs/legal
cat chunk_* > graph.json
cd ../..
```

### 2. Download Training and Test Data

```bash
pip install huggingface_hub

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

repo_id = "yuyangbai/GraphDancer-data"

os.makedirs("data/train", exist_ok=True)
for domain in ["biomedical", "goodreads", "amazon", "legal"]:
    os.makedirs(f"data/test/{domain}", exist_ok=True)

train_file = hf_hub_download(repo_id=repo_id, filename="train/train.parquet", repo_type="dataset")
shutil.copy2(train_file, "data/train/train.parquet")

for domain in ["biomedical", "goodreads", "amazon", "legal"]:
    test_file = hf_hub_download(repo_id=repo_id, filename=f"test/{domain}/test.parquet", repo_type="dataset")
    shutil.copy2(test_file, f"data/test/{domain}/test.parquet")
EOF
```

### 3. Expected Directory Structure

```text
data/
├── graphs/                      # Graph data from GRBench
│   ├── dblp/graph.json          # Academic
│   ├── biomedical/graph.json    # Healthcare
│   ├── legal/graph.json         # Legal (chunked on HF)
│   ├── amazon/graph.json        # E-commerce
│   └── goodreads/graph.json     # Literature
├── train/train.parquet          # Academic training set
└── test/<domain>/test.parquet   # Per-domain test sets
```

## 🚀 Training — Stage 1 (Curriculum-PPO)

Stage 1 trains the policy with executable rule-based rewards on the Academic domain.

### Curriculum-PPO (Proposed Stage 1)

Trains with the biased-mixture curriculum scheduler described in §2.4 of the paper.

```bash
export GRAPH_DIR=./data/graphs
export DATA_DIR=./data/train
export OUTPUT_DIR=./checkpoints/graphdancer_curriculum_ppo

bash scripts/launchers/train_graphdancer_ppo_e2h_mix.sh
```

### Pure Easy-to-Hard (Ablation)

```bash
bash scripts/launchers/train_graphdancer_ppo_e2h.sh
```

### Vanilla PPO Baseline

```bash
bash scripts/launchers/train_graphdancer_ppo.sh
```

## 🚀 Training — Stage 2 (Curriculum-DPO)

Stage 2 refines the Stage 1 checkpoint via DPO on self-generated preference pairs ranked by the six trajectory-level keys defined in §2.3 of the paper with fixed tie-break priority. The full procedure is described in §2.3, Algorithm 1 Phase 2, and Appendix A.2 of the paper.

### 1. Sample M=8 trajectories from the PPO checkpoint

Run the verl rollout entry point in eval-only mode against the Academic training set with `val_kwargs.n=8`, dumping per-trajectory traces to `results.jsonl`. Default decoding: temperature 1.0, top-p 0.95, max_turns 10.

### 2. Build the preference-pair parquet

```bash
python scripts/dpo/build_preference_pairs.py \
    --jsonl_glob "./data/dpo/k8_rollouts/*/results.jsonl" \
    --train_parquet ./data/train/train.parquet \
    --out_parquet ./data/dpo/pair_parquet/pairs.parquet \
    --lex_version v1
```

The script ranks the M=8 trajectories per question by `(EM, EH, VF, -loop_limit, -invalid_tool, -n_graph_rounds)` with fixed tie-break priority and emits one extreme pair per qid (rank-1 vs rank-M), along with character-level `<information>` spans needed for the agent-token mask.

### 3. Run DPO training

```bash
export PAIR_PARQUET=./data/dpo/pair_parquet/pairs.parquet
export PPO_CHECKPOINT=./checkpoints/graphdancer_curriculum_ppo/actor/global_step_200
export OUTPUT_DIR=./checkpoints/graphdancer_curriculum_dpo

bash scripts/launchers/train_graphdancer_dpo.sh
```

The launcher trains for 100 steps with β=0.1, learning rate 2e-7, linear warmup over 5% of steps, and the same E2H biased-mixture curriculum used in Stage 1.

## 📝 Evaluation

Evaluate a trained checkpoint on a specific domain (e.g., Healthcare):

```bash
export GRAPH_DIR=./data/graphs
export DOMAIN=biomedical
export CHECKPOINT=./checkpoints/graphdancer_curriculum_dpo/actor/global_step_100
export BASE_MODEL=$CHECKPOINT

bash scripts/launchers/eval_graphdancer.sh
```

### Scoring

```bash
# Basic metrics
python eval.py --result_file results/biomedical/results.jsonl

# GPT-4 based scoring (requires OPENAI_API_KEY)
python eval.py --result_file results/biomedical/results.jsonl --use_gpt4_score
```

## Acknowledgements

This project builds upon several excellent open-source projects:

- [Graph-CoT](https://github.com/PeterGriffinJin/Graph-CoT) for the graph function implementation and the GRBench benchmark.
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) for inspiring our agentic RL training framework design.
- [veRL](https://github.com/volcengine/verl) for a robust and scalable RL training framework.
- [vLLM](https://github.com/vllm-project/vllm) for efficient and high-throughput LLM inference.

We also thank [Lambda](https://lambda.ai/) for providing GPU resources.

## 📄 License

This repository is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See [`LICENSE`](LICENSE) for the full text.

The `verl/` subdirectory contains upstream code from the [verl](https://github.com/volcengine/verl) project and is licensed under Apache 2.0; the original copyright and license headers are retained.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{bai2026graphdancertrainingllmsexplore,
      title={GraphDancer: Training LLMs to Explore and Reason over Graphs via Two-Stage Curriculum Post-Training},
      author={Yuyang Bai and Zhuofeng Li and Ping Nie and Yu Wang and Jianwen Xie and Yu Zhang},
      year={2026},
      eprint={2602.02518},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02518},
}
```
