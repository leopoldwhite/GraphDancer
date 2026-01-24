<p align="center">
  <img src="assets/images/logo_v3.png" alt="GraphDancer" width="35%">
</p>

<h3 align="center">
GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning
</h3>

<!--- BADGES: START --->
<p align="center">
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B.svg?logo=arxiv" alt="arXiv"></a>
    <a href="https://huggingface.co/papers/XXXX.XXXXX"><img src="https://img.shields.io/badge/Huggingface-Paper-FFD21E.svg?logo=huggingface" alt="Huggingface Paper"></a>
    <a href="https://huggingface.co/GraphDancer"><img src="https://img.shields.io/badge/Huggingface-Model-FFD21E.svg?logo=huggingface" alt="Huggingface Model"></a>
</p>
<!--- BADGES: END --->

**GraphDancer** is a reinforcement learning framework that trains Large Language Models (LLMs) to adaptively interleave natural-language reasoning with graph function execution. By leveraging a **graph-aware curriculum**, GraphDancer enables moderate-sized models (e.g., 3B parameters) to internalize multi-hop reasoning skills and robustly generalize to unseen domains and heterogeneous graph structures.

This repository contains the official implementation of the paper: *GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning*.

## 🌟 Key Features

* **Interleaved Reasoning & Action**: Trains agents to alternate between `<think>` (reasoning) and `<graph>` (function execution) blocks.
* **Graph-Aware Curriculum**: A novel training scheduler that transitions from simple lookup tasks (S-rounds) to complex neighborhood expansions (E-rounds).
* **Rule-Based Reward Shaping**: Proximal Policy Optimization (PPO) integration with format-aware and correctness-based rewards.
* **Cross-Domain Generalization**: Proven effectiveness on the GRBench multi-domain benchmark (Academic, Biomedical, Legal, etc.).

## 📂 Repository Structure

```text
├── graphdancer/        # Core logic for multi-turn generation and tool management
├── verl/               # RL training runtime (PPO implementation)
├── scripts/
│   ├── launchers/      # Entry points for training and evaluation
│   ├── graph_data/     # Utilities for converting raw graph data to parquet
│   └── build_*.sh      # Dataset construction scripts
├── data/               # Placeholder for processed datasets and graph files
└── eval.py             # Evaluation script for computing EM, BLEU, and ROUGE

```

## 🛠️ Installation

Create a virtual environment and install dependencies. We recommend using Python 3.10+.

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

GraphDancer utilizes the graph environments and QA data formats from **GRBench**. We provide pre-processed datasets for quick start.

### 1. Download Data

#### Graph Data (Required)

Download the GRBench graph data from Hugging Face:

```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/yuyangbai/GRBench-copy graphs

# For legal graph, concatenate the chunks
cd graphs/legal
cat chunk_* > graph.json
cd ../..
```

#### Training Data (Required)

Download the pre-processed training data:

```bash
# Download from Hugging Face
git clone https://huggingface.co/datasets/yuyangbai/GraphDancer-data

# Move to correct location
mv GraphDancer-data/train data/train
mv GraphDancer-data/val data/val
```

### 2. Expected Directory Structure

After setup, your `data/` directory should look like this:

```text
data/
├── graphs/                     # Graph data from GRBench
│   ├── dblp/
│   │   └── graph.json         # 15GB - Academic (Computer Science)
│   ├── maple/
│   │   ├── Biology/graph.json       # 6.2GB
│   │   ├── Chemistry/graph.json     # 5.3GB
│   │   ├── Materials_Science/graph.json  # 3.8GB
│   │   ├── Medicine/graph.json      # 6.7GB
│   │   └── Physics/graph.json       # 4.5GB
│   ├── biomedical/
│   │   └── graph.json         # 154MB - Healthcare
│   ├── legal/
│   │   └── graph.json         # 84GB (chunked on HF)
│   ├── amazon/
│   │   └── graph.json         # 20GB - E-commerce
│   └── goodreads/
│       └── graph.json         # 9.1GB - Literature
│
├── train/                      # Training data (Academic domains)
│   └── train.parquet          # 800 samples
│       # Domains: dblp (150), maple_* (650 total)
│       # Difficulty: easy (370), medium (120), hard (310)
│
└── val/                        # Validation data (Biomedical domain)
    └── test.parquet           # 270 samples
        # Domain: biomedical (270)
        # Difficulty: easy (100), medium (150), hard (20)
```

### 3. Training Data Format

Each sample in the parquet files contains:

| Field | Description |
|-------|-------------|
| `prompt` | Formatted instruction with graph definition and few-shot examples |
| `reward_model.ground_truth` | Expected answer for reward computation |
| `extra_info.domain` | Domain identifier (e.g., `dblp`, `maple_Biology`, `biomedical`) |
| `extra_info.difficulty` | Difficulty level: `easy`, `medium`, or `hard` |

### 4. Build from Raw Data (Optional)

If you prefer to build training data from scratch:

```bash
# Organize raw Graph-CoT data into data/processed_data/
# Then run:
bash scripts/build_academic_biomedical.sh
```

## 🚀 Training

We support both standard PPO and our proposed Graph-Aware Curriculum Learning. Configuration is handled via environment variables.

### Graph-Aware Curriculum RL (Proposed Method)

Trains the model using the easy-to-hard curriculum scheduler described in the paper.

```bash
# Optional: customize paths (defaults work if you followed data setup)
export GRAPH_DIR=./data/graphs
export DATA_DIR=./data/train
export VAL_DIR=./data/val
export OUTPUT_DIR=./checkpoints/graphdancer_curriculum

# Optional: specify GPU devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launches training with the curriculum mixture sampler
bash scripts/launchers/train_graphdancer_ppo_e2h_mix.sh
```

### Vanilla PPO Baseline

Trains the model using vanilla PPO without curriculum scheduling.

```bash
export OUTPUT_DIR=./checkpoints/baseline_ppo
bash scripts/launchers/train_graphdancer_ppo.sh
```

### Training Configuration

Key parameters can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_DIR` | `./data/graphs` | Path to graph JSON files |
| `DATA_DIR` | `./data/train` | Path to training parquet files |
| `VAL_DIR` | `./data/val` | Path to validation parquet files |
| `BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Base model for training |
| `GPUS_PER_NODE` | `4` | Number of GPUs to use |
| `OUTPUT_DIR` | `./checkpoints` | Checkpoint save directory |

## 📝 Evaluation

To evaluate a trained model on a specific domain (e.g., Biomedical):

```bash
export GRAPH_DIR=./data/graphs
export DATA_DIR=./data/val
export CHECKPOINT=./checkpoints/graphdancer_curriculum/actor/global_step_100

bash scripts/launchers/eval_graphdancer.sh
```

### Scoring

Generate metrics (Exact Match, BLEU, ROUGE) from the inference results:

```bash
# Basic metrics
python eval.py --result_file results/biomedical/results.jsonl

# GPT-4 based scoring (requires OPENAI_API_KEY)
python eval.py --result_file results/biomedical/results.jsonl --use_gpt4_score
```

## 🙏 Acknowledgements

This project builds upon the shoulders of several excellent open-source projects:

- [Graph-CoT](https://github.com/PeterGriffinJin/Graph-CoT) for their graph function implementation and the GRBench benchmark.
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) for inspiring our agentic RL training framework design.
- [veRL](https://github.com/volcengine/verl) for providing a robust and scalable RL training framework.
- [vLLM](https://github.com/vllm-project/vllm) for efficient and high-throughput LLM inference.


We also thank [Lambda](https://lambda.ai/) for providing GPU resources!
