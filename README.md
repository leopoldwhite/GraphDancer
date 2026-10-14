<p align="center">
  <img src="assets/images/logo_v3.png" alt="GraphDancer" width="35%">
</p>

<h3 align="center">
GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning
</h3>

<!--- BADGES: START --->
<p align="center">
    <a href="https://arxiv.org/abs/2602.02518"><img src="https://img.shields.io/badge/arXiv-2602.02518-B31B1B.svg?logo=arxiv" alt="arXiv"></a>
    <a href="https://yuyangbai.com/graphdancer/"><img src="https://img.shields.io/badge/Website-GraphDancer-E5426E?logo=googlechrome" alt="Website"></a>
</p>
<!--- BADGES: END --->

GraphDancer trains an LLM to interact with a text-attributed graph through a small set
of graph functions, interleaving reasoning with information seeking, using single-stage
PPO with a **graph-aware curriculum**. This repository contains the training and
evaluation code, the fixed data partition, and launchers for every training run
reported in the paper (GraphDancer on Qwen2.5-3B-Instruct and Qwen3-14B, Vanilla PPO,
and the Static Mixture / Hard-to-Easy / Fast Pacing schedule controls).

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing the paper's tables](#reproducing-the-papers-tables)
- [License and acknowledgements](#license-and-acknowledgements)
- [Citation](#citation)

## Overview

**Interaction format.** The policy writes reasoning in `<think>…</think>`, issues
exactly one graph call per step in `<graph>…</graph>`, receives the result in
`<information>…</information>` (injected by the environment, not sampled), and ends
with `<answer>…</answer>`. The four graph functions follow GRBench / Graph-CoT:
`RetrieveNode[text]`, `NodeFeature[node, feature]`, `NeighbourCheck[node, relation]`
and `NodeDegree[node, relation]`. A rollout allows up to 10 tool iterations plus one
final generation with tool execution disabled.

**Reward.** `R(τ) = 0.8C + 0.2F + 0.1H(1−C)(1−F)`, where `C` is normalized exact match
of the last complete answer with the reference, `F` a structurally valid transcript and
`H` an extractable answer block (all binary) (`verl/utils/reward_score/qa_em_format.py`; structure score 0.2, final-format
score 0.1, retrieval score 0).

**Optimization.** PPO with a learned critic and GAE (γ = λ = 1), one rollout per
question, 128 trajectories and one PPO epoch per update, 200 updates, clip 0.2 with a
dual-clip constant of 3, value clip 0.5, entropy bonus 0.001, gradient clip 1,
agent-token-mean loss. Reference-policy KL is disabled (neither in the reward nor as a
loss). Observation tokens (and Qwen3's supplied `<think>` continuation prefix) are
masked from the policy and value losses.

**Graph-aware curriculum.** Every training question carries a fixed structural level
(Easy / Medium / Hard) derived from its question template before training. At update
`t` (with `u_t = t/(T−1)`, `T = 200`) the sampler mixes a Gaussian easy-to-hard
schedule with a fixed level prior:

```
g_t(k) ∝ exp(−(k − μ_t)² / (2σ²)),   μ_t = 2·u_t^κ,   κ = 3, σ = 0.75,   k ∈ {Easy, Medium, Hard}
p_t(k) = (1 − η_t)·g_t(k) + η_t·q(k),  η_t = 0.2 + 0.6·u_t,   q = (0.34, 0.33, 0.33)
```

Per update, level counts are drawn from `Multinomial(128, p_t)` and questions are
taken from per-level shuffled queues that are reshuffled when exhausted
(`verl/experimental/dataset/crl_e2h.py`, class `E2HGaussianSampler`). The schedule
controls replace `p_t`: **Static Mixture** (`static`) uses the mean of the GraphDancer schedule over the
200 updates (fixed proportions), **Hard-to-Easy** (`reverse`) runs it from hard to easy (`t → T−1−t`),
and **Fast Pacing** (`fast`) completes it in the first 100 updates and then keeps its final mixture.
**Vanilla PPO** samples questions uniformly.

## Repository layout

| Path | Contents |
|---|---|
| `graphdancer/llm_agent/` | multi-turn rollout loop (`generation_graph.py`), graph functions and node retriever (`tools/`) |
| `verl/` | fork of [verl](https://github.com/volcengine/verl) 0.5.0.dev with the GraphDancer extensions (GraphDancer uses its PPO/GAE path); entry point `verl/trainer/main_ppo_format.py` |
| `verl/experimental/dataset/crl_e2h.py` | curriculum sampler (GraphDancer schedule and the Static Mixture / Hard-to-Easy / Fast Pacing controls) |
| `verl/trainer/config/ppo_trainer_format.yaml`, `evaluation_ppo_format.yaml` | training / evaluation configs |
| `verl/utils/reward_score/qa_em_format.py` | rule-based reward |
| `data/split701020_v1/` | the fixed 70:10:20 partition (question IDs, frozen levels, prompt templates, hashes) |
| `scripts/data/` | parquet builder (`build_parquets.py`) and the structural labeling rule (`structural_labels.py`) |
| `scripts/train_ppo.sh`, `scripts/runs/` | training launcher and one launcher per reported run |
| `scripts/merge_checkpoint.sh` | FSDP → Hugging Face export of actor checkpoints |
| `scripts/eval_ppo.sh` | verl-native evaluation (one run per graph and decode seed) |
| `scripts/score_rouge.py`, `select_checkpoint.py`, `gptscore.py` | ROUGE-L, validation-based checkpoint selection, GPTScore |
| `tests/test_sampler_schedules.py` | CPU-only checks of the sampler and its controls |

## Installation

The paper's runs used Python 3.10, CUDA 12.4, PyTorch 2.6.0, vLLM 0.8.5.post1,
Ray 2.49.2, Transformers 4.51.3 and flash-attn 2.7.4.post1.

```bash
conda create -n graphdancer python=3.10 -y && conda activate graphdancer
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e . --no-deps        # installs the bundled verl fork and the graphdancer package
```

All launchers run from the repository root and add it to `PYTHONPATH`; every path is
an environment variable with a repository-relative default (`data/…`, `outputs/…`);
evaluation outputs go to `results/` under the repository root. No credentials are
needed except an API key for GPTScore.

## Data preparation

### 1. Questions and the fixed partition

The partition is shipped as identifiers only. `data/split701020_v1/split.jsonl` has one
row per question (1,690 rows) with the GRBench file and `qid`, the partition, a stable
`index`, the frozen structural `level` and the `template_id`, plus a hash of the
question/answer pair. The question text comes from the public GRBench question files:

```bash
huggingface-cli download PeterJinGo/GRBench --repo-type dataset --local-dir data/grbench_raw
python scripts/data/build_parquets.py --grbench-dir data/grbench_raw     # -> data/grbench/
```

The builder reproduces the training and evaluation records exactly (in the row order
used in our runs; both samplers index rows by position) and verifies every record and
every output file against the hashes in `data/split701020_v1/manifest.json`.

| Partition | Graphs | Questions | Output |
|---|---|---:|---|
| Training | Academic (DBLP + 5 MAPLE graphs) 560, Biomedical 189 | 749 (234 Easy / 273 Medium / 242 Hard) | `data/grbench/train/train.parquet` |
| Validation | Academic 80, Biomedical 27 | 107 | `data/grbench/eval/validation/<graph>.parquet` |
| In-domain test | Academic 160, Biomedical 54 | 214 | `data/grbench/eval/test/<graph>.parquet` |
| Unseen-domain test | E-commerce (amazon) 200, Literature (goodreads) 240, Legal 180 | 620 | `data/grbench/eval/ood/<graph>.parquet` |

*Partition.* `data/split701020_v1/split.jsonl` **is** the definition of the partition;
use it as shipped. The following is only a summary and does not regenerate the split.
The 1,070-question Academic + Biomedical pool (GRBench's original "hard" /
out-of-distribution Academic questions are excluded) is divided into roughly 70%
training, 10% validation and 20% in-domain test with per-graph quotas and stratification
by question template. The split was derived from an earlier train/test division of the
same pool: training is a subset of the earlier train side, in-domain test a subset of
the earlier test side, and the 107 validation questions were drawn from both sides
(50 and 57), allocated per graph and template by largest remainder with
`numpy.random.RandomState(20260911)`. Partitions are disjoint by question and share the
underlying graphs. Biomedical is called *Healthcare* in GRBench.

*Structural levels.* `scripts/data/structural_labels.py` contains the labeling rule
(Hard: aggregation, superlative or ranking cue; Medium: several entity slots or an
extra constraint; Easy: one entity slot; unmatched templates default to Medium).
`python scripts/data/structural_labels.py --check` re-applies it to the template texts
in `data/split701020_v1/question_templates.json` and compares with the frozen labels of
all questions that matched a template. The 24 questions whose wording matched no
template (`template_id` 0) keep their frozen labels: 18 are Medium, but 6 are not
(4 Easy, of which 2 are training questions; 2 Hard, one validation and one test).
The shipped `level` field is authoritative. Unseen-domain questions have no level in the
release; levels are used only by the training sampler.

### 2. Graph environments

Download the GRBench graph environments (one `graph.json` per graph) from the original
Graph-CoT / GRBench release (see <https://github.com/PeterGriffinJin/Graph-CoT>) and
arrange them as below. A mirror with this layout is available on Hugging Face:

```bash
huggingface-cli download yuyangbai/GRBench-copy --repo-type dataset --local-dir data/graphs
```

```
data/graphs/
├── biomedical/graph.json                 # GRBench "healthcare"
├── dblp/graph.json
├── maple/{Biology,Chemistry,Materials_Science,Medicine,Physics}/graph.json
├── amazon/graph.json
├── goodreads/graph.json
└── legal/graph.json
```

or point `GRAPH_DIR` elsewhere. The files range from about 0.15 GB (biomedical) to
about 84 GB (legal). Node-retrieval embeddings (`sentence-transformers/all-mpnet-base-v2`,
FAISS on CPU) are computed on first use and cached next to each `graph.json`.
Training keeps the seven training graphs in host memory (plan for roughly 512 GB of
RAM); evaluation loads one graph per run.

## Training

```bash
bash scripts/runs/graphdancer_qwen2.5_3b.sh      # GraphDancer, Qwen2.5-3B-Instruct, 1 node x 4 GPUs
bash scripts/runs/vanilla_ppo_qwen2.5_3b.sh      # Vanilla PPO (uniform sampling)
bash scripts/runs/static_qwen2.5_3b.sh           # Static Mixture schedule control
bash scripts/runs/reverse_qwen2.5_3b.sh          # Hard-to-Easy schedule control
bash scripts/runs/fast_qwen2.5_3b.sh             # Fast Pacing schedule control
bash scripts/runs/pure_e2h_qwen2.5_3b.sh         # optional: pure easy-to-hard (eta = 0); not a run reported in the paper
bash scripts/runs/graphdancer_qwen3_14b.sh       # GraphDancer, Qwen3-14B, 2 nodes x 8 GPUs (see below)
```

Each run script fixes the sampler settings and calls `scripts/train_ppo.sh`, whose
header documents all options (`GRAPH_DIR`, `TRAIN_FILE`, `OUTPUT_DIR`, `N_GPUS_PER_NODE`,
`FREE_CACHE_ENGINE`, …). Additional Hydra overrides can be appended to any command, and
`DRY_RUN=1` prints the fully composed configuration without starting a job. Outputs go to
`outputs/checkpoints/<experiment>/global_step_<N>/` (actor, critic and dataloader /
sampler state every 25 updates), `outputs/rollouts/<experiment>/` (per-question training
rollouts) and `outputs/logs/`.

Settings shared by all runs (see also the paper's hyperparameter table):

| Setting | Value |
|---|---|
| Updates / trajectories per update / PPO mini-batch | 200 / 128 / 128 |
| Actor / critic learning rate, warmup updates | 1e-6 / 1e-5, 56 / 3 (then constant) |
| GAE (γ, λ), policy / value clip, dual clip, entropy, grad clip | (1, 1), 0.2 / 0.5, 3, 0.001, 1.0 |
| Reference KL in reward / loss | off / off |
| Rollout sampling | temperature 0.7, top-p 0.8, top-k 20 |
| Tokens per generation / per observation | 500 / 1,024 |
| Rolling context (training) | 24,576 tokens |
| Initial prompt kept for PPO | 1,024 (Qwen2.5-3B) / 8,192 (Qwen3-14B) |
| Tensor / sequence parallel | 1 / 1 (Qwen2.5-3B), 2 / 2 (Qwen3-14B) |
| Seeds | training rollout seed `VERL_ROLLOUT_SEED=2`; curriculum sampler `RandomState(42)`; uniform sampler `torch.Generator` seed 1 |

**Qwen3-14B (multi-node).** The policy uses thinking mode, and the environment
supplies `<think>` after every observation. Start a Ray cluster across the two nodes
and launch from the head node (the repository, data, graphs and model must be visible at
the same paths on both nodes; the graph environments are loaded by the driver on the head node):

```bash
# head node
ray start --head --port=6379 --num-gpus=8
# second node
ray start --address=<head-ip>:6379 --num-gpus=8
# head node
export RAY_ADDRESS=<head-ip>:6379
bash scripts/runs/graphdancer_qwen3_14b.sh
```

`MODEL` defaults to `Qwen/Qwen3-14B`; our run used Hub revision
`40c069824f4251a91eefaf281ebe4c544efd3e18` (pass a local snapshot path as `MODEL` to pin it).

**Resuming.** Re-running the same command resumes from the latest complete checkpoint
(`trainer.resume_mode=auto`), including the curriculum sampler state.

**Notes.**
- Training has no in-loop validation (`trainer.test_freq=0`, `data.val_files=null`);
  checkpoints are selected afterwards on the validation partition (see below).
- With 749 questions and 128 trajectories per update an epoch is 5 updates, so
  `trainer.total_epochs` must be at least 40 for 200 updates (the launcher sets it to 200).
- `FREE_CACHE_ENGINE=true` (vLLM sleep mode between rollout and update) is what our
  runs used, together with a local fix for a vLLM 0.8.5.post1 memory-pool bug that can
  crash long runs in sleep mode. With an unpatched vLLM either resume after a crash or
  set `FREE_CACHE_ENGINE=false`, which keeps the vLLM engine resident (more GPU memory)
  and does not change the algorithm.
- `python tests/test_sampler_schedules.py` checks the sampler and the controls on the
  frozen training rows (CPU only).

**Exporting checkpoints.** Training keeps the two most recent FSDP checkpoints. To
evaluate intermediate checkpoints, export them to Hugging Face format while training
runs, or ask the trainer to keep them until exported:

```bash
WATCH=1 bash scripts/merge_checkpoint.sh graphdancer-qwen2.5-3b 150 175 200 &   # alongside training
# or: HF_EXPORT_STEPS="150 175 200" bash scripts/runs/graphdancer_qwen2.5_3b.sh
bash scripts/merge_checkpoint.sh graphdancer-qwen2.5-3b 200                     # one-off export
```

Exports are written to `outputs/hf/<experiment>/step_<N>/` and verified.

## Evaluation

Evaluation uses the same verl rollout loop as training (no separate inference
framework), in validation-only mode, with one graph per run and the whole graph's
questions as one batch. The rolling context is 30,000 tokens; all other limits and the
sampling parameters are as in training (temperature 0.7, top-p 0.8, top-k 20). Each
run uses one decode seed (`VERL_ROLLOUT_SEED`).

```bash
MODEL_PATH=outputs/hf/graphdancer-qwen2.5-3b/step_200 \
RUN_TAG=graphdancer-qwen2.5-3b-step200 SETS="test ood" SEEDS=3 \
bash scripts/eval_ppo.sh
# Qwen3-14B (8 GPUs, tensor parallel 2, thinking mode):
BACKBONE=qwen3-14b MODEL_PATH=outputs/hf/graphdancer-qwen3-14b/step_175 \
RUN_TAG=graphdancer-qwen3-14b-step175 bash scripts/eval_ppo.sh
```

Results are written to `results/<RUN_TAG>__<set>__<graph>__s<seed>/<graph>/results.jsonl`
(for MAPLE graphs the inner directory is `maple-<Field>`)
(question, reference answer, extracted model answer and the full interaction trace);
complete runs are skipped when the script is re-run.

Note: `evaluation_ppo_format.yaml` contains `val_kwargs.do_sample: false`, but the
rollout loop issues generation requests without validation metadata, so evaluation
samples with the rollout parameters above. This is the stochastic decoding used for
all reported numbers; do not change it when reproducing them.

**ROUGE-L.** `R-L = 100 × ROUGE-L F1` between reference and extracted answer
(`rouge_score`, no stemming). In-domain pools the 214 Academic and Biomedical test
questions (each question weighted equally); AVG is the unweighted mean of the four
columns (In-domain, E-commerce, Literature, Legal).

```bash
python scripts/score_rouge.py --run-tag graphdancer-qwen2.5-3b-step200 --seed 3
```

**GPTScore.** The percentage of answers an LLM judge marks correct with the GRBench
judge prompt (exactly "Yes"/"No"). Bring your own OpenAI-compatible endpoint; the judge
model is a required argument and nothing is sent without it:

```bash
export OPENAI_API_KEY=...            # and optionally OPENAI_BASE_URL=...
python scripts/gptscore.py --run-tag graphdancer-qwen2.5-3b-step200 --seed 3 \
    --judge-model <judge-model> --cache outputs/gptscore/graphdancer-qwen2.5-3b-step200-s3.jsonl
```

**Checkpoint selection.** Evaluate candidate checkpoints on the validation partition
with decode seeds 1 and 2 and keep the step with the highest mean validation R-L
(ties go to the earlier step):

```bash
for s in 150 175 200; do
  MODEL_PATH=outputs/hf/graphdancer-qwen2.5-3b/step_$s RUN_TAG=graphdancer-qwen2.5-3b-step$s \
  SETS=validation SEEDS="1 2" bash scripts/eval_ppo.sh
done
python scripts/select_checkpoint.py --experiment graphdancer-qwen2.5-3b --steps 150 175 200
```

## Reproducing the paper's tables

1. **Data**: build the parquets and place the graphs (above).
2. **Train** the runs with the launchers in `scripts/runs/`; export the candidate
   checkpoints. In our runs the candidates were updates 150, 175 and 200 for
   GraphDancer (both backbones) and 100, 150, 175 and 200 for the four Qwen2.5-3B
   controls.
3. **Select** each run's checkpoint on the validation partition
   (`select_checkpoint.py`, decode seeds 1 and 2). The selected updates in our runs were
   200 (GraphDancer, Qwen2.5-3B), 175 (GraphDancer, Qwen3-14B), 175 (Vanilla PPO),
   200 (Static Mixture), 150 (Hard-to-Easy) and 150 (Fast Pacing).
4. **Evaluate** the selected checkpoint on the 834 test questions (`SETS="test ood"`)
   and score with `score_rouge.py` and `gptscore.py`. Each GraphDancer and control row
   in the paper is one complete evaluation run with a single decode seed (set via `SEEDS`).
5. The **untrained Qwen3-14B control** uses the same evaluation command with
   `BACKBONE=qwen3-14b MODEL_PATH=Qwen/Qwen3-14B`.

Baseline rows (TextRAG, GraphRAG, ToG-2, Graph-CoT, Graph-Counselor, Graph-MCTS,
Graph-o1) come from the cited papers or from the baselines' own implementations and are
not part of this repository. Because rollouts are sampled, token-level outputs depend on
hardware and library versions, so re-runs can differ from the reported numbers.

## License and acknowledgements

The code is released under the Apache License 2.0 (`LICENSE`); see `NOTICE` for
third-party attributions. The `verl/` directory is a modified copy of
[verl](https://github.com/volcengine/verl) (Apache License 2.0) and keeps the original
copyright headers. The multi-turn rollout loop and exact-match reward follow
[Search-R1](https://github.com/PeterGriffinJin/Search-R1). The graph environments,
graph functions, questions, in-context demonstrations and judge prompt come from
[Graph-CoT / GRBench](https://github.com/PeterGriffinJin/Graph-CoT); please cite the
GRBench paper and follow the dataset's terms when using the data. Qwen2.5 and Qwen3
model weights are subject to their own licenses.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{bai2026graphdancer,
      title={GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning},
      author={Yuyang Bai and Zhuofeng Li and Ping Nie and Jianwen Xie and Yu Wang and Yu Zhang},
      year={2026},
      eprint={2602.02518},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02518},
}
```
