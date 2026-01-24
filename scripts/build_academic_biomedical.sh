#!/usr/bin/env bash
set -euo pipefail

# Build a train/test parquet pair from Graph-CoT processed data.
#
# Expected input layout under $GRAPHCOT_DATA_ROOT (or ./data/processed_data by default):
#   - dblp/data.json
#   - biomedical/data.json
#   - maple/<Domain>/data.json
#
# Output:
#   data/academicTrain_biomedicalValid/{train.parquet,test.parquet}

GRAPHCOT_DATA_ROOT="${GRAPHCOT_DATA_ROOT:-./data/processed_data}"

python scripts/graph_data/build_graph_dataset.py \
  --train_datasets dblp maple \
  --test_datasets biomedical \
  --output_name academicTrain_biomedicalValid \
  --data_root "${GRAPHCOT_DATA_ROOT}"

# If your processed data uses a different difficulty key, override with:
#   --level_key level