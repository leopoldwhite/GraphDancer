## Data directory

This repository ships **code only**. Put your datasets under `data/`.

Expected formats:
- **Training/Eval datasets**: Parquet files referenced by launchers/configs, e.g. `data/<dataset>/train.parquet` and `data/<dataset>/test.parquet`.
- **Graphs**: JSON graphs referenced by `GRAPH_DIR`, e.g. `data/graphs/<domain>/graph.json`.


