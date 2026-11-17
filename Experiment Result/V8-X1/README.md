# ELENDRIL V8-X1 Experiment Results

This directory stores focused reports and curated JSON (`METRICS.json`, `PROVENANCE.json`, `RESULTS.md`) retrieved from verified Kaggle 2xT4 session exports (`ELENDRIL_X1_<SESSION>_ANALYSIS.zip`). Raw ZIPs, large episode dumps, and full checkpoint tensors remain outside Git.

- **Status:** `CPU_QUALIFIED_GPU_PENDING` — Runtime modules (`anra_v5/elendril/v8x/`, `anra_v5/planthir/adapters/x1_s20_adapter.py`, `tools/elendril_v8x_*.py`) and notebook builder (`notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb`) are verified on bounded CPU fixtures only.
- **No GPU training or S20 research claim has been made yet.** `TEMPLATE_METRICS.json` is an unexecuted pre-run scaffold recording the 42-job registry and honest `NOT_RUN` state.
