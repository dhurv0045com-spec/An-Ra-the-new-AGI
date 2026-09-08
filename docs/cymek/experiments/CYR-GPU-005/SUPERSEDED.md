# CYR-GPU-005 — SUPERSEDED BEFORE EXECUTION

Status: **SUPERSEDED_BEFORE_EXECUTION**. No GPU scientific run exists for CYR-GPU-005.

The frozen 005 identity is preserved, not rewritten. Independent post-freeze audit found fatal execution/science defects: Cell 1 did not pass the CUDA device and the runner defaulted to CPU; the orchestrator stopped after the first qualified parent; decision logic could choose a winner from one parent; Cell-0 calibration/resolution was discarded; calibration omitted optimizer stepping and ignored generation cost; the GPU displacement path mixed CUDA and CPU tensors; campaign-level resume and failure-proof packaging were not real.

Because those fixes change executable behavior after preregistration, they are implemented under the new identity **CYR-GPU-006**.
