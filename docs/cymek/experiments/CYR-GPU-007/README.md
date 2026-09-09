# CYR-GPU-007

CYR-GPU-007 supersedes CYR-GPU-006 after the operator's real Colab Cell-0 calibration proved that the fixed 006 worst-case campaign could not fit the 170-minute hard wall on the assigned GPU. No scientific training ran under 006.

007 keeps the same causal science but preregisters hardware-only scope tiers. It prefers the full 3-parent retention+plasticity campaign, can fall back to a complete 2-parent causal campaign, and can finally fall back to replicated same-task retention only. It also calibrates TINY as a last-resort real-V5 development proxy; TINY can never produce a production-facing research candidate.

Operator execution is allowed only after `PREREGISTRATION.json` and `RUN_READINESS.json` bind an immutable executable SHA and readiness is true.

When ready:
1. Open `notebooks/cymek_colab_gpu_research_v7.ipynb` with a GPU runtime.
2. Run Cell 0 only. It writes calibration evidence before resolving. Continue only if it prints `CYR-GPU-007 PREEXECUTION GATE: PASS`.
3. Run Cell 1. Drive stores durable stage evidence/checkpoints.
4. Run Cell 2 and return `CYMEK_GPU_RESEARCH_V7_RESULTS.zip`.

No positive result from this notebook alone authorizes PRE500M, 500M, TPU claims, a production scheduler change, or a 5B corpus build.
