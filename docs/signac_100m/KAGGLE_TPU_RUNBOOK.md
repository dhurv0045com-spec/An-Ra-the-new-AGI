# Kaggle TPU preflight runbook

Notebook: `notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb`.

This notebook is a runtime and bounded model-canary for a Kaggle TPU session. It is designed to make the remaining gap visible and reproducible. It deliberately has no cell that starts long training. Current repository contracts require TPU evidence before the V5 training path can certify XLA execution.

## Before opening the notebook

Attach a Kaggle dataset containing a pinned repository snapshot and the qualified data/evaluation receipts once those exist. Keep Kaggle Output enabled for receipts and canary checkpoints. Record the Git commit and dataset versions in the notebook metadata. Do not put credentials or private data in notebook outputs.

## Run

1. Select a TPU accelerator and enable Internet only if the pinned runtime requires package setup. Record exact Kaggle image and `torch_xla` versions; never silently upgrade dependencies during a registered experiment.
2. Run the repository discovery/import cell. It must resolve one unambiguous repository copy.
3. Run the hardware inventory cell. It requires PJRT to report TPU and exactly eight global devices. Process world size and TPU device count are separate: the interactive process can see eight devices without running an eight-process replicated trainer.
4. Run the small M102 checkpoint/restore/identical-continuation canary, then the one-update 4,096-context canary for M102, the same-depth FFN-aligned challenger, and the fully tiled challenger. Inputs are synthetic, and first-update timing includes XLA compilation. The notebook does not certify peak-memory fit, steady-state throughput, BF16/FP32 parity, production collectives, or multi-process exact resume.
5. Run the static Signac parameter receipt and preflight. M102 must report exactly 101,790,080 parameters. The report currently returns `BLOCKED` because corpus/evaluation receipts and production TPU execution evidence are absent.
6. Save the report, environment output, canary receipts, and notebook version as a small Kaggle Output artifact. Do not treat these plumbing checks as a training result.

## Gate for adding the training cell

The training cell may be added only after model update and exact-resume tests run on the selected TPU topology, a production manifest reaches `RUNNABLE`, Citadel evaluation readiness is green, and a multi-seed control passes the capability sensitivity gate. The trainer must use bounded token/time budgets, periodic atomic checkpoints, restart-safe state, heartbeat/progress receipts, and explicit output sync. A disconnected notebook must not lose or silently double-count work.

The resource estimate in `signac_100m/spec.py` intentionally excludes activations, XLA buffers, compiler padding, and fragmentation. The notebook reports no reliable peak-memory measurement; only an instrumented target receipt can establish batch and context fit. The Kaggle T4×2 S5 run is evidence of that GPU campaign only.
