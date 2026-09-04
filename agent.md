# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches (esoes / triquetra / cymek / ...) are read-only audit inputs —
> never modified, never pushed. A CPU/CUDA run is NEVER a TPU result. No
> fabricated device results. Preregistration and results never share a commit.
> Cycle download ceiling: <10 GB total (target <2 GB); this cycle: 0 bytes.

## STATUS

Citadel SHA: `9fa5e5b` (local; to be pushed this cycle)
Cymek SHA: `105ad22` (read-only; moved +9 since `4abeaeb` — assessed below)

This cycle (Colab-first, platform-neutral core): `origin/cymek` moved +9 commits
(~5000 lines: stream/sourceset/qualify scaffolding, sampler-cursor authority,
tokenizer artifact, eval/experiment harnesses). Audit impact check: `MINI_SPEC`,
`v5_model/*`, optimizer, checkpoint, pack untouched; `causal_lm.py` gained an
optional `eligible` mask (elementwise AND, default path byte-identical) — XLA
audit verdicts hold, no re-audit of the T0 path required. Implemented: (1) platform
identity in every TPU receipt (`platform` colab|kaggle|other via runtime signals +
`CITADEL_PLATFORM` override, never from TPU generation; `accelerator_requested` /
`accelerator_detected` / `xla_device_count`); (2) thin Colab launcher
`notebooks/citadel_colab_tpu.ipynb` (Kaggle notebook kept); receipt-return decision
implemented as operator file transfer (`files.download` cells, no secrets).
Local validation caught + fixed a real regression: new `platform` probe parameter
shadowed the stdlib `platform` module (smoke test crashed) → renamed to
`platform_override`, re-validated (default `other`, explicit + env-var overrides
correct, fail-closed `ABORT_NO_TPU` intact).

## PLATFORM

```text
colab
```

Colab is the first execution surface; Kaggle remains secondary. Core backend
(`citadel_tpu/`) is platform-neutral; only notebook wrappers differ.

## ENVIRONMENT

TPU detected: unmeasured (no device execution from this box)
TPU identity: —
XLA devices: —
Python: —
PyTorch: —
torch-xla: —

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none — no pip installs, no datasets, no checkpoints, no artifacts)
TOTAL_DOWNLOADED_GB = 0.0
```

## T0

Status: NOT_RUN | Loss: — | Grad norm: — | Parameter mutation: —
Checkpoint save: — | Checkpoint reload: — | Wall time: — | Tokens/sec: —

## CALCULATOR

Status: NOT_RUN | Parameters: MINI_SPEC (~1.6M)
Train examples: 0 | Train tokens: 0 | Updates: 0
Loss start: — | Loss end: — | Untrained test accuracy: —
Final test accuracy: — | Checkpoint hash: — | Reload result: —
Tokens/sec: — | Generator: calculator-canary/1.1 (locally validated)

## 5B DATA

```text
NOT_STARTED
```

`0 / 5,000,000,000`. No sample test (no TPU host to feed); sequenced behind T0→T1.

## QUESTIONS FOR OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

No Colab TPU execution yet — T0 (one-update cert) is the gate for everything.

## NEXT ACTION

Run `notebooks/citadel_colab_tpu.ipynb` on a Colab TPU runtime (probe → T0 → STOP unless PASS); transfer back the exact `TPU_ENVIRONMENT.json` / `TPU_ONE_UPDATE.json` files.

## Commit log (latest first, citadel only)

```text
9fa5e5b feat(citadel): add platform identity to tpu receipts
0c199cc feat(citadel): add colab tpu launcher
e6226af docs(citadel): update agent handover
68f49ae fix(citadel): guarantee 50-row commutative slice; enforce probe gate in-driver
```
