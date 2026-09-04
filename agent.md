# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches (esoes / triquetra / cymek / ...) are read-only audit inputs —
> never modified, never pushed. A CPU/CUDA run is NEVER a TPU result. No
> fabricated device results. Preregistration and results never share a commit.
> Cycle download ceiling: <10 GB total (target <2 GB); this cycle: 0 bytes.

## STATUS

```text
citadel:          68f49ae (local; to be pushed this cycle)
origin/citadel:   3a0502b (pre-cycle; no remote movement)
origin/esoes:     85f44b7 (foundation, unchanged)
origin/cymek:     4abeaeb (read-only, unchanged)
origin/triquetra: fa44ea3 (read-only, unchanged)
merge-base(citadel, origin/cymek): 85f44b7 (siblings; never merged)
```

This cycle (small-steps, zero-download validation): validated `calculator_data`
generator (4000/500/500 rows, overlap 0, rebuild-deterministic) and the probe
smoke path locally; confirmed `one_update` aborts fail-closed (`ABORT_NO_TPU`,
no receipt) on a non-TPU box. Found + fixed two defects before any Kaggle run:
(1) commutative heldout slice yielded 6 rows, not 50 (random filter) → now a
deterministic stride guaranteeing exactly 50 (`calculator-canary/1.1`);
(2) in-driver `probe()` result was never enforced (only `main()` raised) → now
`run()` raises `NoTpuError` on `probe_pass == false` (commit `68f49ae`).

## QUESTIONS FOR OPERATOR

1. Receipt return path: after running `notebooks/citadel_kaggle_tpu.ipynb` on
   Kaggle, should the `tpu_receipts/*.json` files come back as pasted text,
   or will you push them from the Kaggle side (Kaggle has no direct remote
   access to this repo)?

## DOWNLOADS

```text
(none — no pip installs, no datasets, no checkpoints, no artifacts)
TOTAL_DOWNLOADED_GB = 0.0
```

## TPU STATUS

```text
IMPLEMENTED_NOT_RUN
```

Kaggle execution: `NOT_EXECUTED_ON_KAGGLE`. One-update: NOT_RUN. Calculator:
NOT_RUN. C1: preserved, NOT executed.

## CALCULATOR STATUS

```text
parameters: MINI_SPEC (~1.6M; P35 explicitly not the bring-up model)
training examples: 0 | training tokens: 0 | updates: 0
loss start: - | loss end: -
untrained held-out accuracy: unmeasured | final held-out accuracy: unmeasured
checkpoint hash: - | reload status: -
wall time: - | steady-state tok/s: -
generator: calculator-canary/1.1 (validated locally: overlap 0, deterministic)
```

## 5B DATA STATUS

```text
NOT_STARTED
```

`0 / 5,000,000,000` real train tokens. No sample-pipeline test this cycle (no
TPU host to feed); correctly sequenced behind T0→T1→T2.

## BIGGEST BLOCKER

No Kaggle TPU execution yet — T0 (one-update cert) is the gate for everything.

## NEXT ACTION

Run `notebooks/citadel_kaggle_tpu.ipynb` cells 0–3 on a Kaggle TPU notebook
(env probe → T0 one-update); return `TPU_ENVIRONMENT.json` /
`TPU_ONE_UPDATE.json` or the abort + log hash.

## Key facts (do not re-derive blindly)

- XLA shims, architecture unchanged: XLA AdamW + `xm.optimizer_step`/`mark_step`;
  `packed_layout` + loss-count + hashing on CPU host; no `data_ptr()` on XLA;
  `torch_module`-injection is the port seam. No JAX/TF. No `torch_xla` in `v5_*`.
- Top device unknown: SDPA bool-mask lowering per torch-xla version.
- Buckets 512/1024/2048/4096: bring-up on 512 fixed-batch only until 8-device passes.
- Strongest negative result: `production_scoring_mode = null` (gates all learned-cognition comparisons).

## Commit log (latest first, citadel only)

```text
68f49ae fix(citadel): guarantee 50-row commutative slice; enforce probe gate in-driver
3a0502b docs(citadel): update agent.md handover
68ca5af docs(citadel): add kaggle tpu launcher notebook
0e4827e feat(citadel): add kaggle tpu bootstrap path
594fc77 docs(citadel): switch execution target to kaggle tpu
203ff60 docs(citadel): tpu-first override audit and milestone preregistration
```
