# Cymek failures (what broke, why, what changed)

> **HISTORICAL / APPEND-ONLY — NOT CURRENT-STATE AUTHORITY (phase-3 consolidation, 2026-09-24).** This branch-local ledger preserves its original entries; do not use its older status wording to classify current outcomes or schedule work. Read the [`source manifest`](../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.md), [`consolidated evidence ledger`](../../research/EXPERIMENT_EVIDENCE_LEDGER.md), [`phase-3 decision model`](../../research/RESEARCH_DECISION_MODEL.json), and [`current state`](../research/CURRENT_STATE.md) for current evidence. Historical entries are not rewritten here.

## Closure cycle (local, throttled)
- CUDA OOM on frozen 32k microsteps (RTX 4050 6 GB): environmental,
  classified DOES_NOT_FIT; larger GPUs UNMEASURED; TPU UNMEASURED.
- Demand planner restarted counters from zero on resume (found live):
  fixed with restored-counter planning + unit proof.
- Resume published against grandparent (found live, prior cycle): fixed
  with explicit resume parent.
- Window-vs-loss eligible-token mismatch (found live, prior cycle):
  predict-with-loss-rule + backend cross-check.
- Overwrote tracked mixture.py mid-cycle: restored and merged properly.
- Test-pack bucket drift (BPE rate misestimated twice): measured
  calibration with exact-full rows.

## CYR-GPU-001 outcomes
- PENDING_OPERATOR_EXECUTION. Failures append here with FAILURE.json refs.
