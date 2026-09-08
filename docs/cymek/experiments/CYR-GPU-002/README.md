# CYR-GPU-002 — Colab GPU research campaign (preregistered, PENDING)

Status: PENDING_OPERATOR_EXECUTION. Nothing here reports measured outcomes.
CYR-GPU-001 is SUPERSEDED_BEFORE_EXECUTION (see ../CYR-GPU-001/SUPERSEDED.md;
its preregistration is immutable history — do not run it).

- PLAN.md — frozen design (question through next-bottleneck).
- PREREGISTRATION.json — hash-bound prereg (code/notebook/plan/model/
  tokenizer/manifests/seeds/arms/thresholds/cadence/budget-rule/runtime/
  metrics/gates). Verified by notebook CELL 0 before any training.
- THREATS.md — confounds and mitigations.
- README.md — this file: open the Colab link, choose GPU runtime, run
  CELL 0 → CELL 1 → CELL 2, return CYMEK_GPU_RESEARCH_V2_RESULTS.zip.

## Result bundle (operator returns)
SESSION_MANIFEST.json, ENVIRONMENT.json, PREREGISTRATION.json,
RESOLVED_PREREGISTRATION.json, CALIBRATION.json, DATA_MANIFEST.json,
SPLIT_MANIFEST.json, SMOKE.json, NEGATIVE_CONTROL_TESTS.json,
ACQUISITION/, PAIR_QUERY/, LR_RETENTION/, TRANSFER/, SCALE/, REDTEAM/,
CHECKPOINTS/ (receipts only, no tensor payloads), DECISION.json,
FAILURE.json if applicable.
