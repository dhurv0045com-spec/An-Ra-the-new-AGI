# CYR-GPU-001 — Colab GPU research tournament (preregistered, PENDING)

Status: PENDING_OPERATOR_EXECUTION. Nothing in this directory reports
measured tournament outcomes.

- PLAN.md — frozen design (question through next-bottleneck).
- PREREGISTRATION.json — hash-bound prereg (code/notebook/plan/model/
  tokenizer/manifests/seeds/arms/thresholds/cadence/budget-rule/runtime/
  metrics/gates). Verified by notebook CELL 0 before any training.
- THREATS.md — confounds and mitigations.
- README.md — this file: open the Colab link, choose GPU runtime, run
  CELL 0 → CELL 1 → CELL 2, return CYMEK_GPU_RESEARCH_RESULTS.zip.

## Result bundle (operator returns)
SESSION_MANIFEST.json, ENVIRONMENT.json, RESOLVED_PREREGISTRATION.json,
DATA_MANIFEST.json, SPLIT_MANIFEST.json, SMOKE.json, RESULTS/BASELINE.json,
QUERY_FACTORIAL.json, PAIR_SAMPLER.json, LR_TOURNAMENT.json,
SCALE_TRANSFER.json, REDTEAM/, DIAGNOSTICS/, CHECKPOINT_RECEIPTS/,
DECISION.json, FAILURE.json if applicable.
