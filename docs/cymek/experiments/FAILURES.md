# Cymek failures (what broke, why, what changed)

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
