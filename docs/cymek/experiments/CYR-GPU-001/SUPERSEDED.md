# CYR-GPU-001 SUPERSEDED — verified defect record (superseded before execution)

Status: SUPERSEDED_BEFORE_EXECUTION. The original preregistration is
preserved untouched; this file records why. Each item was verified against
the committed tree (not blindly accepted). Verdicts: CONFIRMED (present),
REJECTED (not reproduced), PARTIAL.

## Deterministic defects — all CONFIRMED except noted
- A (CYR_DEV_WORLDS import): REJECTED as stated — the actual defect class
  (missing/threaded names: CYR_EVAL_DENSE_WORLDS, CheckpointStore,
  reload_model, row_width) was confirmed instead by static analysis and
  fixed in v2 (static_check.py now gates this mechanically).
- B (blind_gap signature): CONFIRMED — rewritten in v2 (worlds-only).
- C (arm_payloads CheckpointStore binding): CONFIRMED — bound in v2.
- D (row_width not threaded): CONFIRMED — threaded in v2.
- E (controller stops observing after switch): CONFIRMED — v2 observes in
  BOTH states with persisted history (re-entry possible, unit-tested).
- F (sustained history resets per chunk): CONFIRMED — v2 keeps rolling
  history across chunks/checkpoints (unit-tested).
- G (LOW arm absent from prereg): CONFIRMED — v2 prereg enumerates the
  full arm set; a test asserts prereg == executed sets.
- H (seed 303 outside preregistered set): CONFIRMED — v2 uses CYR_SEEDS
  only; seed-set equality tested.
- I (larger-proxy stub): CONFIRMED — v2 executes the next proxy for real
  or writes LARGER_PROXY_NOT_EXECUTED_TIMEBOX (never "available").
- J (ByteTokenizer calibration): CONFIRMED — v2 calibrates with the
  production 24,576 tokenizer on the resolved proxy/shape/backend.
- K/L (dev labeled train-exact; no sustained gate): CONFIRMED — v2 keeps
  TRAIN acquisition, DEV_CONTROLLER, DEV_MEASUREMENT, SEALED_RESERVED
  separate with onset+confirmation in updates/tokens/wall time.
- M (side-specific padding breaks query-only purity): CONFIRMED — v2
  renders ONE shared latent context per world; variants differ only as
  named; rows use normal batch padding + eligible masks.
- N (generic baselines): CONFIRMED — v2 implements task-aware baselines
  against the task grammar with precomputed expected scores (oracle test).
- O/P (teacher-forced primary; free-gen spot): CONFIRMED — v2 primary is
  batched free-generation complete exact (+both-correct, EOS/stop rates);
  teacher forcing is diagnostic only.
- Q (intentional overlap in real ledger): CONFIRMED — v2 records
  negative-control tests under NEGATIVE_CONTROL_TESTS only.
- R/S (tail-masked shell; COMPLETE after failure): CONFIRMED — v2
  notebook uses subprocess.run(check=True) with returncode asserts; no
  bare COMPLETE prints after shell.
- T (no full-chain test): CONFIRMED — v2 adds the compressed S0–S4
  plumbing test (TINY, tiny data, ~6 updates; allowed as plumbing).
- U (fork state continuation): PARTIAL — v2 forks restore model +
  optimizer bytes (proven equal post-load) with fresh research counters
  (receipted); production TrainingState continuity is intentionally NOT
  claimed for research arms (documented boundary).

## Design corrections in v2 (beyond defect fixes)
- Dose floor 2–6M tokens/arm; wall-time resolver (115-min training split
  30/25/30/15); RESEARCH_SMALL proxy; proxy-downshift before replication
  cuts. CYR-001's 60–400k budgets could not have tested generalization.
- Full campaign uses the real production tokenizer everywhere; precision
  discovered (FP32/BF16/AMP) and frozen pre-outcome.
- In-training deadlines with TIMEBOX checkpoints; Drive mirroring with
  EPHEMERAL warning fallback; V2 bundle layout; development claim ladder
  (no TPU/production levels from GPU).
- Sealed split generated, hashed, and NEVER consumed by control/selection.
