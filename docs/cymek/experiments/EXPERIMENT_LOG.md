# Cymek experiment log (chronological, append-only)

## 2026-09-07 — closure cycle (branch cymek-500m-readiness)
- Bucket-pure execution, durable milestones, mixture scheduler, eval
  hooks, XLA adapter (pending), persistent mirror, import-plane fix.
  See experiment.md E1–E11 and
  artifacts/v5/cymek_500m_closure_test_receipt.json.

## 2026-09-08 — CYR-GPU-005 freeze cycle
- CYR-GPU-004 SUPERSEDED_BEFORE_EXECUTION (fatal fork-contract defect
  confirmed against live run_v4: independent acquisitions per LR, dead
  parent snapshot, no future-stream identity). See
  CYR-GPU-004/SUPERSEDED.md.
- Production XLA accumulation defect fixed (per-microstep all-reduce of
  the ACCUMULATED buffer scaled early microstep gradients by powers of
  the replica count); one collective at the accumulation boundary now,
  with a CPU oracle + negative regression. Hardware status stays
  IMPLEMENTED_PENDING_PRE500M_TPU.
- CYR-GPU-005 executable freeze (COMMIT A) + hash-bound preregistration
  (COMMIT B). RUN_READINESS=true. GPU evidence: NONE YET.

## CYR-GPU-001 — status: SUPERSEDED_BEFORE_EXECUTION
- Preregistered design + tournament runner committed; never executed;
  superseded by the 002 rebuild (shared contexts, free-gen primary,
  dose resolver).

## CYR-GPU-002 — status: SUPERSEDED_BEFORE_EXECUTION
- Tournament stack + cycle test receipt exist (artifacts/v5/
  cyr_gpu_002_test_receipt.json); superseded by 003 without operator
  execution.

## CYR-GPU-003 — status: SUPERSEDED_BEFORE_EXECUTION
- Pre-execution audit found 13 defects (3 blockers) before any GPU
  execution; superseded by 004.

## CYR-GPU-004 — status: SUPERSEDED_BEFORE_EXECUTION
- Corrected runner still launched independent acquisitions per LR and
  never restored the parent snapshot; killed by the 005 audit. See
  CYR-GPU-004/SUPERSEDED.md.

## CYR-GPU-005 — status: PREREGISTERED, PENDING_OPERATOR_EXECUTION
- Shared-parent fork campaign (HIGH / LOW / FIXED_TIME / HYST from ONE
  G90 parent per seed), hash-bound preregistration, RUN_READINESS=true.
- No outcomes yet. Live results append below with dates when the
  operator returns the bundle. Do NOT edit preregistered sections
  after execution starts.
