# Agent brief: CYMEK research + readiness handoff

Branch: `cymek-500m-readiness`. Date: 2026-09-08. No training started.

## CURRENT EXPERIMENT
CYR-GPU-005 — shared-parent LR-retention fork campaign
(`docs/cymek/experiments/CYR-GPU-005/`). CYR-GPU-001/002/003/004 are
SUPERSEDED_BEFORE_EXECUTION (none ever ran on a GPU; 004 was killed by
the 005 pre-execution audit after its fork contract proved broken).
Runners live under `v5_experiments/cyr_gpu005.py` (pure core) and
`anra_v5/cyr_gpu005_run.py` (torch orchestrator); the prior
cyr_tournament/cyr_execute stacks remain for CYR-GPU-002 evidence.

## SOFTWARE
- 500M campaign engine: bucket-pure microsteps, accumulation accounting,
  trainer state machine, durable milestones + persistent mirror, mixture
  scheduler, eval hooks. 50/50 entry tests.
- XLA accumulation boundary FIXED this cycle: one gradient SUM collective
  per logical update (the per-microstep reduce that scaled early
  microstep gradients by powers of R is gone; AST-guarded, oracled,
  negative-regressed).
- CYR-GPU-005 fork contract enforced in code: ONE acquisition per seed,
  byte-identical fork restorations (PARENT_EQUIVALENCE.json), shared
  future stream with consumed-batch SHA equality, actual-token
  targeting, one absolute campaign deadline.

## EVIDENCE
- GPU evidence: NONE YET (no CYR experiment has executed on a GPU).
- Local: full suite green at the freeze (receipts
  `artifacts/v5/cyr_gpu_005_test_receipt.json` and the refreshed closure
  receipt, both bound to the preregistration commit); plumbing E2E runs
  the whole campaign on TINY with the labeled PLUMBING_SMOKE_ONLY gate
  override; static check 0 findings.
- Audits (remote state wins): Arkenstone@fc3e689 — ARK-007R replicated
  (LOW protects: 0/12 vs 9/12 collapse), ARK-010 recovery (HIGH 8/9 vs
  LOW 2/9), ARK-009 transfer gate NOT qualified (order-reversal
  confound), ARK-011 executed + Discovery-V6 receipts hash-valid
  (14/14). BRAMASTRA@4655733 — chief D02 horizon audit;
  discovery_dev_701 comparisons. Matrix:
  `docs/cymek/research/CROSS_BRANCH_EVIDENCE_AUDIT.md` (update pending
  next cycle for the new SHAs).
- Literature check: plasticity framing supported; no external candidate
  admitted (fashion bar not met).

## CURRENT HYPOTHESES (CYR-GPU-005, preregistered, UNEXECUTED)
- H-primary: hysteretic HIGH↔LOW retains like LOW and re-acquires like
  HIGH; beats both fixed arms on qualified parents.
- H-fixed-time: switching HIGH→LOW at τ=0.5 of ACTUAL continuation
  tokens captures LOW's protection — if it matches HYST, "state" is
  mostly time.
- H-null: all four arms indistinguishable — protects the 500M recipe.
- Adjudication: near-freezing vs consolidation read from the mandatory
  displacement/moment ledger, never assumed.

## GPU EXPERIMENT READY?
YES — RUN_READINESS.json: ready=true. Notebook
`notebooks/cymek_colab_gpu_research_v5.ipynb` (3 thin cells), prereg
`docs/cymek/experiments/CYR-GPU-005/PREREGISTRATION.json` (hash-bound to
the executable freeze; two-commit protocol with CELL 0 checkout+verify),
section-39 evidence bundle, failure-proof packaging. Status:
READY_FOR_OPERATOR_COLAB_GPU_RUN. Local validation this cycle: unit
suites + TINY plumbing smoke only (no training runs on the operator
machine; full mode fails closed without CUDA).

## TPU STATUS
IMPLEMENTED_PENDING_PRE500M_TPU. Zero TPU evidence. CPU oracles do NOT
certify TPU. GPU results never count as TPU evidence. PRE500M only
after CYR-GPU-005 execution + result audit + exact-SHA audit + Citadel
independent audit.

## DATA STATUS
DATA_NOT_READY (500M supply). CYR-GPU-005 uses rendered T2 synthetic
worlds with a manifest bound to the exact rows (full-mode manifest SHA
in the preregistration) — production corpus untouched.

## 500M STATUS
NOT AUTHORIZED (main_training_authorized=false; E1–E6 pending; see
`artifacts/v5/launch_readiness.json`).

## NEXT OPERATOR ACTION
Open the Colab link (only valid because RUN_READINESS=true), choose GPU
runtime, run CELL 0 → must print `CYR-GPU-005 PREEXECUTION GATE: PASS`
→ CELL 1 → CELL 2, return CYMEK_GPU_RESEARCH_V5_RESULTS.zip. Do NOT run
PRE500M. Do NOT build the 5B corpus.
