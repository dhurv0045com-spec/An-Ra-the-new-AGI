# Gandiva engineering status

## Latest verified build — 23 September 2026

The latest code revision, `ab580776c4fab6398f23d503413eace41d50dbe7`, fixes the
cognition prefix contract: auxiliary channels use the canonical initial-state
prefix, while trajectory answer batches retain their full history. E3 now uses
the same canonical prefix. The regression suite confirms those contexts remain
distinct and correctly encoded.

The source-bound, no-update build report is
[`reports/FINAL_K8/gandiva-cognition-final-a6e99c9b-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-final-a6e99c9b-20260923/build_verification.json).
It reports all F01–F24 checks passing, all seven pytest groups passing, the
notebook interface exercise passing (24 cells; 12 Python cells; no hard-coded
paths or shell magics), and zero optimizer updates. The verifier completed in
113.375 seconds. Its source-closure SHA-256 is
`88e130fdc65c21afc97f92f61e62c610365643839a5ae4a127a4b1b7adccb73c`; the
verified full-bundle identity is
`79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`.

Focused validation after the cognition fix passed **74 tests, 1 skipped, and 8
subtests**. The report marks the source tree dirty because the pre-existing
user edit in `tests/test_research_k8_real.py` was preserved. The implementation
did not stage or replace that edit.

## Owner experiment and limits

The normal self-sustaining Kaggle notebook is
[`../notebooks/bramastra_k8.ipynb`](../notebooks/bramastra_k8.ipynb). The
notebook contract is build-verified, but the actual two-T4 E0 runtime gates
G01–G04 remain pending an owner Kaggle session. Those gates must establish both
devices, real updates and exact resume, measured throughput, and live allocation
before E1–E6. No optimizer updates, GPU qualification, learned result, or AGI
capability are claimed by this build.

The separate 100M TPU path remains a preflight and engineering design. It has
not been trained or qualified on Kaggle TPU and is not part of the normal K8
notebook campaign. See
[`TPU_100M_COGNITION_PROGRESS.md`](TPU_100M_COGNITION_PROGRESS.md).

## Continue from here

Use [`FINAL_K8_PROGRESS.md`](FINAL_K8_PROGRESS.md) for the detailed progress
cursor and [`FINAL_EXPERIMENT_EXECUTION.md`](FINAL_EXPERIMENT_EXECUTION.md) for
the active F01–F24 build contract. Keep generated failure reports as audit
evidence, do not overwrite earlier reports, and preserve the user's uncommitted
test edit.
