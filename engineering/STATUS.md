# Gandiva engineering status

## Four-mode cognition matrix and full build — 23 September 2026

The latest no-update diagnostic covers all 12 registered E2 family/mode
pairs: rule-inquiry, inventory, and program across `b-policy`, `b-workspace`,
`b-planner`, and `b-memory`. Each trace has a two-call cap, except bounded
planning at four calls. The `b-memory` index is sourced only from the training
split. Coverage, genuine model-call provenance, memory reads, eval mode,
discarded gradients, and zero optimizer updates are checked.

The first expanded run caught an integration error: the live episode now
requires the data directory, while the integrated rehearsal still called the
old zero-argument helper. That failed report remains preserved. The rehearsal
now receives the verified bundle for this production diagnostic and continues
to build its separate small fixture bundle for phase wiring. The fixed,
source-bound report
[`reports/FINAL_K8/gandiva-cognition-modes-fixed-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-modes-fixed-20260923/build_verification.json)
passes F01–F24, all seven test groups, all seven real-interface exercises, and
the 24-cell owner-notebook interface check. Duration was 223.015 seconds; the
integrated rehearsal took 55.578 seconds. Data identity is
`79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`, source
closure is
`9f41d8e6424e719eba5c286a8f9a81521d35335575e41ff2f9f178e7f998307c`, and
report identity is
`0f2455901507661219e7d1926a0da4c432796bdcec287db89026d3fbf20e81b9`. The
full handoff records exact commands and outcomes:
[`reports/FINAL_K8/gandiva-cognition-modes-fixed-20260923/HANDOFF.md`](reports/FINAL_K8/gandiva-cognition-modes-fixed-20260923/HANDOFF.md).

This matrix is execution evidence, not a positive cognition result. The model
was randomly initialized: all planner traces ended unsuccessfully after model
parse failures and fallback actions, while the other modes exhausted their
small call caps without an environment outcome. `b-memory` read two training
records per trace. No local training or GPU qualification occurred;
optimizer updates remain zero. The user's existing edit to
`tests/test_research_k8_real.py` remains preserved and excluded from scoped
publication. Runtime gates G01–G04 still require the owner Kaggle two-T4 E0
run.

## Three-family cognition rehearsal — 23 September 2026

Source revision `ec39a32811492f31047f504e4d066c8bbc74936e` expands the
zero-update production trace and integrated E2 fixture rehearsal to every
registered family: rule-inquiry, inventory, and program. The random-initialized
model made four real model-origin calls per family under the explicit cap; each
trace truncated with unknown success. The model stayed in evaluation mode,
retained zero gradients, and committed zero optimizer updates. This is honest
integration evidence, not evidence of useful cognition.

The new full build report
[`reports/FINAL_K8/gandiva-cognition-all-families-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-all-families-20260923/build_verification.json)
passes F01–F24 and all seven check groups in 81.985 seconds. Its data identity
is `79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`; source
closure is `e603b5bbc6d3b7e4e7841482fe0b2327d9d8e253c015f5ef0f765956abb4f378`.
The report records `dirty=true` because the pre-existing user edit to
`tests/test_research_k8_real.py` was preserved. The fixture rehearsal now has
three matched groups across the E2 modes; those rows are explicitly fixture
evidence. Owner runtime gates G01–G04 still require the real Kaggle two-T4 E0
session. See
[`reports/GANDIVA_F6_FAMILY_REHEARSAL_20260923/HANDOFF.md`](reports/GANDIVA_F6_FAMILY_REHEARSAL_20260923/HANDOFF.md).

## Cognition privacy follow-up — 23 September 2026

Follow-up commit `719bcbcf5b84e80900e87f21a0cea2f49d550ff1` adds two
regressions for private-state isolation: changing unreceived private fields
cannot change the world-model prompt, and two environments with opposite hidden
answers produce the same prompt but opposite outcomes. It also fixes a false
positive in the older C04 diagnostic: the original check compared two
depth-one roots rather than observing a depth-two call. The corrected probe
exercises one real second-depth prediction. Its six recorded defect flags are
false on this revision.

Focused CPU-only checks passed: 50 cognition tests, including all 11 planning
tests, plus 7 RSI integration tests. No optimizer update, GPU run, or learned
capability was involved. This closes those regression gaps only; it is not a
complete acceptance of the broader cognition-foundation F1–F6 contract. See
[`reports/GANDIVA_COGNITION_FOLLOWUP_20260923/HANDOFF.md`](reports/GANDIVA_COGNITION_FOLLOWUP_20260923/HANDOFF.md)
and the [compact probe receipt](reports/GANDIVA_COGNITION_FOLLOWUP_20260923/probe_v2.json).

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
