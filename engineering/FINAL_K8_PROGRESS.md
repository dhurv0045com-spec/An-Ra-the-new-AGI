# FINAL-K8 progress cursor

## 23 September continuation: all-family cognition verification

Commit `ec39a32811492f31047f504e4d066c8bbc74936e` removes a family-coverage
gap in both the no-update production trace and integrated E2 fixture rehearsal.
The verifier now runs the random-initialized decoder on rule-inquiry, inventory,
and program tasks, with a four-model-call cap per family. Each trace contains
real model-origin calls and truncates with unknown success; the decoder remains
in evaluation mode, has no retained gradients, and commits zero optimizer
updates. This confirms the public-input/trace wiring under a bounded diagnostic,
not competence or a positive learning result.

The integrated fixture rehearsal also selects all three E2 families across its
matched modes. The fresh full build report is
[`reports/FINAL_K8/gandiva-cognition-all-families-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-all-families-20260923/build_verification.json):
F01–F24 pass, all seven check groups pass, zero optimizer updates, duration
81.985 seconds, data identity
`79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`, and
source closure
`e603b5bbc6d3b7e4e7841482fe0b2327d9d8e253c015f5ef0f765956abb4f378`. The
source tree is marked dirty solely because the user's pre-existing edit to
`tests/test_research_k8_real.py` remains. Test-only fixture rows do not qualify
as learned evidence.

The F1–F6 semantic suite passed 37 tests with `ResourceWarning` promoted to an
error, after closing two test file handles. Its E2 fixture trace now asserts
all three family identities. Review notes are in
[`reports/GANDIVA_F6_FAMILY_REHEARSAL_20260923/HANDOFF.md`](reports/GANDIVA_F6_FAMILY_REHEARSAL_20260923/HANDOFF.md).
This closes a verification-coverage hole; the broader cognition/RSI system is
still not scientifically qualified. The owner must still run Kaggle E0 gates
G01–G04 on two T4s.

## 23 September follow-up: public-state privacy and diagnostic validity

Commit `719bcbcf5b84e80900e87f21a0cea2f49d550ff1` adds a direct world-model
prompt invariant and an end-to-end private-world twin regression. With the same
public task and history, hidden answers `north` and `south` yield opposite
environment outcomes while the pre-observation model prompts remain identical.
No private answer is injected into inference to obtain this result.

The historical C04 source probe was also inaccurate: its eight-action/eight-node
setup never expanded a depth-two child, then compared two depth-one root states.
The updated v2 probe creates a two-root, three-node search and records an
actual depth-two call. On this revision, its six legacy semantic defect flags
are all false; one depth-two call and all eight depth-one roots were observed.
The earlier `BASELINE.json` remains unchanged.

Verification used only small CPU test fixtures:

- `python -B tests/test_research_cognition_planning.py` — 11 passed.
- `$env:PYTHONPATH = (Get-Location).Path; python -B -m unittest discover -s tests -p 'test_research_cognition*.py'` — 50 passed.
- `$env:PYTHONPATH = (Get-Location).Path; python -B -m unittest discover -s tests -p 'test_research_gandiva_rsi_cognition.py'` — 7 passed.
- `$env:PYTHONPATH = (Get-Location).Path; python -B engineering/cognition_foundation_20260914/probe.py` — all six defect flags false, zero optimizer updates.

This is partial cognition evidence, not acceptance of every F1–F6 condition and
not a learned capability result. The world-model remains randomly initialized
unless the owner later runs the campaign; no local optimizer update or GPU run
occurred. Continue by auditing the remaining F1–F6 semantics against production
consumers and the active K8 contract, then expand paired CPU traces only where
the audit finds gaps.

## 23 September latest continuation: cognition prefix contract

Code revision `ab580776c4fab6398f23d503413eace41d50dbe7` repairs an
inconsistency between first-decision cognition channels and trajectory answer
batches. The initial-state channels now encode the goal and initial budgets;
trajectory answers continue to encode received history and remaining budgets.
E3 tool training uses the canonical initial prefix, with an explicit context
selector so the two batch types cannot be confused silently. A focused
regression test covers both prefixes.

Validation after this fix: **74 passed, 1 skipped, 8 subtests passed**. The
full zero-update build verifier then passed F01–F24 in 113.375 seconds, with all
seven pytest groups and the notebook exercise passing. Its report is
[`reports/FINAL_K8/gandiva-cognition-final-a6e99c9b-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-final-a6e99c9b-20260923/build_verification.json).
Verified bundle identity:
`79c9706d122050cc1e8f5e6a3363af68005fc1206db0451b40ea807121c8d7f8`; source
closure:
`88e130fdc65c21afc97f92f61e62c610365643839a5ae4a127a4b1b7adccb73c`. The
report records a dirty tree because the pre-existing user edit in
`tests/test_research_k8_real.py` was preserved. The new fix does not stage or
replace it.

This is build verification, not an experiment result: optimizer updates remain
zero. The normal two-T4 Kaggle campaign still needs owner E0 checks G01–G04 for
device mapping, real update/resume, measured runtime, and live allocation. The
100M TPU path is still preflight-only and has no real Kaggle TPU qualification.
Continue from the active F01–F24 contract; do not report this as AGI or a
learned capability result.

## 23 September continuation: cognition and 100M Kaggle TPU

The latest source-bound zero-update report is
[`reports/FINAL_K8/gandiva-cognition-tpu-r6-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-tpu-r6-20260923/build_verification.json).
It passes F01–F24 for the existing 6,493,952-parameter K8 campaign, with zero
local optimizer updates and source closure
`3fa4bc14f649741a80b34e7931e83228e232077aadf014746d450db89b3b50a1` at
revision `f7d10985eaee0dcaadbe1db5480e46f683cfe8ba`. G01–G04 still require
the owner's actual two-T4 E0 run. The verifier says `dirty=true` because the
user's notebook and K8 test edits remained present and were deliberately
excluded from the implementation commit; the report is not a claim that the
whole working tree was clean.

This continuation adds append-only, serializable belief-revision audit records
for applied evidence, duplicate replays, and finite-support model mismatches.
The record retains prior, raw/effective likelihood, posterior, reliability,
and episode-local references; restore validates event order and references.
It is an auditable reference updater, not a learned belief module. TPU runtime
helpers now broadcast master weights and verify per-rank initialization
receipts; focused tests cover scalar buffers and malformed/divergent receipts.
Device routing selects the BF16 autocast mode required by K8Trainer on XLA,
fixing a trainer-construction blocker in the generic non-CUDA path. Focused
validation passed 106 tests and 16 subtests. The source-bound verifier is the
complete current K8 contract check; it did not instantiate or train the 100M
model.

The 100M/cognition details and remaining work are tracked in
[`TPU_100M_COGNITION_PROGRESS.md`](TPU_100M_COGNITION_PROGRESS.md). The normal
two-T4 notebook remains the existing owner experiment and is not changed by
this implementation commit. The `tpu_100m` profile is still configuration-only
and no Kaggle TPU campaign consumer or real TPU qualification exists. Continue
with a dedicated TPU preflight/consumer and real eight-core measurement; this
update is a checkpoint, not task completion.

Implementation revision: `f7d10985eaee0dcaadbe1db5480e46f683cfe8ba` on
`Gandiva`, based on `BRAMASTRA`. The user's changes to
`notebooks/bramastra_k8.ipynb` and `tests/test_research_k8_real.py` were
preserved and excluded from this commit.

Active assignment: FINAL-K8, requirements F01–F24. The older ten-day schedule
and 15 September source review are history; their defect lists and readiness
labels predate the current implementation.

## Completed and verified

- The normal self-sustaining notebook
  [`notebooks/bramastra_k8.ipynb`](../notebooks/bramastra_k8.ipynb) now clones
  `Gandiva` by default, generates its own complete bundle if no Kaggle input is
  attached, and follows the registered 480-minute / two-T4 plan: 450 minutes
  for training and 30 minutes reserved for export.
- The data contract now contains 4096 tool-training rows and 256 held-out
  compositions, allowing E3-T0 to consume any calibrated target up to 4000
  without replacement or synthetic padding. Full bundle identity:
  `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`.
- `validate` passed with no issues. The generated 29-file bundle is 73,321,319
  bytes; its ZIP is outside Git at
  `C:\Users\ankit\AppData\Local\Temp\bramastra-k8-data-6fb94b7018406632.zip`
  (integrity checked, 2,171,411 bytes).
- Fresh `verify-build --no-updates` passed all F01–F24 checks and the integrated
  rehearsal against the committed implementation. Report:
  [`reports/FINAL_K8/gandiva-cognition-final-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-final-20260923/build_verification.json).
  Its source closure is `cb005c2cbb3a02da888a5cf271805b6751954e403cbee1de6794ba82482963b7`,
  data identity is `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`,
  and measured verifier duration is 130.563 seconds. All seven groups passed:
  foundation (75 passed, 3 skipped), data/splits (34), experience/codec (65),
  checkpoint/ledger (39 passed, 1 skipped), gate/readiness (10), phase
  contracts (91 plus 12 subtests), and statistics/evaluation (53). F01–F24
  are all PASS, local optimizer updates are zero, and evidence-backed
  `ready_for_owner_experiment` is true. The generated readiness and handoff
  are beside the report; runtime checks G01–G04 remain mandatory in E0.
- Cognition is now exercised on the production E2 episode path. E2 builds a
  frozen exemplar index solely from the exact `training` split and compares
  `b-memory` against the same learner/environment/seed in `b-policy`.
  Retrieved content is inserted through the shared renderer, ineligible
  scopes are filtered before ranking, per-decision record aliases and byte-
  codec token costs are traced, and an overflow drops whole records rather
  than slicing evidence. New focused coverage:
  `python -m pytest tests/test_research_gandiva_rsi_cognition.py -q
  --maxfail=1` → 6 passed. This establishes execution and leakage boundaries,
  not that memory improves task accuracy.
- RSI evidence is stronger and more honest: E5's compiled objective weights
  now reach support windows; currently unsupported supervision terms are
  rejected instead of silently ignored. M2 applies its declared 0.5 clip
  bound. Proposer/successor captures are checked against exact restorable
  payloads and provenance; P0/P1/P_fixed checkpoint ancestry and the fixture-
  labeled M24 generation receipt chain are recorded. Gated migration probes
  preserve global RNG, model mode, and existing gradients. These changes have
  focused regression coverage in the new Gandiva test file and the full
  verifier groups.
- All 12 notebook code cells compile and the 24-cell notebook parses as JSON.

## Remaining owner work and limits

Attach no inputs if desired: the normal notebook clones the pinned branch and
builds its data automatically, so it needs Kaggle Internet access and a
`GPU T4 x2` session. Run all cells in order. The single E0 gate must validate
both actual T4s, full-profile update/resume, measured timing, and the live
allocation before E1–E6 can train. No local optimizer update, CUDA qualification
or learned campaign was run; no AGI or scientific outcome is claimed.

Evidence and implementation notes are in the report directory above. The
compact bundle
manifest/audit are committed under `engineering/final_delivery/data/`; the
full bundle remains outside Git. Do not overwrite old reports. Continue any
new work from this cursor and the active FINAL-K8 contract.
