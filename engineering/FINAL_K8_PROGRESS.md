# FINAL-K8 progress cursor

## 23 September cognition runtime integration

The current zero-update build report is
[`reports/FINAL_K8/gandiva-cognition-runtime-r2-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-runtime-r2-20260923/build_verification.json).
It passes F01–F24 for the existing 6,493,952-parameter K8 campaign, records
zero local optimizer updates, and has source closure
`cd4b02d77bbde04772989b6c8e8533a1d1d0c076a4d897e4d6289351aa69c8e6` and data
identity `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`.
It took 103.921 seconds. G01–G04 still require the owner's actual two-T4 E0
qualification; this report is not evidence of GPU qualification or an AGI
result.

E2's live `b-workspace` episode now records received observations through the
typed `CognitiveWorkspace` ledger. The prompt gets episode-local evidence
aliases; the durable cognitive snapshot preserves ancestry outside the model
input, conflict and supersession state, step count, and a content identity.
Same-time contradictory values remain conflicting, while later valid-time
observations mark older records superseded, including when a conflict exists
at an intermediate time. The build verifier's foundation
group now includes `tests/test_research_cognition_runtime.py`; focused cognition
and episode checks pass (110 tests, 16 subtests), and the O04/O05 operational
selection passes (7 tests). No optimizer update was run.

This implementation keeps the existing normal two-T4 K8 notebook and its E2
comparison plan. The separate `tpu_100m` profile remains configuration-only;
it is not consumed by the campaign and has not been TPU-qualified.

## 23 September continuation: cognition and 100M Kaggle TPU

Latest build evidence:
[`reports/FINAL_K8/gandiva-cognition-tpu-r3-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-tpu-r3-20260923/build_verification.json).
All F01–F24 pass for the existing 6,493,952-parameter K8 campaign, with zero
local optimizer updates; G01–G04 still require the owner's real two-T4 E0
run. This does **not** establish 100M TPU readiness.

The 100M/cognition continuation and next implementation steps are tracked in
[`TPU_100M_COGNITION_PROGRESS.md`](TPU_100M_COGNITION_PROGRESS.md). This turn
adds provenance-aware Bayesian belief revisions and bounded/provenance-safe
cognitive rendering, the configuration-only 100,334,720-parameter `tpu_100m`
profile, an XLA replica runtime and no-padding shard loader, and global
token/action/value/pair objective normalization with pair activations flushed
between base and counterfactual passes. Focused regression checks pass: 152
passed, 1 skipped, 16 subtests; no local optimizer update was run. The current
host has 2.87 GiB available RAM, no CUDA, and no PyTorch/XLA, so the 100M
profile was not instantiated locally.

**Important boundary:** `notebooks/bramastra_k8.ipynb` and the E1–E6 campaign
consumer still target the existing two-T4 K8 profile. No production Kaggle
TPU entry point currently consumes the new sampler/backend or trains the 100M
profile. Build a dedicated TPU preflight/consumer and qualify the real eight
cores, memory, initialization, checkpoint/resume, and throughput before calling
that path ready. Continue the active objective; this update is a checkpoint,
not task completion.

Implementation revision: `bd6bfb091b68679fcfdd9f544384b3b8cbd9c0f1` on
`Gandiva`, based on `BRAMASTRA`. The only pre-existing local edit is the user's
change to `tests/test_research_k8_real.py`; it was preserved and not included
in the implementation commit.

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
