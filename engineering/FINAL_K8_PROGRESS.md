# FINAL-K8 progress cursor

Source revision: `be946613d415751bf8cd6e227d7624040f1c9d6a` on `Gandiva`, based
on `BRAMASTRA`. The only pre-existing local edit is the user's change to
`tests/test_research_k8_real.py`; it was preserved and not included in the
Gandiva implementation commit.

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
  rehearsal. Report:
  [`reports/FINAL_K8/gandiva-20260923/build_verification.json`](reports/FINAL_K8/gandiva-20260923/build_verification.json).
  Seven groups passed: foundation (75 passed, 3 skipped), data/splits (34),
  experience/codec (65), checkpoint/ledger (39, 1 skipped), gate/readiness (10),
  phase contracts (91 plus 12 subtests), statistics/evaluation (53). Zero
  optimizer updates were committed. `ready_for_owner_experiment` is
  evidence-backed true; E0 checks G01–G04 remain mandatory at runtime.
- All 12 notebook code cells compile and the 24-cell notebook parses as JSON.

## Remaining owner work and limits

Attach no inputs if desired: the normal notebook clones the pinned branch and
builds its data automatically, so it needs Kaggle Internet access and a
`GPU T4 x2` session. Run all cells in order. The single E0 gate must validate
both actual T4s, full-profile update/resume, measured timing, and the live
allocation before E1–E6 can train. No local optimizer update, CUDA qualification
or learned campaign was run; no AGI or scientific outcome is claimed.

Immutable evidence is stored in the report directory above. The compact bundle
manifest/audit are committed under `engineering/final_delivery/data/`; the
full bundle remains outside Git. Do not overwrite old reports. Continue any
new work from this cursor and the active FINAL-K8 contract.
