# FINAL-K8 engineering status

## 23 September cognition integration update

Use the fresh report
[`reports/FINAL_K8/gandiva-cognition-runtime-r2-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-runtime-r2-20260923/build_verification.json)
for the current implementation closure
`cd4b02d77bbde04772989b6c8e8533a1d1d0c076a4d897e4d6289351aa69c8e6` and data
identity `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`.
It passes F01–F24 with zero local optimizer updates. The production E2
workspace-policy arm now consumes a typed, persistent-per-episode evidence
ledger with explicit conflict/supersession status. G01–G04 still require the
owner's two-T4 E0 run, so build readiness remains distinct from accelerator
qualification and learned results.

## 23 September continuation note

The K8 verifier now has a newer zero-update report at
[`reports/FINAL_K8/gandiva-cognition-tpu-r3-20260923/build_verification.json`](reports/FINAL_K8/gandiva-cognition-tpu-r3-20260923/build_verification.json):
F01–F24 pass for the existing 6,493,952-parameter K8 architecture; real
hardware checks G01–G04 remain pending. Cognition and the experimental 100M
Kaggle TPU path continue in
[`TPU_100M_COGNITION_PROGRESS.md`](TPU_100M_COGNITION_PROGRESS.md). The new
`tpu_100m` profile and XLA trainer/data helpers are not yet wired into the
production E1–E6 campaign or a Kaggle TPU notebook, and have not been run on a
TPU. Do not interpret K8 build verification as 100M TPU qualification.

## Current source and readiness

Active execution branch: `Gandiva`, based on `BRAMASTRA`. The implementation at
`be946613d415751bf8cd6e227d7624040f1c9d6a` passed a fresh zero-optimizer-update
build verification with all F01–F24 requirements passing. The source-code
closure is `fa9a41453967e5ae89c3b8a4c12484ef47f2be1985a27bfb655f05426fc1974b`.
The immutable report is
[`reports/FINAL_K8/gandiva-20260923/build_verification.json`](reports/FINAL_K8/gandiva-20260923/build_verification.json).

The owner-facing notebook is
[`../notebooks/bramastra_k8.ipynb`](../notebooks/bramastra_k8.ipynb). It defaults
to cloning `Gandiva`, builds the full data bundle if no Kaggle Dataset is
attached, and uses one two-T4 allocation capped at 480 minutes (450 training,
30 export). No local optimizer updates or accelerator campaign were run.

The matching offline bundle identity is
`6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`; its
compact manifest and audit are in `final_delivery/data/`. The local bundle ZIP
is outside Git, as required by the delivery contract. E0 must still qualify the
owner's actual pair of T4s, precision, resume, timing, and live allocation
(G01–G04). A passing build report is not an AGI or learned-result claim.

## Resume instructions

Continue from [`FINAL_K8_PROGRESS.md`](FINAL_K8_PROGRESS.md) and the active
contract [`FINAL_EXPERIMENT_EXECUTION.md`](FINAL_EXPERIMENT_EXECUTION.md). The
15–24 September handoff is supporting history; do not rely on its old status
snapshots. Preserve the pre-existing uncommitted change to
`tests/test_research_k8_real.py` unless its owner directs otherwise.
