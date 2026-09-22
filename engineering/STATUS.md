# FINAL-K8 engineering status

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
