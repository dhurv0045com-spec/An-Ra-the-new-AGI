# RUN_EXPERIMENT — owner operating guide (FINAL-K8)

Everything except the final GPU launch is already done. The single owner
action is the two-T4 Kaggle notebook session (one allocation, 480 minutes,
training stops at 450, export reserve 30).

## Inputs (verified)

- Source: branch `Gandiva`; the fresh no-update verification is archived at
  `engineering/reports/FINAL_K8/gandiva-cognition-final-20260923/build_verification.json`.
  It verifies source revision `bd6bfb091b68679fcfdd9f544384b3b8cbd9c0f1`,
  all F01–F24, and zero local optimizer updates. The adjacent
  `BUILD_READINESS.json` and `HANDOFF.md` were generated from that report.
- Cognition/RSI changes and their measured verification are documented in
  `engineering/reports/FINAL_K8/gandiva-cognition-final-20260923/COGNITION_RSI_IMPLEMENTATION.md`.
- Data: the offline bundle `bramastra-k8-data` (identity
  `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`),
  generated with the registered command below and validated
  (`validate_bundle`: hashes, splits, information-sufficiency witnesses).
  Keep it outside Git; its manifest/audit are committed under
  `engineering/final_delivery/data/`.

Reproducible generation command (already executed; rerun only to rebuild):

```
python -m bramastra_lab.research.campaigns.k8 prepare --out <offline-bundle> --training-mechanisms 4096 --controller-mechanisms 256 --development-mechanisms 256 --confirmation-mechanisms 128 --tool-mechanisms 4096 --tool-heldout 256 --meta-train 24 --meta-validate 6 --meta-confirm 6
```

## Notebook

`notebooks/bramastra_k8.ipynb` — Run all cells in order on a Kaggle
session with two T4s. The cells:

1. discover the checkout and print source/GPU identities (no hardcoded
   paths; operator inputs are the repo/data/output locations only),
2. validate the bundle,
3. run `verify-build` (pre-allocation, zero optimizer commits) and refuse
   to continue unless the build report verifies,
4. run the E0 hardware-qualification gate (same allocation),
5. run the full campaign on the SAME allocation, stopping on any failed
   qualification,
6. summarize, 7. export (partial runs export honestly).

## Manual equivalent (without the notebook)

```
python -m bramastra_lab.research.campaigns.k8 validate --bundle <offline-bundle>
python -m bramastra_lab.research.campaigns.k8 verify-build     --data <offline-bundle> --report-dir <new-build-report-dir> --no-updates
python -m bramastra_lab.research.campaigns.k8 run --mode e0  --run-dir <run>     --data <offline-bundle> --max-wall-minutes 480     --devices cuda:0,cuda:1 --precision fp16_autocast
python -m bramastra_lab.research.campaigns.k8 run --mode full --run-dir <same-run>     --data <same-bundle> --max-wall-minutes 480     --devices cuda:0,cuda:1 --precision fp16_autocast
python -m bramastra_lab.research.campaigns.k8 summarize --run-dir <same-run>
python -m bramastra_lab.research.campaigns.k8 export --run-dir <same-run> --out <new-export>
```

`run --mode e0` and `--mode full` share the original deadline: E0
qualification is automatic inside the 480-minute window and a failed gate
stops dependent training with evidence preserved.

## Runtime environment

`pip install -r engineering/final_delivery/runtime-constraints.txt`
(keeps the image's Kaggle-compatible Torch).

## What remains GPU-only (by design)

E0 hardware qualification gates G01-G04 (two distinct T4s, full-profile
gradients/updates + fresh-process resume, measured update/evaluation
costs, live allocation + identity verification) and the actual learned
outcomes of E1-E5. They are listed as `runtime_checks_pending` in the
build report; they are not local blockers.

The cognition memory arm is an experimentally active implementation, not a
claimed performance gain: E2 compares matched `b-policy` and `b-memory`
episodes. The Kaggle result must decide whether training-only lexical memory
helps on held-out tasks. Likewise, E5's RSI method-selection receipts do not
establish recursive self-improvement until an owner run produces valid learned
evidence and independent confirmation.

## Recovery

- A failed E0: preserve the run directory and stop; do not re-launch
  without reading the preserved evidence.
- A crashed session: the ledger keeps per-phase receipts; rerun
  `run --mode full` with the SAME run dir ONLY if the ledger deadline is
  still live (no allowance reset), else keep the partial export.
- Never overwrite an existing run directory or build report.
