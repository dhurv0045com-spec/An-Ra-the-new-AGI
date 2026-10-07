# RUN_EXPERIMENT — owner operating guide (FINAL-K8)

Everything except the final GPU launch is already done. The single owner
action is the two-T4 Kaggle notebook session (one allocation, 600 minutes,
training stops at 570, export reserve 30).

## Inputs (verified)

- Source: branch `BRAMASTRA`, revision recorded in
  `engineering/reports/FINAL_K8/build_verification.json`
  (`source_identity.git_head` + `source_closure_sha256`).
- Data: the offline bundle `bramastra-k8-data` (identity
  `ab2bd6fa0efe227a8bb4aed1a70112f3698c097a5e5ffcedffcaf080af3207f0`),
  generated with the registered command below and validated
  (`validate_bundle`: hashes, splits, information-sufficiency witnesses).
  Keep it outside Git; its manifest/audit are committed under
  `engineering/final_delivery/data/`.

Reproducible generation command (already executed; rerun only to rebuild):

```
python -m bramastra_lab.research.campaigns.k8 prepare --out <offline-bundle>   --training-mechanisms 4096 --controller-mechanisms 256   --development-mechanisms 256 --confirmation-mechanisms 128   --tool-mechanisms 4096 --tool-heldout 256   --meta-train 24 --meta-validate 6 --meta-confirm 6
```

## Notebook

`notebooks/bramastra_k8.ipynb` — Run all cells in order on a Kaggle
session with two T4s. The cells:

1. discover the checkout and print source/GPU identities (no hardcoded
   paths; operator inputs are the repo/data/output locations only),
2. **acquire the data automatically**: attached dataset first (rejected if
   its tool cardinality cannot satisfy E3), else the registered bundle is
   downloaded from the pinned GitHub release
   (`releases/tag/k8-data-v1`, SHA-256-verified, unpacked losslessly
   under `/kaggle/working`), else generated in-session,
3. validate the bundle,
4. run `verify-build` (pre-allocation, zero optimizer commits) and refuse
   to continue unless the build report verifies,
5. start the auto-safety thread: run-directory snapshots every 10 minutes
   and at each phase boundary into `/kaggle/working` (recovery ZIPs with
   ledger/phase outputs/logs; heavy payloads excluded),
6. run the E0 hardware-qualification gate (same allocation),
7. run the full campaign on the SAME allocation, stopping on any failed
   qualification; E1/E3/E4 additionally checkpoint every 200 steps/updates,
8. summarize, 9. export (partial runs export honestly), 10. xprobe,
11. package `results-*.zip` — in Save & Run All mode it lands in the
    version's Output tab automatically; in interactive mode a FileLink is
    displayed, and the final safety sweep cell writes one last snapshot.

No manual dataset attachment is required. Losing your own network
connection does not stop a Save & Run All commit; the run continues
server-side and the Output is saved when it completes.

## Manual equivalent (without the notebook)

```
python -m bramastra_lab.research.campaigns.k8 validate --bundle <offline-bundle>
python -m bramastra_lab.research.campaigns.k8 verify-build     --data <offline-bundle> --report-dir <new-build-report-dir> --no-updates
python -m bramastra_lab.research.campaigns.k8 run --mode e0  --run-dir <run>     --data <offline-bundle> --max-wall-minutes 600     --devices cuda:0,cuda:1 --precision fp16_autocast
python -m bramastra_lab.research.campaigns.k8 run --mode full --run-dir <same-run>     --data <same-bundle> --max-wall-minutes 600     --devices cuda:0,cuda:1 --precision fp16_autocast
python -m bramastra_lab.research.campaigns.k8 summarize --run-dir <same-run>
python -m bramastra_lab.research.campaigns.k8 export --run-dir <same-run> --out <new-export>
```

`run --mode e0` and `--mode full` share the original deadline: E0
qualification is automatic inside the 600-minute window and a failed gate
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

## Recovery

- A failed E0: preserve the run directory and stop; do not re-launch
  without reading the preserved evidence.
- A crashed session: the ledger keeps per-phase receipts; rerun
  `run --mode full` with the SAME run dir ONLY if the ledger deadline is
  still live (no allowance reset), else keep the partial export.
- Never overwrite an existing run directory or build report.
