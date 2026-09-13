# V5.1 Canary-v2 — Operator Runbook

This runbook is intentionally fail-closed. It does not authorize changing the preregistration, scientific endpoint, output treatment, or evaluation thresholds from Colab.

## Persistent root

Use a dedicated Drive root:

`/content/drive/MyDrive/CYMEK/V5_1_CANARY_V2`

Never point V2 at the V1, R1C, or ARK-020 roots.

The operator notebook sets `V51_CANARY_V2_ROOT` **before importing** the runner. Persistent artifacts then live under that root:

- `training.jsonl`, `development.jsonl`, `sealed.jsonl`
- `receipts/`
- `state/`
- `SEALED_CONSUMPTION.json` after finalization begins

## Frozen execution order

1. Verify T4/CUDA.
2. Checkout the exact `CANARY_V2_EXECUTABLE_COMMIT` in detached HEAD state.
3. Verify critical Git blob identities.
4. Run the static V2 validator and the V1+V2 CPU qualification tests.
5. Mount Drive and set the dedicated root.
6. `--mode prepare`: deterministically generate fresh V2 splits, run contamination checks and shortcut baselines, and write the DATA receipt. Any exact/normalized/group collision or shortcut score >=0.35 aborts.
7. `--mode preflight --cuda`: one update through the real production backend.
8. `--mode scan`: obtain `START`, `RESUME`, `COMPLETE`, or `FAIL_CLOSED`.
9. `--mode run --cuda`: fixed 360-update endpoint, FP32, checkpoint every 24 updates. A rerun after interruption resumes the same lineage.
10. `--mode evaluate --cuda`: development diagnostics only.
11. `--mode finalize --cuda`: only after update 360. This creates the sealed-consumption marker before touching the sealed rows and then writes `FINALIZATION.json` exactly once.

## Resume semantics

The target `--updates` value is the **total endpoint**, not “another 360.” A checkpoint at update 144 resumes updates 145..360. The epoch for global zero-based update `u` is `u // windows_per_epoch`; the position inside that deterministic sampler epoch is `u % windows_per_epoch`. Scheduler position remains cumulative-token indexed across epoch and process boundaries; there is no epoch rewarm.

The training receipt is merged by global update number. A conflicting duplicate row or any gap fails closed. The checkpoint store remains the V5 production `CheckpointStore`; V2 does not introduce a parallel checkpoint format.

## Sealed-test firewall

`evaluate` never consumes sealed rows. `finalize` creates `SEALED_CONSUMPTION.json` with status `STARTED` **before** sealed inference. If the process dies after that marker but before `FINALIZATION.json`, do not rerun sealed evaluation. The sealed set is considered compromised for a second look; an explicit amendment with a fresh sealed identity is required.

If `FINALIZATION.json` already exists, finalization returns `COMPLETE` without touching sealed data again.

## Fixed scientific constraints

- Rung A only: 10,227,456 parameters.
- 360 updates exactly.
- 4,096 real tokens/update; total 1,474,560.
- canonical tied full softmax only.
- CUDA FP32 substantive run; BF16 is rejected for V2.
- V1 formation thresholds are unchanged.
- no outcome-based stopping.
- no MASK_4096 / inactive-offset rescue.

## Interpreting the result

`CANARY_V2_PASS`: all mechanical gates and all retained formation gates pass.

`CANARY_V2_FAIL_FORMATION`: execution/integrity gates pass, but at least one formation gate fails.

`CANARY_V2_FAIL_ENGINEERING`: a required execution/integrity gate fails.

`CANARY_V2_INCONCLUSIVE`: reserved for an externally interrupted run that has valid resumable state but has not reached the fixed endpoint. Do not convert an incomplete session into a scientific result.

No V2 outcome can issue `READY_FOR_500M`.
