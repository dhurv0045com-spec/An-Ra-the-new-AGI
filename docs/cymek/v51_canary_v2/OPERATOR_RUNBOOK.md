# V5.1 Canary-v2 — Operator Runbook

This runbook is intentionally fail-closed. It does not authorize changing the preregistration, scientific endpoint, output treatment, or evaluation thresholds from Colab.

## Canonical launcher and frozen science identity

Use only:

`notebooks/cymek_colab_v51_canary_v2_t4.ipynb`

Colab:

`https://colab.research.google.com/github/dhurv0045com-spec/An-Ra-the-new-AGI/blob/cymek-v51-canary-v2/notebooks/cymek_colab_v51_canary_v2_t4.ipynb`

The branch may advance for operator-only notebook, documentation, or audit hardening. The scientific executable does **not** follow branch HEAD. The launcher must detach to exactly:

`4470a34b7e2d4b2d673c328ef962c84d8e075b89`

Any different scientific commit is a different executable identity and requires an explicit review/amendment rather than an implicit rerun. The launcher also verifies critical Git blob identities before permitting the experiment to proceed.

## Persistent root

Use a dedicated Drive root:

`/content/drive/MyDrive/CYMEK/V5_1_CANARY_V2`

Never point V2 at the V1, R1C, or ARK-020 roots.

The operator notebook sets `V51_CANARY_V2_ROOT` for every runner subprocess. Persistent artifacts then live under that root:

- `training.jsonl`, `development.jsonl`, `sealed.jsonl`
- `receipts/`
- `state/`
- `EXECUTABLE_BINDING.json`
- `PREREGISTRATION_FROZEN.json`
- `SEALED_CONSUMPTION.json` after finalization begins
- `OPERATOR_RESULT.json` after successful finalization

Do not delete or replace state/receipt files to force a restart. Resume the same bound lineage.

## Frozen execution order

1. Verify T4/CUDA.
2. Checkout scientific commit `4470a34b7e2d4b2d673c328ef962c84d8e075b89` in detached HEAD state.
3. Verify critical Git blob identities.
4. Run the static V2 validator and the V1+V2 CPU qualification tests.
5. Mount Drive, establish/verify the executable binding, and set the dedicated root.
6. `--mode prepare`: deterministically generate fresh V2 splits, run contamination checks and shortcut baselines, and write the DATA receipt. Any exact/normalized/group collision or shortcut score >=0.35 aborts. If the bound DATA receipt already exists, preserve it rather than regenerating it.
7. `--mode preflight --cuda`: one update through the real production backend. V2 executes preflight in an isolated temporary checkpoint root and distinct `v51-canary-v2-preflight` lineage; it must report `scientific_state_untouched=true`.
8. `--mode scan`: obtain `START`, `RESUME`, `COMPLETE`, or `FAIL_CLOSED`.
9. `--mode run --cuda --updates 360 --checkpoint-every 24`: fixed 360-update FP32 endpoint. A rerun after interruption resumes the same lineage; `--updates 360` is the total endpoint, not an additional 360 updates.
10. `--mode evaluate --cuda`: development diagnostics only after the endpoint.
11. `--mode finalize --cuda`: finalization follows without outcome-based branching on development metrics. It creates the sealed-consumption marker before touching sealed rows and writes `FINALIZATION.json` exactly once.
12. Package operator receipts into `CYMEK_V51_CANARY_V2_RESULTS.zip`. Checkpoint state and raw train/dev/sealed rows remain outside the result bundle.

## Resume semantics

The target `--updates` value is the **total endpoint**, not “another 360.” A checkpoint at update 144 resumes updates 145..360. The epoch for global zero-based update `u` is `u // windows_per_epoch`; the position inside that deterministic sampler epoch is `u % windows_per_epoch`. Scheduler position remains cumulative-token indexed across epoch and process boundaries; there is no epoch rewarm.

The training receipt is merged by global update number. A conflicting duplicate row or any gap fails closed. Before each checkpoint publication, V2 first persists a training trace through that same update. If a checkpoint publication fails after the trace write, resume trims the trace back to the last durable checkpoint and deterministically replays the uncheckpointed suffix. A checkpoint ahead of its durable training history is not accepted.

The checkpoint store remains the V5 production `CheckpointStore`; V2 does not introduce a parallel checkpoint format.

## Sealed-test firewall

`evaluate` never consumes sealed rows. `finalize` first checks all pre-sealed mechanical gates, then creates `SEALED_CONSUMPTION.json` with status `STARTED` **before** sealed inference. If the process dies after that marker but before `FINALIZATION.json`, do not rerun sealed evaluation. The sealed set is considered compromised for a second look; an explicit amendment with a fresh sealed identity is required.

If `FINALIZATION.json` already exists, finalization returns `COMPLETE` without touching sealed data again. The operator notebook likewise treats an incomplete sealed marker as fail-closed rather than attempting automatic recovery.

## Fixed scientific constraints

- Rung A only: 10,227,456 parameters.
- 360 updates exactly.
- 4,096 real tokens/update; total 1,474,560.
- canonical tied full softmax only.
- CUDA FP32 substantive run; BF16 is rejected for V2.
- fresh V2 seed and fresh train/development/sealed split identities.
- V1 formation thresholds are unchanged.
- no outcome-based stopping.
- no MASK_4096 / inactive-offset rescue.
- no tokenizer promotion, PRE500M authorization, 250M run, or 500M run.

## Interpreting the result

`CANARY_V2_PASS`: all mechanical gates and all retained formation gates pass.

`CANARY_V2_FAIL_FORMATION`: execution/integrity gates pass, but at least one formation gate fails.

`CANARY_V2_FAIL_ENGINEERING`: a required execution/integrity gate fails.

`CANARY_V2_INCONCLUSIVE`: reserved for an externally interrupted run that has valid resumable state but has not reached the fixed endpoint. Do not convert an incomplete session into a scientific result.

A pass authorizes only the preregistered next experiment, `CS-TRANSFER-001`. It does not authorize production scaling by itself. No V2 outcome can issue `READY_FOR_500M`.
