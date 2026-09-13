# CS-TRANSFER-001 operator runbook

Status before qualification: `IMPLEMENTED_NOT_EXECUTED`.

Canonical executable module: `anra_v5.cs_transfer_001_run_v3`.

Dedicated persistent root:

`/content/drive/MyDrive/CYMEK/CS_TRANSFER_001`

Never reuse Canary-v2, R1C, ARK, or another CS-TRANSFER root.

## Required order

1. Checkout the exact frozen `CS_TRANSFER_001_EXECUTABLE_COMMIT` in detached HEAD state.
2. Verify critical Git blob hashes.
3. Run `python tools/validate_cs_transfer_001.py`.
4. Run the dedicated CS-TRANSFER CPU qualification tests plus the relevant production backend/checkpoint tests.
5. Mount Drive and set `CS_TRANSFER_001_ROOT` to the dedicated root **before importing the runner**.
6. Run `python -m anra_v5.cs_transfer_001_run_v3 --mode protocol` and preserve the effective protocol SHA.
7. Run `--mode prepare`. This is CPU-only and must finish before any scientific GPU update. It generates the fresh candidate worlds, applies Amendment 1, selects the shared `<4096` token rows, attacks shortcuts/contamination, writes exact split rows, pack identity, DATA receipt, and PROTOCOL receipt.
8. Inspect the DATA receipt mechanically only. If any family/split acceptance is `<15%`, any required count is unavailable, any selected content token is `>=4096`, or a predictive shortcut is `>=0.35`, **stop before GPU**. Do not relax the screen after seeing it fail.
9. Run `--mode preflight --cuda`. Both physical arms must execute a real production-backend update with finite loss/gradients, valid clipping, exact parameter counts, and matched initialization. Preflight must not create scientific arm checkpoints.
10. Run `--mode scan`. It must report `START`, `RESUME`, `FINALIZE`, or `COMPLETE`; any `FAIL_CLOSED` is an operator stop.
11. Execute all eight mandatory arms sequentially. Recommended fixed order for operational simplicity: pair 0 V4096, pair 0 V24576, then pairs 1–3 in the same order. Arm order is not a scientific variable because each arm has its own frozen model/order seed and state root.
12. Re-run `--mode scan` after each arm. Never delete a partial arm. Reissuing the same arm command restores its last durable checkpoint and continues to update 480.
13. After all eight arm results are COMPLETE, run `--mode development`. This freezes the paired development aggregate and the preregistered development verdict before sealed consumption.
14. Run `--mode finalize --cuda` exactly once. Finalization writes `SEALED_CONSUMPTION.json` with `STARTED` before any sealed inference, evaluates every endpoint checkpoint, writes `FINAL_RESULT.json`, then changes the marker to `CONSUMED_AND_FINALIZED`.
15. Package receipts/traces/results and their hashes. Checkpoint objects stay in Drive and should not be copied into the compact result ZIP.

## Mandatory arms

| Pair | Model seed | Order seed | Arms |
|---|---:|---:|---|
| 0 | 4811 | 7811 | PHYS_4096, PHYS_24576 |
| 1 | 4812 | 7812 | PHYS_4096, PHYS_24576 |
| 2 | 4813 | 7813 | PHYS_4096, PHYS_24576 |
| 3 | 4814 | 7814 | PHYS_4096, PHYS_24576 |

Each arm is exactly 480 updates / 1,966,080 real tokens. Checkpoints and development evaluations occur every 60 updates, with an additional update-0 development point.

## Resume semantics

`run-arm` always means “reach the fixed update-480 endpoint,” not “run another 480.” A compatible endpoint arm returns COMPLETE without retraining. A checkpoint ahead of the durable training trace is rejected. A trace ahead of the checkpoint is trimmed and replayed. If a crash occurs after a fixed checkpoint publication but before its development evaluation, the restored checkpoint is evaluated before the next update.

Do not manually rename, delete, or edit files under an arm's `state/` or `receipts/` directories to force resume.

## Sealed-test ambiguity

If `SEALED_CONSUMPTION.json` exists with `STARTED` but no `FINAL_RESULT.json`, the sealed set has been touched and the canonical finalizer refuses a second look. Do not delete the marker. A fresh sealed identity and explicit amendment/new campaign is required.

## Result interpretation

The primary statistic is the mean matched-pair identity formation-AUC gap `PHYS_4096 - PHYS_24576`. Endpoint identity exact-token+EOS is co-primary. The full verdict taxonomy is frozen in the preregistration. Sealed evaluation can upgrade a development `SUPPORTED_PHYSICAL_CLASS_SPACE` result to `STRONG_PHYSICAL_CLASS_SPACE`; it cannot rescue a null/partial development result by changing thresholds.

A supported result says only that physical tied class space has a causal formation effect on this controlled shared-token surface. It does not justify a 4,096-token production tokenizer. A null result is equally useful because it deprioritizes physical vocabulary as the cause of Canary-v2's identity bottleneck.

No outcome authorizes PRE500M, 250M, 500M, cognition, or AGI claims.
