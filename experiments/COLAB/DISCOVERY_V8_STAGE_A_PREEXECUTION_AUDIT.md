# Discovery V8 Stage A — pre-execution audit (ARK-017)

## What is being launched

`experiments/COLAB/arkenstone_ark017.ipynb` — pinned one-click T4 launcher for
`experiments/ARK-017/run_ark017.py` (mechanism factorial: update magnitude vs
invariant-supporting replay, on the reliable ARK-015 failure generator).

## Pinned execution identity

| item | value |
|---|---|
| PINNED_RUNNER_COMMIT | `59e1b805b7d93b7f2e1e9d3ea66b34c4fabca9c8` (commit that introduces `run_ark017.py`) |
| ARK-017 PLAN commit | `4d145288215c304310253a93881073d5d1a03800` — verified ancestor of pin |
| MASTER V8 PLAN commit | `6f0a38088e966494bb7caf2f3b49ea235c84f971` — verified ancestor of pin |
| Plan SHAs embedded in runner | match the commit hashes above; runner prints both at startup |
| Default budget | 180 minutes (T4 safety budget, not a minimum) |

## Runner self-checks (fail-closed)

- `--expected-head` must equal the pinned commit or the run aborts before training.
- Smoke-test mode exercises GPU forward/loss, exact fork restore, optimizer
  delta-step, and the applied-delta cap primitive; any failure aborts before the
  full campaign.
- SEALED data is measurement-only; qualification uses CONTROL robustness only
  (`canonical >= .90, order_only >= .85, query_order >= .85`).
- Every outcome writes a canonical JSON receipt; unexpected exceptions write
  `ARK-017_FAILURE_RECEIPT.json`; all receipts are zipped into
  `ARKENSTONE_ARK017_RESULTS.zip` for independent hash revalidation on import.

## Compile gate

The notebook py_compiles the full import chain before running:
`discovery_v7_common.py`, `run_ark017.py`, `run_ark015.py`, `run_ark011.py`,
`run_ark014.py`, `run_ark001.py`, `lib/ark_tasks.py`.

## Frozen design (from PLAN.md, unchanged)

- Binding manifest: `fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3`
  (identical to ARK-014/015).
- Acquisition seeds `2601, 2702, 2803`; continuation seeds `10801, 10802`.
- Arms: canonical-only HIGH, LOW reference, HIGH capped to LOW movement,
  HIGH + 1/16 order-diverse replay, HIGH cap + replay, full-augmented HIGH reference.
- 16k max acquisition, 8k continuation, eval every 200 steps, batch 64.

## Verdicts the campaign is allowed to return

`UPDATE_MAGNITUDE_SUFFICIENT`, `REPLAY_SUPPORT_SUFFICIENT`, `JOINT_CONTROL_REQUIRED`,
`BOTH_SUFFICIENT`, `MIXED`, `LOW_EVENT_OR_BUDGET_BLOCKED` — exactly one, written by
the runner with the evidence that produced it. No post-hoc reinterpretation.
