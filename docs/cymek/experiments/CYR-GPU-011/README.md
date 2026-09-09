# CYR-GPU-011

Long exposure-matched capability-emergence bridge for Google Colab GPU.

## Current state

**EXECUTED / COMPLETE.** Raw result and post-run audit are preserved in:

- `RESULT.md`
- `artifacts/v5/cyr_gpu_011_result_receipt.json`
- `artifacts/v5/CYMEK_GPU_RESEARCH_V11_RESULTS.zip`

Bundle SHA-256: `fbec390f66223a19a998db6519f42c046ad8cb0a345c205b168f5bd1a86668e5`.

Official preregistered verdict: `NO_G90_WITH_INCOMPLETE_EXPOSURE`.

The production bridge nevertheless completed 100% of the ARK-002B semantic exposure box (1,152,000 row presentations / 18,000 batch-64 updates) and remained at 0% held-out STANDARD and 0/48 SEALED despite confirmed train M99. The compact bridge timeboxed at 44.89% exposure, reached sustained G50 and finished at 56.47% DEV_MEASUREMENT STANDARD exact-with-EOS, but did not reach G90.

Read `RESULT.md` before using raw structural flags: the raw `COMMUTATION_INVARIANCE=true` label is rejected by the post-run audit because reversing operands changes the task's first-operand OOD tens-band axis. The observed 100% COMMUTED score is therefore not a clean invariance result.

## Why V11 existed

CYR-GPU-009 and the superseded V10 design were semantically underexposed relative to the successful Arkenstone ARK-002B reference. V11 used the exact ARK-002B frozen manifest and measured exposure in semantic row presentations, targeting the same 1,152,000-row box when wall time permitted.

## Experiment order

1. `COMPACT_BRIDGE` — real Cymek V5 4L/128w, exact ARK data, 19-symbol arithmetic representation, canonical Cymek causal objective, max 25 minutes.
2. `PRODUCTION_PRIMARY` — same V5 geometry/data/objective with frozen 24,576-token tokenizer, remaining science wall.
3. `PRODUCTION_REPLICATION` — only if primary qualified G90 and at least 40 minutes remained. It was not launched because primary never reached G90.

The compact stage was a bridge, not an exact ARK reproduction: ARK-002B supervised an answer-prefix BOS while Cymek's canonical objective excludes BOS targets. V11 deliberately kept Cymek's objective rather than changing production semantics.

## Primary gate

Candidate-free exact answer with valid EOS stop. Controller G90 required three consecutive DEV_CONTROLLER evaluations >=.90 and final DEV_MEASUREMENT/STANDARD exact-with-valid-EOS >=.90.

No bridge qualified G90.

## Diagnostics

STANDARD, COMMUTED, LOCALITY, CARRY, TRIPLE_ADD, THREE_DIGIT and VERBAL were reported separately. They characterize controlled transfer and are not an aggregate reasoning/AGI score.

## Evidence boundary

No result from this experiment authorizes broad reasoning claims, TPU equivalence, PRE500M, 500M training, production promotion, or 5B-corpus work.

## Next highest-information move

Do not rerun all of V11. Production already completed full semantic exposure. The clean next closure is to continue/recreate only the compact bridge to the full 1,152,000 row presentations with corrected structural probes. That resolves whether Cymek can enter the delayed ARK-like generalization regime under compact representation before spending another long session on more complex mechanisms.
