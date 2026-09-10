# ARK-017 V2 PRE-EXECUTION RE-AUDIT — 2026-09-10

**Status:** STATIC RE-AUDIT PASS / GPU SMOKE STILL REQUIRED / SCIENTIFIC RESULT NOT EXECUTED

This re-audit was performed after an unrelated R1 (`CYR-GPU-012-R1`) pre-execution unit-test fixture failed before training. The purpose is to verify that ARK-017 V2 does not contain the same class of launch-gate error and that the current Colab launcher still resolves its frozen runner correctly.

## Frozen identity

- Scientific runner commit: `377c4743f8017e3455f576eafb75bb8ab9c50284`
- Runner path: `experiments/ARK-017/run_ark017_v2.py`
- Runner Git blob SHA: `ea8d595f605f2fc258561297978280b7d051921b`
- Colab path: `experiments/COLAB/arkenstone_ark017_v2.ipynb`
- Readiness record: `experiments/ARK-017/RUN_READINESS_V2.json`
- Current live Arkenstone head before this audit: `708564bed852c2d97eb9d7ccea27d279a108d19d`
- Frozen runner was 27 commits behind that live head and remained an ancestor, so the launcher's `--depth 80` clone can currently resolve it.

## Static launch-path checks

PASS:

1. The notebook pins the exact same runner commit as `RUN_READINESS_V2.json`.
2. It checks out that commit detached and asserts `git rev-parse HEAD` equals the frozen SHA before smoke or training.
3. It `py_compile`s the V2 runner and all critical inherited runner modules before execution.
4. Unlike R1, ARK-017 V2 has no synthetic runtime-resolver pytest fixture whose guessed throughput can fail the launch gate; therefore the R1 fixture error is not present here.
5. The first scientific executable action is a real CUDA smoke test, and the full campaign is blocked unless its return code is zero.
6. The smoke verifies the frozen ARK-014 binding manifest, exact 1/16 non-identity replay treatment, non-replay row identity, CUDA forward/backward, model+optimizer snapshot restoration, deterministic next update from identical forks, multi-step LOW movement trace consumption by capped-HIGH, and finite optimizer state.
7. The full campaign pins `--expected-head` to the same scientific commit.
8. JSON partial receipts and final ZIPs are copied to Google Drive while the campaign is running.

## Scientific-design checks

PASS at the static level:

- fresh acquisition seeds: `2601, 2702, 2803`;
- continuation order seeds: `10801, 10802`;
- acquisition uses order-augmented binding training;
- each primary continuation arm starts from the same acquisition snapshot;
- primary arms remain HIGH, LOW reference, HIGH+CAP1X, HIGH+exact noncanonical 1/16 replay, CAP1X+replay, and augmented-HIGH reference;
- LOW is used to materialize the prospective applied-delta trace before all other arms restart from the same fork;
- CONTROL determines qualification/failure logic while SEALED is measurement-only;
- primary mechanism verdict is computed before any optional efficiency screen can be interpreted;
- 1/32 and 1/64 replay or CAP4X/CAP16X are secondary only.

## Remaining operational limitation

The `240` minute value is a campaign safety budget, **not an OS-level hard timeout**. Budget checks occur before acquisitions / matched continuation sets, but an already-started set is allowed to finish; therefore actual wall time can exceed 240 minutes if a set begins near the remaining-time boundary. This does not invalidate the scientific comparison, but the operator should not interpret `240` as a guaranteed maximum elapsed time.

The runner also preserves partial receipts rather than resumable per-arm model checkpoints. A runtime loss in the middle of a six-arm matched set may require repeating that incomplete set. Do not treat a Drive partial JSON as an exact optimizer-resume checkpoint.

## Verdict

**No R1-like pre-execution fixture defect was found. ARK-017 V2 remains READY FOR OPERATOR GPU SMOKE.**

This is not evidence that the GPU smoke will pass on every Colab environment and is not a scientific result. If the smoke fails, stop before the full campaign and preserve the exact traceback/output; do not modify thresholds or treatments after observing scientific outcomes.
