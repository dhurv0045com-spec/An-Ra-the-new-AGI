# HORM-001 experiment.md

**Date:** 2026-09-17
**Branch:** cymek-beta (worktree C:\Users\ankit\Downloads\An-Ra-cymek-beta)
**Hardware used:** CPU only, 2 threads, RTX 4050 (6 GB) untouched. System RAM free at start: ~4.9 GiB of 15.3.

## Result
- Verdict: **WIRING_VERIFIED_PENDING_TRAINING_EVIDENCE** (claim level: wiring-only)
- Control (inert projection): output bitwise-equal to baseline; scale exactly 1.0.
- Treatment (active projection, same seed): scale 1.01203, bounded in [0.8, 1.2],
  finite, max output difference 0.00183 vs control baseline.
- Tests: 9 passed (tests/test_hormonal_state.py). Import boundaries: PASS.
- Frozen spec + launch gate untouched: model_spec.py DFDCD883...,
  launch_readiness.json 95B2331A... unchanged.

## Honesty note
No 250M training was run. No capability evidence exists. This is a working,
tested, wired-in mechanism pending training-scale evidence, per the
IMPLEMENTED_PENDING_EVIDENCE convention.
