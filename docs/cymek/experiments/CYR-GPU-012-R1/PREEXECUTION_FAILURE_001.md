# CYR-GPU-012 / R1 — PREEXECUTION FAILURE 001

**Date:** 2026-09-10  
**Stage:** frozen executable verified; unit-test gate failed before CUDA calibration and before any scientific model update.  
**Scientific result:** **NONE / NOT EXECUTED**.

## Symptom

The operator Colab verified frozen executable `d6953af2b0af64439dd9fd9ac0b2bc4987ed9c97`, then the command

```text
python -m pytest tests/test_v5_cyr_gpu012_r1.py tests/test_v5_cyr_gpu011.py tests/test_v5_cyr_gpu011_entry.py -q
```

returned non-zero and the notebook correctly stopped.

## Root cause

Two **synthetic runtime-resolver unit-test fixtures** were numerically inconsistent with the resolver they were testing. The resolver intentionally inflates projected runtime by `1.35x`, adds evaluation/finalization overhead, and protects the 170-minute science window inside the 175-minute campaign wall. The old synthetic rates labeled as “two pairs fit” and “one pair fits” did not actually satisfy those inequalities, so the assertions were wrong while the fail-closed resolver behavior was correct.

This was a test-fixture defect, not a scientific-model failure.

## Repair

Only synthetic calibration rates in `tests/test_v5_cyr_gpu012_r1.py` were changed:

- two-seed fixture: V19/V24576 rates now make two complete matched pairs conservatively fit;
- one-seed fixture: rates now make one complete pair fit while two do not;
- fail-closed fixture remains intentionally too slow.

No scientific code, dataset, model geometry, intervention, seeds, optimizer, objective, batch, 8,000-update endpoint, 512,000-row endpoint, primary metric, thresholds, or claim ceiling changed.

## New frozen executable

`1fa417260416e31097c8166770c69d2d7d8af168`

New test blob:

`584dec5afb8d67b5ee75759af1668c3d17c102d4`

The preregistration and readiness receipts were refrozen to this executable before any scientific execution. The operator should rerun Cell 0 from the canonical R1 notebook; it will fetch the updated branch and verify the new executable before proceeding.
