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

# HORM-002 experiment.md addendum

**Date:** 2026-09-17 (evening session)
**Branch:** cymek-beta (worktree C:\Users\ankit\Downloads\An-Ra-cymek-beta)
**Hardware used:** CPU only, 2 threads, no CUDA. Free RAM ~5.2 GiB of 15.3; CPU load 46 percent before run; 8-update A/B finished in <1.5 s total.

## Correction to HORM-001 record
HORM-001's smoke multiplied attention SUB-LAYER OUTPUT by the scale; it did
not scale attention query logits. ANALYSIS.md and experiment.md for HORM-001
are accurate as "wiring verified" but the integration point was the residual
stream, not attention temperature. HORM-002 implements and measures the
preregistered attention-scale mechanism (query logits scaled after
RoPE+QK-norm, bounded 0.8-1.2) via v5_identity/attention_patch.py.

## Measured result (RESULT_horm002_ab.json, sha256 24ce92ab...)
- Matched A/B through real ProductionTrainingBackend, seed 707001, 8 updates.
- Control (inert): final loss 6.250078; scale 1.0 each update.
- Treatment: final loss 6.250082; scales alternate 1.00497/1.00679.
- max |dLoss| = 1.38e-05; effect direction NOISE_DOMINATED at this scale.
- Integration canary: patched forward differs from baseline; restore()
  returns bitwise-equal baseline.

## Verdict
MECHANISM_PRESENT_AT_MINIATURE_SCALE; NOISE_DOMINATED at 8 updates.
No capability, quality, or production-scale claim is made.

## Honesty note
Measured, not replicated (single seed, single A/B pair). Implementation is
runtime-patched, not merged into v5_model; the frozen spec path is untouched
(model_spec.py DFDCD883..., launch_readiness.json 95B2331A... unchanged).
