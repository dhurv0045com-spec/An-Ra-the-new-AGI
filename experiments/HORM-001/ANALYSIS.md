# HORM-001 ANALYSIS

**Status: IMPLEMENTED_PENDING_EVIDENCE (wiring verified; no training-scale evidence)**

## What was actually run
- Tiny CPU smoke (2 threads, seed 707): frozen counterfactual pair.
  Control: hormonal projection inert (`raw_alpha=0`) -> scale exactly 1.0,
  output bitwise-equal to baseline (`max_abs_difference = 0.0`).
  Treatment: same seed/data, active projection -> scale 1.01203, bounded,
  finite, `max_abs_difference = 0.00183`.
- 9 unit tests pass (`tests/test_hormonal_state.py`).
- Import boundaries: PASS. Frozen `V5A_250M` spec file hash unchanged
  (DFDCD883...); `launch_readiness.json` unchanged (95B2331A...).

## What was NOT run
- No 250M training. No G90/capability probes. No held-out eval.

## Honest verdict
`WIRING_VERIFIED_PENDING_TRAINING_EVIDENCE`, claim level `wiring-only`.
Per the honesty convention, no capability or quality claim is made.
A negative/positive training-scale result remains open future work.

# HORM-002 ANALYSIS

**Status: MEASURED_AT_MINIATURE_SCALE (mechanism-presence evidence; no capability claim)**

## What was actually run (2026-09-17)
- Preregistered PLAN amendment for HORM-002 written before the A/B run.
- Integration canary probe (temp, CPU): patched model forward differs from
  baseline (max abs 2.7e-4 at raw_alpha=1.0); patch.restore() returns output
  bitwise-equal to baseline.
- Matched A/B: real ProductionTrainingBackend, tiny spec
  (e4dc015a...), seed 707001, 8 updates x 256 real tokens, CPU 2 threads.
  - Control (raw_alpha=0, scale 1.0): losses [6.2417, 6.2386, 6.2501, 6.2413,
    6.2444, 6.2448, 6.2342, 6.2501]; final 6.250078.
  - Treatment (raw_alpha=0.35, scales 1.00497/1.00679 alternating): final
    6.250082; max |dLoss| = 1.38e-05; all finite.
- Receipt: experiments/HORM-001/RESULT_horm002_receipt.json (source-bound).

## Interpretation (inference, clearly labeled)
- The bounded modulation measurably enters the loss trajectory (max diff
  1.38e-05 > float32 epsilon ~1e-7), consistent with H1.
- Direction of effect at this scale: treatment final loss is 3.8e-06 HIGHER
  than control. This is NOT evidence of benefit; at 1e-6 relative scale with
  8 updates it is noise-dominated. No quality claim is made.

## What this does NOT establish
- Any capability, G90, production-scale, or long-horizon training effect.
- Any claim that hormonal modulation improves or harms training.

## Verdict
MECHANISM_PRESENT_AT_MINIATURE_SCALE; effect direction/sign at this scale is
NOISE_DOMINATED. Training-scale evaluation remains open future work.
