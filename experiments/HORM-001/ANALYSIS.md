# HORM-001 ANALYSIS

## Evidence correction (supersedes the historical conclusions below)

HORM-001 scaled attention outputs, not queries. Its smoke does not verify
the attention-temperature hypothesis. HORM-002's original run and three-seed
replication scaled queries using an unchanged baseline `patch.state`.
The alternating scales in their logs were calculated from a separate state
that never reached the model. Keep those JSON artifacts as historical records;
they do not demonstrate dynamic hormonal conditioning.

The HORM-002 plan was appended after the diagnostic result existed in temp.
The replication's sign-consistency readout was not preregistered in PLAN.md.
Claims of prospective registration below are withdrawn. Three inconsistent
signs and comparison with float32 epsilon establish neither statistical noise
nor a clean negative finding about training benefit.

A focused repair now updates the state actually captured by attention before
the backend step and logs that same scale. The schedule deliberately resets
state each update and alternates synthetic success/failure fixtures; these are
not outcomes from a live verifier. Persistent session-state checkpoint/resume,
a learnable projection, and a launchable hormonal sibling remain unsupported.

Validation: `python -m pytest tests/test_hormonal_state.py
 tests/test_hormonal_integration.py -q -p no:cacheprovider` passed 12 tests in
7.45 seconds with CPU threads capped at two. The new regression executes eight
tiny treatment updates and verifies that every layer reads exactly the logged
scale. This is a correctness check, not a new scientific A/B result.

Historical text follows unchanged for auditability; the corrections above
supersede its stronger claims.


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

# HORM-002 3-seed replication addendum (2026-09-17 evening)

## What was run
`run_horm002_ab.py --replication`: seeds 707001/707002/707003, identical
protocol to the primary A/B (real backend, CPU 2 threads, no CUDA).
Receipt: RESULT_horm002_replication.json (sha256 c280c860...).

## Primary readout (preregistered): sign consistency of mean loss difference
- Seed 707001: -1.61e-06 (treatment LOWER)
- Seed 707002: +2.98e-07 (treatment HIGHER)
- Seed 707003: -7.15e-07 (treatment LOWER)
- consistent_sign_across_seeds: FALSE -> INCONSISTENT_OR_NOISE_DOMINATED

## Interpretation
At 8 updates / 2048 tokens, the hormonal query-scale modulation produces only
~1e-06-scale loss-trajectory perturbations whose sign flips with the seed.
H1's mechanism-presence component is replicated (all pairs show nonzero,
finite differences; all losses finite). H1's implicit direction question is
answered: NO DIRECTIONAL EFFECT is resolvable at this scale. This is a clean,
complete negative at miniature scale per the repo's negative-results
convention.

## Verdict
MECHANISM_PRESENT (replicated across 3 seeds); DIRECTIONAL_EFFECT:
NONE_RESOLVABLE_AT_MINIATURE_SCALE. Any benefit/harm question requires
training-scale compute and is out of scope without authorization.

# HORM-003 ANALYSIS (2026-09-18)

**Status: PROSPECTIVE RUN EXECUTED; verdict NOT_SUPPORTED (per preregistered rules)**

## What was run
- PLAN.md HORM-003 section written before execution (00:05:10+05:30);
  runner mode added and guard-tested (overwrite refused; --force archives
  .previous) before the real run.
- 5 fresh seeds (707011-707015), corrected state binding (regression test
  enforces logged scale == scale read by every attention layer), CPU only,
  2 threads, no CUDA, real ProductionTrainingBackend path.
- Result: RESULT_horm003_prospective.json (sha256 e196a6af...), provenance-
  bound (runner + v5_identity sources + torch/python/platform).

## Primary readout (preregistered): sign consistency >= 4/5
- 707011: +1.91e-06 (treatment higher)
- 707012: -7.15e-07 (treatment lower)
- 707013: -2.21e-06 (treatment lower)
- 707014: +4.89e-06 (treatment higher)
- 707015: +3.58e-07 (treatment higher)
- 3/5 positive, median +3.58e-07 -> NOT_SUPPORTED under H1 (needed >=4/5).

## Interpretation (inference, labeled)
With the state-binding bug fixed, the dynamic hormonal modulation still
produces only ~1e-06-scale loss-trajectory perturbations at 8 updates /
2048 tokens, with mixed signs. Under the preregistered decision rule, the
directional hypothesis is NOT SUPPORTED at this scale. This is a real,
prospective negative result under the repo's negative-results convention -
it is a complete answer to the question asked, not a suppressed finding.

## What would change the verdict
- Longer horizons / more tokens per arm where per-update perturbations can
  compound (requires compute authorization).
- A live verifier signal driving appraisal (synthetic fixtures here).

## What this does NOT establish
- Any claim that hormonal modulation helps or harms training at any scale.
- Nothing about capability, G90, or production behavior.
