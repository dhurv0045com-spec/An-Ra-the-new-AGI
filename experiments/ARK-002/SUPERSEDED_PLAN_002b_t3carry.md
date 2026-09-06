# SUPERSEDED — NOT EXECUTED

This T3-carry plan was written before the ARK-002B/ARK-003 mission reprioritized replication and causal acceleration. Retained as a candidate follow-up (S4 boundary diagnostic). Never executed.

# ARK-002b — TIER DOSE-RATIO SWEEP: the carry boundary (preregistered)

## Objective
Measure the memorize->generalize dose ratio at the next tier: two-digit
ADDITION WITH CARRY (citadel tier-3 semantics: compositional). ARK-002a
established the ratio ~10-20x for no-carry (T2). The question: does the ratio
explode at the carry boundary (predicting citadel T1D budgets cannot reach OOD
lift-off at T3+), or stay bounded?

## Arm
- T3-COMPACT: two-digit add, carry allowed (ones pairs unconstrained), same
  structural tens-band holdout (train tens 1..5, test tens 6..7), pool 500,
  seed 13, 30-min wall box, otherwise frozen from the ARK-001 harness.
- If OOD lift-off occurs within the box, replicate on seed 29 (second run).

## Metrics (frozen)
train exact, OOD exact curve, memorization dose (train lift-off), OOD onset +
saturation dose, per-position accuracy (ones vs tens vs carry-dependent),
majority/marginal baselines.

## Predictions
- P1 (ratio explodes): OOD stays near 0 through the box while train lifts off
  early -> the carry/binding step multiplies the generalization dose; T1D's
  45-min arms would need redesign (teacher/curriculum first).
- P2 (ratio bounded): OOD transitions within the box like T2 -> tier dose
  ratios are quantitatively trackable; T1D viable as designed.

## Falsification
An OOD curve indistinguishable from T2's (transition < ~10k steps) refutes P1.

## Novelty test
A measured carry-boundary dose ratio is new for the program either way; a
P1 outcome is the more decision-relevant negative for T1D.
