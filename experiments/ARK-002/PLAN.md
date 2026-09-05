# ARK-002a — GROKKING SATURATION + LIFT-OFF REPLICATION (preregistered)

## Objective
1. Does T2-COMPACT's structural-band OOD exact saturate at 1.0 with extended
   budget, or plateau (partial algorithm extraction)?
2. Does the ARK-001 lift-off result replicate on an independent seed (29)?

## Design
- Arm R1: T2-COMPACT, seed 13, 30-min wall box (2.5x the ARK-001 box that cut
  the transition at 0.365 and climbing). Readout: full OOD curve + saturation.
- Arm R2: T1-COMPACT, seed 29, same 12-min box as ARK-001. Readout: lift-off
  step replication (predicted ~200).
- Everything else frozen from ARK-001 (same harness sha lineages recorded in
  results).

## Predictions
- P1: OOD saturates at 1.0 (full algorithm extraction) — if plateau < 0.9, the
  extracted "algorithm" is incomplete and teacher decomposition gains priority.
- P2: lift-off at 200 +- 200 steps.

## Falsification
- R2 failing to lift off within budget undermines the ARK-001 lift-off claim
  (seed sensitivity) and demotes Claim 1 to UNSTABLE.
- R1 OOD curve diverging wildly from ARK-001's (different shape, not a
  continuation) suggests the transition is seed-dominated noise.

## Novelty test
Saturation point + replication converts Claim 2 (grokking decomposition) from
NOVEL_CANDIDATE to REPLICATED for this program.
