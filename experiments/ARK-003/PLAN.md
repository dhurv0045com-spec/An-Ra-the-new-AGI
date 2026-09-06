# ARK-003 — TEACHER DECOMPOSITION AT THE GROKKING BOTTLENECK (preregistered)

## Objective
Does mixing digit-level decomposition teacher rows into the carry-tier stream
shift the OOD transition earlier or raise its plateau? ARK-001/002b located the
bottleneck: the binding-heavy tens column (with carry) is what groks late.

## Design (single variable: teacher rows in the training stream)
- CONTROL: T3-COMPACT, seed 13, 30-min box — REUSED from ARK-002b R3 (same
  harness lineage, same seed/budget; its receipt is the control).
- TEACHER: identical, except ~40% of training rows carry a decomposition
  suffix using the SAME vocabulary plus one new token ';' (added to BOTH
  arms' vocabularies; unused in control data):
    "34 + 62 = 96 ; 4+2=6 ; 3+6+0=9"   (ones rule ; tens rule incl. carry)
  Supervision covers answer + teacher suffix; eval is unchanged ordinary
  exact match on clean rows; the OOD tens band is structurally held out.
- All else frozen: seed 13, pool 500, 30-min box, same eval cadence.

## Predictions
- P1 (teacher helps): teacher arm's OOD onset earlier than control's and/or
  final OOD higher at box end; per-position tens accuracy rises first.
- P2 (teacher is just more rows): no material difference — the transition is
  governed by exposure dose to the core mapping, not decomposition.
- P3 (teacher hurts generalization): teacher arm memorizes decomposition
  formats without extracting the rule (OOD equal or worse) — format imitation.

## Falsification
If the two OOD curves are statistically indistinguishable through the box,
P2 stands and teacher rows are PARKED at micro scale.

## Novelty test
Citadel T1D arm C designed teacher rows for arithmetic but never executed;
testing them at the identified bottleneck (tens/carry) with band-OOD curves
is new for the program either way.
