# ARK-002a ANALYSIS

## R1 — T2 saturation (seed 13, 30-min box, 10,176 steps)
OOD (structural tens-band) exact curve: ~0.00 through step 4200 (memorization
phase; train exact 1.0 from step 800) -> noisy transition (0.24 / 0.42 / 0.52 /
0.43 / 0.23 / 0.56 / 0.73 between steps 4800-8400) -> **0.98-1.00 saturated by
step ~9000**. Final per-position: tens 1.0, ones 1.0 — full two-digit no-carry
algorithm extracted. Prediction P1 (saturation at 1.0) CONFIRMED.

## R2 — lift-off replication (seed 29)
Lift-off step **200** (identical to seed 13), final train exact 1.0.
Prediction P2 CONFIRMED.

## Interpretation
- The memorize->generalize dose ratio is ~10-20x on this task family.
- The transition is noisy/chaotic, not monotone — checkpoint choice near the
  transition is high-variance; evaluation at a single late checkpoint can
  under- or over-state capability. Program implication: developmental
  trajectories need dense checkpoint sampling through the transition zone.
- The ones-column rule (55 combos, local) extracts early; the tens-column rule
  (requires binding both operands' tens digits) is what groks — mechanism-level
  support for digit-decomposition teachers (citadel T1D arm C) targeting the
  binding-heavy positions.

## Verdicts
- ARK-001 Claim 1 (lift-off dose): REPLICATED (seeds 13 and 29 agree exactly).
- ARK-001 Claim 2 (memorize-then-grok decomposition): REPLICATED and EXTENDED
  (saturation at 1.0 established; noisy-transition character documented).
- M-001 verdict: REPLICATED as a measurement instrument.
