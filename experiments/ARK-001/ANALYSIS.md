# ARK-001 ANALYSIS (post-execution)

## Results (all arms hit the preregistered 12-min wall box; doses recorded)

| Arm | Task | Vocab | Params | Lift-off step (train>=0.9) | Final train exact | Final OOD exact |
|-----|------|-------|--------|-----------------------------|--------------------|------------------|
| T1-COMPACT | single-digit add (100 combos) | 19 | ~0.8M | **200** | 1.00 | 1.00 (in-dist) |
| T2-COMPACT | two-digit no-carry, structural tens-band holdout | 19 | ~0.8M | 400 | 1.00 | **0.365** |
| T0-COMPACT | x+0, x*1 (memorization probe) | 19 | ~0.8M | 400 | 1.00 | 1.00 (in-dist) |
| T1-COMPACT-LARGE | single-digit, width 256 | 19 | ~2.5M | 200 | 1.00 | 1.00 |
| T1-BYTE | single-digit, frozen 24,576 byte-level vocab | 24576 | ~3.9M | **200** | 1.00 | 1.00 |

Seed 13 everywhere. T2 OOD trajectory (step -> test exact): 0.06@200 -> 0.00@400-800
(the memorization phase) -> 0.01@1000 (train lift-off) -> ~0.03-0.055 plateau ->
0.365 by the wall box. Per-position data in RESULT.json.

## Findings

1. **Lift-off exists and is fast at micro scale.** Flat answer-only CE lifts
   single-digit addition to 100% exact in 200 steps (~30s CPU, ~12,800 token
   exposures). H-FLOOR (optimization/capacity pathology as a universal explanation
   of citadel T1/T1C) is REFUTED at micro scale: the stack learns symbolic
   input->answer mappings readily when the task is small enough.
2. **The bottleneck is generalization, not optimization.** T2 memorizes 500
   two-digit pairs perfectly (lift-off 400) while structural-band OOD sits at
   ~0-6% through the memorization phase and only climbs late (0.365 by box end):
   the model fits a lookup table first; algorithmic extraction emerges late and
   gradually with sustained exposure (grokking-adjacent dynamics, measured here
   for the first time in this program).
3. **Vocabulary representation is not a first-order variable at micro scale.**
   The frozen 24,576 byte-level vocabulary lifted off at the SAME step (200) as
   the 19-symbol compact vocabulary. H-REPR refuted for this regime; T1D arm E's
   masked-vocab diagnostic is de-prioritized.
4. **Width 2x changed nothing** (lift-off 200 vs 200) at this scale.

## Decision (per the preregistered next-decision table)

"Arm 1 lifts off, arm 3 partially does (late OOD emergence)" -> **ARK-002**:
locate the memorize->generalize transition on the T1->T2 boundary as a function
of (digit count, pool size, exposure dose), and test whether teacher
decomposition (digit-level supervision) or curriculum shifts the OOD emergence
point. This directly parameterizes citadel T1D arms B/C before any TPU session:
their tier lift-off readout should add per-band OOD curves, because our evidence
says train-side lift-off and OOD lift-off are DIFFERENT events.

## Honesty notes

- Single seed (13); the lift-off effect size (200 vs never) is large, but
  replication is ARK-002's first obligation before any claim is promoted.
- T2 "OOD" is a structural tens-band holdout; ones-digit sub-combinations
  overlap with training. The measured 36.5% is partial algorithm extraction,
  not pure OOD transfer. Per-position decomposition (receipt) quantifies this.
- All arms hit the wall box: doses are capped, and every "no lift-off" style
  claim would be budget-relative. Here all claims are positive lift-off events,
  so the cap does not weaken the conclusions.
- The T1-BYTE harness asymmetry (PAD vs BOS prefix) was caught by the
  impossible loss-0/exact-0 signature and fixed before analysis; the original
  artifact is superseded and documented.
