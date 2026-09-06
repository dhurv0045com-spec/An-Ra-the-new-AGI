# ARK-002B ANALYSIS (seeds 29 and 47, frozen commutation-free manifest)

| Run | init/order seed | lift-off | M99 | G50 | G90 (sustained) | final OOD (box end) |
|-----|------------------|----------|-----|-----|------------------|----------------------|
| historical (002a, seed 13) | 13 | 400 | 1000 | ~6000* | ~9000-10000* | 1.00 (*pre-erratum split; comparable shape) |
| 002B-1 | 29 | 400 | 1000 | 15000 | NOT_DEMONSTRATED in box | 0.924 (climbing) |
| 002B-2 | 47 | 400 | 1000 | 11000 | **12000 (DEMONSTRATED)** | 1.00 |

## Findings
1. **Qualitative replication: YES.** Both fresh seeds show memorize-first
   (M99 at step 1000) -> long OOD delay -> late climbing transition reaching
   0.92-1.00 by box end. The phenomenon is not a seed-13 artifact.
2. **Quantitative dose is seed-variable**: sustained G90 at ~12k steps (s47),
   not reached by 18k (s29, still climbing). Across seeds 13/47/29 the
   transition dose spans roughly 9k-18k steps (~2x spread). Single-seed dose
   claims must carry this variance band.
3. **The commutation-free manifest did not remove the phenomenon** — the
   ARK-001/002a shape survives the stricter holdout (the 002a absolute OOD
   numbers were mildly optimistic, as the erratum predicted, but the
   transition is real).
4. Sustained-metric discipline worked as designed: G90 for seed 29 is honestly
   NOT_DEMONSTRATED despite a 0.924 final snapshot — max-snapshot claims would
   have overstated it.

## Verdict per mission section 4
The qualitative memorize->generalize transition is REPLICATED FOR THIS TASK
FAMILY (3 seeds total). Next: ARK-003 causal acceleration.
