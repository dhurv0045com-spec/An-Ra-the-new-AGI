# NOVELTY REGISTER (Arkenstone)

| Claim | Prior search | Class | Evidence | Status |
|-------|--------------|-------|----------|--------|
| No branch has measured a lift-off dose (steps/exposures to first train-exact >= 0.9) on any task | citadel EXPERIMENTS_BRIEF + T1C tables record only end-state exact; T1D plan has no lift-off metric; cymek receipts contain no capability lift-off | NEW_MEASUREMENT (candidate) | ARK-001 | VERIFIED-FOR-THIS-REPO |
| Binding-v2 qualification survives trained escalation baseline | cymek test-asserted only, no receipt | REPRODUCTION + EXTENSION | BINDING-V2-REDTEAM | EXECUTED |
| Per-position digit accuracy decomposition of exact-match failure | absent from branch receipts | NEW_MEASUREMENT (candidate) | ARK-001 per-position curves | EXECUTED |
| Tens-selectivity precursor | reanalysis inverted direction and showed marker-not-precursor | FALSIFIED (as precursor) / REPRODUCTION (as marker) | ARK-004A-R | VERDICT B+C |
| Matched post-G90 low-LR retention protection on canonical Micro T2 | earlier Arkenstone work had only single/smaller evidence; LR scheduling broadly known | NEW_EMPIRICAL_DISCOVERY (program-local), intervention family ALREADY_KNOWN | ARK-007R: 3 fresh acquisitions × 4 paired orders; HIGH collapse 9/12, LOW 0/12 | REPLICATED_MICRO_TASK_EFFECT |
| Post-collapse high-LR reacquisition advantage over immediate low LR | no prior Arkenstone recovery tournament | NEW_EMPIRICAL_DISCOVERY (program-local) | ARK-010: sustained G90 recovery HIGH 8/9 vs LOW 2/9 | SUPPORTED_PATTERN |
| HIGH acquire/recover -> LOW retain controller | derived from ARK-007R + ARK-010; adaptive-LR family broadly known | NEW_EMPIRICAL_DISCOVERY (program-local), intervention family ALREADY_KNOWN | ARK-011: HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6 across 3 fresh acquisitions; risk diff -0.50 | DIRECTLY_SUPPORTED_MICRO_T2 |
| Perfect canonical held-out binding exact coexisting with poor order robustness | ARK-008 did not acquire; ARK-009/014 use fact-set-disjoint split | EXTENSION / NEW_EMPIRICAL_BOUNDARY | ARK-009 + ARK-014 canonical arm | EXECUTED |
| Exact recovery-state threshold for LOW switch | thresholds 0.75/0.85/0.90/0.95 screened on selected events | NEW_HYPOTHESIS / MECHANISTIC_EXTENSION | ARK-012 | NOT_DEMONSTRATED; threshold aliasing |
| LOW same-task retention protection extends to no-replay task shifts | no prior direct cross-task test in Arkenstone | EMPIRICAL_BOUNDARY | ARK-013: all arms lost sustained T2 while training T3 only | NEGATIVE_BOUNDARY_AT_MICRO |
| Deterministic fact-order augmentation repairs robust binding acquisition | order augmentation/invariance broadly known | ALREADY_KNOWN_FAMILY / EXTENSION | ARK-014: canonical ~0.33 order robustness vs augmented sealed ~0.987 and qualification at 1.8k | SUPPORTED_SCREEN_ONE_SEED |
| Non-arithmetic transfer of LR retention protection | ARK-009 blocked by robustness; ARK-014 qualified robust subject but generated no failures | EXTENSION | ARK-014 HIGH 0/3 failures, LOW 0/3 | NOT_DEMONSTRATED |
