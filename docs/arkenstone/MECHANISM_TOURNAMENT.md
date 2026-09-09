# MECHANISM TOURNAMENT (Arkenstone)

Verdicts: UNTESTED / TESTING / FAILED / TENTATIVE / SUPPORTED / REPLICATED /
NOVEL_CANDIDATE / PROMOTED_TO_CORE / REJECTED / SUPERSEDED / ALREADY_KNOWN /
REPRODUCTION_ONLY / EXTENSION_ONLY. Failed mechanisms stay here forever.

| ID | Mechanism | Novelty | Prior status | Hypothesis (observed failure -> why it should help) | Verdict |
|----|-----------|---------|--------------|------------------------------------------------------|---------|
| M-001 | Lift-off dose mapping (measurement instrument) | NEW_MEASUREMENT | No branch measured lift-off dose | T1/T1C train-exact ~0 everywhere -> first establish WHERE lift-off exists at all | REPLICATED for T1 lift-off; T2 memorize→generalize trajectory replicated; G90 dose seed-variable |
| M-002 | Curriculum (easy→hard tiers) | ALREADY_KNOWN | ARK-003 executed | staged exposure | FAILED_AT_MICRO |
| M-003 | Micro-teacher rows | ALREADY_KNOWN | ARK-003 executed | teacher decomposes binding-heavy position | NO_EFFECT_IN_BUDGET; compute confound retained |
| M-004 | Vocab reduction for symbolic tasks | EXTENSION | ARK-001 | dead-vocab embedding dilutes capacity | REJECTED_AT_MICRO_SCALE |
| M-005 | Query-swap auxiliary objective | ALREADY_KNOWN | designed elsewhere | binding failures -> query-conditioning pressure | UNTESTED |
| M-006 | Recurrent/universal blocks, adaptive depth, memory tokens | ALREADY_KNOWN families | no direct Arkenstone test | architecture may change computation | PARKED pending stronger causal need |
| M-007 | Interference-pair binding generator | REPRODUCTION_ONLY | cymek | stronger binding data | REPLICATED by Arkenstone red team |
| M-008 | Column-selectivity precursor | NEW_PROGRAM_MEASUREMENT | ARK-004A | factorization precedes OOD | NOT_SUPPORTED as precursor; transition marker only |
| M-009 | Post-G90 low-LR protection | NEW_EMPIRICAL_DISCOVERY program-local; LR scheduling known | ARK-006/007 | reduce instability after acquisition | **REPLICATED at Micro T2** by ARK-007R: HIGH 9/12 vs LOW 0/12 |
| M-010 | State-dependent LR recovery/retention controller | NEW_EMPIRICAL_DISCOVERY program-local; adaptive LR known | ARK-010 | HIGH for movement/recovery, LOW after recovery | **SUPPORTED at Micro T2** by ARK-011: HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6 |
| M-011 | Non-arithmetic transfer of retention protection | EXTENSION | ARK-014 had zero-event retention screen | if effect is broader than arithmetic it should protect a robust non-arithmetic capability under informative stress | **SUPPORTED under controlled Micro distribution narrowing by ARK-015:** 3 fresh parents, 8 matched pairs, NARROW_HIGH 8/8 failures vs NARROW_LOW 0/8; risk diff -1.0 |
| M-012 | Recovery-state switch threshold | NEW_HYPOTHESIS / MECHANISTIC_EXTENSION | ARK-010/011 | LOW switch may depend on state threshold | SCREENED; exact threshold NOT_SUPPORTED by ARK-012 aliasing/non-monotonicity |
| M-013 | Adaptive stability–plasticity controller | NEW_HYPOTHESIS | ARK-013 | preserve old while acquiring new | INCONCLUSIVE: new T3 skill never qualified; no controller claim |
| M-014 | Invariance retention under presentation narrowing | NEW_TRANSFER_TEST / program-local empirical discovery | ARK-014 robust subject | removing presentation diversity should pressure learned order invariance | **SUPPORTED by ARK-015.** HIGH narrowing failed 8/8 while LOW and full-augmentation HIGH failed 0/8. Canonical exact stayed 1.0 while invariant performance eroded. |
| M-015 | State-conditional applied-update trust region | NEW_HYPOTHESIS; trust-region/update-clipping families known | ARK-016 | if LOW works through movement, movement-matched HIGH should protect | **INCONCLUSIVE_LOW_EVENT_RATE.** Only 1/12 T2 event opportunities produced a qualifying fork and all four arms were stable. No mechanism credit. |
| M-016 | Continued invariant-supporting replay | ALREADY_KNOWN replay/augmentation family; new causal role in this program | ARK-015 augmented-HIGH remained stable despite very large path | a small amount of broad-support data may prevent specialization-driven capability narrowing without freezing movement | **TESTING / PREREGISTERED ARK-017.** 1/16 replay independently manipulated from update cap. |
| M-017 | Plasticity × data-support interaction | NEW_HYPOTHESIS / MECHANISTIC_EXTENSION | ARK-015 shows LOW and full augmentation both protect; large movement alone insufficient | failure may require both high plasticity and insufficient current support for the invariant | **TESTING / PREREGISTERED ARK-017** with 2×2 cap/replay factorial plus references. |
| M-018 | Real-data-substrate retention transfer | EXTENSION | all prior causal work uses controlled Micro-from-scratch subjects | same phenomenon should survive on a model first pretrained from real text if it reflects general training dynamics | **PREREGISTERED ARK-018; UNTESTED.** User ~1GB Drive corpus, ~20–25M proxy, held-out real-text NLL + controlled invariant capability. |
| M-019 | Capability Guardian closed-loop controller | SYSTEM_EXTENSION / NEW_HYPOTHESIS; adaptive control/replay families known | no integrated old-skill/new-skill controller result | activate the prospectively chosen protection only when CONTROL probes detect old-capability erosion, preserving plasticity when safe | **PREREGISTERED ARK-019; BLOCKED on ARK-017/018 evidence.** |

## Current tournament leader

The strongest current mechanistic statement is intentionally weaker than a universal optimizer law:

> **Capability narrowing appears to depend on an interaction between how plastic the model remains and whether current training continues to support the broader capability.**

ARK-017 is the causal-credit test. ARK-018 is the real-data substrate test. ARK-019 is the algorithm/infrastructure promotion gate.
