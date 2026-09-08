# ARK-011 ANALYSIS — state-conditional LR control after recovery

## Status

**EXECUTED. PREREGISTERED VERDICT: `SUPPORTED_ADAPTIVE_PROTECTION`.**

All three fresh acquisition seeds reached sustained OOD_CONTROL G90. Across the 12 frozen continuation opportunities, 6 produced a collapse followed by HIGH-LR recovery and therefore reached the decisive matched recovery fork. Four collapse states did not recover inside the preregistered box and two orders did not produce a collapse.

The six decisive forks were all OOD_SEALED-qualified at the recovery checkpoint and span all three independent acquisition seeds.

| primary sealed endpoint | HIGH_CONTINUE `1e-3` | SWITCH_LOW `1e-5` |
|---|---:|---:|
| recurrent instability | **3/6** | **0/6** |
| mean RET90 | 0.856 | **1.000** |
| mean exact/area | 0.958 | **0.997** |
| mean final exact | 0.971 | **0.997** |

Paired risk difference `LOW - HIGH = -0.50`. Discordance was 3 cases HIGH unstable / LOW stable and 0 reverse. By independent acquisition checkpoint: seed 1313 was 2/3 vs 0/3, seed 1414 was 1/2 vs 0/2, and seed 1515 was 0/1 vs 0/1.

This satisfies every preregistered support condition: >=4 sealed-qualified forks, >=2 independent acquisitions, >=2 HIGH recurrent-instability events, risk difference <= -0.30, and no majority checkpoint-level reversal.

## Scientific interpretation

ARK-007R showed that LOW LR protects an already-generalized T2 solution. ARK-010 showed that LOW LR is usually poor for reacquisition after an instability event. ARK-011 now closes the causal loop on the same Micro T2 family: **use HIGH LR while reacquiring, then switch LOW only after recovery is confirmed**. Under matched future minibatches this controller reduced recurrent instability relative to simply remaining HIGH.

This is stronger than the earlier correlation/pattern because the two arms fork from the exact same recovered checkpoint and differ only in LR. The primary endpoint is sealed from the controller.

## What is demonstrated

**DEMONSTRATED at Micro T2 scale:** after a prospectively observed instability and HIGH-LR reacquisition, switching from `1e-3` to `1e-5` at confirmed recovery reduces subsequent recurrent-instability risk under matched continuation data.

## What is not demonstrated

- non-arithmetic transfer;
- benefit on natural-language pretraining;
- compatibility with Cymek's WSD production schedule;
- a universal optimizer law;
- improved ability to learn a new skill while retaining an old one;
- AGI.

The six forks are clustered within three independently acquired checkpoints and must not be treated as six independent model replications.
