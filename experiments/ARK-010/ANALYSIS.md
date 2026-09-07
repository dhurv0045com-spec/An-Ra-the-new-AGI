# ARK-010 ANALYSIS — recovery after post-G90 instability

## Status

**EXECUTED. ORIGINAL LOW-LR RECOVERY HYPOTHESIS NOT SUPPORTED.**

Nine prospectively captured HIGH-LR collapse-confirmation states from ARK-007R were forked onto the same unused future continuation tail.

| metric | HIGH continue `1e-3` | LOW recovery `1e-5` |
|---|---:|---:|
| sustained G90 recovery | **8/9** | **2/9** |
| mean RET90 | **0.611** | 0.111 |
| mean OOD area | **0.909** | 0.788 |
| mean final OOD | **0.949** | 0.812 |

By acquisition: seed 909 HIGH 3/3 vs LOW 0/3; seed 1010 HIGH 3/4 vs LOW 1/4; seed 1111 HIGH 2/2 vs LOW 1/2.

This falsifies the simple recovery rule “collapse -> immediately lower LR.” Instead, HIGH LR usually reacquired G90, while LOW LR often left the model in the degraded state.

Combined with ARK-007R, the evidence is state-dependent: LOW protects an already-generalized solution; HIGH is usually better for reacquisition after an instability event.

This motivates, but does not demonstrate, an adaptive policy: **HIGH for acquisition/recovery -> LOW after confirmed capability for retention**. The decisive HIGH-until-recovered-then-LOW arm has not yet been run.

Because 8/9 HIGH arms recovered, `collapse90` should be interpreted as a post-G90 instability episode, not irreversible forgetting.
