# ARK-012 ANALYSIS — recovery-switch threshold map

## Status

**EXECUTED SELECTED-EVENT SCREEN. Preregistered verdict: `TIME_NOT_STATE_SCREEN`.**

Two of the three frozen historical high-instability source combinations completed before the V6 allocation gate; the third was budget-blocked. Two executed sources are sufficient to avoid the preregistered low-event-rate classification, but this remains selected-event evidence and must not be counted as fresh instability incidence.

Mean OOD_SEALED area across the two executed sources:

| schedule | mean sealed area |
|---|---:|
| HIGH_CONTINUE | 0.915 |
| LOW_IMMEDIATE | 0.862 |
| SWITCH_75 | 0.884 |
| SWITCH_85 | **0.940** |
| SWITCH_90 | **0.940** |
| SWITCH_95 | 0.916 |

The thresholds do not produce a clean monotonic state ordering. On source 909/2702 every threshold from 0.75 through 0.95 triggered at the same 600-step confirmation because recovery was rapid relative to the 200-step evaluation cadence. On source 1010/2702, 0.75 switched at 600, 0.85 and 0.90 both at 3200, and 0.95 at 7000. Immediate LOW failed to recover on the difficult source, while the 0.85/0.90 schedules produced substantially better sealed area than immediate LOW and slightly better mean area than HIGH_CONTINUE.

## Interpretation

The screen does **not** establish a precise capability threshold. It does support a weaker operational boundary already suggested by ARK-010/011: switching LOW before meaningful recovery can trap a degraded state, while waiting until recovery is stronger can help. However, the tested thresholds alias heavily under the evaluation cadence, and the best pair (0.85/0.90) is a two-source selected screen rather than a replicated optimum.

Therefore the correct result is `TIME_NOT_STATE_SCREEN`, not "0.90 is optimal." A future threshold study would need denser recovery measurement and fresh sources if the exact switch rule matters.

## Demonstrated / not demonstrated

**SUPPORTED SCREEN:** immediate/very-early LOW can be worse than delayed switching after recovery on selected high-instability sources.

**NOT DEMONSTRATED:** a universal state threshold, an optimal 0.85/0.90 policy, fresh incidence, transfer, or scale generality.
