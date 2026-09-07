# ARK-007R ANALYSIS — fresh-checkpoint replication of post-G90 LR protection

## Status

**EXECUTED. REPLICATED MICRO-TASK RETENTION EFFECT.**

This analysis is grounded in the imported V5 Colab receipt. Binding preregistration: `3d98103cacc38177390df78ee0eff402da687fcf`; pre-execution V5 addendum: `6809bfe8ca70661a4f8b2cd42679db594b944668`.

## Fresh acquisitions

| seed | G90 onset | G90 confirmation | supervised tokens |
|---:|---:|---:|---:|
| 909 | 16,200 | 16,600 | 4,249,600 |
| 1010 | 19,200 | 19,600 | 5,017,600 |
| 1111 | 21,400 | 21,800 | 5,580,800 |

Each checkpoint was forked over continuation seeds `2701..2704`; HIGH and LOW consumed the identical frozen continuation order within each pair.

## Primary result

- 12 matched pairs.
- HIGH `1e-3`: **9/12 collapse90 events (75%)**.
- LOW `1e-5`: **0/12 events (0%)**.
- Risk difference LOW−HIGH: **−0.75**.
- Discordant HIGH-collapse/LOW-stable pairs: **9**; reverse discordance: **0**.
- Mean RET90: HIGH `0.675`, LOW `1.000`.
- Mean OOD area: HIGH `0.917`, LOW `0.991`.
- Mean final OOD: HIGH `0.934`, LOW `0.992`.

Replication by independent acquisition checkpoint: HIGH collapse rates were `3/4`, `4/4`, `2/4` for seeds 909/1010/1111; LOW was `0/4` on all three.

## Interpretation

**DEMONSTRATED for the Micro T2 family:** after confirmed structural generalization, `LR=1e-5` strongly reduces sustained post-G90 instability relative to `LR=1e-3` under matched continuation data.

The 12 forks are nested within only 3 independent acquired models, so they are not 12 independent model replications.

The strongest parsimonious red-team explanation is near-freezing: mean final relative parameter displacement was about `0.379` at HIGH versus only `0.008` at LOW. This may be stabilization by very small updates rather than a special consolidation mechanism.

Not demonstrated: universal optimizer law, transfer beyond T2 arithmetic, V5-Core readiness, or a distinct consolidation computation.

**Claim level:** `REPLICATED_MICRO_TASK_RETENTION_EFFECT`.
