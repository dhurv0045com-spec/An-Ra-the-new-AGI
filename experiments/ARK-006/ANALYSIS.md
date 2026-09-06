# ARK-006 ANALYSIS — LR dose-response on the decaying seed (606)

## Result: the transition has an LR threshold

| LR multiplier | Actual LR | RET90 | Collapse | Final OOD |
|---------------|-----------|-------|----------|-----------|
| × 0.001 | 1e-6 | 1.0 | none | 0.985 |
| × 0.01 | 1e-5 | 1.0 | none | 1.0 |
| × 0.1 | 1e-4 | 0.191 | 21200 | 0.822 |
| × 1.0 (control) | 1e-3 | 0.143 | 19200 | 0.822 |

The threshold sits between 1e-5 and 1e-4. At or below 1e-5, the generalized
solution is fully stable. At 1e-4, it collapses — nearly as badly as at the
full 1e-3 learning rate.

## Interpretation
The post-generalization decay is **caused by continued high-LR optimization**,
not by weight decay (ARK-005 C null), not by lack of consolidation machinery
(ARK-005 D null), and not by an inherent metastability of the generalized
state (the state is perfectly stable at low LR). The learning rate that is
correct for *learning new things* is destructive for *holding learned things*.

This is the micro-scale retention law for An-Ra:
**retain = drop LR below ~1e-5 at the generalization transition.**

## Practical implication
A training schedule should detect the generalization transition (via the
column-selectivity precursor or sustained OOD thresholds) and drop LR by
>= 100x. This is directly actionable for Citadel T1D and any production run.

## What this does NOT establish
- Whether the same threshold holds at P35/V5-A scale (different optimizer
  dynamics, different parameter counts).
- Whether the collapsed state can be recovered by dropping LR after collapse
  (only tested pre-emptive drop).
- Whether this interacts with weight decay at lower LR (C had wd=0 AND
  full lr; the confound was not separated).
