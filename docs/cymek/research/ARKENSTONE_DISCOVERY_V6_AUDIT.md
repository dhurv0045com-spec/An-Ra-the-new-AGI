# Arkenstone Discovery V6 audit for Cymek

Date: 2026-09-09

Read-only source branch: `Arkenstone`
Audited source HEAD: `6acd9dcbdd28d00f387ffcd004253a813aca4b66`
Discovery V6 source bundle SHA256: `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`

This audit is derived from the live Arkenstone branch, especially `experiments/ARK-011..014/{RESULT.json,ANALYSIS.md,PLAN.md}` and `experiments/COLAB/results/v6/DISCOVERY_V6_VALIDATED.md`. Raw/imported receipts take precedence over stale summary prose.

## Evidence matrix

| Finding | Status | Evidence | Cymek consequence |
|---|---|---|---|
| Post-recovery HIGH→LOW protection on T2 | DEMONSTRATED_MICRO | ARK-011: 6 sealed-qualified forks across 3 acquisitions; HIGH recurrent instability 3/6 vs SWITCH_LOW 0/6, risk difference -0.50 | Keep state-dependent LR as a serious challenger, but do not promote it from Micro T2 directly. |
| Exact capability threshold for switching | NOT_DEMONSTRATED | ARK-012: 0.85/0.90 looked best on two selected sources, but threshold times aliased and ordering was non-monotonic | Cymek must test policy/state dependence, not claim that 0.90 is an optimal universal threshold. Fixed-time control remains mandatory. |
| LOW solves cross-task interference | REFUTED_AS_GENERAL_RULE | ARK-013: under 12k T3-only updates with no T2 replay, all arms lost sustained T2; LOW only slowed loss | Same-task retention evidence cannot be treated as continual-learning evidence. Transfer must measure old-skill retention while a genuinely new skill is learned. |
| Plasticity cost of LOW/adaptive policy | UNRESOLVED | ARK-013 primary test was inconclusive because FIXED_HIGH never acquired sustained T3 G90 and ADAPTIVE never switched | Cymek transfer stage must use a reliably learnable new task and compare acquisition speed/quality from baseline vs retained state. |
| ARK-009 was query-blindness | REJECTED_AS_SPECIFIC_DIAGNOSIS | ARK-014 isolated the failure: canonical/query-only were perfect while order-only/query+order were brittle | Non-arithmetic transfer must use orthogonal query/order metrics; a canonical-only heldout score is insufficient. |
| Order augmentation repairs robust binding acquisition | DEMONSTRATED_SCREEN | ARK-014: canonical training did not qualify; order-augmented training qualified at 1.8k and sealed order-only/query+order were ~0.987 | Use deterministic order augmentation for the new-skill transfer subject so the transfer stage is not blocked by a known presentation-order failure. |
| Non-arithmetic LR-retention transfer | UNRESOLVED | ARK-014: HIGH 0/3 failures and LOW 0/3 failures | Do not spend Cymek GPU budget merely repeating a zero-event HIGH-vs-LOW continuation. Use non-arithmetic binding primarily as a plasticity/transfer test, with old-skill retention measured concurrently. |
| LOW mechanism is consolidation rather than near-freezing | UNRESOLVED | ARK-007R had very small LOW parameter displacement relative to HIGH; later V6 does not separate LR from update magnitude | Cymek must report parameter displacement/path and optimizer/gradient diagnostics and keep the near-freezing alternative alive. |

## Important documentation inconsistency

`docs/arkenstone/COGNITION_BOTTLENECK_GRAPH.md` is stale relative to the live V6 evidence: it still describes ARK-011 as an unexecuted prediction and ranks its execution as the next test. The live experiment log and validated ARK-011 result show that edge is already executed and supported at Micro T2. Cymek therefore must not use that bottleneck graph as the current authority.

## Revised Cymek experimental priority

CYR-GPU-006 should answer three separable questions in one bounded Colab session:

1. **Scale-transfer / policy:** on the real Cymek V5 implementation, do HIGH, LOW, fixed-time decay and hysteretic HIGH↔LOW differ in same-task retention across independent G90 parents under identical continuation data?
2. **Mechanism red-team:** if LOW/adaptive protects, is the effect accompanied by near-freezing-scale parameter movement, or does a state/timing policy outperform controls at comparable useful movement?
3. **Plasticity + non-arithmetic transfer:** a preregistered adaptive candidate must learn an order-robust non-arithmetic binding task from at least two independent parent states while old T2 retention is measured. The transfer task must use deterministic order augmentation and report canonical, query-only, order-only and query+order metrics separately.

The transfer candidate must be chosen prospectively from prior evidence (`HYSTERETIC_HIGH_LOW`), not selected after observing CYR-GPU-006 arithmetic outcomes. Arithmetic outcome selection and transfer measurement therefore remain statistically distinct.

## Claim boundary

Even a positive CYR-GPU-006 result is only a GPU proxy result on the Cymek V5 implementation. It cannot certify TPU behavior, cannot silently modify the production WSD schedule, and cannot authorize the 500M run. A production scheduler proposal still requires later TPU semantic confirmation and an explicit promotion decision.
