# Cross-branch evidence audit — CYR-GPU-006

Audit date: 2026-09-09. Raw result receipts are authoritative over older prose summaries.

- Arkenstone live authority: `6acd9dcbdd28d00f387ffcd004253a813aca4b66`.
- Validated Discovery V6 bundle SHA256: `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`.
- BRAMASTRA live head observed: `415250f44179f3310e5dc55addb21722290604fa`. No new BRAMASTRA execution claim is admitted here merely from its newer documentation commit.
- Canonical Cymek remains the production authority; Arkenstone/BRAMASTRA are evidence mines, not merge targets.

## Current evidence matrix

| Finding | Raw authority | Status | What the result actually shows | Main limitation | CYR-GPU-006 consequence |
|---|---|---|---|---|---|
| LOW LR protects an already acquired T2 capability | ARK-007R historical replication + ARK-011/RESULT.json | REPLICATED / DEMONSTRATED AT MICRO T2 | ARK-011 produced 6 sealed-qualified recovery forks across 3 acquisitions; HIGH recurrent instability 3/6, SWITCH_LOW 0/6; risk difference `LOW-HIGH=-0.50`; LOW sealed RET90=1.0 | Micro arithmetic only; LOW may work partly by near-freezing | Keep LOW as a retention control, never call it a universal consolidation law |
| HIGH is useful for acquisition/recovery | ARK-010 + ARK-011 acquisition/recovery path | REPLICATED AT MICRO T2 | ARK-011 acquired all 3 fresh subjects at HIGH and only forks after HIGH recovery | Same task/scale; does not define a universal LR | Acquisition stays HIGH; LOW is never used from random initialization |
| Exact switch threshold is known | ARK-012/RESULT.json | REJECTED | Selected-event screen returned `TIME_NOT_STATE_SCREEN`; several thresholds collapsed to identical switch times and one source separated them | Only two executed sources; third budget-blocked | Preserve FIXED_TIME control and treat hysteresis thresholds as an experimental policy, not discovered constants |
| LOW solves cross-task forgetting | ARK-013/RESULT.json | REJECTED | Under 12k pure new-skill updates with no T2 replay, every arm lost sustained T2 retention; LOW only slowed interference | New skill itself failed to reach the intended acquisition gate | Transfer stage must test plasticity + old-skill retention jointly and include a fixed replay condition rather than assuming LR alone solves interference |
| Non-arithmetic binding failure was query blindness | ARK-014/RESULT.json | PARTLY REJECTED | Canonical-only subject reached canonical/query-only 1.0 but order-only/query+order ~0.33; deterministic order augmentation qualified at step 1800 with sealed order-only/query+order ~0.987 | One non-arithmetic family | Use order-augmented binding and keep canonical/query/order/query+order measurements orthogonal |
| LOW protects non-arithmetic binding | ARK-014/RESULT.json | INCONCLUSIVE | Three matched robust-binding pairs had HIGH failures 0 and LOW failures 0; both RET_QUALIFIED=1.0 | Zero-event screen cannot identify protection | Do not repeat the same retention screen; use binding as a plasticity/transfer subject instead |
| EOS supervision is mechanically required | BRAMASTRA terminal experiment, previously audited | DEMONSTRATED MECHANICAL CONTRACT | With EOS supervision complete stopping changed from 0/32 to 32/32 in the bounded lab | Small lab; not an AGI result | Keep content+EOS loss and candidate-free valid-stop evaluation in Cymek |
| Aggregate accuracy can hide query/order shortcuts | BRAMASTRA binding diversity + ARK-014 | REPLICATED DIAGNOSTIC LESSON | ~headline competence can coexist with zero pair/query robustness; ARK-014 isolates order sensitivity cleanly | Synthetic binding tasks | Require orthogonal counterfactual diagnostics; never promote from canonical accuracy alone |

## CYR-GPU-006 design decisions derived from this evidence

1. **Primary causal unit = one acquired parent.** Each seed acquires once at HIGH. Every retention arm restores identical model+optimizer bytes and consumes the same future batch hashes.
2. **Four retention policies remain informative:** HIGH, LOW, FIXED_TIME HIGH→LOW, HYSTERETIC HIGH↔LOW. FIXED_TIME is mandatory because ARK-012 weakens any claim that a particular state threshold is uniquely causal.
3. **A single parent cannot win.** The arithmetic verdict requires at least two independent contract-valid parents with paired margins and no material reversal.
4. **Transfer is prospectively fixed, not post-hoc.** The transfer candidate is `HYSTERETIC_HIGH_LOW`; comparator is equal-age/equal-exposure `LOW_CONTINUE` from the same parent. Both then move to HIGH LR on the same robust-binding stream.
5. **Transfer includes replay.** Primary transfer batches are fixed at 18 order-augmented binding rows + 2 old-T2 rows. Receipts report actual token fractions; the 10% figure is row-count only. This is motivated by ARK-013's demonstrated no-replay interference boundary, not by a claim that 10% is optimal.
6. **Binding subject is order-augmented.** Canonical/query-only/order-only/query+order remain separate. This incorporates ARK-014 rather than rerunning its known brittle canonical-only setup.
7. **Claim ceiling stays low.** A positive CYR-GPU-006 result is GPU V5-proxy development evidence. It cannot authorize a production scheduler change, PRE500M, or the 500M campaign without exact-SHA review, Citadel audit, and real TPU confirmation.

## Evidence statuses that remain open

- Universal or scale-invariant LR law: **NOT DEMONSTRATED**.
- Exact hysteresis thresholds: **NOT IDENTIFIED**.
- Plasticity advantage of adaptive retention over LOW: **UNRESOLVED; CYR-GPU-006 targets this directly**.
- Natural-language pretraining transfer: **UNTESTED**.
- Production WSD compatibility: **UNTESTED**.
- TPU behavior: **NO EVIDENCE**.
