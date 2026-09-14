# UNIFIED EVIDENCE MAP (Arkenstone)

Rule: execution artifacts beat prose. Every entry carries evidence strength and scope. This map is reconciled through the validated MASTER GPU V5 campaign (ARK-007R/009/010).

## DEMONSTRATED / REPLICATED

| Claim | Evidence | Scope / caveat |
|---|---|---|
| V5 training substrate constructs/trains on CPU/CUDA and has bounded V5-A canaries | cymek receipts | engineering substrate only; not a scientific full V5 training claim |
| T1/simple symbolic lift-off exists at micro scale | ARK-001 + replication | hundreds of steps; not evidence of broad cognition |
| T2 memorize -> delayed structural-OOD transition exists | ARK-002a/002B/004A | replicated qualitative transition; timing seed-variable |
| Memorization timing does not explain G90 timing | ARK-004A | M99 vs G90 timing decoupled in observed seeds |
| Column-selectivity is not a validated precursor | ARK-004A-R | original direction was inverted; retained only as transition marker |
| Post-G90 instability is real | ARK-004A/005/007/007R | instability event can later recover; not automatically permanent forgetting |
| LOW LR protects an already-generalized T2 state under matched future minibatches | ARK-007R | 3 fresh acquisitions × 4 paired continuations; HIGH collapse90 9/12, LOW 0/12, risk diff -0.75; Micro T2 only |
| After a defined instability event, continued HIGH LR usually reacquires G90 better than immediate LOW LR | ARK-010 | HIGH recovery 8/9 vs LOW 2/9 across 9 sources nested in 3 acquisitions; state-dependent pattern, not universal law |
| ARK-009 ordinary fact-set-disjoint canonical held-out exact can reach 1.0 | ARK-009 seeds 1201/1202 | strict robustness qualification still failed; does not establish robust variable binding |

## FAILED / NEGATIVE

| Attempt / claim | Evidence | What it rules out |
|---|---|---|
| Curriculum accelerates T2 transition | ARK-003 B | rejected at tested micro regime |
| Equal-wall aligned teacher accelerates T2 | ARK-003 C/D | no acceleration in budget; step confound prevents stronger general claim |
| Weight-decay removal prevents retention loss | ARK-005 | not supported |
| EMA consolidation prevents retention loss | ARK-005 | not supported |
| LR `1e-4` is sufficient protection | ARK-006 | not sufficient on tested decaying trajectory |
| Higher early tens-selectivity predicts earlier G90 | ARK-004A-R | falsified as stated |
| Immediate LOW LR is best after collapse | ARK-010 | falsified on Micro T2: HIGH recovery 8/9 vs LOW 2/9 |

## TENTATIVE / OPEN

| Claim / hypothesis | Evidence | Status |
|---|---|---|
| LOW LR protects by a specific consolidation mechanism rather than near-freezing | ARK-007R displacement: LOW final relative movement ~0.008 vs HIGH ~0.379 | OPEN; near-freezing is strong alternative explanation |
| HIGH acquire/recover -> LOW retain is an effective adaptive controller | ARK-007R + ARK-010 | NEW_HYPOTHESIS; decisive switch arm unexecuted |
| Continuation order is a causal source of retention variability | matched continuation studies support sensitivity | supported directionally but should be tested with explicit order-only causal tournament if elevated to a general claim |
| LR-retention protection transfers outside arithmetic | none qualified yet | NOT_DEMONSTRATED |

## ARK-009 TRANSFER DIAGNOSTIC WARNING

ARK-009's implemented `query-swap` changes both query identity and fact order (reversal). Therefore the low diagnostic score cannot be assigned specifically to query conditioning. Before retrying transfer, split into:
- `QUERY_ONLY`: same facts/order, different query;
- `ORDER_ONLY`: same query, permuted facts;
- `QUERY+ORDER`: both changed.

Until that repair, ARK-009 should be read as `TRANSFER_GATE_NOT_QUALIFIED`, not as a failure of the LR-retention effect to transfer.

## CURRENT BOTTLENECKS

1. **Transfer qualification:** find a non-arithmetic task the Micro model can acquire under clean orthogonal robustness diagnostics, then run the matched LR-retention fork.
2. **State-dependent control:** test HIGH-until-recovery -> LOW-retain against HIGH-only and LOW-only from the same collapse state and same future data.
3. **Mechanism identification:** distinguish near-freezing from genuine consolidation by matching/parameterizing update magnitude, displacement, or intermediate LR regimes.
4. **Scale:** only after transfer survives should any intervention be considered for P35/V5 Core.

## CLAIM BOUNDARY

No Arkenstone result yet demonstrates a general AGI mechanism, general optimizer law, or V5-Core-ready schedule. The strongest executed scientific claim is a **replicated Micro-T2 causal retention effect** plus a **state-dependent post-collapse recovery asymmetry**.
