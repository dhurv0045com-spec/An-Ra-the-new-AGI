# X1 A0 | Does a relational slot update improve a pointer learner?

**Decision, 18 November 2026.** Retain P as the comparator. R missed the registered development threshold. Both S20 models learned on D1-H. The strongest measured vulnerability was transfer to a held-out presentation of the same world. The role of the relational path on multi-hop worlds remains **CAUSE UNRESOLVED**.

| Matched init/order seed | P final macro accuracy | R final macro accuracy | R minus P, percentage points |
|---|---:|---:|---:|
| 7 | 59.51% | 63.63% | +4.12 |
| 19 | 65.89% | 59.98% | −5.90 |
| 41 | 60.68% | 61.33% | +0.65 |
| Mean | 62.02% | 61.65% | −0.38 |

Each final score covers 384 base worlds, six linked views each, balanced over 12 families. The gate requires a mean gain of at least five points, two positive seeds and no drop worse than five points. Only the positive-seed condition passed. The archived world-cluster bootstrap gives a 95% interval of −1.82 to +1.06 points conditional on these runs. Across three paired training seeds the t interval is much wider, −13.02 to +12.27 points. Seed changes affect initialization and order together. This experiment cannot attribute seed 19’s reversal to either one.

**Learning and cost.** P had 19,947,925 parameters and R had 20,444,310. Both received 2,368 updates per seed at effective batch 16. On the *same* 576 milestone views, including a matched subset at the final checkpoint, mean accuracies were:

| Update | 0 | 592 | 1,184 | 1,472 | 2,368 |
|---|---:|---:|---:|---:|---:|
| P | 1.85% | 38.25% | 47.45% | 56.25% | 65.10% |
| R | 1.85% | 38.54% | 55.90% | 58.56% | 66.15% |

R led by 8.45 points midway. Most of that lead disappeared by the final matched subset, and the full development cohort slightly favored P. Mean first-to-last 100-update loss fell from 1.830 to 0.441 for P and 1.828 to 0.493 for R. Recorded job durations summed to 3,752 seconds for P and 4,893 for R, about 30% more worker time for R. These receipts cover concurrent work on two T4s. Summing them gives worker time, not session elapsed time.

**Where behavior changed.** Averaged over seeds, P scored 74.4% on the canonical view and 16.1% on the held-out narrative rendering. R scored 69.8% and 25.4%. The difference begins before many final answers: the first executed operation disagreed with the recorded ideal operation on 14.2% of canonical versus 45.5% of narrative episodes for P, and 16.2% versus 24.7% for R. This diagnoses sensitivity to the rendering intervention. Wording, token positions, prompt length and learned tool selection remain competing explanations. On X10 (missing evidence or an equal-authority conflict), P and R attained just 22.7% and 23.3%. On five-hop depth stress they reached 58.6% and 58.3%. R scored lower on the seven-family mechanistic counterfactual both-correct endpoint in every seed.

**What the interventions establish.** Disabling copy reduced subset accuracy by 52–58 points for P and 48–67 points for R. Removing R’s relational contribution, permuting edges or applying the sham produced zero net accuracy change in all three R runs on this subset. The evaluator takes the first 48 development views, all from X01 (eight base worlds times six views). The registered seven-family mechanistic counterfactual endpoint excludes X01 entirely. Aggregate accuracy can also hide opposing per-world changes. These receipts show that copy contributed on the sampled task. They leave open whether R uses its edges elsewhere.

**Audit and correction.** The analysis ZIP passed the project verifier. The original offline analyzer regenerated the D1-H training identity and found zero contract violations in 27,648 milestone/development traces. A separate replay regenerated development and depth worlds, checked ordered cohort identities and all six update journals, and rescored all 31,488 recorded action sequences for terminal status, supported correctness, lucky guesses and charged cost. All outcomes and milestone/depth summaries matched (family-rate averaging tolerance 1e−14). This archive contains checkpoint manifests. Neural inference and weight-file integrity require the corresponding full archive. The archived seed-statistics helper received two incomplete rows per seed and returned INVALID_INFERENCE. The repaired adapter supplies one complete observation per paired seed. Correcting it leaves the development decision unchanged.

**Next experiment.** Use the existing frozen checkpoints first. Cross sham, zero-R, permuted edges and copy disabled with the canonical/narrative views of identical worlds. Include all seven mechanistic families, X10, and the depth suite. Record paired changes in operations, answer values, target log-probability and supported accuracy per base world and seed. If lesions change predictions primarily on deeper worlds, test that interaction with the registered R-noedge and capacity-matched P-capR comparisons in A2. If R and sham remain indistinguishable while copy removal has a large effect, measure the norm and gradient of the R contribution before spending another training budget. For the rendering gap, change one presentation feature at a time on the same worlds and check whether first-operation errors recover. Keep follow-up development analyses separate from any later confirmation.

Reproduce with `python -m tools.rescore_x1_a0 PATH_TO_ANALYSIS_ZIP METRICS.json` from the repository root. Per-seed outcomes and immutable source/archive identities are in the adjacent JSON files.
