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

## Detailed findings by checkpoint, task and matched world

### Where seed 19 changed the result

All six runs used the standard training objective. The table follows the same 576 development views throughout training. The full-cohort score in the opening table uses 2,304 views at the final checkpoint.

| Model and seed | Update 0 | 592 | 1,184 | 1,472 | 2,368 | Mean loss, first → last 100 updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P, 7 | 0.35% | 32.64% | 42.36% | 47.40% | 62.50% | 1.802 → 0.439 |
| R, 7 | 0.35% | 35.42% | 56.94% | 61.46% | 68.23% | 1.797 → 0.469 |
| P, 19 | 0.00% | 33.33% | 46.35% | 64.93% | 68.40% | 1.873 → 0.446 |
| R, 19 | 0.00% | 31.08% | 55.03% | 55.90% | 64.24% | 1.871 → 0.570 |
| P, 41 | 5.21% | 48.78% | 53.65% | 56.42% | 64.41% | 1.815 → 0.437 |
| R, 41 | 5.21% | 49.13% | 55.73% | 58.33% | 65.97% | 1.816 → 0.441 |

On seed 19, R led P at update 1,184. By update 1,472 P had reached 64.93% while R stood at 55.90%. Both continued improving through the last checkpoint. A diagnosis of training collapse would miss the observed crossover. The checkpoint spacing leaves its precise timing open.

The following counts pair P and R by base world and view at the final checkpoint. Each row sums to 2,304. The six views of one world share structure, so these counts describe disagreements and cannot be treated as independent training replications.

| Seed | Both correct | Only P correct | Only R correct | Both wrong |
| --- | ---: | ---: | ---: | ---: |
| 7 | 1189 | 182 | 277 | 656 |
| 19 | 1239 | 279 | 143 | 643 |
| 41 | 1248 | 150 | 165 | 741 |

Seed 41 illustrates why to keep the traces. A net gain of 15 correct episodes combines 165 R-only successes with 150 P-only successes. On seed 19, the corresponding balance reverses to 143 versus 279. Replaying these discordant worlds with frozen weights gives a targeted way to study which decisions change.

### Transfer between presentations

| View on the same base world | P | R | R − P (points) |
| --- | ---: | ---: | ---: |
| Held-out narrative | 16.15% | 25.43% | +9.29 |
| Canonical | 74.39% | 69.79% | -4.60 |
| Entity/code permutation | 56.51% | 61.55% | +5.03 |
| Irrelevant facts | 74.65% | 69.27% | -5.38 |
| Relevant counterfactual | 75.09% | 71.09% | -3.99 |
| Changed fact order | 75.35% | 72.74% | -2.60 |

| Model | Both renderings correct | Canonical only | Narrative only | Both wrong |
| --- | ---: | ---: | ---: | ---: |
| P | 150 | 707 | 36 | 259 |
| R | 225 | 579 | 68 | 280 |

Each rendering comparison pairs 384 base worlds within each of three seeds. P lost a correct canonical solution on 707 world–seed pairs and gained one on 36. R lost 579 and gained 68. Alternate wording therefore changes which particular worlds the models solve. R recovers some robustness while its canonical accuracy falls.

### Depth, counterfactuals and abstention

| Depth stress, links followed | P | R | R − P (points) |
| --- | ---: | ---: | ---: |
| 1 | 87.50% | 84.11% | -3.39 |
| 2 | 83.85% | 81.25% | -2.60 |
| 3 | 89.32% | 86.72% | -2.60 |
| 4 | 70.57% | 71.09% | +0.52 |
| 5 | 58.59% | 58.33% | -0.26 |

The separate depth suite contains 64 base worlds per hop count, each evaluated on two linked views. Both models score higher at three hops than at two, then decline at four and five. The worlds also differ between hop counts, so the curve alone cannot identify a computation limit.

| Matched endpoint | P | R | R − P (points) |
| --- | ---: | ---: | ---: |
| Canonical and changed answer both correct, seven mechanistic families | 80.80% | 74.70% | -6.10 |
| Canonical, permuted and irrelevant-fact views all correct | 51.30% | 52.34% | +1.04 |
| Unanswerable, missing evidence or tied authority | 22.74% | 23.26% | +0.52 |
| Unanswerable after available tools | 40.80% | 53.99% | +13.19 |

The three-view conjunction is scored on these A0 checkpoints, which trained on D1-H. It is an evaluation of invariance. An effect of training with a paired-view objective requires the separate objective comparison.

### How to decide the follow-up

1. Retrieve the full A0 checkpoints and match each final checkpoint to the SHA in the curated metrics. Reproduce intact actions on the archived worlds before interpreting any changes under intervention.
2. Evaluate identical worlds under intact R, sham, zero relational contribution, permuted edges and disabled copy. Balance base worlds across the seven mechanistic families, hop counts and renderings. Save paired action changes even if aggregate accuracy stays constant.
3. A repeatable loss under relational intervention relative to sham, concentrated on multi-hop or counterfactual pairs, would support a functional contribution there. If only copy removal moves the answers, test whether the relational path alters target log-probabilities without changing the winning action. If tool decisions already diverge at the first step, isolate the renderer fields naming available actions and the goal.
4. Choose the next training comparison from those outcomes. A localized relational effect motivates the capacity and edge controls in A2. A first-operation rendering defect motivates a paired-renderer intervention with fixed facts, training budget and seeds. Use a fresh held-out renderer for confirmation after choosing the intervention.

The tables use the [curated per-job metrics](METRICS.json). The two disagreement tables additionally pair final episodes from the owner archive identified in [provenance](PROVENANCE.json), using seed, base-world ID and view. Accuracy counts use the independently rescored supported-correct verdict. The matched analyses are retrospective development findings.
