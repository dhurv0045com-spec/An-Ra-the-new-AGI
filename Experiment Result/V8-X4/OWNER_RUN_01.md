# ELENDRIL V8-X4 | Owner run 01 | 27 November 2026

**X4 repaired a measurable defect in X3 and isolated a repeatable data-recipe effect. The evidence favors keeping the S20 reference model for the next comparison.** The fixed-400 R-Ref recipe beat the fresh-stream version at every seed on both held-out evaluations. A family-only predictor nevertheless beat 26 of the 27 configurations on mean unfamiliar-world accuracy. No checkpoint met the registered competence threshold. The strongest lead is to change which problems the learner sees and what the evaluation rewards.

## Run and custody

The FULL export was 6,095,440,291 bytes, SHA-256 `9c816cbd5a65126f51b715e2ecccb3096d11ea2354a3e4c6d758272234276840`. The recorded source was `4e7b59e8fc25f23f731ddf1dd2e19c16600ccc34`. All 909 manifest-listed files passed size and SHA-256 verification. All 81 per-job receipts matched the job registry and aggregate receipt. Final checkpoint pointers, manifest digests and component hashes agreed. All 81 jobs have both held-out profiles. Their update journals contain 82,944 consecutive entries in total, 10 reported automatic mixed-precision skips and zero nonfinite total losses. Summed per-job wall time was 15,694.779 seconds (4.36 worker-hours), excluding separate campaign-level evaluation and export. S50 had zero registered jobs.

The design had 27 configurations, three fitted seeds per configuration, 1,024 scheduled updates per fit. C1–C3 used 7, 19 and 41. C4 used 101, 202 and 303. Each split's macro score weights the 22 families equally: six examples per family on development, eight on confirmation and eight on unfamiliar worlds (OOD). All models reused those test cases. I recomputed the scores and all 11 registered development contrasts from the underlying receipts and also paired the held-out comparisons by seed. The metrics contain all 27 arms, seed-level deltas, family outcomes and checkpoint hashes. The X3 battery had different tasks, generators and denominators; its aggregate score cannot serve as an X4 baseline.

## What the comparisons establish

| Configuration | Confirmation | Unfamiliar worlds |
|---|---:|---:|
| Family-only predictor, fitted on generated train cases | 36.93% | 35.23% |
| C1 Base / byte cross-entropy | 39.77% | 9.47% |
| C1 R-Ref / byte cross-entropy | 43.37% | 21.21% |
| C2 R-Ref / D2-ExactCF / fresh | 45.45% | 21.78% |
| C2 R-Ref / D4-ECCO / fresh | 42.05% | 18.75% |
| C2 R-Ref / D4-ECCO / fixed-400 | 58.14% | 41.86% |
| C2 R-Ref / D4-ECCO / fixed-4k | 58.33% | 33.14% |
| C2 R-Ref / D4-ECCO / fixed-40k | 58.14% | 27.27% |
| C3 P1-OpBank / dense control | 45.64% | 26.89% |
| C3 P1-OpBank | 44.32% | 22.92% |
| C4 P1-OpBank, new seeds | 40.15% | 21.78% |
| C4 integrated model, same new seeds | 38.64% | 18.56% |

**H1, sparse operator bank.** Its dense control led on both held-out means. Two of three seeds favored the dense control on OOD. The sparse operation has no reproducible advantage here. R1-Alpha was a different mixed result: it lost to its dense comparator at every seed on confirmation but gained 7.58 points on average on OOD. Check which families generated this difference before building on it.

**H2, error-conditioned objective.** On fresh data, D4 trailed D2-ExactCF on both held-out means. Unlike X3, the six fresh curricular objective arms shared exactly the same ordered case-ID digest and eligible target-byte total within each seed. This gives the negative comparison more weight. The pair builder usually infers a relevant change or an equivalence from neighboring examples' target strings. That does not establish that a single causal fact changed. The exposure receipts recorded zero irrelevant-change pairs across all jobs. The reported pair score measures success on the implemented pairs.

**H3, fresh experience and curriculum.** Fixed-400 exceeded fresh by 16.10 percentage points on confirmation and 23.11 on OOD. Out of 176 OOD cases, fixed/fresh correct counts were 77/42, 77/17 and 67/40 for seeds 7, 19 and 41. On fresh data the primitive-to-composition and random-interleaved recipes tied at 18.75% mean OOD. “Random interleaved” selects phase 3 throughout and the family rotation remains deterministic. This comparison does not isolate ordering at matched phase exposure.

**H4, integrated architecture.** New-seed C4 offers no consistent improvement over its standalone OpBank comparator. In the five archived depth probes, the measured accuracy stayed unchanged at 0, 1, 2, 4 and 8 reasoning steps within each probe. Those are small selected probes. No depth benefit or architecture promotion is supported.

**H5, unified causal output.** The archived tiny-model integrity test passed for 16 architectures and detected its deliberately leaky control. I repeated the X3-style suffix intervention on trained S20 weights: fixed-400 R-Ref seed 7 and integrated seed 101, one development example each from C03, C04, C09 and C10. With identical public prompt and answer prefix, swapping only future payload characters produced **zero change** in the first-payload-byte log distribution in all eight comparisons. Enabling the deliberate leak on the same weights produced changes of 0.44–13.35 in maximum absolute log probability. This is direct evidence for the repair in these cases. C1 R-Ref also beat Base on OOD at all three seeds (+9.66, +10.80, +14.77 points), while P-Ref tied R-Ref on mean OOD. X4 has no otherwise-identical legacy dual-head arm, so a causal accuracy gain from output unification alone remains open.

## Three checks that change the interpretation

**1. The generator made many new prompts while repeatedly selecting the same task variants.** I regenerated the complete seed-7 schedules for fixed-400 and fresh and reproduced the archived exposure manifests exactly. Both presented 16,384 examples. Fixed-400 used 1,200 IDs across three phase pools and 1,155 distinct prompts. Fresh used 16,384 IDs and 14,118 distinct prompts. The fresh pair generator asks for indices 0 and 1 each time. It computes an advancing instance index and never passes it to the generator.

| Actual training coverage, seed 7 | Fixed-400 | Fresh |
|---|---|---|
| Grid operations | 7 | Horizontal and vertical flips only |
| Novel-rule demonstration counts | 0, 1, 4, 16, 64 | 0, 1 |
| Self-check problem types | Easy, hard, missing premise | Easy, hard |
| Language–math presentations | 37 | 744 |
| Language–math units observed | Tokens only | Tokens only |

The fixed pool also has a defect: it builds cases grouped by family then truncates the combined list. Its last family, language–math, received only 37 of the seed-7 presentations. Neither schedule showed any exact prompt overlap with confirmation or OOD in this replay. Some canonical structural hashes did match (fixed 6/6 cases on confirmation/OOD; fresh 8/7), which warrants family-specific interpretation. The archived custody receipt used separately generated small split batteries. The fixed-40k label likewise describes a configuration; its receipts show 13,850–13,962 distinct IDs actually presented. The results establish an advantage for the implemented fixed recipe. Repetition and coverage remain confounded.

**2. A cheap shortcut explains much of the attainable headline score.** For each public family I selected the most frequent typed target on 32 separately generated train cases, with lexicographic tie-breaking. This predictor ignores the body of every test prompt. Scored through the public parser it got 65/176 confirmation cases and 62/176 OOD cases right. It solved every held-out X05–X10 case by family identity alone. Fixed-400 beat it on OOD at all three seeds by 15, 15 and 5 cases. One encouraging counterexample is C08 memory revision: fixed-400 averaged 91.7% OOD where the shortcut got 0/8. This is performance on the supplied revision problems. Reuse of experience across independent episodes needs a separate test.

**3. Presentation alone changed trained behavior.** I loaded the fixed-400 and fresh R-Ref seed-7 checkpoints for CPU inference. For the first four OOD worlds in each of X01–X12, I verified identical facts, goals, permissions and gold answers across rendering templates 0 and 3. The 48-world paired intervention changed only the public rendering.

| Seed-7 checkpoint | Correct, template 0 | Correct, template 3 | Correct → wrong | Wrong → correct |
|---|---:|---:|---:|---:|
| Fixed-400 | 41/48 | 32/48 | 11 | 2 |
| Fresh | 31/48 | 19/48 | 12 | 0 |

Presentation sensitivity is demonstrated for these two checkpoints and worlds. This intervention does not explain the entire campaign-wide OOD gap. The shortcut and renderer probes are post-hoc diagnostics.

## Unresolved capabilities and qualification

* All 81 models got **0/8 in each of grid transformation, compositional instructions and language–math** on each held-out split. These are eight unique cases per family per split, reused across models. Metacognitive self-check averaged only 1.39% confirmation and 1.54% OOD.
* Zero checkpoints qualified. The highest individual confirmation accuracy was 61.93% against the 70% requirement, and the highest both-correct pair score was 35.19% against 65%. The archive labeled 31 checkpoints preliminary and 50 unqualified. Its preliminary labels compare everyone with the first sorted Base checkpoint (seed 19), so those labels cannot substitute for paired replication.
* The four five-mode evaluations covered three Base seeds and one R-Ref seed. All four recorded zero persistent-feedback gain, zero pre/post adaptation gain and zero tool-free and cross-domain transfer accuracy. Neither the integrated model nor OpBank received this evaluation. The separate L0–L5 profiles cover one checkpoint per configuration. Their “exposed train” pool comes from a generic split battery, without verifying membership in each job's actual schedule.
* The oracle's 100% integrity score establishes generator/scorer consistency. The oracle accepts the full case object, including latent specifications and, in some branches, the stored gold answer. Independent public-prompt solvability is still unverified. The neural held-out evaluator receives public prompts.

## Decision for the next owner-run experiment

Keep R-Ref at S20. First fix the instance-index bug and allocate fixed-pool capacity by family before truncation. Audit the **realized** schedule for operation, demonstration count, renderer, target type, distinct prompts, and target bytes. Construct labeled counterfactuals by editing one known fact, and verify the intended effect with an independent public-input solver.

Then run a three-seed **2 × 2** on the same backbone and objective: narrow versus broad task-variant coverage, crossed with repeated versus newly sampled worlds. Match examples, eligible target bytes, family weights, difficulty, renderer and update count across arms. Preserve the old fixed-400 results as a reference. Advance only if a gain repeats on fresh, balanced worlds against both the family-only predictor and its matched training control. Score counterfactual pairs whose correct answers differ within a family. Retain paired alternate renderings and report all three zero-score families separately. Check C08 with the same prompt shape but swapped memory events. This makes a positive result interpretable and a negative result actionable.

[Compact metrics and probe measurements](OWNER_RUN_01_METRICS.json)
