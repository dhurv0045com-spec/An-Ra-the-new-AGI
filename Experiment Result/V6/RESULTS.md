# Experiment Result — Elendril V6

**Status:** Archived research outcome (not a new research plan).  
**Study:** Elendril V6 Kaggle T4×2, 15 preregistered primary panels P01–P15 and four exploratory probes S16–S19.  
**Evidence inspected:** User-uploaded `ELENDRIL_V6_ANALYSIS.zip` (11 reports; 8.5 MiB), ZIP SHA-256 `939b3389f236a803444e40386b76a5eae1ca04b98fc45e0862689ec3a6b4a011`.  
**Result classification:** P01–P15 = 2 PASS, 10 FAIL, 3 NOT_QUALIFIED; S16–S19 = four completed **exploratory** probes, not four scientific hypothesis passes.  
**Code provenance:** V6 run operator source was pinned to Git commit `381195e17b0252220115c64eafb2e9f67d58416b` as shipped in the auto-run notebook; evaluation is procedural/development evidence, **not** an externally sealed AGI benchmark.

## What 'finished' means

The decision report records `campaign_status: COMPLETE`, and all 14 training-arm records have `status: COMPLETE`. This **does not mean all 19 hypotheses passed or that the full six-hour budget was spent**. The six hours were a maximum execution budget. All 14 listed training runs are the small S3 profile, at one epoch each; there are no completed S8/S20 training records in the supplied `all_arm_results.json`. The formation threshold was 90% overall accuracy, at least 80% per task kind and no invalid actions, on **both seeds**. Even the best individual S3 result (62/112, or 55.4%) did not reach the gate. More expensive confirmation/scaling work was therefore **not qualified**, rather than being attempted and yielding negative results.

The 33-event experiment journal includes verified exports and some `EXPORT_DEFERRED` checkpoint/snapshot coordination events. These warrant engineering follow-up but do not, on their own, prove a Python failure caused the measured cognition failures. Do not interpret a failure to satisfy a scientific gate as a crashed training job.

## Results of all 19 panels

| ID | Status | Observation from uploaded results | Meaning / unresolved cause |
|---|---|---|---|
| P01 — Positive-control learnability | PASS | 14/14 on the small positive-control rawtrainval check; only 6/14 on distinct validation (1 invalid) | Small memorized control succeeds; separate generalization is weak. The PASS is not a 90% full-task formation pass. |
| P02 — Known-evidence answering | FAIL | 1/16 correct; 10 unnecessary queries when zero were needed; 1 invalid | Known evidence is not reliably converted to an answer; observed unnecessary queries. Cannot determine from aggregate scores alone whether binding, output decoding or policy dominates. |
| P03 — Missing-evidence query | FAIL | 16/16 useful first queries; 2/16 correct terminal answers | The model often makes the appropriate first query but cannot reliably use the returned evidence or emit the correct terminal answer. |
| P04 — Conflicting-evidence query | FAIL | 16/16 useful first queries; 4/16 correct answers | Query decision and final conflict resolution are separated: the latter remains weak. Timestamp/source binding is a diagnostic hypothesis, not a demonstrated unique cause. |
| P05 — Null and unavailable abstention | PASS | 32/32 correct abstentions; zero queries; zero invalid | Bounded abstention behavior works on these held-out procedural cases; this does not validate broader understanding. |
| P06 — Post-query revision | FAIL | 0/32 complete intervention pairs both followed the changed observation | No demonstrated causal updating under the paired payload-swap requirement; cannot attribute uniquely to input conditioning vs action/answer generation. |
| P07 — Cost-aware thinking and queries | FAIL | Model utility 0.3107/episode vs fixed 0.9714, always-query 0.9429, no-query 0.7679 | Learned decisions underperform simple engineered controls at registered query cost. No learned cost-aware superiority. |
| P08 — Uncertainty/self-error detection | NOT_QUALIFIED | Descriptive Brier 0.0174, ECE 0.0491, AUROC 0.9965 on 112; report labels DIAGNOSTIC | A descriptive confidence score is not independently qualified calibration or self-awareness. Separate validation fitting, comparison and interpretation prerequisites remain unresolved. |
| P09 — Legal action and EOS integrity | FAIL | 1 invalid action in 112 episodes; 39/112 correct | The registered zero-invalid requirement fails. Invalid action alone does not explain all poor accuracy. |
| P10 — Query swap/query blindness | FAIL | 0/16 paired query-swap cases both correct; aggregate 3/32; 2 invalid | No robust response to counterfactually changed query payloads; policy may be query-blind or fail later output generation. |
| P11 — Temporal/recency shortcuts | FAIL | 0/16 ordered intervention pairs both correct; 1/32 stress answers; 12 invalid | Temporal/source-order interventions fail; observed invalid actions also confound attribution. |
| P12 — Representation and distractor OOD | FAIL | Aliases 4/112, templates 28/112, distractors 13/112, combined 13/112 | Strong sensitivity to changed surface forms; does not establish which tokenizer, representation or training-data factor causes it. |
| P13 — Compositional 2-to-3 hop | NOT_QUALIFIED | 2-hop prerequisite failed; 3-hop exploratory score 0/16 | Three-hop generalization cannot be meaningfully claimed before 2-hop acquisition; observed exploratory result still retained. |
| P14 — Frozen-weight memory benefit | FAIL | Memory on/off/irrelevant/prefetch all 0/16; 0/16 causal-swap pairs followed | No attributable memory benefit. Engineered prefetch cannot be counted as model-selected retrieval; a zero-baseline also limits causal interpretation. |
| P15 — Sequential retention and transfer | NOT_QUALIFIED | No independently acquired A/B stages and matched C trajectories | Retention/transfer was not evaluable under its prerequisites, not evidence that a learned skill was later forgotten. |
| S16 — Thinking-budget sensitivity | COMPLETE_EXPLORATORY | 1 decision 8/28; 2 decisions 11/28; 4 decisions 11/28 | A modest gain from 1 to 2 decisions, none beyond 2 in this sample; more computation does not prove general metacognition. |
| S17 — Fixed symbol-permutation probe | COMPLETE_EXPLORATORY | Baseline 11/28; fixed permutation 10/28; output equivariance 0.3571 | Weak equivariance in one predefined bijection. One permutation does not establish arbitrary invariance. |
| S18 — Stale-memory interference | COMPLETE_EXPLORATORY | Baseline 1/4 and stale 1/4; stale consumed fewer requests/positions | Only four cases; unchanged accuracy is inconclusive about resistance to stale memory. |
| S19 — Query-cost sensitivity | COMPLETE_EXPLORATORY | Model mean utility 0.3107 at cost 0.1; fixed 0.9714; at cost 1 model -0.0268 vs no-query 0.7679 | Repricing a frozen policy reveals cost sensitivity of utility, not adaptive learning or a reoptimized cost-aware policy. |

## Training comparisons — all 14 completed small-model arms

All values below are drawn from `all_arm_results.json` for **validation** on 112 examples per run; values are not independently sealed results. The profile was S3 and each run used one epoch. Utilities shown are totals, not per-episode normalized numbers.

| Arm | Seed | Validation correct | Utility (sum) |
|---|---:|---:|
| A1 | 7 | 39/112 | 34.8 |
| A1 | 19 | 43/112 | 38.7 |
| A2 | 7 | 39/112 | 34.7 |
| A2 | 19 | 43/112 | 38.8 |
| A3 | 7 | 39/112 | 39.0 |
| A3 | 19 | 62/112 | 62.0 |
| A4 | 7 | 38/112 | 33.4 |
| A4 | 19 | 42/112 | 37.4 |
| A5 | 7 | 38/112 | 33.7 |
| A5 | 19 | 41/112 | 36.8 |
| R0 (legacy ASCII) | 7 | 48/112 | 44.8 |
| R0 (legacy ASCII) | 19 | 54/112 | 50.8 |
| R1 (UTF8 ASCII) | 7 | 41/112 | 37.8 |
| R1 (UTF8 ASCII) | 19 | 48/112 | 44.6 |

**Measured observation:** A3 seed 19 reached 62/112 while A3 seed 7 reached 39/112. The large seed dependence makes the single best score insufficient to justify claims of stable learning. The strongest result still fails the ≥90% formation gate. R0 legacy-ASCII comparisons are not evidence of a general advantage across matched token/computation budgets; the original comparison report explicitly notes representation/work differences.

## What the results do and do not explain

**Direct observations:** Basic positive-control learning and null/unavailable abstention can succeed. Appropriate first queries in P03 and P04 do not result in reliable final answers. Evidence-swaps, source/time changes, representation variation and memory interventions are not reliably handled. Baseline engineered controllers outperformed the neural policy in the tested cost-sensitive worlds.

**Plausible mechanisms still needing discrimination:** inadequate supervision of post-query answer transitions; incomplete binding between newly observed payload and target symbol; high sensitivity to text serialization/tokenization; sparse one-epoch optimization; and invalid-action/EOS control. The archive alone does not establish one exclusive root cause, nor does it establish that a software bug is responsible for the cognition scores. Those are hypotheses, **not measured results**.

**Limitations:** There was no S8/S20 scaling comparison after the failed formation gate; P08/P13/P15 were not qualified by their required prerequisites; S16–S19 are low-power exploratory or post-hoc frozen-model contrasts. The positive-control P01 PASS is 14/14 on its small training-associated task, but the distinct validation portion was only 6/14 with one invalid output. No independent held-out external benchmark or AGI claim is warranted.

## Evidence custody and future entries

This Markdown report is a **compact, source-grounded archive of findings**, not the full experiment output. The original uploaded 11-file analysis ZIP and larger Kaggle checkpoint ZIP should be retained separately: the 70 MB `all_arm_results.json` and model weights are **not embedded in this GitHub document**. That distinction prevents a summary from being mistaken for a reproducibility-complete artifact.

The analysis ZIP contained these 11 source files:
`all_arm_results.json`, `diagnostics_15panels.json`, `supplemental_experiments.json`, `diagnostics_by_checkpoint.json`, `comparisons.json`, `decision_matrix.json`, `experiment_ledger.jsonl`, `data_manifest.json`, `checkpoint_manifest.json`, `protocol.json`, `device_inventory.json`.

For future experiments, add a separate, dated run folder under `Experiment Result/` with its report, source commit, source artifact checksums and any appropriate compact, reproducible evidence. Preserve failed, skipped and unqualified results without relabeling them.
