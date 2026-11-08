# Independent V7.1 research review — 2026-11-09

**Scope:** Evidence-grounded inspection of the `Elendril` branch at `a2e8c0107ecf486acf2c4ef6d3677bd73ee47399`, owner core run `20c80f66e76e4b589aed2053afbddffc` (runnable source `6104ffd7134cb4cf4b3f340bd060d24a1e2dd200`), recorded GitHub files and GitHub Actions. **No training, neural checkpoint inference, or raw ZIP byte rehashing was performed for this review.** This is an independent interpretation and arithmetic cross-check of the archived machine-readable `METRICS.json` and selected raw case records, not a second neural reproduction.

## Delivery and provenance

- The owner-run `FINAL_STATUS.json` records `COMPLETE`, core phase `EXECUTED`, 9/9 finished jobs, no recorded worker errors, and both exports `EXPORT_VERIFIED`. These are the source-run receipts, not evidence that this reviewer restored the optimizer or inspected the FULL archive.
- GitHub has the original `ELENDRIL_V71_ANALYSIS.zip` (22,696,077 bytes), `FINAL_STATUS.json`, `METRICS.json`, `FOCUSED_CASES.json`, `TRAJECTORIES.csv`, `RESULTS.md`, a plot and regeneration code. The owner-reported analysis SHA256 is `5f198fbed35c022b2c114e8cd11eb747bcba73c06916f79169e3f61169ff62c5`; the reported full-archive SHA256 is `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19` for 1,958,029,910 bytes. The FULL archive was not provided in GitHub.
- The archived offline audit reports 55 suite-report action replays plus four positive-control records, with narrowly bounded cross-runtime rounding allowances on six derived Brier values. This is a reported replay result, **not neural re-inference**. The `METRICS.json` file contains 144 job/cohort rows and 198 learning-trajectory rows; selected sums were independently recalculated here.
- The latest GitHub Linux workflow at this review's branch head passed 347 tests and 14 subtests, with one warning from PyTorch's missing optional NumPy during CI. Successful CPU/CI tests verify engineering paths, not higher model competence.
- Same-device, fresh-process GPU recovery and positive canaries are reported successful on both T4s. **Independent FULL-checkpoint custody and fresh-VM recovery remain unverified.**

## What ran versus what was merely designed

The executed phase was **core only**: 3 S3 training methods × seeds 7/19/41, each four epochs, all 3,084,800-parameter models. The notebook explicitly defaults `PHASE='core'`. The proposed 12-run S8/S20 scale phase, nine-run continual-learning phase, optional original-V7 frozen phase, 4× data study, pointer architecture and genuine novel-C transfer **did not execute**. They are not negative empirical outcomes.

The six uniform/blocked CE runs reproduce the original V7 seed-matched recorded weight fingerprints and confirmation traces. They are useful reproducibility evidence, **not six additional independent seed replications**. The three intermediate-supervision runs represent the principal newly trained comparison. Total reported work is 23,412 optimizer updates and 782,496 supervised tokens. Controller time is 3,815.15 seconds (63.59 minutes) before final export; summed worker reservation is 6,761.61 seconds (112.69 GPU-minutes), not kernel-only utilization.

## Recomputed confirmation performance

All scores below come from the committed `METRICS.json` `rows` for the same 112 synthetic confirmation worlds per seed. Three model seeds do not create 336 independent worlds.

| Method | Correct across 3 seeds | Mean accuracy | Interpretation |
|---|---:|---:|---|
| Uniform CE | 207/336 | 61.61% | Seed-unstable comparator |
| Blocked CE | 270/336 | 80.36% | Retain as current reference |
| Blocked + intermediate supervision | 269/336 | 80.06% | Not better overall |

Intermediate supervision required 3,068 versus 2,368 optimizer updates per run (+29.6%) and 109,280 versus 75,776 supervised tokens (+44.2%). A different target distribution and extra work prevent treating changed training loss as comparable evidence of better learning. It improves seed 7 relative to blocked CE but worsens seeds 19 and 41. **No mechanism or model upgrade is qualified from this result.**

## Findings across research families

**A. Composition — benchmark and acquisition failure are confounded.** Blocked CE gets 3/48 on independent one-hop chains and 0/48 on two-hop, so the new chain distribution fails *before* composition is isolated. Intermediate gets 22/48 on four-hop versus blocked CE's 4/48, but the deterministic last-visible-row / max-timestamp control is correct on all 64 new hop worlds. The generated edges and ticks advance in path order, creating a noncompositional shortcut. Four-hop gains are not proof of multi-hop computation. The newly implemented 512-record, 48-contrast same-world follow-up dataset is **not yet evaluated by a frozen neural checkpoint**.

**B. Evidence / balanced counterfactuals — narrow positive result.** Both-correct balanced-payload pairs total 22/48 for uniform, 45/48 for blocked CE, and 47/48 for blocked intermediate across three seeds. A model can follow some supplied factual changes; these pairs share synthetic world/task authorship and do not prove generalized causal reasoning.

**C. Memory — strong correction-specific regression.** The original-memory condition scores 45/48 for blocked CE and 48/48 for intermediate, while adding a stale same-source row followed by a newer correction lowers these to 28/48 and 2/48. This is not explained simply as 'memory absent'—retrieval and public observation can occur. The archived audit notes only 7/46 incorrect intermediate corrected-memory answers exactly copy the stale value; no single copying explanation is proven. Memory-removed and irrelevant variants deliberately leave some targets unattainable; an incorrect final answer there is not automatically unsafe abstention.

**D. Action decisions — availability checking is not robust.** Across blocked CE seeds, 96/96 cases where needed query channels are unavailable include invalid QUERY attempts despite `Query=0`. Both memory-removed and irrelevant-memory variants produce 48/48 invalid retrieval attempts for blocked CE. A grammar-constrained decoder can remove illegal actions, but its intervention is engineered output restriction, not learned decision competence; final answer quality typically remains weak. All methods still exhibit unnecessary requests on some ordinary cases.

**E. Robustness — severe surface dependence.** Blocked CE across three seeds: baseline 270/336 (80.36%); aliases 78/336 (23.21%); distractors 65/336 (19.35%); changed templates 42/336 (12.50%). In matched row-order variants the recorded report separates 270/336 original from 139/336 reversed (41.37%). This robustly demonstrates distribution/serialization sensitivity in the procedural suite, not its unique internal neural cause. One documented case queries the same source, observes the newer `A=J` fact, but answers `G` instead of `J` when rows are reversed.

**F. Training stability — extra supervision is not a universal remedy.** The CE baselines replicate V7. Intermediate training improves some seeds and balanced-payload scores but introduces or worsens memory-correction and surface robustness failures. The final 28-world training monitor is not evidence of comprehensive training-set mastery. A later checkpoint losing a case does not itself prove a catastrophic forgetting mechanism.

**G. Retention and transfer — not tested.** Nine continual branches and their acquisition qualifications exist in the protocol/code, but **no B-stage training, retention or forward-transfer result** was recorded. The independently novel C transfer stage is deferred.

## Why the learner failed: evidence versus hypotheses

**Demonstrated behavioral boundaries:** surface-order sensitivity, legal-action availability failures, stale-correction regressions, difficulty on newly generated simple chains and stable benchmark-specific evidence responsiveness.

**Plausible, unproven causes:** dependence on row positions or recency cues, brittle token/string representation, inadequate optimization on unfamiliar formats, conflicts between supervision objectives, insufficient relation binding and poor learned action-availability conditioning.

**What must not be claimed:** a proven internal root cause, causal memory mechanisms, learned multi-hop composition, successful larger-model scaling, catastrophic forgetting, independent sealed generalization or AGI.

The most revealing pattern is **not simply 'more hops are hard'**: small changes in serialization, aliases and new single-hop distributions can break behavior that looks strong in original V7-like worlds. A larger model on the existing confounded hop benchmark could obtain impressive scores through a shortcut and still teach us little about composition.

## Recommended next research sequence

1. **Custody first:** preserve and independently validate the original 1.96 GB FULL archive; restore frozen S3 checkpoint components and run identity/finite-tensor checks before asserting checkpoint reuse. Keep archived original ZIP and receipts unchanged.
2. **Disentangle presentation from reasoning:** execute the already built 16-world × 32-cohort frozen-checkpoint follow-up (512 records; 48 preregistered one-factor contrasts) under matching public information. Compare timestamp range, row order and removal of irrelevant rows, with original two-hop, first-hop, clamped and direct-lookup tasks. Record case-level actions, costs and paired differences. These are 16 worlds reused across conditions, not 512 independent worlds. Follow with a larger, independently authored holdout after designing an adversarial no-shortcut route generator.
3. **Validate shortcuts explicitly:** randomize edge order/ticks independently of path, test last-row/max-timestamp/copy-only baselines and include clamped versus true multi-hop with matched formatting; do not promote a neural score when simple public heuristics solve the benchmark.
4. **Control memory corrections fairly:** run an equal-length, equal-position neutral-history comparator versus stale/corrected same-key history; confirm retrieval exposure. Distinguish conflicting binding from added context. The current follow-up memory design is proposed, not implemented.
5. **Repair action availability under correct attribution:** compare raw neural decisions with legal-action-constrained decoding and an appropriately supervised availability-aware policy. Measure needless queries, INVALID outcomes, query economics and downstream answers; the constrained decoder is a mechanics control.
6. **Preserve strong abilities while broadening evaluations:** retest balanced payloads and real abstention behavior as the surface distribution changes. Stratify non-null, null, cost, accuracy and robustness results; report regressions as prominently as gains.
7. **Then study scale responsibly:** the registered S8/S20 × uniform/blocked × three-seed phase would add 12 GPU training runs. Before spending it, qualify capacity and benchmark validity; report parameter-, token-, and GPU-cost-adjusted improvements. If running an exploratory size pilot earlier, register it separately and do not contaminate the original confirmation protocol. Continual learning should follow independently demonstrated A/B acquisition and proper exposure controls.

**Overall judgment:** Excellent evidence and archive engineering, clear improvement over V6's simple PASS/FAIL reporting, and useful narrow evidence-reading acquisition. The present evidence does **not** justify promoting intermediate supervision or claiming compositional learning. The highest-value next step is to fix the interpretability of evaluation and examine frozen checkpoints before another broad scaling campaign. No changes to the training code or historical results were made by this review.
