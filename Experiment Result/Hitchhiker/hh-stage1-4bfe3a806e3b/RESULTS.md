# Hitchhiker Stage 1 S3 screening — results

Analyzed 2026-11-15 by Codex, with Terra completing the curated report.

**Outcome: all 15 jobs completed, but no intervention passed the locked screening gate.** This was a validation screen, not proof that the interventions can never work. The largest mean primary gain came from C, but it was inconsistent across seeds and accompanied by an original-task regression above the allowed threshold.

## Scope and primary results

Five arms (baseline B0 and A, B, AB, C) ran at seeds 7, 19 and 41. Each job recorded 2,368 updates, 37,888 examples and 75,776 supervised tokens. The primary metric is accuracy on the entity-permutation cohort (2,048 episodes per model). Averages are over three matched seeds; gains and regressions are percentage points (pp).

B0 repeats the original static corpus; A uses fresh procedural worlds; B consistently relabels entities; AB combines freshness and relabeling; C changes coverage of task, evidence-completeness and action-availability conditions. These are the tested treatments, not all eight mechanisms from the original proposal.

| Arm | Seed 7 | Seed 19 | Seed 41 | Mean | Mean gain vs B0 | Worst original-kind regression | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | 4.59% | 0.68% | 0.83% | 2.03% | — | — | baseline |
| A | 7.91% | 0.00% | 27.69% | 11.87% | +9.83 pp | 9.09 pp | failed |
| B | 8.94% | 0.00% | 12.35% | 7.10% | +5.06 pp | 16.67 pp | failed |
| AB | 17.53% | 0.00% | 11.33% | 9.62% | +7.58 pp | 18.18 pp | failed |
| C | 32.67% | 14.70% | 0.00% | 15.79% | +13.75 pp | 36.36 pp | failed |

All four failed both the requirement for positive gains at every matched seed and the maximum 5 pp original-kind regression limit. The recorded gate has no locked winner. World-cluster intervals in the accompanying metrics are conditional on these three seeds; strong average contrasts do not override the prespecified seed consistency or regression criteria. C is a coverage package trained with pure cross-entropy, not a mask or penalty intervention. Supervised output counts match, but input compute does not exactly match: for seed 7, B0/B used 8,914,404 input positions, A/AB 8,915,589, and C 8,487,984, about 4.8% fewer than B0.

## What the apparent four-hop success means

C seed 7 answered all 256 ordered four-hop cases correctly, but this is not evidence of learned traversal. On the same 256 worlds in the row-order factor, original order scored 256/256, reverse 0/256 and random 81/256. Across all 768 recorded predictions, the answer was exactly the object of the final serialized fact. This is direct evidence of a positional shortcut on this specific cohort, not a characterization of every model. On the unperturbed hop cohorts this model scored 62/256 at hop 1, 0/256 at hop 2, 16/256 at hop 3 and 256/256 at hop 4; it abstained on 194, 256, 240 and 0 cases respectively. On these same four-hop worlds the distractor, source-renaming and large-timestamp perturbations each scored 0/256. The terminal-edge-change panel scored 256/256 but also changed the last row, so it does not distinguish traversal from copying the last fact.

For the row-order audit, the recorded prompt was parsed for four serialized facts and the recorded answer compared with the final fact's object. Counts and method are preserved in [METRICS.json](METRICS.json); individual prompts and full traces remain in the owner archive.

## Follow-up and boundaries

The selection receipt admits **C+penalty** and **D-external** as diagnostic follow-ups. It withholds AB+C and S20 contrast because no arm passed screening. Admission is a selection decision, not evidence those follow-ups were run or would succeed. D and E–H, Stage 0, S8 and S20 were not executed in this Stage 1 archive. A useful future test should randomize fact order and place the target edge independently of position, retaining the perturbation failures as well as successes.

## Experiment definition and model

This experiment asks whether data freshness, entity relabeling, or supervision coverage improves transfer to changed relational tasks over a static baseline. The [original proposal](../../../research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md) and [reviewed design](../../../research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md) are historical hypotheses and plans; this report and the executed protocol determine what actually happened. Later selection receipts supersede any tentative design recommendation.

S3 is a 3,084,800-parameter dense decoder transformer: UTF-8 byte vocabulary 260, width 256, four layers, four query heads and one key/value head, head dimension 64, feed-forward width 768, and context 512. It has tied embeddings, rotary positions, QK normalization and zero dropout. The registered recipe uses cross-entropy, learning rate 1e-4, four epochs, batch 16, microbatch 2, and matched initialization/order seeds 7, 19 and 41. Training data has 9,472 samples; 2,368 updates correspond to 37,888 examples. The B0 arm repeats static original-distribution worlds; A uses fresh procedural worlds; B relabels entity names consistently within an episode; AB combines A and B; C changes context-completeness and availability coverage while remaining static and pure CE. These are interventions, not established cures for memorization.

## Evaluation design and denominators

The validation suite has 256 generated worlds **per hop** and 20,992 recorded cases per model. Factor cohorts share underlying worlds and include multiple variants; their cases must not be treated as independent new worlds. A world identifier also does not guarantee a distinct normalized semantic state. The primary `factor-entity-permutation` cohort has 2,048 cases: original and permuted entity presentations across hops, not renamed-only accuracy. Separate validation evaluates 80 original tasks and 576 health-probe cases (256 belief revision, 256 missing bridge, 64 distractor trap); preservation was not qualified. Structural holdouts and confirmation suites were prepared but not evaluated in these jobs.

| Validation cohort | Cases per model |
| --- | ---: |
| hop-1, hop-2, hop-3, hop-4 | 256 each |
| factor-entity-permutation | 2,048 |
| factor-row-order | 3,072 |
| factor-distractors | 2,048 |
| factor-source-renaming | 2,048 |
| factor-timestamps | 2,048 |
| factor-answer-edge-swap | 2,048 |
| factor-answer-query-swap | 2,048 |
| factor-alias-clause | 4,096 |
| factor-wording | 512 |

The gate required at least +10 pp mean primary gain, positive differences for all three matched seeds, no original-task kind regression exceeding 5 pp, a positive world-cluster confidence interval, and at least 256 development worlds per hop. A and B and AB miss the mean threshold; C exceeds it. All four violate seed consistency and original-kind regression. For A/B/AB seed 19 the matched difference is −0.68 pp; for C seed 41 it is −0.83 pp. The worst regressions compare a same-seed original task kind with B0; those kinds contain only 11 or 12 cases each, so a single case changes the rate substantially. The uncertainty intervals cluster variants by world, not by distinct normalized content, and do not make three seeds a large independent sample.

## Every job: hop and original-task performance

Entries are correct cases, each hop out of 256 and the original validation set out of 80. Primary results in the earlier table are a different, pooled factor cohort.

| Arm | Seed | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 7 | 0 | 42 | 0 | 0 | 65 |
| B0 | 19 | 0 | 0 | 8 | 0 | 67 |
| B0 | 41 | 0 | 1 | 9 | 0 | 67 |
| A | 7 | 0 | 82 | 0 | 0 | 66 |
| A | 19 | 0 | 0 | 0 | 0 | 68 |
| A | 41 | 68 | 27 | 191 | 0 | 67 |
| B | 7 | 0 | 96 | 0 | 0 | 66 |
| B | 19 | 0 | 0 | 0 | 0 | 69 |
| B | 41 | 0 | 60 | 69 | 0 | 67 |
| AB | 7 | 0 | 69 | 110 | 0 | 64 |
| AB | 19 | 0 | 0 | 0 | 0 | 68 |
| AB | 41 | 93 | 24 | 0 | 0 | 66 |
| C | 7 | 62 | 0 | 16 | 256 | 69 |
| C | 19 | 1 | 0 | 1 | 147 | 67 |
| C | 41 | 0 | 0 | 0 | 0 | 64 |

## Health probes and interpretation

All 15 models failed the combined missing-bridge gate. All 15 had zero both-correct timestamp revision pairs out of 64; cross-source paired success was 7/64 for C seed 19 and 0/64 for the other 14. Distractor-trap accuracy ranged from 0/64 to 8/64. Preservation is **NOT_QUALIFIED**, not a failure: independently acquired A/B stages and matched C trajectories were unavailable. Health results, original-task by-kind counts, invalid actions and all cohort breakdowns remain in the metrics JSON.

The row-order comparison directly tests a positional shortcut in C seed 7; matching the last fact in all 768 predictions establishes an exact behavioral equivalence on these examples. It does not establish the model's internal mechanism, prove every model uses the shortcut, or eliminate other cues such as timestamp, source string, wording, or end-of-prompt proximity. The original 80-case accuracy alone can coexist with severe transfer failure and should not be read as broad reasoning competence.

## Proposed next diagnostic, not yet executed

Before further training, obtain the verified FULL checkpoint bundle and frozen source, restore the models, and evaluate matched novel prompts that move the answer-bearing edge independently of row position, timestamps, source labels and query location. Compare predictions against explicit competing heuristics: final fact object, highest timestamp object, fixed answer, nearest-to-query fact, and actual chain traversal. Include paired terminal-edge and query changes, distractors, missing bridges and invalid-action rates; report each seed and each failure. This discriminates mechanisms more directly than another aggregate accuracy. Raw first-token distributions and availability-only/grammar comparisons require checkpoint inference and are not recoverable from these action traces alone. Stage 0 on the earlier S20 checkpoints likewise remains pending; the supplied archive cannot settle it.

## Verification and provenance

The supplied analysis ZIP has SHA-256 `104f9ef675564bc12f10832ee8c56aeb3b7b95a6dff29589971bb5e2a4615bdb`; source commit `6859904a02c8083c4a0ab320c9c4a511899f73e4`, protocol SHA-256 `11be27cf15a268501bb0c552438b2e12c0d02a3756e771160017e9d37043687f`, run `hh-stage1-4bfe3a806e3b`. The inventory and member hashes verified. The analyzer reconstructed the evaluation suites, independently rescored recorded raw actions for **255 reports / 324,720 episode evaluations**, audited **35,520 update rows**, and reproduced all gate receipts exactly. This is replay of recorded outputs, not fresh model inference. The supplied archive is analysis-only: the full checkpoint ZIP and final status were not supplied, so checkpoint restore, regeneration of logits, and final dual-export completion cannot be certified here.

The ZIP is **121,196,749 bytes** (121.2 MB decimal, 115.6 MiB), containing 121 members and 739,511,710 expanded bytes. Fifteen `final_eval.json` files account for 120,024,620 compressed bytes (99.0%) and 732,551,200 expanded bytes: repeated prompts, episodes and full action traces dominate the size, rather than model weights. The original ZIP remains at `C:/Users/ankit/Downloads/ELENDRIL_HITCHHIKER_ANALYSIS.zip`, outside the results folder. The curated metrics preserve per-job and failure details without copying the raw archive.

For a smaller future export, emit a default compact summary containing provenance, cohort and per-seed metrics, gates, failures and audit hashes; make raw episode traces an optional separately retained evidence bundle. Deduplicate shared worlds and prompts by stable identifiers and store model-specific actions once. Keep every failure row and enough hashes and metadata to reconnect summaries to the original evidence; do not sacrifice replayability silently.

To reproduce the recorded-output audit from the repository root, with the original archive available locally:

```powershell
python -m tools.analyze_elendril_hitchhiker_results --zip "C:/Users/ankit/Downloads/ELENDRIL_HITCHHIKER_ANALYSIS.zip" --output "<output>/METRICS.json"
```

The analyzer reconstructs the 13 suite cohorts, original validation cases and health probes and checks recorded actions and receipts; it does not perform fresh inference. The analyzer also reproduces the row-order shortcut audit by parsing the C seed 7 raw recorded prompts and comparing answers to the final serialized fact. The reproducible analyzer output is the principal source for the tables; the full per-episode evidence is in the external ZIP.
