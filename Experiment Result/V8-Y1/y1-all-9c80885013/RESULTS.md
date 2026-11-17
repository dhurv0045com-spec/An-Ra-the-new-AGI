# V8-Y1 results: candidate treatments failed; benchmark shortcuts limit interpretation

Analyzed **2026-11-18** by Codex with Terra reviews. Run `y1-all-9c80885013`.

**Conclusion:** the experiment completed its selected stages, but demonstrated neither a successful new recipe nor a benefit from scaling S3 to S20 under the tested training setup. Balanced coverage greatly increased unnecessary abstention; the models were largely insensitive to changed queries in the dual-chain test. More importantly, the generator contains an unregistered timestamp shortcut that solves the entire primary panel, and querying places the answer in the final public fact. These findings make fixing the experiment's ability to distinguish mechanisms the next priority.

This is an analysis of the supplied evidence and executed source. No local neural inference or training was performed. Old evidence and code remain unchanged. Archive integrity, replay consistency, scientific success and durable checkpoint delivery are separate claims.

## What ran, and what did not

| Stage | Supplied evidence | Status and interpretation |
|---|---|---|
| D0/D1 | All 15 historical Hitchhiker S3 models; Y1 raw outputs and diagnostic logits | Executed. Historical reproduction receipts match original aggregate counts; no local fresh restore performed here. |
| T0 | Seed 7, 256 updates; recorded 127/128 memorized training families, final loss 0.25170 | Recorded PASS (99.22% vs 95%). T0 raw evaluation predictions are absent, so its accuracy cannot be independently replayed from these ZIPs. |
| T1 | Four arms × three seeds; 12 fresh S3 training jobs | All completed 2,368 updates; every candidate failed the registered screening gate. |
| T2 | Confirmation data is prepared, no selected winner | Not run. A confirmation fixture in an archive is not a completed confirmation experiment. |
| T3 | F00/F11 × three seeds at S20; six saved S3 models re-evaluated on the same separate follow-up split | Executed as failure diagnosis, not promotion of F11. No consistent capacity benefit. |
| Group-loss continuation | Bottleneck admission receipt | Not admitted: constrained answer/pair prerequisites failed. |
| D2 / T4 / larger models | Probe dataset only; no execution evidence for probes/patches/preservation training/60M–90M | Not established as executed. A prepared dataset is not a result. |

There are **40 recorded job directories**, comprising 19 new training jobs (one T0, twelve T1, six S20), 15 old-model diagnostic jobs and six S3 follow-up re-evaluations. This is not 40 new trained models. There are 39 full prediction bundles. Each has 11,840 Y1 cases plus 80 historical-continuity cases. The raw-action audit described below covers those bundles; the additional D0 historical reproduction is represented by aggregate receipts.

The budget ledger records about **241.04 wall minutes** for the combined run: diagnosis 42.34, core 96.78 and follow-up 100.22 minutes, with orchestration overhead. Do not add the parent `all` duration to its child stages. Two GPUs were recorded; wall time is not the same as measured busy GPU-hours.

## Controlled S3 screen

F00 repeats static worlds with imbalanced availability coverage. F10 changes route-world freshness. F01 changes the coverage mixture. F11 combines both. All use the redesigned rendering and post-query supervision; F00 is not the earlier Hitchhiker B0.

| Arm | Data / coverage | Seed 7 | Seed 19 | Seed 41 | Mean primary |
|---|---|---:|---:|---:|---:|
| F00 | Static / imbalanced | 41.89% | 22.46% | 30.66% | 31.67% |
| F10 | Fresh / imbalanced | 9.28% | 17.09% | 27.34% | 17.90% |
| F01 | Static / balanced | 7.42% | 1.46% | 9.28% | 6.05% |
| F11 | Fresh / balanced | 5.08% | 4.39% | 6.35% | 5.27% |

The primary is raw exact terminal success on known-evidence answerable cases, macro-averaged across hops 1–4 and base/row-permutation presentations: 512 families, two presentations each, 1,024 case evaluations per model. It is not overall accuracy pooled with easy abstention or query conditions.

Against F00, freshness alone lost **13.77 percentage points**, balanced coverage alone lost **25.62 pp**, and their combination lost **26.40 pp** on average. Every candidate lost at every paired seed. The factorial interaction is +12.99 pp, which means the combined loss is less than the sum of the two losses; it does **not** make F11 successful. These are observed arm contrasts under this dataset and optimizer, not universal judgments against fresh data or balanced supervision. The generator also uses different initial corpus seed tags for static and fresh arms (`y1_data.py:1491–1497`), and fresh tags include the arm name. The three model seeds share those arm-specific corpus realizations; they are not three independently sampled data replicates. Therefore corpus draw and refresh schedule are not perfectly isolated. A tighter next comparison should share epoch-zero worlds and use separate controlled refresh seeds thereafter.

The registered gate required ≥10 pp mean gain, positive gain at every seed, ≥10 pp advantage over the strongest registered shortcut, per-seed causal-pair/action guards, and no original-kind regression over 5 pp. The declared strongest shortcut was last-row object at 24.80%. All candidate mean primaries were below that rule; all failed the edge/query pair requirements. Coverage arms improved unavailable-case abstention but did not preserve reliable direct answering. Some original-kind regressions also exceeded the limit: F10 seed 7 memory −85.55 pp, F01 seed 19 memory −34.38 pp and F11 seed 19 known-value −5.08 pp. These are comparative preservation-suite scores, **not a sequential forgetting experiment**.

The recorded family-bootstrap intervals for candidate minus F00 means are entirely negative: F10 [−16.05, −11.43] pp, F01 [−27.60, −23.63] pp, F11 [−28.58, −24.35] pp. They condition on three trained seeds and this generator, cluster related variants by family, and do not establish a population-level result over all initializations or tasks.

## What the failures actually look like

On 512 answerable base cases per seed, F11 immediately abstained on **461, 460 and 431** cases (90.04%, 89.84%, 84.18%). F00 abstained on only 39, 47 and 32. Thus coverage changed behavior substantially, but towards abstaining even when the public facts were sufficient.

Changing only the queried chain in the same two-chain graph should change the answer. F11 instead emitted identical complete recorded raw output sequences on **256/256, 256/256 and 255/256** query pairs (seeds 7/19/41). All three had **zero both-correct pairs**. This is direct behavioral insensitivity on this panel, not proof every internal representation ignores the query. It includes both repeated wrong answers and repeated abstentions; [BEHAVIOR.json](BEHAVIOR.json) separates them and preserves concrete case IDs.

The conditional answer diagnostic does not support the proposed simple action-competition rescue. For F11, the correct symbol ranked first among A–P on only 23.63%, 21.29%, 24.22% of answerable base cases. Constrained dual-query both-correct rates were 0%, 0%, 0.39%; constrained edge rates were 0.78%, 1.17%, 10.55%. Removing the ability to abstain therefore does not reveal consistently correct answers or causal behavior. Answer-set probabilities are conditional on restricting the vocabulary; they are not raw answer probabilities. No claim of a uniquely identified executive or binding circuit follows.

## S20 did not rescue the tested recipe

S3 and S20 below are evaluated on the **same capacity-follow-up split**, not compared across development and follow-up data.

| Arm | Seed | S3 primary | S20 primary | S20 minus S3, pp |
|---|---:|---:|---:|---:|
| F00 | 7 | 40.72% | 9.96% | -30.76 |
| F00 | 19 | 23.83% | 24.22% | +0.39 |
| F00 | 41 | 26.66% | 16.99% | -9.67 |
| F11 | 7 | 4.20% | 0.29% | -3.91 |
| F11 | 19 | 3.32% | 3.12% | -0.20 |
| F11 | 41 | 6.54% | 3.42% | -3.12 |

F00 mean falls from **30.40% to 17.06%** (−13.35 pp); F11 falls from **4.69% to 2.28%** (−2.41 pp). F00 seed 19 is the sole positive paired primary change, +0.39 pp. Every S20 model has zero both-correct dual-query pairs. This does not establish that additional capacity cannot help: size changes width/depth, the learning rate and update budget were held fixed rather than independently optimized per scale, and compute was not matched. It establishes that this S20 configuration/training schedule did not solve the measured failure.

The source defines S3 as 3,084,800 parameters (4 layers, width 256, 4 query/1 KV heads, FFN 768, context 512); S20 is 19,871,616 (7 layers, width 512, 8 query/2 KV heads, FFN 1,408, context 640). Both use UTF8 byte vocabulary 260, tied embeddings and the existing dense decoder. Training is FP32 AdamW, betas 0.9/0.95, epsilon 1e−8, decay 0.1 with existing group exclusions, clip 1, batch 16/microbatch 2, peak LR 1e−4 and cosine decay to 0.1× peak without warmup. Each full job logs 2,368 updates, 37,888 decision-state examples and 75,776 supervised action/EOS tokens. Input lengths and actual compute can differ; equal target exposure is not equal compute.

## New findings: the instrument admits shortcuts

### 1. Timestamp rank completely reveals the primary answer

`anra_v5/elendril/y1_data.py:339–350` sorts sampled timestamps, assigns the earliest ones to successive chain edges, and gives later timestamps to distractors. Consequently this public-input rule needs no query traversal:

> Read the requested hop count h; sort the public facts by timestamp; output the object of the h-th earliest fact.

Recomputing that rule against the **actual archived fixtures** yields **1,024/1,024 primary cases correct in each of development, confirmation and capacity follow-up**—3,072/3,072 across the three panels. The shared worlds are not 3,072 independent graph families. Row shuffling cannot defeat this cue; shifting timestamps preserves their rank. This was not among the registered shortcut baselines. It is a post-hoc audit finding, not a newly preregistered gate.

The rule uses public hop count and timestamps and ignores which node was queried. It proves a shortcut exists in the fixture; it does **not** prove the neural models use that rule. Their low scores do not make the design safe for future higher scores. Even 100% on this primary panel would not uniquely identify graph traversal. The original plan specifically required timestamp rank to vary independently of answer-edge identity; the implementation did not meet that requirement.

### 2. Querying always returns the terminal answer edge

`y1_data.py:352–364` removes the terminal edge, rather than varying the missing bridge's depth. The response contains that one terminal edge. The episode appends it to the public events, and the renderer preserves event order (`cognition.py:286–291`, `hitchhiker_data.py:144–165`). A query-then-copy-the-final-object policy can therefore succeed without integrating a newly supplied internal bridge.

Across the actual main-base query-needed plus available missing-bridge fixtures, **512/512 cases in each split** return one event whose object is the gold final answer. The 100% query-needed scores of several balanced-coverage models are compatible with this shortcut. This audit demonstrates the competing explanation; it does not establish those models' exact internal algorithm.

### 3. Changed answers concentrate on A/B

The replacement-value selector chooses the first permitted letter from A–P (`y1_data.py:601`, `:773`). Development main terminal-change golds are **A:441, B:71** out of 512; the edge-panel changed golds are **A:217, B:33, C:4, D:2** out of 256. Confirmation and capacity fixtures show similar skew (exact counts in BEHAVIOR.json).

These counterfactual panels are not balanced arbitrary-value binding tests. Preserve their observed failure, but balance changed values in the next frozen fixture and compare both endpoints of each pair. Do not relabel the current measurements after the fact or silently replace the registered denominator.

### 4. Train/held-out exclusion was not actually assured by the cited check

`y1_data.py:1447–1450` decodes framed sample content including the supervised target and compares it with a set of bare held-out prompts. That comparison does not establish the promised prompt exclusion. The broader custody audit compares evaluation suites, not the generated training decision states.

This is a defect in the assurance, **not proof that train/test overlap occurred**. This review did not reconstruct all training streams to quantify actual overlap. Before the next run, hash prompt-only token prefixes at the explicit target boundary and separately compare normalized instantiated fact/query/availability states, including post-query states. Keep all variants/decision states from a family in the same split; report topology overlap honestly rather than treating a small chain grammar as structurally disjoint.

## Historical checkpoints and comparison with Hitchhiker

All 15 D0 receipts report exact agreement with published per-hop and original-task **aggregate counts**, with no listed mismatch blockers. The earlier C seed 7 signature (62/256, 0/256, 16/256, 256/256 by hop; 69/80 original) is reproduced in its receipt. That is consistent with use of the intended checkpoints, but does not independently certify per-case identity or local fresh inference here.

Old Hitchhiker models evaluated on the new Y1 primary:

| Historical arm | Seed 7 | Seed 19 | Seed 41 | Mean |
|---|---:|---:|---:|---:|
| B0 | 1.37% | 4.79% | 7.13% | 4.43% |
| A | 0.59% | 1.17% | 6.54% | 2.77% |
| B | 2.73% | 3.03% | 2.25% | 2.67% |
| AB | 3.61% | 3.52% | 10.94% | 6.02% |
| C | 9.38% | 9.18% | 9.47% | 9.34% |

Historical C's 9.34% mean on this new endpoint is distinct from its old 15.79% entity-permutation score. The suites and denominators differ; do not present their difference as improvement or regression on a fixed benchmark. Y1's new F00 is better on this endpoint, but several training/data choices changed together, so this cross-generation comparison cannot isolate the causal contribution of any one change.

## Independent verification and what remains unverified

- Both archive manifests and listed member hashes passed. Of 117 shared members, 115 are byte-identical; only the inventory-specific `BUNDLE_MANIFEST.json` and `SHA256SUMS` differ.
- **39 prediction bundles / 464,880 recorded episodes** were replayed from raw tokens through the public environment, with **zero outcome-row mismatches**. Available summary fields and stored family clusters also match.
- Registered development heuristics were independently recomputed; the T1 gate and its family-bootstrap outputs match exactly: **FAIL, no winner**. Confirmation/follow-up registered heuristics were not recomputed by this audit; the new timestamp and query-payload cue checks cover all three suites.
- **19 update logs / 42,880 updates** have contiguous expected indices and finite logged numeric fields. This checks logged evidence, not every intermediate gradient/tensor.
- All numeric arrays in **39 NPZ logit sidecars** are finite, with expected 1,024×260 base-logit shape. **39,936 first-token argmax comparisons** match recorded base outputs, with none missing. This verifies consistency of saved logits/actions, not regeneration of logits from weights.
- Small negative controls detected both a changed stored correctness flag and a changed raw token. No broad training/test suite was rerun for this analysis.

| Archive | SHA256 |
|---|---|
| SUMMARY | `9844ad2015c6c3c9b9bf8ffd8864b8bded9d37ec001c89bc73ac4f64c8cad1bf` |
| EVIDENCE | `4f76ac0d3a1fc67fac2705325c0909f791325696f9a1162c9e86c783c717173f` |

A material reporting distinction: the shipped campaign's `_replay_phase_evidence` (`tools/elendril_y1_campaign.py:315–331`) increments a receipt count and checks prediction-file existence. It does not replay actions or compare metrics. Therefore the archive's `independent_replay: PASS, verified_jobs: 40` was not itself proof of replay. The existing offline analyzer aggregates stored `correct` flags. The new audit instead reparses recorded raw tokens, runs them through the public environment and recomputes outcome rows before deriving metrics. This is independent of stored outcome flags, but still uses the source evaluator semantics; it is not a separately implemented formal oracle.

The Y1 FULL checkpoint archive and external FINAL_STATUS were not supplied or found in the top-level Downloads listing. Model tensors, optimizer state and fresh-session restoration therefore cannot be verified from these two files. The profile receipt reports S3 recovery verified but **S20 recovery_verified=false**, despite overall profile status PASS. Treat that as an unqualified S20 recovery check, not evidence of corrupted weights. D2 layer probes/causal patches were not established as executed. Neither successful byte checks nor action replay repairs generator confounds.

## Complete per-job behavioral comparison

Percentages are from the respective development (T1) or follow-up (S20/S3-ref) split. Edge = all three original/fixed/moved cases correct in a 256-family panel. Dual = both queries correct in 256 families. Query success and unavailable abstention use the source-defined solvability cohorts, which include preservation cases; they are not identical to a main-only denominator. False-abstain is immediate abstention on 512 answerable base cases. Same-output dual counts identical full recorded raw token sequences, including abstentions.

| Job | Primary | Edge | Dual | Query success | Unavailable abstain | False-abstain | Same-output dual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1-S3-F00-s19 | 22.46% | 0.00% | 0.39% | 41.58% | 47.46% | 47/512 | 242/256 |
| T1-S3-F00-s41 | 30.66% | 8.59% | 0.78% | 60.76% | 32.07% | 32/512 | 245/256 |
| T1-S3-F00-s7 | 41.89% | 13.67% | 0.39% | 66.36% | 37.70% | 39/512 | 226/256 |
| T1-S3-F01-s19 | 1.46% | 0.00% | 0.00% | 53.12% | 98.98% | 475/512 | 243/256 |
| T1-S3-F01-s41 | 9.28% | 0.39% | 0.00% | 100.00% | 94.80% | 383/512 | 255/256 |
| T1-S3-F01-s7 | 7.42% | 3.12% | 0.00% | 100.00% | 96.72% | 399/512 | 253/256 |
| T1-S3-F10-s19 | 17.09% | 8.59% | 0.00% | 30.99% | 48.95% | 52/512 | 248/256 |
| T1-S3-F10-s41 | 27.34% | 5.08% | 0.00% | 66.32% | 21.45% | 18/512 | 252/256 |
| T1-S3-F10-s7 | 9.28% | 0.39% | 0.00% | 12.11% | 77.58% | 207/512 | 251/256 |
| T1-S3-F11-s19 | 4.39% | 0.00% | 0.00% | 90.45% | 98.75% | 460/512 | 256/256 |
| T1-S3-F11-s41 | 6.35% | 3.91% | 0.00% | 100.00% | 96.05% | 431/512 | 255/256 |
| T1-S3-F11-s7 | 5.08% | 0.39% | 0.00% | 100.00% | 98.59% | 461/512 | 256/256 |
| S20-S20-F00-s19 | 24.22% | 4.30% | 0.00% | 35.59% | 51.80% | 79/512 | 251/256 |
| S20-S20-F00-s41 | 16.99% | 5.08% | 0.00% | 27.69% | 58.71% | 114/512 | 246/256 |
| S20-S20-F00-s7 | 9.96% | 0.00% | 0.00% | 16.15% | 70.43% | 177/512 | 255/256 |
| S20-S20-F11-s19 | 3.12% | 0.00% | 0.00% | 41.88% | 99.10% | 473/512 | 252/256 |
| S20-S20-F11-s41 | 3.42% | 2.34% | 0.00% | 99.39% | 98.59% | 466/512 | 256/256 |
| S20-S20-F11-s7 | 0.29% | 0.00% | 0.00% | 18.66% | 99.65% | 505/512 | 249/256 |
| S3-ref-S3-F00-s19 | 23.83% | 0.78% | 0.00% | 41.54% | 45.04% | 60/512 | 240/256 |
| S3-ref-S3-F00-s41 | 26.66% | 8.59% | 0.00% | 58.25% | 30.23% | 33/512 | 243/256 |
| S3-ref-S3-F00-s7 | 40.72% | 17.58% | 0.39% | 69.75% | 34.34% | 51/512 | 235/256 |
| S3-ref-S3-F11-s19 | 3.32% | 1.17% | 0.00% | 89.80% | 97.46% | 470/512 | 254/256 |
| S3-ref-S3-F11-s41 | 6.54% | 2.73% | 0.00% | 100.00% | 94.92% | 438/512 | 256/256 |
| S3-ref-S3-F11-s7 | 4.20% | 0.00% | 0.00% | 100.00% | 98.75% | 469/512 | 256/256 |

The evidence preserves every recorded failure. METRICS.json retains per-seed outcomes, guards, raw-replay discrepancies if any, training-log checks and sidecar checks; BEHAVIOR.json retains diagnostic summaries, cue counts, label histograms and paired-output examples. Raw archives remain outside Git.

## Next action: repair the discriminating test before another training matrix

1. **Preserve these results and obtain the Y1 FULL checkpoint bundle.** Do not spend another training campaign merely to regenerate existing models. Verify its source/protocol and manifest identities against the receipts, then restore the existing F00/F11 seeds for frozen diagnostics.
2. **Freeze a new diagnostic fixture under a new version/hash.** Start with 128 known-evidence families per hop (512); for each retain a base plus independently randomized timestamp ranks, row order, and a same-graph changed query with a different correct answer. All 2,048 cases must have independent oracle truth and paired family IDs. Give each graph two disjoint chains so the changed query holds depth fixed. Randomize source labels independently. Reject generated cases that exceed context; do not truncate.
3. **Repair missing-bridge tests.** Use 64 families per hop (256), with available/unavailable pairs. Sample missing edge depth uniformly; for an internal missing edge, include the terminal edge publicly. Randomize returned-evidence placement in the post-query view and retain a matched control with the same facts already present. Measure pair-both-correct and explicit post-query state equivalence. This distinguishes terminal copying from integrating an internal link.
4. **Balance counterfactual values and expand cue controls.** Balance original and changed answers across A–P subject to unequal pair answers. Evaluate timestamp-rank, first/last object, query-blind fixed-position, post-query last-object and oracle controls before neural inference. The independent oracle must get every supported case right. A shortcut rule must not accidentally determine all golds; publish its actual conditional baseline instead of assuming 1/16 chance for every problem.
5. **Repair custody and replay assertions.** Test deliberate train/eval collisions, modified raw tokens, fabricated correct flags, missing sidecars and incomplete checkpoint state. Publish explicit pass/fail receipts for those failures; separate file-existence checks, recorded-action replay and fresh neural inference.
6. **Re-evaluate frozen models first.** Keep raw, constrained-answer and legality controls separate. If query changes still leave outputs unchanged and answer ranks remain weak after cue removal, use one matched intervention aimed at query-conditioned binding (for example, same-graph/different-query paired supervision versus CE on the identical examples) before adding more mechanisms. If value discrimination is robust but action choice fails, revisit the separately controlled group-loss experiment. Do not assume a new head, recurrence or larger model is the cause-specific remedy without this evidence.

Steps 2–4 are a proposed next diagnostic, **not executed or retrospectively substituted for Y1**. A later training campaign needs its own preregistration and compute forecast. The useful outcome of Y1 is a concrete failure map and a better discriminating test, not a claim that the full research direction is disproven or that AGI has been demonstrated.

## Provenance and reproduction

Executed source: `60e63d8025259ab4e579f28bb32f66b4189dc0ee`. Protocol: `3809785b27aa4feca1c0abfc8c81e9ed6dd69e48ff3c4bc0093481a647322cac`. Run: `y1-all-9c80885013`. Review worktree began at `ec12238e`; its Y1 source/parser/evaluator files are unchanged from the executed source (later changes add unrelated infrastructure). User plan snapshot: `GEMINI_V8_Y1_EXECUTABLE_PLAN.md`, SHA256 `30c2ba5864a0e02d6078f10a57367f001c6678f97884825003dc610fe7310b35`.

Archive identities are in METRICS.json. SUMMARY is **123,196 bytes**; EVIDENCE is **64,957,331 bytes**. The small summary meets the requested compact-delivery goal. The larger separate evidence bundle contains fixtures, all predictions, training logs and binary logit sidecars; it is not 65 MB of model weights. No source ZIP or model weights are committed with this report.

From repository root:

```powershell
python -m tools.audit_elendril_y1_results --evidence "C:/Users/ankit/Downloads/ELENDRIL_V8_Y1_EVIDENCE.zip" --summary "C:/Users/ankit/Downloads/ELENDRIL_V8_Y1_SUMMARY.zip" --output "<scratch>/METRICS.json"
python -m tools.analyze_elendril_y1_behavior --archive "C:/Users/ankit/Downloads/ELENDRIL_V8_Y1_EVIDENCE.zip" --output "<scratch>/BEHAVIOR.json"
```

Reproduction reads existing records and regenerates summaries/cue audits. It does not retrain models, require the FULL checkpoint ZIP, or claim fresh logits were computed.
