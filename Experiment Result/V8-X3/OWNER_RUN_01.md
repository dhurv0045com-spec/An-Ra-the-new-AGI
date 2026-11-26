# ELENDRIL V8-X3 | Owner run 01 | 2026-11-26

**Decision. Keep S50 closed.** All 45 registered S20 fits finished. The D3-Hybrid recipe improved exact answers on held-out worlds across all three seeds in the integration comparison. The experiment also exposed two obstacles to explaining that gain. D2 and D3 trained on substantially different examples, and the training forward pass can use future target bytes. The released checkpoints can produce correct, structured outputs on familiar problems. Transfer to fresh problems is the failure that now needs explaining.

## What ran

The submitted FULL ZIP is 3,353,723,964 bytes. SHA-256 `dac5a275c23e6ada643aac0fad4f2da5ac0a8837418a11a3cd49d5d179387277`. Campaign source `7654543bfa882bf3f9afe9fe1eae940b0d18b515`. All 515 manifest-listed members verified. The 45 per-job receipts agree with the aggregate job file, registered specs, final checkpoint pointers, content digests and both held-out result files. Every journal contains steps 1–1024 in order. The 46,080 entries include 18 reported AMP skips and zero nonfinite total losses. Summed per-job wall time was 12,746 seconds (3.54 worker-hours). This sum excludes separate evaluation and export work and does not measure elapsed Kaggle session time.

There were 15 S20 configurations, each at seeds 7, 19 and 41. The confirmation lock admitted 45/45. Each model saw the same 200 confirmation and 200 out-of-distribution (OOD) cases, 20 per family per split. Development used 120. The 12 conditional S50 jobs used zero GPU seconds. The separate R0 ledger still credits 49/192 historical X2 obligations and lists 125 mandatory and 18 conditional as pending. Its next runnable ID is `X1-U1-S20-U-4-standard-s19`. These 45 receipts discharge the S20 design registry only.

## Exact answers

Values are percentages, averaged equally over 10 families and three seeds. Differences below are percentage points. Shared worlds make pooling model-case rows as independent trials inappropriate.

| Stage and arm | Development | Confirmation | OOD |
|---|---:|---:|---:|
| S0 R / D2 | 11.39 | 10.50 | 6.33 |
| S0 R-noedge / D2 | 10.83 | 9.83 | 6.33 |
| S1 P / D3-Hybrid | 18.89 | 12.17 | 11.17 |
| S1 P / NoAux | 16.94 | 14.67 | 10.67 |
| S1 P / NoSurgery | 16.39 | 12.67 | 9.67 |
| S1 R / D3-Hybrid | 18.33 | 15.17 | 9.67 |
| S1 R / NoAux | 14.72 | 14.50 | 10.33 |
| S1 R / NoSurgery | 14.72 | 11.33 | 9.17 |
| S2 P-R-Iter / D3-Hybrid | 13.33 | 11.33 | 10.50 |
| S2 additive-memory control | 15.28 | 13.83 | 8.67 |
| S2 fixed-halting control | 15.83 | 13.67 | 9.67 |
| S2 copy-disabled control | 16.67 | 13.00 | 9.67 |
| S2 relation-disabled control | 14.44 | 14.00 | 9.33 |
| S3 P-R-Iter / D2 | 9.72 | 8.17 | 4.67 |
| S3 P-R-Iter / D3-Hybrid | 17.50 | 16.67 | 10.83 |

**The recipe gain repeated.** With the S3 architecture held fixed, D3-Hybrid gained 4.5, 8.5 and 12.5 points on confirmation for seeds 7, 19 and 41. On OOD the gains were 5.5, 6.0 and 7.0 points. **The architecture result was mixed.** In S2 all four controls beat the integrated architecture on confirmation. The integrated architecture led all four on OOD by 0.83–1.83 points. The gate recorded a pooled development deficit of 3.89 points versus P and an absent-answer deficit of 3.42 points. It pooled different stages and recipes by architecture, so those gate values measure the implemented admission rule only.

All 45 fits scored **0/20 on grid transformations, compositional instructions and language–math transfer**, separately on both held-out sets. All scored 0/12 on those development families too. Novel-operation induction had the highest pooled family score (37.4% confirmation, 21.6% OOD). This metric can credit appropriate responses to ambiguous cases. It does not by itself establish acquisition of a hidden rule. Every family's underlying counts are in the metrics.

Four selected **seed-19** checkpoints received the five-mode probes. All had zero persistent-state accuracy and zero reset effect; all scored zero on the tool-free transfer probe and 100% with the supplied oracle. All four scored zero on the cross-domain probe. Post-adaptation accuracy ranged from 25% to 50% on eight C01 queries. A small post-update score alone cannot establish that updating improved the model. The independent oracle scored 100% and the best archived inexpensive shortcut scored 11.5% on the separate 200-case development preflight. This was a different sample from the 120-case training-job development evaluation.

## What the checkpoints reveal

**Familiar versus fresh problems.** Using the pinned implementation and FP32 CPU inference, I loaded the final S3 D3-Hybrid seed-7 weights. The model reproduced the exact emitted strings for the first 10 archived development mistakes. Using a fixed RNG (20261126), I then sampled three cases per family from training IDs actually presented to that model and three per family from development. The same public decoder and scorer gave **28/30 on exposed training cases and 4/30 on fresh development cases**. In the three families with zero held-out performance the model solved **9/9 familiar and 0/9 fresh** cases. A global inability to emit or score these formats cannot explain that pattern. The probe covers one checkpoint and 60 cases. Memorization, inadequate rule coverage, and brittle representations are still competing explanations.

**The D2 comparison changes the data.** Rebuilding the production sampler at all three registered order seeds showed that each 25-batch cycle gave D2 200 presentations covering **100 distinct cases**. D3-Hybrid gave 400 presentations covering **329, 319 and 323** distinct cases out of 400. The training loop recycles these prepacked batches. The D2 branch duplicates paired cases and drops the independent cases; changing the objective also changes the effective minibatch size and training exposure. The observed S3 contrast identifies a recipe effect. It cannot assign that gain to gradient surgery or any single objective term. See `_resolve_batch_order_and_offset` in `experiments/x3/d3_objectives.py` and the update loop in `experiments/x3/module.py`.

**Future answer bytes influence earlier predictions during teacher forcing.** The packer appends the correct target. The slot reader in `experiments/x3/models.py` can attend to every non-padding token, and its pooled output is added to each byte-generation position. I tested the trained S3 Hybrid checkpoints at all three seeds. For one development ANSWER case per family, I kept the prompt, the literal ANSWER prefix including its colon, the action position and the sequence length fixed. I replaced only the subsequent payload with equal-length characters. At the position predicting the first payload byte, the causal backbone activation remained identical on all 30 comparisons. The complete model’s byte logits changed by more than 0.00001 on 5/30 comparisons; its action logits changed beyond that tolerance on 17/30. Identical-suffix duplicate controls had zero byte-logit difference. The largest byte-logit change was 0.05364 and no first-byte argmax flipped. This establishes a target-dependent training path. The owner’s held-out scores used free generation from the public prompt. The intervention does not quantify how much the training defect cost those scores.

**Two summary contrasts are miswired.** In the pinned source, `X3_PAIRED_CONTRASTS` supplies one objective to both sides of the R Hybrid-vs-NoAux and Hybrid-vs-NoSurgery comparisons. Both reported zero by comparing each control with itself. Using the actual S1 receipts gives +3.61 points on development for both, +0.67 and +3.83 on confirmation, and −0.67 and +0.50 on OOD respectively. Using stage as part of the key also matters. The S2 and S3 copies of the nominally identical hybrid had different endpoints. Their logged byte losses first differ at updates 2, 2 and 11 for seeds 7, 19 and 41. The available journals do not isolate why. The discrepancy warrants a same-seed repeatability check before interpreting small effects. Calibration numbers also use confidence from the auxiliary action head alongside correctness of generated bytes; those are two different outputs.

## Next decisive experiment

1. **Repair and gate the training path.** Restrict slot/pointer reads to public prompt positions for a fixed evidence read, or implement an explicitly causal per-position read. Specify which one the decoder uses at each generation step. Check full teacher-forced versus prefix-only byte logits with identical available context, future-suffix invariance for action and byte logits, padding invariance, and exact replay on the unchanged historical scorer. Use identical-input and deliberately leaky controls. Fix the reporting join to require distinct stage/architecture/objective/seed keys. Preserve this run as historical evidence.
2. **Separate coverage from the objective at S20.** Hold the architecture, three initializations, token budget, optimizer, training worlds, validation worlds and decoding fixed. Cross low/high distinct-world coverage with D2/D3 losses using explicit manifests and the same number of examples per step in every arm. Within each coverage condition feed both objectives the exact same ordered batches and target tokens; vary only the prescribed auxiliary/surgery terms. Preserve linked-pair structure in both schedules. Record realized unique IDs, repeats, eligible tokens, skipped updates and compute. Include a same-seed duplicate and a small legacy-versus-causal-read bridge on identical data to estimate the repair’s effect.
3. **Use the existing failures as diagnostic tasks.** For C03, C04 and C10, compare exposed training worlds, unseen worlds rendered in the same format, and the same latent problem re-rendered in a new format. Record first wrong output byte, parse validity, operation choice, teacher-forced likelihood and free-generation exactness. Pair cases by latent problem and compare across seeds. Once this report has informed design decisions, use fresh locked confirmation worlds for promotion. Favor broader coverage if it helps both objectives; credit D3-specific learning only if a matched gain survives on fresh worlds. Keep S50 closed until the integrated arm beats its declared controls on a preregistered held-out endpoint.

**Causal conclusion.** The reporting bug, the exposure difference and the future-suffix dependency are demonstrated. The extent to which each training issue caused failure on unfamiliar worlds is unresolved. The next GPU allocation should distinguish those explanations.

[Compact metrics, per-job checkpoint hashes, sampling coverage and local probe results](OWNER_RUN_01_METRICS.json)
