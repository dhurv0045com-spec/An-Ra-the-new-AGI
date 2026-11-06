# Next experiment decision

**Update after ELD-DEV-002.** The paired-sampling comparison below has been executed. The clean associative model passed on the original two-fact surface and showed uneven robustness to aliases and distractors. The next study should preregister a capacity-matched dense comparator and lexical-relation/copy-gate ablations on a larger generator with disjoint latent worlds, variable distractor counts and independent key/separator aliases. Measure exact + EOS and paired query accuracy by world and cost. Keep staged transfer gated on separately qualified skills. See the [executed comparison](ASSOCIATIVE_RESULTS.md).

## Historical decision before ELD-DEV-002

## 1. Test whether training preserves the query-swap signal

The read-only diagnostic places the first bottleneck at in-sample acquisition. Each clean seed attained exactly the positional baseline, 84/168, with 0/84 both-correct training swaps. The next cheapest discriminating intervention is matched paired sampling. It changes batch co-occurrence while preserving the symbolic representation, task family, training exposure and objective. The hypothesis is that presenting both query views with identical facts within each update helps the CE trainer learn query conditioning.

Prospective draft. Build a frozen multiset of complete swaps per seed and epoch, using eight pairs per 16-row update. Produce the treatment stream by keeping pair members together and the control stream by shuffling the exact same epoch multiset with pair membership hidden from batching. Use full matched epochs or truncate at a shared completed epoch. Match initialization, row exposure histogram, AdamW, CE, LR, total tokens, updates, probe work and generation cap. Record realized pair co-occurrence in both arms. Before training, assert that both facts and answer lengths match within each pair, the queries and targets differ, and all sample indices fit the frozen family. Preserve the old registered ELD-DEV-001 results as context.

Use seeds 7, 19 and 41 initially with one CPU thread and a cap of 400 updates per run. A 350-update equivalent six-arm binding run on this host would take about 72 seconds of measured in-process time based on ELD-DEV-001. Use a three-minute per-process stop and preserve every checkpoint and prediction. Set the exact epoch count before running and use one held-out snapshot schedule for both arms. Primary endpoints are training both-correct and held-out both-correct. The first gate is at least 60% in-sample both-correct for each treatment seed. Then require at least 60% development both-correct, at least 75% exact EOS and a positive paired improvement of at least 20 percentage points in development both-correct in each seed. No improvement, a failed manipulation check or failure at in-sample acquisition falsifies or invalidates the corresponding staged claim. Seven development worlds limit the result to a screen. Later estimate cluster uncertainty on a newly generated panel before a claim about transfer.

If batching leaves the floor intact, preregister a distinct two-choice conditional ranking term and test it against CE under matched paired batches and compute. The inherited Cymek challenger takes three negatives and has a separate protocol. A two-value task needs its own explicit one-distractor loss and gradient-sign test. Maintain free-generation scoring.

## 2. Recover historical S5 Output in parallel

Recover the original checkpoint-bearing Output. A verified original 16-state recovery opens the read-only METRIC-RES protocol. The v12 partial archive is insufficient and historical evidence rejects repeating broad softmax or physical vocabulary sweeps.

## 3. Advance to A/B/C only after formation

Qualify three individually learnable skills under disjoint world groups. Preregister A retention after B with an A-only drift control, B acquisition against fresh B, unseen A+B compositions with separate primitive prerequisites, and C after A/B versus fresh C. For forward transfer, compare matched cumulative training and evaluation work first, then show a secondary curve at equal C exposure. Compare no protection, fixed replay and bounded memory interventions only after the unprotected arms form both skills. Use matched seeds and fixed task/cost identities. The longitudinal scorer calculates paired cluster contrasts once those prerequisites are met.
