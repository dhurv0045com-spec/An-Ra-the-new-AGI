# V7.1 core: verified delivery, narrow learning, brittle generalization

Analyzed 2026-11-09 (Asia/Calcutta). Run `20c80f66e76e4b589aed2053afbddffc`; source `6104ffd7134cb4cf4b3f340bd060d24a1e2dd200`. **All nine jobs completed. CAUSE UNRESOLVED for the underlying learning failures.** This is useful positive and negative evidence, not a capability promotion.

## Custody and reproduction

The original owner-supplied analysis ZIP was verified byte-for-byte: 22,696,077 bytes, SHA256 `5f198fbed35c022b2c114e8cd11eb747bcba73c06916f79169e3f61169ff62c5`. Its complete inventory, CRCs, manifest SHA and SHA256SUMS matched the supplied final-status receipt. Original ZIPs and full receipts are kept outside the current Git tree; [PROVENANCE.json](PROVENANCE.json) retains only essential hashes, source identities and verification status. Reconstruction source files matched the pinned Git blobs after LF normalization. The reconstructed suite matched the raw manifest: **1,120 case records over 288 unique worlds**, reused across interventions and models.

Independent offline replay passes for **55 suite reports plus four positive-control records**, including prompts, actions, outcomes and resource accounting. Six derived Brier fields differ across runtimes by at most `2.7755575615628914e-17` (one binary64 ULP). The offline adapter declares a two-ULP allowance only for derived Brier scores; entropy retains the historical verifier's bound. Original bytes, confidence values, raw actions, counts and all other checks remain strict. This is not regenerated neural inference.

The FULL ZIP is **reported** verified by Kaggle: 1,958,029,910 bytes, SHA256 `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19`. It was not supplied locally. We therefore cannot independently restore tensors, recompute logits or verify fresh-VM numerical continuation. Preserve/download that archive before disposing of Kaggle output. Both physical T4s have saved positive-learning and fresh-process same-device recovery PASS receipts; four production-shaped recovery probes also passed.

Reproduce the compact audit from the repository root:

```text
python -m tools.analyze_elendril_v71_results --zip PATH_TO_OWNER_ANALYSIS.zip --final PATH_TO_OWNER_FINAL_STATUS.json --output NEW_OUTPUT --v7-zip PATH_TO_V7_ANALYSIS.zip --data-dir PATH_TO_MATCHING_DATA
```

The two final arguments are optional: V7 supplies receipt comparisons; the data directory supplies the canary/monitor datasets for four additional learning-control replays. The 55 suite reports are reconstructed without externally supplied data. [METRICS.json](METRICS.json) retains per-cohort counts, seed outcomes, decoder contrasts, failures, provenance and compute; its trajectories field retains epoch 0/1/2/4 observations. [FOCUSED_CASES.json](FOCUSED_CASES.json) records the concrete failures below. Original plots and duplicate CSV exports are excluded from this results folder.

## What was learned, and what was not

Each confirmation seed sees the same 112 worlds. Means below are descriptive across three initializations; model-case sums are not independent world samples.

| Training regimen | Seed 7 | Seed 19 | Seed 41 | Mean correctness | Non-null model-case correctness |
|---|---:|---:|---:|---:|---:|
| Uniform CE | 91/112 | 68/112 | 48/112 | 61.6% | 99/228 |
| Blocked CE | 83/112 | 93/112 | 94/112 | 80.4% | 162/228 |
| Blocked intermediate supervision | 91/112 | 91/112 | 87/112 | 80.1% | 162/228 |

The six CE models have exactly matching **reported weight fingerprints and confirmation prompts/actions/probabilities** relative to V7. This supports deterministic reproduction of the known seed-dependent findings. It is not six additional independent initializations, and tensor fingerprints have not been independently recomputed from the absent FULL archive.

Intermediate supervision improves seed 7 but loses ground in seeds 19/41; aggregate non-null confirmation correctness is unchanged. It uses **3,068 updates and 109,280 supervised tokens per model**, versus CE's **2,368 and 75,776**: 29.6% more updates and 44.2% more supervised tokens. Its lower teacher-forced loss has a changed target distribution, so it is not evidence of superior general learning. **Do not promote it over blocked CE.**

Balanced payload swaps provide positive evidence of narrow evidence dependence: both answers in each pair are correct for **45/48** blocked-CE model-pairs and **47/48** intermediate model-pairs, versus **22/48** uniform model-pairs. This is stronger than merely issuing QUERY, but remains a related synthetic evidence-reading task. Grammar leaves all nine base scores unchanged and changes confirmation by at most one answer; it does not explain away the main deficits.

## Failures localized by actual interventions

**Serialization sensitivity is demonstrated; its learned internal cause is unresolved.** Blocked CE scores **270/336** on original row order and **139/336** after reversal. Across the same three seed blocks, original-only successes are 46/46/49 and reversal-only successes 4/4/2. The facts and intended answers are unchanged. This is a controlled behavioral effect of serialization for these checkpoints/worlds, not proof of a particular neural heuristic.

For blocked seed 19, world `918eae8f...` asks the current value of A. Both variants QUERY the same source and receive `A=J@366/s0`. The original rendering then answers **J**; the reversed rendering answers **G**, the older `A=G@361/s0` value. The first action is the same QUERY; the first observed behavioral divergence is the next ANSWER after the evidence is present. This localizes the error beyond evidence release. A position/recency shortcut is plausible, not proven to explain all failures.

**Memory correction regresses sharply under intermediate training.** Blocked CE scores **45/48** on original memory and **28/48** with an older wrong row plus a later same-source correction. Intermediate scores **48/48** and **2/48**, respectively. For intermediate seed 7, the trace successfully retrieves `G=A@656/map` and `G=M@657/map`, with `L=G@656/map` already visible, then answers **L** instead of M. Retrieval occurred and the relevant observations reached the prompt; the observed failure is post-retrieval answer generation/conditioning. It is not simply absent memory. Only 7/46 intermediate wrong corrected-memory answers equal the stale value, versus 2/20 for CE, so a universal "copies the old row" explanation is unsupported. Extra history length, conflicting bindings, positional cues and changed training dynamics remain competing explanations.

**Action availability does not generalize reliably.** Blocked CE gets **94/96** advertised-needed query cases correct, but **0/96** when the needed channel is unavailable; all 96 unavailable cases contain invalid query attempts despite `Query=0`. Removed/irrelevant memory conditions each produce **48/48 invalid RETRIEVE outcomes** for blocked CE. Those memory cases intentionally retain a now-unavailable latent target: their correctness deficit is not a safe-abstention score, but the invalid action is a real observed failure. Grammar suppresses invalid actions while generally leaving poor answer correctness. Engineered legal-action constraints must remain labeled as mechanics.

**The independent chains do not isolate composition.** Blocked CE gets **0/48** on the new two-hop cohort and **0/96** across its original/clamped variants. One-hop accuracy is also only **3/48**. Compared with training, these chains simultaneously introduce sparse graphs, longer timestamp ranges and ordered edges; one-/three-/four-hop instructions also change. The near-zero clamped result cannot establish that multi-hop computation alone is the bottleneck.

Intermediate models get **22/48** four-hop answers versus blocked CE's **4/48**, but a newly audited deterministic shortcut gets **64/64** across the four hop cohorts: simply copy the last visible row's value, or the maximum-timestamp value. Their rows and timestamps were generated in path order. This proves the benchmark admits a non-compositional solution; it does not prove the neural models used that shortcut. Preserve these outcomes as diagnostic evidence, not a composition achievement.

## Compute, limitations and next decision

There were **23,412 optimizer updates and 782,496 supervised tokens** across nine 3,084,800-parameter models. Recorded controller time is **3,815.15 seconds (63.59 minutes)** before final reporting/replay/export. Summed worker reservations, including canaries/profiling/recovery overhead, are **6,761.61 seconds (112.69 GPU-minutes)**; summed update-loop wall time is 5,237.00 seconds. These are measured process times, not kernel-only utilization or metered Kaggle quota. No local research training was performed.

No scale, continual or original-V7 frozen phase ran. There is no measured forward transfer, adaptation to query prices, architecture self-improvement, independent sealed evaluation or established internal root cause. Training accuracy was not measured for all new objectives, so we cannot assume the learner mastered training data before failing generalization.

**Next: diagnose fixed checkpoints before larger training.** Implemented follow-up data holds the same 16 route worlds fixed while crossing timestamp range, irrelevant rows and serialization order, with original two-hop, first-hop, clamped and direct-lookup questions. There are **512 records, 48 predeclared one-factor contrasts**, and a passing exact disjointness check against the actual 7,335-world training set. These are controlled development interventions, not 512 independent worlds or an external benchmark. See [NEXT_EXPERIMENT.md](../../../research/v7.1/NEXT_EXPERIMENT.md).

The dataset and nine focused regression checks pass. Frozen neural execution and its cloud wrapper remain pending FULL-checkpoint custody and hardware qualification. **GO for preserving/auditing checkpoints and this diagnostic iteration; NO-GO for promoting intermediate supervision or launching scale/continual campaigns from these scores alone.**
