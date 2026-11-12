# V7.16 — S20 scale results

All six S20 runs completed. V7.16 is the owner-requested archive label for the S20 continuation of the registered V7.15 scale matrix. Runtime source is `4cc1e993c09d1475a3bc1e250fda58909d539407`, run `51f3aeb7cb604c6d8c80f6ae4cd9f3ff`, with the inherited V7.1 protocol and export filenames. Uniform and blocked CE each ran at seeds 7, 19 and 41 for four epochs on two Tesla T4s. The six S8 and six S20 jobs have now executed. Together with the earlier six S3 CE references they provide the planned three-size comparison.

## What happened

Each model has 19,871,616 parameters and used fresh weights, 2,368 updates, 75,776 supervised tokens and 8,914,404 unpadded input positions. Training data and the 1,120-record evaluation suite match the earlier comparisons. Its 288 worlds recur across interventions. Confirmation results below reuse the same 112 worlds at every seed. Means describe three initializations.

| Profile and order | Seed 7 correct /112 | Seed 19 | Seed 41 | Mean | Non-null model-case correct /228 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S3 uniform | 91 | 68 | 48 | 61.6% | 99 |
| S8 uniform | 49 | 72 | 44 | 49.1% | 57 |
| S20 uniform | 72 | 52 | 49 | 51.5% | 65 |
| S3 blocked | 83 | 93 | 94 | 80.4% | 162 |
| S8 blocked | 79 | 85 | 94 | 76.8% | 150 |
| S20 blocked | 54 | 79 | 78 | 62.8% | 103 |

S20 uniform improves over S8 in seeds 7 and 41, but loses at seed 19. S20 blocked loses to S8 blocked at every seed. Within S20, blocked loses to uniform at seed 7 and wins at seeds 19/41. The architecture profiles change width, depth and attention configuration as well as parameter count. The measured result is that this larger profile, under the fixed exposure and recipe, fails to improve aggregate confirmation. Different data budgets or optimization settings remain prospective hypotheses.

Blocked seed 7 illustrates incomplete acquisition. Its base non-null score at epochs 1/2/4 is 4/71, 3/71 and 5/71. Seeds 19 and 41 finish at 39/71 and 34/71. Uniform seed 41 remains at 8/71 at both epochs two and four. Full training-set accuracy was not recorded, so these observations do not isolate a train-versus-test generalization gap.

## What worked and what failed

- Balanced payload pairs are both correct for 22/48 blocked S20 model-pairs and 11/48 uniform model-pairs. Earlier blocked scores are 36/48 at S8 and 45/48 at S3. Evidence-responsive behavior exists on a subset and deteriorates in aggregate under this scale recipe.
- Blocked S20 scores 29/48 on original memory and 9/48 on stale memory followed by a later correction. The corresponding S8 scores are 41/48 and 28/48. Removed and irrelevant memory each lead to 48/48 invalid outcomes at S20 blocked. These inherited memory panels retain the unavailable latent target, which limits interpretation of their correctness scores.
- Reversing the same public rows reduces blocked S20 from 201/336 to 120/336. The corresponding original/reversed totals are 251/336 and 137/336 at S8, and 270/336 and 139/336 at S3. Ordering sensitivity persists.
- Blocked S20 scores 65/96 on advertised-needed queries and 0/96 when the channel is unavailable. All 96 unavailable cases have invalid outcomes. Raw operation choice remains sensitive to availability changes.
- Raw S20 achieves 0/96 across the independent two-hop cohort and 0/192 on the original/clamped two-hop pair panel across all six models. On the independent three-hop cohort, raw correctness is 0/96 and grammar correctness is 5/96. Grammar removes illegal actions with little recovery of correct terminal values. The inherited sparse-chain cohorts also change timestamp and distractor distributions and admit copying shortcuts, so they cannot uniquely diagnose composition.
- A concrete first divergence appears in blocked seed 41 case `65ab8687…`. The prompt contains `A→E→J→O`, advertises `Query=0 Memory=0`, and asks for three links from A. The first proposal is RETRIEVE with recorded confidence about 0.994. The controller rejects it. The episode fails before an answer can be generated. The full selected prompt and action are preserved in the curated metrics.

**CAUSE UNRESOLVED.** Candidate explanations include acquisition and optimization difficulty, representation sensitivity, weak relation binding and learned action selection. The fixed-budget profile comparison establishes observed outcomes, while a controlled mechanism intervention is still needed to distinguish those explanations.

## Time, recovery and verification

Recorded controller time was 3,975.00 seconds, or **66.25 minutes**, before final reporting/replay/export. S8 recorded 45.68 minutes, so S20's observed controller duration was approximately 1.45 times S8. It fit the 200-minute campaign allowance. The conservative forecast was 128.74 minutes. Summed worker reservations were 127.95 GPU-minutes, including controls and recovery work. Six S20 update loops took 5,440.18 seconds versus S8's 3,933.42 seconds, a 38.3% increase across the respective sessions. Both scale batches used 454,656 supervised tokens in total. These process timings exclude some orchestration and finalization costs and do not measure GPU kernel utilization.

The supplied ZIP has 16,055,744 bytes and SHA-256 `9ff25533dfd2e197457604970c8c005ca26191f505db3b1f011de2297cd256d1`. Inventory, CRCs, hashes and manifest verification passed for all 112 manifested files. Source, batch identity, all six complete workers, initial/epoch 1/2/4 reports, grammar pairing and every update log were checked. Independent replay passed **38 suite reports plus four positive-control records**. Sixteen derived Brier fields differed across runtimes by at most `1.11e-16`, within the existing two-ULP rule. Raw confidence values, actions and counts retain strict verification.

Both GPU lanes report positive learning and recovery PASS, and both S20 production-shaped probes report recovery PASS. The separately produced FINAL_STATUS receipt and FULL checkpoint ZIP have not been supplied for this analysis. Final dual-export certification, independent tensor restoration and fresh-VM recovery therefore remain pending. The original ZIP stays outside Git. [Curated measurements and provenance](METRICS.json)

Reproduce the audit from the repository root using a new output filename. The optional data directory must contain matching canary and monitor datasets.

```text
python -m tools.analyze_elendril_v715_s8 --profile S20 --zip PATH_TO_ANALYSIS.zip --output NEW_METRICS.json --data-dir PATH_TO_MATCHING_DATA
```

## Next experiment and execution

Retain S3 blocked as the reference. The completed scale matrix gives no basis for promoting S20 under this recipe. The next diagnostic question is whether direct lookup and one-hop behavior survive matched changes in row order, timestamps and clutter, followed by shortcut-resistant composition and equal-length memory correction tests. Use the existing frozen V7.15 diagnostic phase on the original S3 core checkpoints. A subsequent learning experiment should change one factor, such as row-order augmentation, with a matched unchanged blocked-CE comparator and fresh held-out worlds. Additional exposure for S20 is a separate hypothesis requiring its own fixed-budget controls.

Preserve this S20 run's FULL ZIP and FINAL_STATUS from Kaggle. For the frozen diagnostic run, use the V7.15 general notebook with `PHASE="diagnostics"` and `PARENT_ARCHIVE` pointing to the original V7.1 S3 core FULL ZIP whose SHA-256 is `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19`. Enable T4×2 and Internet. The S20 FULL archive has a different lineage and cannot satisfy that S3 loader. [Launch instructions](../../../research/v7.15/RELEASE.md)
