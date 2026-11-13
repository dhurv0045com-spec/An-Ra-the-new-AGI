# V7.15 S8 scale results

The six-job S8 batch executed on two Kaggle Tesla T4s. With equal training exposure, the S8 profile scored below the earlier S3 profile under both curricula. Blocked ordering improved S8 confirmation at every tested seed. The larger profile still showed substantial row-order sensitivity, failures to revise memory, and illegal actions when evidence was unavailable.

Run `3981e34a2ffe4b4b9528b4b4d588883b` used source `e5779922a0072a070f64356f38ea47090e543d6a`, the published V7.15 pin. The `scale-s8` notebook phase uses the inherited V7.1 training protocol and export names. It ran uniform and blocked CE at seeds 7, 19 and 41 for four epochs each. The archive evaluates the same 16 V7.1 cohorts, with 1,120 records from 288 distinct worlds. The V7.15 frozen diagnostics use a separate suite.

## Confirmation and cost

Each seed saw the same 112 confirmation worlds. Means summarize the three seeds. The 336 model-case scores per row reuse worlds.

| Profile and training order | Seed 7 | Seed 19 | Seed 41 | Mean | Non-null correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| S3 uniform, V7.1 reference | 91 | 68 | 48 | 61.6% | 99/228 |
| S8 uniform | 49 | 72 | 44 | 49.1% | 57/228 |
| S3 blocked, V7.1 reference | 83 | 93 | 94 | 80.4% | 162/228 |
| S8 blocked | 79 | 85 | 94 | 76.8% | 150/228 |

- At S8, blocked order adds 30, 13 and 50 correct answers relative to uniform, respectively. The paired outcomes contain 99 blocked-only successes and six uniform-only successes across the three seed blocks.
- S8 changes confirmation by −42, +4 and −4 answers under uniform order and −4, −8 and 0 under blocked order, compared with the same seed and regimen at S3. The comparison changes the architecture profile, including width, depth and attention configuration, together with parameter count. This leaves open whether a different exposure or optimization recipe would help S8.
- All six S8 models queried all 32 required confirmation cases and correctly abstained on all 36 null cases. Each used 43 queries on confirmation. The change in overall accuracy therefore tracks the answerable cases.
- Each 7,970,944-parameter S8 model used 2,368 updates, 75,776 supervised tokens and 8,914,404 unpadded input positions. The matched S3 models had 3,084,800 parameters and the same examples, updates, token exposure and logged padded attention pair counts. The S8 update loops took 3,933.42 seconds in aggregate versus 3,176.94 seconds for the six S3 CE models, a 23.8% increase in recorded wall time. This comparison spans separate Kaggle sessions.
- The S8 controller recorded 45.68 minutes before final reporting and export. The summed worker reservation was 86.94 GPU-minutes including control and recovery work. The capacity forecast was 98.86 minutes for the phase. These are process measurements and a planning estimate.

## Where performance broke

- **Row order.** For blocked S8, reversing the same public rows drops accuracy from 251/336 to 137/336. There are 126 original-only successes and 12 reversed-only successes. This intervention demonstrates sensitivity to serialization. The corresponding S3 blocked contrast is 270/336 to 139/336.
- **Evidence and correction.** Blocked S8 answers both versions of 36/48 balanced payload pairs correctly, compared with 11/48 for S8 uniform and 45/48 for S3 blocked. Blocked S8 scores 41/48 on original memory and 28/48 on stale memory with a later correction. Removed and irrelevant memory each produce 48/48 invalid outcomes for blocked S8. These inherited panels retain the latent target after removing relevant evidence. Their invalid actions are measurable, while safe abstention needs the newer diagnostic targets.
- **Availability and decoding.** With the needed query channel advertised, blocked S8 solves 81/96 crossed cases. Without that channel, it solves 0/96 and produces 94 invalid outcomes. Across the 6,720 final model-case records, engineered grammar reduces invalid outcomes from 861 to zero. It rescues one confirmation answer across six models. On the independent three-hop cohort, correctness rises from 1/96 to 6/96. The legality intervention changes action validity much more than answer accuracy.
- **First observed divergence.** In blocked seed 41 case `65ab8687…`, the model sees `A→E→J→O` with `Query=0 Memory=0` and issues RETRIEVE before producing an answer. Across all six S8 models, 92/96 three-hop episodes start with RETRIEVE. Blocked S8 also scores 1/48 on the independent one-hop cohort and 0/96 on the clamped/original two-hop pair panel. The inherited chains admit last-row and maximum-time copying. These results leave both the failure of simple lookup under the changed surface and the internal cause of the routing deficit unresolved.
- **Learning trajectory.** On the base cohort, blocked seed 41 reaches 51/71 non-null at epoch two and 52/71 at epoch four. Blocked seed 7 rises from 13/71 to 29/71 over the same interval. Uniform seeds 7 and 41 finish at 8/71 and 10/71. The epoch snapshots show uneven acquisition. The archive lacks a matched training-accuracy trajectory that would separate optimization limits from generalization limits.

**CAUSE UNRESOLVED.** The observed differences support retaining S3 blocked as the reference for the next intervention and keeping S8 as the measured scale comparison.

## Verification and custody

The supplied 16,023,595-byte analysis ZIP passed inventory, CRC and member SHA-256 checks for 112 manifested files. Its SHA-256 is `08cca3edcec070b85fac752faf89f156b85fb7b7fc2fbd9b02c908461175c02b`. The six complete workers, registered batch identity, initial and epoch 1/2/4 evaluations, final decoder pairing, and complete update logs were checked against the pinned protocol. Both T4 lanes report passing positive learning controls and fresh-process recovery. The S8 production probes also report exact same-device recovery.

Independent CPU replay passed for 38 suite reports and four additional positive-control records using matching local datasets. Eight derived Brier values differed by at most `1.39e-17` and passed the existing two-ULP portability rule. Replay verifies saved prompts, actions and outcomes. Tensor restore and fresh neural inference require the FULL archive. The analysis ZIP also predates the separate final delivery receipt, so dual-export completion and fresh-VM recovery remain unverified from the supplied files.

The original ZIP stays in owner custody. The curated [measurements and provenance](METRICS.json) include per-seed metrics, trajectories, costs, archive hashes and a selected failure. To regenerate them from the repository root with a fresh output path:

```text
python -m tools.analyze_elendril_v715_s8 --zip PATH_TO_ANALYSIS.zip --output NEW_METRICS.json
```

Add `--data-dir PATH_TO_DATA` with matching `canary.json` and `monitor.json` to reproduce the four additional positive-control replays.

## Next experiment and execution

**Experiment.** Run the existing frozen diagnostics on the original V7.1 S3 core checkpoints. Start with same-world direct lookup and one-hop results across row order, timestamp and distractor changes. Use the shortcut-resistant route suite to see whether a two-hop deficit persists when simple lookup succeeds. Compare equal-length neutral and stale memory histories and compare raw with grammar-constrained action validity. This gives a basis for selecting one training change, such as row-order augmentation, and testing it against S3 blocked at equal exposure on fresh held-out worlds. A same-suite frozen S8 comparison requires extending the current S3-bound harness.

**Execution.** Preserve this S8 run's `ELENDRIL_V71_FULL.zip` and `FINAL_STATUS.json` from its Kaggle output to verify complete delivery. For the next run use the published V7.15 notebook, set `PHASE="diagnostics"`, attach the original V7.1 **S3 core FULL** archive through `PARENT_ARCHIVE`, select T4×2 with Internet and Run All. Match its SHA-256 to `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19`. If that parent is unavailable, the predeclared `scale-s20-seed7`, `scale-s20-seed19` and `scale-s20-seed41` batches can each run in a separate session without inputs to finish the size comparison. Keep the six S20 jobs and their results together when interpreting scale. Continual and historical frozen phases remain unexecuted in the supplied evidence. [Notebook and release instructions](../../../research/v7.15/RELEASE.md)
