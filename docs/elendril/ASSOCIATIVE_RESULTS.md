# ELD-DEV-002 results

The registered comparison ran on source `77edcbd8dce7b73d7562e619351a9cc496c6ef84` with fresh weights in all 15 arms. The inherited model is 42,512 parameters. The associative model is 44,106, an increase of 1,594 (3.75%). This was a matched-data, matched-update screen.

| Method | Training both correct /84 by seed | Development both correct /28 by seed | Development exact + EOS /56 by seed |
| --- | --- | --- | --- |
| Shuffled CE | 32, 0, 0 | 10, 0, 0 | 38, 24, 20 |
| Paired CE | 0, 1, 1 | 0, 0, 0 | 26, 12, 7 |
| Paired contrast | 0, 1, 0 | 0, 0, 0 | 26, 22, 20 |
| Associative | **84, 84, 84** | **28, 28, 28** | **56, 56, 56** |
| Associative with permuted labels | 84, 84, 0 | 28, 28, 0 | 56, 56, 0 |

Columns use seeds 7, 19 and 41 in order. Training panels are scored against the clean rule. Paired CE and paired contrast failed the preregistered formation gate. The associative arm reached 28/28 at update 84 and stayed there in all three seeds, clearing the formation and clean architecture-delta thresholds. The preregistered separation from the permuted-label arm failed in two seeds. Permutation flipped 6, 10 and 11 of the 21 training worlds, respectively. The majority clean rule agreed with the first two outcomes, while the third seed's clean score was zero. These three observations motivate the separately documented rule-direction diagnostic.

| Stress panel | Associative clean both correct /28, seeds 7, 19, 41 |
| --- | --- |
| Renamed keys and separator | 28, 0, 14 |
| Additional distractor fact | 19, 25, 14 |
| Query neutralization | 0, 0, 0 |

All arms scored zero both-correct under query neutralization. The clean head's success on original held-out value worlds survived changes in rendering inconsistently. Attributing the gain to a particular subcomponent of the head requires separate ablations. A claim about equal-compute architecture advantage requires a capacity- and cost-matched baseline.

## Exposure and cost

The paired arms recorded 16,128/16,128 colocated rows per method across the three seeds. Shuffled CE recorded 1,495/16,128. Rows were repeated exactly twice per exposure block in every arm. Batch-local repeat counts were 376 for shuffled CE and 362 for paired CE over the three seeds.

The serial CPU batch used 5,040 updates, 1,128,960 non-BOS input/target tokens and 161,280 supervised answer/EOS targets. All arms ran on one CPU thread. In-process wall time was 94.18 seconds and process CPU time was 91.58 seconds. Evaluation used 19,300 forwards over 268,416 input positions. Total clean associative in-process time was 21.58 seconds versus 16.96 seconds for dense paired contrast. System-available memory stayed at or above 4.23 GiB in the launcher samples. The first logger sampled the Python launcher shim. A separate process-tree monitor sampled the final five runs and saw a maximum aggregate resident set of 293.2 MiB. The RAM figure applies to that sampled subset.

All 135 primary and stress scoring snapshots were independently rescored. The benchmark's seven latent development worlds are reused across seeds and render panels. The outcomes are a controlled development screen. The appropriate next test is a capacity-matched ablation of the lexical relation and copy gate on a larger, globally disjoint generator with renamed keys and variable distractors.


## Rule-direction follow-up

ELD-DEV-002I deliberately reversed the two labels in every training pair. All three fresh associative runs reached 84/84 both-correct on reversed training truth and 28/28 on reversed development truth. The identical predictions scored 0/56 exact and 0/28 both-correct on clean development truth. The learned read path can follow either coherent two-fact rule. This clarifies the original permutation outcome while leaving its preregistered separation failure on record.

The diagnostic added 1,008 updates, 225,792 training tokens and 14.08 seconds of measured in-process wall time. It reused the registered model source and the same ordered prompt exposure. The [diagnostic protocol](ASSOCIATIVE_INVERSION.md) was defined after observing the first seed's noisy-label result and before running the inversion arms.


## Verification and custody

The focused ELENDRIL and inherited-contract suite passed 56 tests. An independent 21-update run restarted in a new process after update 3. Its final model, optimizer, RNG, scheduler, cursor, ledger, per-update telemetry and all final predictions matched the uninterrupted run. The campaign audit recomputed every exposure block, verified paired stream identity, and rescored all inversion panels.

The [custody manifest](../../artifacts/elendril/eld-dev-002/manifest.json) binds 24 receipt and resource files and 75 checkpoint inventories. Model and optimizer payloads are in the sibling local directory C:/Users/ankit/Downloads/Elendril-ELD-DEV-002. The raw checkpoints have local custody. All registered and diagnostic training ran on CPU.
