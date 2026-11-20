# V8-X2 owner run 1 · The pointer, the capacity control, and the B0 stop

Reviewed 20 November 2026. Campaign status **PARTIAL**. The compact archive delivers two complete original X1 comparisons, A1 and A2. It reaches the first B0 baseline and stops. The new X2 mechanisms, including selective replacement, T13 rule learning and T14 information seeking, have no training result in this delivery.

## What the completed jobs show

Each row reports supported terminal accuracy at update 2,368. Each seed saw the same 384 development worlds in six linked presentations (2,304 episodes). Figures are percentages. The standard and narrative columns each use 384 presentations per seed. Five-hop uses a separate depth suite. The last column requires correct answers on both sides of a relevant evidence change among eligible world pairs. P and R come from the previously audited A0 results; the other four arms are from this delivery.

| Architecture | Seed 7 | Seed 19 | Seed 41 | Mean | Standard rendering | Narrative | Five-hop | Evidence-change pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 31.38 | 32.64 | 18.92 | 27.65 | 30.21 | 18.92 | 21.09 | 10.21 |
| F-cap | 28.69 | 32.64 | 22.96 | 28.10 | 31.16 | 10.76 | 22.14 | 11.35 |
| P (A0) | 59.51 | 65.89 | 60.68 | **62.02** | 74.39 | 16.15 | 58.59 | 80.94 |
| P-capR | 56.55 | 64.97 | 55.69 | 59.07 | 68.66 | 16.23 | 55.73 | 74.79 |
| R (A0) | 63.63 | 59.98 | 61.33 | 61.65 | 69.79 | 25.43 | 58.33 | 75.21 |
| R-noedge | 63.85 | 59.85 | 62.02 | 61.91 | 72.22 | 16.75 | 59.90 | 78.44 |

**The clearest comparison is P against F-cap.** The models have 19,947,925 and 19,947,988 parameters, respectively, a difference of 63. P led by 33.93 percentage points on average and by more than 30 points on every paired seed. It also solved both members of 80.94% of eligible evidence-change pairs, versus 11.35% for F-cap. Increasing F's active capacity alone moved mean accuracy by only +0.45 points, with mixed seed effects. Combined with the earlier frozen copy-path intervention on its restricted subset, the evidence favors the pointer readout as an architectural contribution on this benchmark. It leaves open whether the model learned abstract relations or exploited visible candidate tokens.

**R's edges have a narrower case.** Removing cross-slot messaging while keeping active slot aggregation gave nearly identical mean development accuracy. R minus R-noedge was −0.22, +0.13 and −0.69 points across the three seeds. The no-edge control also solved more evidence-change pairs on all three seeds. Narrative rendering was different. R's advantage there averaged 8.68 points, with seed differences of +5.73, +22.66 and −2.34. That makes the renderer a worthwhile target for a frozen causal test. With three seeds there is insufficient evidence for a general rendering benefit, equivalence across settings, or a mechanism for solving five-hop tasks. More active width on P also failed to help at this horizon: P-capR trailed P on every seed.

**Where to look in the traces.** Across the 6,912 final development views per architecture, the archived first-divergence classifier counted 2,700 wrong-value/same-operation cases for F and 2,568 for F-cap, compared with 1,415 for P-capR and 1,214 for R-noedge. Action selection also remains a problem. R-noedge made 1,091 unnecessary initial actions. These counts pinpoint behavior for follow-up; they cannot identify a latent representation failure. The three training seeds are the replication units. Views of the same world are linked.

## Why the run stopped and what survived

The prepared queue contained 84 scheduled units. The session admitted 43 and deferred 41. Six two-worker waves completed all twelve A1/A2 jobs. In the seventh wave the B0 workers for seeds 7 and 19 both exited at the initial u0000 evaluation. The exception was `AttributeError: '_S' object has no attribute 'base_spec'`. The legacy X1 evaluator constructs a stand-in with architecture and sequence length while the V8 byte policy immediately calls `v8_spec.base_spec.sha256()`. It also reads vocabulary size and context length. Both traces identify the same code path. Neither failed seed has a training journal or an evaluable B0 endpoint here.

The controller recorded `FAIL_FAST_ALL_WORKERS_IN_WAVE_FAILED`. Its final session status lists 12 completed and 31 unfinished admitted jobs, making **72 remaining across the initial queue**. The standalone admission receipt records a denied larger batch, while the session admission plan records the accepted 43-job prefix. The wave logs establish the immediate stop reason. Summed recorded wave durations were 8,557.51 seconds (2 h 23 min). The completed jobs totalled 16,614.08 worker-seconds (4.62 worker-hours); neither figure measures total notebook elapsed time. The preflight independently matched 10 canary predictions across two lineages. The B0 evaluation path still escaped that check.

## What I verified

The supplied file is 20,746,157 bytes, SHA-256 `5888d77f499317fc4b6b165462b3ab0dd0a74421114cb06a80a34b4274eccceb`. I checked all 268 manifested members by size and SHA-256 and verified the separate checksum index. Manifest coverage accounts for every ZIP entry besides the manifest itself. These checks establish internal integrity. An externally signed owner digest was unavailable.

I matched all 12 job identities to the original registry, checked all 28,416 numbered updates for continuity and finite logged losses, regenerated the original training identity and evaluation worlds, and checked ordered world IDs. I replayed **62,976 recorded action sequences** against the project verifier: 27,648 early milestone episodes, 27,648 final development episodes and 7,680 depth episodes. All terminal verdicts and summary statistics matched the archive within floating-point tolerance. This is recorded-action verification. The compact delivery contains weight manifests but no production checkpoint tensors for fresh neural inference or restart from those weights.

Campaign source `596c5e3dd07cb30bd636b68250d11d960adf67ae`. Legacy worker source `8e62ca0f2046c0f07910fdd20abee40508631227`. Registry digest `179b29ab1bb35cc6568858b4b5efa1b477f38e171ab4cfa8bf9bcf768e14deb1`. Review code at Elendril `5e77fbc51fa10d68353595e62daff543204d148a`. A0 comparison data come from the committed [A0 metrics](../V8-X1/A0/METRICS.json).

## Next decisions

1. **Execution.** Repair the legacy B0 adapter in a separately pinned compatibility revision. Pass the real model identity, vocabulary and sequence/context limits through the byte policy. Exercise actual u0000 evaluation and a tiny update → checkpoint → restore → evaluation flow before allowing two B0 lanes. Compare byte decoding and terminal actions with the reference policy for the same weights and public prompts. Exercise J's initial evaluation too. Import the 12 completed results through validated job identities and archive hashes, verify the new queue excludes all 12, and resume at B0 seed 7. A new source pin must retain the original scientific configuration and disclose the adapter change. Identical rerun of the failed notebook is a **NO-GO**.
2. **Science.** Use P as the reference reader. The smallest decisive follow-up is a paired edit to an interior edge with a plausible wrong terminal token still visible. Query the changed path and an untouched path from the same world, counterbalance row order and symbols, and compare intact P, pointer-disabled P and an active matched readout on shared frozen checkpoints where retrievable. If the learner tracks the distractor or changes both answers, its high original score has a narrower interpretation. For R, compare intact, edge intervention and sham on the same public histories under both renderers and identify the first changed action. These tests can proceed independently of B0 once appropriate weights are available. Leave T13/T14 and the delta-memory hypothesis explicitly pending.
