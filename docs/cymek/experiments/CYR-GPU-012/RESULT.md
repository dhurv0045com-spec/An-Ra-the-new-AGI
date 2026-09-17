# CYR-GPU-012 RESULT — compact full-exposure closure

**Verdict: `NO_G90_AT_FULL_EXPOSURE` (preregistered Branch B).**
Frozen executable: `1a1624ed4c62e1a176dc4bed97e628f0b648a31b` (isolated descendant of V11's `0a97257e2b38db6dfa85cc6e58da0697591dde6b`; diff limited to corrected probes, `stop_on_g90`, current-streak qualification, operator, tests, PLAN — no model/optimizer/data-math changes).
Result receipt: `artifacts/v5/cyr_gpu_012_result_receipt.json`. Raw bundle outside git at `C:/Users/ankit/cyr012-evidence/full01` (`CYR_GPU_012_RESULTS.zip`, sha256 `d2da6d18847c846bd1865d4069c5afe46309ced339350a0115e1c23f3049f2cc`; bundle, all 8 JSON payloads, checkpoint model/optimizer hashes, and PLAN hash independently re-verified post-run).

## What ran

Real compact bridge geometry recreated exactly per preregistration: 987,392 parameters, 19-symbol CompactCharTokenizer, model seed 3301, order seed 4701, batch 64, canonical AdamW (lr 1e-3, betas .9/.95, eps 1e-8, semantic weight-decay grouping, clip 1.0), FP32 + CUDA BF16 autocast, same ARK-002B manifest (bytes verified against the frozen V11 git blob), 18,000 updates / 1,152,000 presentations / 16,128,000 real tokens, eval every 200 updates, no G90 early stop. Hardware: local NVIDIA RTX 4050 Laptop GPU (6 GiB), torch 2.11.0+cu128, wall 54.14 min. **Timing is not T4-comparable.** Resource guards stayed green (min free RAM 3.13 GiB, max temp 57°C).

## Headline result

- Train probe (memorization): first M99 threshold hit at update 600; M99 confirmed at update 1,400, matching V11's compact confirmation. Final exact 100% on the 100-row training probe; accuracy was not continuously 100% afterward. This is not an evaluation of all 500 training rows.
- **DEV_CONTROLLER (held-out): final 0/64. Nine of the 90 scheduled evaluations were nonzero; maximum observed 12.5%.** No G50 or G90.
- **DEV_MEASUREMENT STANDARD: 0%.** Max observed: 9.41%. Final ones-digit accuracy 54.12%, tens-digit 0%.
- G50 never reached, G90 never reached. SEALED_RESERVED 0/48 (compact subject; development-reserved, not an unseen holdout).
- LOCALITY 0% both-exact (relation consistency 33.3%), CARRY 0%, TRIPLE_ADD 0%, THREE_DIGIT 0%.

## The one striking exception

**Corrected within-band commutation (COMMUTATION_MATCHED_BAND): 52/52 pairs exact in both orders — 100% both-exact-with-EOS, per band and aggregate.** These are held-out operand pairs inside trained tens bands 1–4, a probe family V11 never measured. Meanwhile the unseen-band family (bands 6/7, sums >99): 0/50 pairs, 36% consistency.

So this run exhibits *narrow, order-symmetric in-band pair transfer coexisting with zero STANDARD generalization*. This is diagnostic partial transfer per the preregistered decision table — it does not redefine the primary endpoint and is not a generalized-arithmetic claim.

## Reproducibility divergence from V11 (unresolved, explicitly recorded)

V11's compact (nominal seeds 3301/4701, same core training source, T4) reached G50 at update 2,200 and finished at controller 54.7% / STANDARD 56.5% after 8,081 updates. This run's maximum controller score was 12.5% through 18,000 updates. The independent read-only source audit found no concrete training-path defect explaining this difference: acquisition selects the same tokenizer, optimizer shim, backend, sampling algorithm, seeds and cadence. Deliberate changes concern probes, qualification, full-dose stopping and resource controls.

**The historical partial-generalization trajectory was not reproduced; the cause is unresolved.** Hardware, kernels and CPU-thread configuration differ, and V11's calibration phase was omitted, but none is established as causal or ranked by a controlled test. Initialization is explicitly seeded within fork_rng, so skipped calibration is not by itself evidence of shifted initial tensors. Follow-up: the original V11 ZIP was located in Downloads and its SHA-256 verified against the V11 receipt. Both raw acquisition records contain the same full precomputed sampling-index digest (`12e471f4eda1d20ae9fb9a29cb645e3795b04f88cb738aaf59f720e28f5560ea`). Their recorded initial L2 norms also match (46.88997268676758), but the initial tensors were not directly compared. Scores already differ at the first scheduled observation, update 200: V11/local controller 9.375%/4.6875%, training probe 65%/82%. This bounds the first observed divergence, not the first differing optimizer step. Matching index digests do not independently prove matching dataset row contents or initial tensors. Source equivalence does not prove numerical equivalence; this result does not establish environment sensitivity as the cause.

## Decision-tree consequences (preregistered, both branches honored)

- **Branch B triggered:** this fixed dose was insufficient for this seed/configuration/run — giving the compact representation the entire ARK-002B box did not produce G90 here. This does not rule out larger doses or other seeds. Per PLAN, next discovery attention goes to Cymek-vs-Arkenstone objective/BOS supervision, initialization, optimizer grouping/precision, and architecture — not vocabulary alone.
- Branch A (representation-factorial CYR-GPU-013) is **not** triggered by this run.
- The V11-vs-012 reproducibility gap must be closed (or explained) before any factorial design consumes it as ground truth.

## Post-run engineering verification

The final checkpoint was reloaded with its model hash verified. All 253 registered rows (104 matched-band commutation, 85 STANDARD, 64 controller) reproduced saved predictions exactly, with identical outputs at batch sizes 1 and 32. The 52 canonical matched-band pairs had zero overlap with all original train/controller/measurement/reserved partitions. Final controller replay was 0/64; paired replay was 52/52. This rules out the tested checkpoint-loading/batching mismatch, not every possible evaluator defect.

A separate six-prompt spacing diagnostic recorded first-token logits and outputs. Removing spaces broke all four selected in-band examples; neither of two selected STANDARD examples became correct. These selected post-hoc samples are engineering observations, not a new endpoint or a causal explanation of the V11 discrepancy. External diagnostic: `C:/Users/ankit/cyr012-evidence/pair_replay.json`; script: `check_pair_replay.py` in the same directory. Free GPU memory after release: 4.91 GiB.

Focused verification: 22 tests passed in 7.31 seconds across the V11 core/runtime and CYR-012 closure/evidence tests. These include bounded real-model CPU execution, full-dose stopping, pair exclusions, negative controls and bundle-hash rejection. No additional scientific training was run for this verification. The frozen operator lacks a finalization reserve within its acquisition deadline; the observed 54.14-minute acquisition-through-reserved-evaluation wall stayed below the 90-minute budget. Startup and ZIP packaging are excluded from that measured wall.

## Scope limits

- Not proof that representation doesn't matter (one seed, one environment, unreproduced V11 trajectory).
- Not a production or TPU result. Production promotion, PRE500M, 500M: all unauthorized.
- The historical production 0% at full exposure stands as recorded in V11; it was not rerun.
