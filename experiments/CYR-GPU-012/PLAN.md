# CYR-GPU-012 — compact full-exposure closure

Status at registration: PREEXECUTION. No CYR-GPU-012 measured training has run.
PRODUCTION_PROMOTION_FORBIDDEN. No production, TPU, PRE500M, 500M or AGI authorization.

## Hypothesis and falsification (registered before execution)

Hypothesis: "The compact representation, given full ARK-002B exposure (1,152,000 presentations / 18,000 updates), will reach G90 on the corrected structural probe battery; production's 0% result at the same exposure is representation-dependent, not exposure-dependent."

Falsification condition: "If compact also fails to reach G90 at full exposure, the exposure-sufficiency explanation is ruled out, and the bottleneck is not simply 'compact needs more data.'"

Scope: this falsifies sufficiency of THIS fixed 18,000-update dose for THIS seed/configuration, not all larger doses or all seeds. Historical production and compact use different model seeds, sequence lengths and hardware; a positive implicates, but cannot prove causality of, representation alone. More presentations repeat the same 500 training rows, not new unique data.

## Discovery and frozen controls

Read full V11 RESULT.md, result receipt, PLAN.md, cyr_gpu011.py and cyr_gpu011_run.py. V11 frozen executable is the existing git commit 0a97257e2b38db6dfa85cc6e58da0697591dde6b. Current branch cyhex-hermes HEAD 35b7661 contains later clipping certificate tolerance changes; execute an isolated checkout of V11 plus explicitly recorded research-only changes instead. Do not include HORM work or prior cyr_gpu012_r1 factorial scaffolding. A new isolated executable commit and source manifest will be frozen before measured training; no push is authorized or needed.

V11 compact: 987392 parameters, 8081 updates, 517184 rows, 7240576 real tokens, M99=1400, G50=2200, G90=null, controller=.546875, STANDARD=.5647058823529412, ones=.8, tens=.5647058823529412 (85 examples). Production: 4130688 parameters, 18000 updates, 1152000 rows, 9216000 tokens, M99=2200, G50/G90=null, controller/STANDARD=0; ones=.12941176470588237, tens=0 (85). Production triple-add=.041666666666666664 and verbal=.0625: NOT all diagnostics were zero. V11's 8383.781 seconds is the combined session, not separately measured production-only wall.

## Exact subject and procedure

Recreate compact only, model seed 3301, CPU sampling generator seed 4701. Batch64, 18000 updates, no early stop on G90; record transient crossings and final confirmation separately. Same real V5 4L/128w/Q4/KV2/head32/FFN512/context512, tied embeddings, QK norm, 19-symbol CompactCharTokenizer. Canonical AdamW lr=.001 constant, betas=(.9,.95), eps=1e-8, semantic grouping (normalization no decay, other matrices .1 decay), global gradient clip1. FP32 parameters and optimizer, CUDA BF16 autocast as V11. No activation checkpointing. Same render/loss: BOS + prompt + answer + EOS; supervise answer digits and EOS, not BOS. Same 500 train rows and hash-partitioned controller64/measurement85/reserved48 from docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json. Verify actual manifest bytes against frozen git blob, not only its declared split hash. Sampling with replacement; 1152000 int64 indices ~9 MiB. No data/optimizer/architecture tuning.

Evaluate every 200 updates, greedy candidate-free eight-new-token cap. M99=train[:100] >=.99 three consecutive evaluations; G50=controller>=.5 three consecutive; G90=controller>=.9 three consecutive AND contemporaneous STANDARD>=.9. Require current streak, not a historical stale confirmation. Final primary pass requires the last three controller evaluations >=.9 AND final STANDARD>=.9. Measurement never changes training or stopping. Reserved48 evaluated once on final checkpoint after decision, and labeled historically consumed development-reserved, not a newly sealed independent test.

## Corrected commutation design

Keep STANDARD, LOCALITY, CARRY, TRIPLE_ADD, THREE_DIGIT unchanged for V11 comparison. Keep legacy reversed STANDARD only as LEGACY_ORDER_ASYMMETRY, never an invariance claim. Its old row metadata must be swapped along with prompt.

New COMMUTATION_MATCHED_BAND: all unordered distinct pairs a<b, both operands in the SAME tens band t in {1,2,3,4}, ones sum<=9 and total<=99, excluding canonical pairs in ALL original train/controller/measurement/reserved partitions. Enumerate ascending t,a,b. There are 52 eligible pairs (14/14/13/11 by band), evaluated in BOTH orders (104 rows). Reversal preserves each operand's tens band exactly; no carry or output-length shift. This is held-out-pair transfer within trained tens bands, NOT unseen-band extrapolation.

New COMMUTATION_OOD_BAND: all unordered distinct pairs a<b with both operands in the SAME tens band t in {6,7}, ones sum<=9. Evaluate both orders. This holds unseen-band placement fixed under reversal. Inevitably sums exceed99; explicitly a joint unseen-band + output-length challenge, NOT directly comparable to original two-digit STANDARD and not used to redefine its G90. Report per-band and aggregate paired content consistency, EOS-valid consistency, and both-exact-with-EOS. Constant wrong outputs can have consistency1 but must score both-exact0; only both-exact>=.9 supports the narrow commutation flag. No broad invariance claim. Probe rows/manifest/hash frozen before training; tests enforce band invariance, pairing, arithmetic, no contamination and negative controls.

No aggregate structural reasoning score. 'G90 on the battery' means the original controller+STANDARD endpoint, with corrected orthogonal probe families reported separately; it does not mean every novel family is >=.9.

## Exhaustive decision branches

Precedence: invalid identity/nonfinite/runtime/evaluation evidence => INVALID_OR_INCOMPLETE, neither A nor B. Fewer than18000 updates or1152000 rows => INCOMPLETE_EXPOSURE, neither branch (even if transient G90).

A: full exposure, last3 controller>=.9, final STANDARD>=.9. Verdict COMPACT_G90_FULL_EXPOSURE. Historical production STANDARD remains0; do not rerun it. Prepare CYR-GPU-013/PLAN.md only, first one-variable factorial cell: output-softmax competition over the SAME 24576-row char-embedded model, same active token IDs, same matched initial tensors/optimizer/stream, compare full output support vs 19-active-token support. This isolates output competition conditional on the fixed large embedding; it does not isolate vocabulary size itself. Plan only, not execute.

B: full valid exposure with neither controller sustained nor final STANDARD>=.9 => NO_G90_AT_FULL_EXPOSURE. Report partial scores literally (including70%); rule out THIS dose being sufficient; redirect next discovery toward Cymek/Arkenstone objective/BOS supervision, initialization, optimizer grouping/precision, architecture, not vocabulary alone. Representation can still matter.

AMBIGUOUS: only one of sustained controller or final STANDARD passes, or G90 occurred transiently but final confirmation fails => AMBIGUOUS_PARTIAL_OR_TRANSIENT_G90. No factorial launch and no clean null; report both endpoints and all families, propose a preregistered independent-seed confirmation before causal follow-up. Family-specific>=.9 with primary failure is diagnostic partial transfer, not G90. Primary pass with other families failing is A with explicitly narrow scope; no generalized arithmetic claim.

## Real differences to investigate only after A

Beyond vocabulary19 vs24576: char vs production segmentation/number atomization and special IDs; 14 vs8 real tokens/row in V11, position distances and answer-target lengths; tied input/output table987392 vs4130688 total parameters (3143296 difference); softmax competitors; independent seeds3301 vs3401; initialization RNG consumption differs with embedding shape even if seed matched. Core geometry/objective/grouping/LR same. Local RTX4050 vs historical T4 is an additional uncontrolled hardware/precision-kernel factor. These are candidate causes, not isolated facts of causation.

## Budget, safety and evidence

Preflight: Ryzen7 170 CPU load13%, RAM free6.86GiB, RTX4050 Laptop6141MiB idle0MiB/46C; torch2.11.0+cu128, CUDA12.8 available after clearing PYTHONPATH/PYTHONHOME. Timing is NOT T4-comparable. Expected compact wall ~60min, hard scientific wall90min incl evaluation, reserve5min packaging; total95min. Separate engineering calibration <=5min (seed9911, not science; discarded weights). CPU threads2, one GPU model at a time, no dataloader workers. Memory model+optimizer well under100MiB, activation budget conservatively<1GiB. CUDA allocator cap3GiB; require>=3GiB system free RAM and>=4GiB GPU free before launch; periodically abort if RAM<2GiB or GPU temperature>=85C. No batch reduction or altered precision to rescue a failed feasibility test. Timebox or safety abort is incomplete, not a negative. Source and bundle stored outside repository; raw ZIP/checkpoints never git-staged.

Receipt: field-compatible V11 sibling with experiment/schema version changed; compact fields preserved, historical production explicitly source-labeled; real execution GPU/VRAM/torch/wall; bundle byte count and SHA256; git frozen_executable_sha; prediction receipts, manifest and PLAN hashes, source change list. Commit only scientific source/freeze artifacts if needed for requested frozen SHA, never unrelated user files. Distilled RESULT.md and agent.md updated after outcome or explicit execution-pending state. No invented receipt or completed result if blocked.

Deliberate executable changes: sibling corrected battery/scoring/operator; acquisition option to disable G90 early stop while preserving default V11 behavior; qualification checks current streak. Resource guard around real updates. No model/training math modification. V11 clipping certificate tolerance remains frozen; an actual rejection is reported, not silently relaxed.
