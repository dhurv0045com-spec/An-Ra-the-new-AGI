# CYR-GPU-011 — EXPOSURE-MATCHED CAPABILITY-EMERGENCE BRIDGE

## Status

PREEXECUTION. Do not run until a separate hash-bound `PREREGISTRATION.json` exists and `RUN_READINESS.json` says ready.

## Objective

Determine whether the real Cymek V5 4L/128w architecture can enter the delayed memorize→generalize regime demonstrated by Arkenstone ARK-002B when semantic exposure and compact task representation are brought close to that protocol, then measure how much harder the same task is under Cymek's real 24,576-token representation.

## Prior

- CYR-GPU-009: TINY memorized but did not reach candidate-free held-out G90 under ~250k semantic row presentations per parent.
- ARK-002B: 4L/128w compact model, batch64, up to 18k updates = 1,152,000 row presentations. Seed47 sustained G90 ~12k; seed29 ended at OOD .924 without sustained G90 at 18k. Qualitative delayed transition replicated.
- CYR-GPU-010: superseded before execution because batch16 × 18k would expose only 288k rows, 25% of the ARK reference box.

## Primary questions

1. **Architecture/task bridge:** under the exact ARK-002B data, compact 19-symbol representation, ARK-comparable semantic dose, but Cymek's real V5 architecture/objective, does held-out G90 emerge?
2. **Production-representation bridge:** under the same task, V5 geometry and Cymek objective, does G90 emerge with the frozen 24,576-token tokenizer within the available wall and actual semantic exposure?
3. **Structural characterization:** if held-out generalization emerges, which controlled transformations transfer with it?

## Data

Exact copied ARK-002B frozen manifest:

- task: two-digit no-carry addition, structural tens-band holdout
- train 500 / test 197
- split SHA256 `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`
- source blob `6c46fdf90139526b00e9041af2d511ed0ac24270`
- canonical train/test pair overlap 0

The 197 original test rows are partitioned deterministically before outcomes by hash into:

- DEV_CONTROLLER = 64
- DEV_MEASUREMENT = 85
- SEALED_RESERVED = 48

## Model geometry

Both stages use real Cymek `ModelSpec` + `v5_model.core.initialize()`:

4L / width128 / Q4-KV2 / head32 / FFN512 / context512 / tied embeddings / QK norm / no dropout.

- COMPACT_BRIDGE: vocab19, exactly 987,392 parameters.
- PRODUCTION_BRIDGE: vocab24,576, exactly 4,130,688 parameters.

## Optimizer

HIGH LR = `1e-3` for this controlled development experiment.

AdamW values are the canonical Cymek constants: betas `(0.9,0.95)`, eps `1e-8`, weight decay `0.1`, with Cymek's semantic parameter grouping and clip/backend behavior. Research compatibility code may accept these values redundantly but must reject any drift and restore the canonical API after use.

## Compact objective boundary

The compact bridge uses the same 19-symbol alphabet as ARK-002B but **does not pretend to reproduce Arkenstone's loss exactly**.

Arkenstone's helper encoded an answer-prefix BOS and supervised that BOS. Cymek's canonical causal objective hard-excludes BOS targets. V11 therefore keeps normal Cymek rendering:

`BOS + prompt + answer digits + EOS`

and supervises answer digits + EOS only. This is a deliberate residual difference. No production objective code is modified for the research experiment.

The compact bridge should therefore be interpreted as: *real Cymek V5 on exact ARK data with compact representation and exposure*, not as an exact ARK reproduction.

## Semantic exposure

ARK reference maximum = **1,152,000 row presentations**.

Target update count is a function of selected batch:

- batch64 → 18,000 updates
- batch32 → 36,000 updates
- batch16 → 72,000 updates

Actual wall may truncate the target. Every result records row presentations, ARK exposure fraction, updates, real tokens and wall time. A null is interpreted relative to actual exposure, not nominal steps.

## Hardware resolver

Cell 0 calibrates COMPACT and PRODUCTION at batch64/32/16 using:

- real model construction;
- real tokenizer/render path;
- real optimizer step;
- candidate-free generation;
- peak VRAM.

The resolver is outcome-blind. It selects the batch expected to maximize semantic exposure within the fixed stage wall, with only a small preference for batch64 when competitive. It never blocks simply because the full target cannot fit.

## Wall allocation

Hard **scientific campaign** wall: 175 minutes. Cell 0 preflight/calibration occurs before the campaign and is expected to remain short; it is not counted as scientific training evidence.

- packaging reserve: 5 minutes
- COMPACT_BRIDGE: max 25 minutes; stops early only on qualified G90
- PRODUCTION_PRIMARY: receives remaining science wall
- PRODUCTION_REPLICATION: starts only if primary qualifies and >=40 minutes remain

This is progressive. Completed evidence is retained even when a later stage timeboxes.

## Seeds

Fixed before outcomes:

- compact model seed 3301, order seed 4701
- production primary model seed 3401, order seed 4701
- production replication model seed 3402, order seed 4702

## Candidate-free capability gates

Evaluation cadence is every 12,800 semantic row presentations, matching 200 ARK batch64 updates in exposure units. All capability and structural candidate-free generation uses a fixed eight-new-token cap.

Milestones:

- M99: train probe >=.99, sustained 3 evaluations
- G50: DEV_CONTROLLER >=.50, sustained 3 evaluations
- G90 controller confirmation: DEV_CONTROLLER >=.90, sustained 3 evaluations
- G90 qualification: controller confirmation has occurred **and** contemporaneous DEV_MEASUREMENT/STANDARD exact-with-valid-EOS >=.90

Controller confirmation is recorded as a transition event but does **not** stop training by itself. A bridge stops early only at qualified G90; otherwise it continues to its semantic-exposure or wall limit. Final decision logic rechecks measurement support defensively.

## Structural battery

Measurement-only; never changes training or stage selection.

- STANDARD: DEV_MEASUREMENT no-carry holdout
- COMMUTED: same examples, operands reversed
- LOCALITY: paired ±1 no-carry counterfactual; score numeric usability, relation consistency, both-exact
- CARRY: unseen carry mechanism
- TRIPLE_ADD: unseen three-addend composition
- THREE_DIGIT: no-carry length extrapolation
- VERBAL: unseen natural-language rendering; production tokenizer only

Threshold flags are reported separately. No aggregate reasoning score is allowed.

Final controller/structural/sealed candidate-free rows are preserved with canonical SHA-256 prediction receipts. Research checkpoints preserve model/optimizer byte SHA-256 values and counters along with their durable Drive paths; the returned ZIP contains the receipts, not duplicated tensor binaries.

## Primary verdict logic

- two production subjects qualify → `PRODUCTION_REPRESENTATION_G90_REPLICATED_DEVELOPMENT`
- primary production qualifies → `PRODUCTION_REPRESENTATION_G90_SINGLE_SEED_DEVELOPMENT`
- compact qualifies, production null but production exposure < compact G90 exposure → `PRODUCTION_UNDEREXPOSED_RELATIVE_TO_COMPACT_G90`
- compact qualifies, production null with exposure >= compact G90 exposure → `EXPOSURE_MATCHED_BRIDGE_DIVERGENCE_COMPACT_G90_PRODUCTION_NO_G90`
- neither qualifies and both reach >=95% ARK exposure → `NO_G90_AT_NEAR_ARK_REFERENCE_EXPOSURE`
- neither qualifies under lower exposure → `NO_G90_WITH_INCOMPLETE_EXPOSURE`
- production qualifies without compact → `PRODUCTION_G90_WITHOUT_COMPACT_G90`

`EXPOSURE_MATCHED_BRIDGE_DIVERGENCE` implicates the compact-vs-production representation burden inside Cymek; it does not isolate vocabulary size, token segmentation, or embedding-parameter burden individually.

## Success / null / refutation

**Strongest useful positive:** production primary + replication both qualify, with clean exposure and structural receipts. Still development evidence only.

**Useful bridge positive:** compact qualifies. This shows Cymek V5 can enter the task's generalization regime under compact representation; it does not prove exact ARK replication.

**Useful negative:** compact receives near-full reference exposure and fails. This shifts attention toward objective/architecture/optimization/precision/init differences rather than dose alone.

**Non-result:** production null under less exposure than compact needed for G90. Report underexposed, not divergence.

## Abort conditions

Abort/fail closed on:

- no CUDA;
- executable/prereg/readiness/hash mismatch;
- tokenizer identity drift;
- ARK manifest split/blob drift;
- data firewall violation;
- live parameter-count mismatch;
- optimizer hyperparameter drift;
- corrupt checkpoint/result path;
- unhandled runtime exception.

Failures must package partial evidence and `FAILURE.json` before re-raising when possible.

## Claim boundary

CYR-GPU-011 cannot authorize broad reasoning/AGI claims, production promotion, PRE500M, TPU certification, the 500M campaign, or any 5B-corpus action. Structural probes characterize one controlled task family only.
