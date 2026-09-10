# CYR-GPU-012 / R1 — REPRESENTATION OUTPUT-BURDEN CAUSAL SCREEN

**Status:** PREREGISTRATION CANDIDATE — execute only after `PREREGISTRATION.json` and `RUN_READINESS.json` are green.  
**Hard wall:** 175 minutes including a 5-minute packaging reserve.  
**Claim ceiling:** controlled development mechanism evidence only.

## Question

CYR-GPU-011 showed a dramatic condition-specific divergence: the exact 19-symbol compact representation reached 56.47% held-out STANDARD at 44.89% of the ARK-002B semantic exposure box, while Cymek's 24,576-token production representation reached 0% held-out STANDARD and 0/48 SEALED after 100% exposure. That result changed several representation variables together.

R1 asks a narrower causal question:

> **If arithmetic tokenization, active token IDs, model block geometry, objective, optimizer, batch, examples, order and shared initialization are fixed, is enlarging only the tied embedding/output class space enough to suppress early held-out capability formation?**

## Primary intervention

All primary arms use the exact CYR-GPU-011 ARK-002B manifest, batch 64, 8,000 updates = **512,000 semantic row presentations**, and the exact active compact IDs `0..18`.

For each matched seed:

- `CHAR_V19`: vocabulary 19; exact compact reference.
- `CHAR_V24576`: vocabulary 24,576, but tokenizer still emits only IDs `0..18`; IDs `19..24575` are unused output/embedding classes.

The Transformer blocks and the first 19 embedding rows are copied from the same V19 reference initialization. Every non-embedding tensor is therefore byte-identical at step 0 across the matched pair; only the additional tied embedding/output rows exist in the large-vocabulary arm.

This intervention deliberately isolates **tied output/embedding class-space burden as a bundle**. Because Cymek ties input embedding and output projection, R1 cannot separately attribute an effect to parameter capacity versus softmax competition.

## Replication and runtime resolver

Fresh model seeds: `3511`, `3512`.  
Fresh order seeds: `5801`, `5802`.  
Primary endpoint never changes to make a run fit.

Cell 0 calibrates V19, V4096 and V24576 at fixed batch 64 before scientific outcomes. The resolver uses only throughput/generation speed and a 1.25 safety factor:

- if two complete V19-vs-V24576 pairs fit the 175-minute wall, run both seeds;
- otherwise run one complete pair;
- if even one complete pair cannot fit, **fail closed** rather than lower exposure or change batch;
- optional `CHAR_V4096` on seed 1 may run only after the primary pair(s) are protected by the wall budget.

Completed arms on Drive are reused on rerun if and only if their exact arm identity and fixed endpoint match. An incompatible existing arm is never silently overwritten.

## Fixed endpoint and metrics

Primary endpoint: **512,000 row presentations**. This is chosen prospectively because it is approximately the regime where CYR-GPU-011's compact bridge had already entered strong partial generalization while production remained at floor.

Primary metric: `DEV_MEASUREMENT/STANDARD complete_exact_with_valid_stop` at exactly 512k rows.

Secondary diagnostics:

- DEV_CONTROLLER trajectory;
- train-probe M99/G50 timing;
- tens/ones digit exactness;
- locality counterfactual relation consistency;
- carry, triple-add and three-digit transfer as separate diagnostics;
- parameter displacement;
- matched-initialization SHA-256 receipts.

The old `COMMUTED` label from CYR-GPU-011 is not treated as a clean invariance claim because the post-run audit showed operand reversal changes the task's OOD role. No aggregate “reasoning score” is allowed.

## Preregistered decision rule

For each complete seed pair define:

`gap = STANDARD(CHAR_V19) - STANDARD(CHAR_V24576)` at 512k rows.

Thresholds:

- compact signal floor: `CHAR_V19 >= 0.45`;
- strong causal gap: `gap >= 0.30`;
- practical equivalence band: `|gap| <= 0.10`.

Verdicts:

- both seeds meet signal floor and strong gap → `REPLICATED_OUTPUT_VOCAB_BURDEN_SUPPORTED_AT_512K_ROWS`;
- both seeds meet signal floor and equivalence band → `REPLICATED_OUTPUT_VOCAB_BURDEN_NOT_PRIMARY_AT_512K_ROWS`;
- one complete seed gives the analogous single-seed label;
- otherwise → `MIXED_OR_INTERMEDIATE_REPRESENTATION_EFFECT`;
- no complete primary pair → `INCONCLUSIVE_NO_COMPLETE_PRIMARY_PAIR`.

The historical CYR-GPU-011 production-BPE result is reported as an external anchor but is **not used by this primary verdict**.

## What changes our mind

If V24576 collapses while V19 forms capability under identical active tokenization and matched shared initialization, large tied output/embedding class-space burden becomes a causally supported explanation of the early regime divergence. Next experiment should then separate tied embedding capacity from output-softmax competition and test a production-compatible factorization.

If V24576 tracks V19 closely, raw class-space size is not the primary cause; BPE segmentation/number atomization or another correlated representation factor becomes the next target. If neither V19 seed reproduces the expected >=0.45 signal, do not interpret the large-vocab comparison; first explain the compact instability/seed effect.

## Authorization boundary

No R1 result alone authorizes a production tokenizer replacement, PRE500M, 500M training, TPU equivalence, broad reasoning, or AGI claims. A mechanistic winner must be prospectively replicated in a production-compatible representation and then on a larger/real-data substrate before promotion.
