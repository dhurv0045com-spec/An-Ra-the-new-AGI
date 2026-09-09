# ARK-017 — FACTORIAL DISSECTION OF INVARIANCE EROSION

## Status

**PREREGISTERED BEFORE IMPLEMENTATION / EXECUTION.**

## Question

ARK-015 gave Arkenstone a reliable non-arithmetic failure generator: after robust order-invariant binding was acquired, canonical-only continuation at HIGH LR caused SEALED robustness failure in 8/8 matched runs, while LOW and fully order-augmented HIGH caused 0/8 failures.

ARK-017 asks the causal question that ARK-016 could not answer because its T2 event rate was too low:

> Is preservation caused mainly by restricting applied parameter movement, by continuing to supply evidence for the learned invariant, or by an interaction between both?

The experiment must separate these levers while holding semantic training examples, acquisition checkpoints, optimizer family, and evaluation firewall fixed.

## Frozen subject

Reuse the exact ARK-014/015 binding construction and manifest SHA256:

`fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3`

- 400 train fact-sets;
- 50 CONTROL fact-sets;
- 50 SEALED fact-sets;
- all 3 queries per fact-set;
- no train/test or CONTROL/SEALED fact-set overlap;
- deterministic order augmentation;
- diagnostics: CANONICAL, ORDER_ONLY, QUERY_ONLY, QUERY_ORDER.

SEALED is measurement-only and may never determine optimization, intervention activation, arm inclusion, or early stopping.

## Model / acquisition

Same Micro architecture and AdamW contract as ARK-015.

Fresh acquisition seeds:

`2601, 2702, 2803`

Acquisition regime: full deterministic `ORDER_AUGMENTED` training at LR `1e-3`.

CONTROL qualification: 3 consecutive evals satisfying CANONICAL >= .90, ORDER_ONLY >= .85, QUERY_ORDER >= .85.

Maximum acquisition: 16,000 optimizer steps; eval every 200.

At qualification, snapshot exact model + optimizer + CPU/CUDA RNG. Record SEALED once for measurement only.

## Continuation streams

Fresh semantic-order seeds:

`10801, 10802`

Maximum matched sets: 3 independent acquisition checkpoints × 2 orders = **6**.

Continuation horizon: **8,000 optimizer steps**. ARK-015 observed HIGH failure confirmations from 600 to 5,600 steps, so 8k covers the demonstrated event window without spending 12k per arm.

Each matched set materializes one semantic-example stream. Every arm consumes identical semantic IDs at every step. Only presentation rendering and/or applied update magnitude differ according to the frozen arm definition.

## Arms

The LOW arm is executed first only to materialize the per-step applied-delta reference trace. Every other arm starts again from the exact same acquisition snapshot and sees the same semantic stream.

1. `NARROW_HIGH`
   - canonical-only rendering;
   - LR `1e-3`;
   - no applied-delta cap.

2. `NARROW_LOW_REFERENCE`
   - canonical-only rendering;
   - LR `1e-5`;
   - records exact applied-delta norm for every optimizer step.

3. `NARROW_HIGH_CAP1X`
   - canonical-only rendering;
   - LR `1e-3` and HIGH optimizer state;
   - after each AdamW step, globally rescale the applied parameter delta so its L2 norm is at most the matched LOW_REFERENCE delta for that exact step;
   - optimizer moments are not rescaled or replaced.

4. `NARROW_HIGH_REPLAY_1OF16`
   - LR `1e-3`, uncapped;
   - identical semantic IDs;
   - exactly 4 of the 64 examples in each batch (1/16 = 6.25%) are rendered using deterministic alternative fact orders; the other 60 remain canonical;
   - replay positions and alternative permutations are hash-deterministic from acquisition seed, continuation seed, absolute step and batch position.

5. `NARROW_HIGH_CAP1X_REPLAY_1OF16`
   - same 1/16 invariant-supporting presentation replay as arm 4;
   - HIGH LR/optimizer state;
   - same matched LOW per-step applied-delta cap as arm 3.

6. `AUGMENTED_HIGH_REFERENCE`
   - LR `1e-3`, uncapped;
   - full deterministic order augmentation as during acquisition;
   - specificity / upper-support reference, not part of the core 2×2 mechanism attribution.

## Primary endpoint

SEALED robust-qualification failure: first 3 consecutive evals for which the SEALED checkpoint fails the same robust gate used in ARK-015.

Report for each arm:

- failure count/risk;
- robust-retention fraction;
- ORDER_ONLY and QUERY_ORDER area/final;
- canonical area/final;
- failure onset/confirmation;
- cumulative applied path;
- raw optimizer path;
- relative displacement;
- preclip gradient norms;
- cap-fire fraction where applicable;
- supervised tokens.

## Causal contrasts

Only matched sets whose acquisition snapshot is SEALED-qualified are primary.

Define `R_X` as failure risk of arm X over primary matched sets.

### Movement rescue

`movement_rescue = R_NARROW_HIGH - R_NARROW_HIGH_CAP1X`

### Data-support rescue

`support_rescue = R_NARROW_HIGH - R_NARROW_HIGH_REPLAY_1OF16`

### Joint rescue

`joint_rescue = R_NARROW_HIGH - R_NARROW_HIGH_CAP1X_REPLAY_1OF16`

## Preregistered interpretation flags

Event sufficiency requires:

- at least 5 primary matched sets;
- all 3 fresh acquisition seeds represented;
- NARROW_HIGH failure risk >= .60.

If event sufficiency fails: `INCONCLUSIVE_LOW_EVENT_RATE`.

Otherwise:

- `UPDATE_MAGNITUDE_SUFFICIENT` if movement_rescue >= .50, CAP1X risk <= LOW risk + .20, and CAP1X median cumulative path is within [0.7, 1.3] × LOW path.
- `DIVERSITY_SUPPORT_SUFFICIENT` if support_rescue >= .50, replay risk <= LOW risk + .20, and REPLAY arm mean cumulative path >= .50 × NARROW_HIGH path.
- `JOINT_CONTROL_REQUIRED` if neither single-lever criterion fires but joint_rescue >= .50 and joint risk <= LOW risk + .20.
- `BOTH_LEVERS_SUFFICIENT` if both single-lever criteria fire.
- `MECHANISM_MIXED_OR_UNRESOLVED` otherwise.

`AUGMENTED_HIGH_REFERENCE` is expected to remain stable based on ARK-015. If it fails in >= .40 of primary matched sets, attach `REFERENCE_INSTABILITY_WARNING` and weaken all data-support interpretations.

## What changes our training design

- Movement sufficient -> candidate applied-update trust region / state-dependent update budget.
- Data support sufficient -> candidate capability-aware replay or mixture floor.
- Joint required -> hybrid update-budget + replay controller.
- Both sufficient -> select by stability/plasticity efficiency in ARK-019 rather than assuming either is universally superior.

## Claim boundary

ARK-017 is still a Micro controlled-task mechanism experiment. Positive results do not authorize production changes, 500M training, or a universal cognition law. Its job is causal credit: identify which control variable deserves to be carried into a larger/real-data proxy.
