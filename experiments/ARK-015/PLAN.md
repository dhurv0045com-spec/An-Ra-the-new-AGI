# ARK-015 — NON-ARITHMETIC INVARIANCE RETENTION UNDER DISTRIBUTION NARROWING

## Status

**PREREGISTERED BEFORE IMPLEMENTATION / EXECUTION.**

## Why this experiment exists

ARK-014 repaired the non-arithmetic binding subject: deterministic fact-order augmentation produced a checkpoint that was robust to canonical order, reversed order, query changes, and query+order changes. But the retention transfer screen was uninformative because HIGH `1e-3` and LOW `1e-5` both remained qualified on all 3 continuation orders. There was no failure contrast.

The next transfer test should not merely run the same zero-event continuation for longer. It should apply a natural, reproducible training-distribution shift that creates pressure to lose a learned invariant without corrupting weights or changing answer semantics.

ARK-015 therefore asks:

> After robust order-invariant binding has been acquired with order augmentation, does narrowing continuation training to canonical fact order cause HIGH-LR training to lose order robustness more often than LOW-LR training, under identical semantic minibatches and a sealed evaluation firewall?

This is the user-proposed non-arithmetic transfer retry, sharpened into a causal invariance-retention stress test.

## Prior evidence / branch boundary

Read-only evidence used to design this experiment:

- Arkenstone ARK-014: order augmentation qualified a robust binding subject, but HIGH and LOW had 0/3 retention failures.
- Cymek `cymek-500m-readiness` audit at branch HEAD `125b25c19204cce1994deebbfc4957119f2ae31f`: non-arithmetic transfer remains unresolved and zero-event HIGH-vs-LOW repetition is low information.

ARK-015 modifies only `Arkenstone`. It does not modify Cymek or authorize any production scheduler change.

## Frozen task family

Reuse the exact ARK-014 symbolic binding construction:

- keys: 0..5;
- values: 0..5;
- each semantic example contains 3 key=value facts and queries one key;
- complete fact-set split before query expansion;
- task seed: `4242`;
- 400 train fact-sets;
- 50 CONTROL fact-sets;
- 50 SEALED fact-sets;
- all 3 queries per fact-set;
- zero train/test fact-set overlap;
- zero CONTROL/SEALED fact-set overlap.

The implementation must reproduce the ARK-014 task-manifest SHA256:

`fbc8605dc1cfc19ac8692d2b2c6338fcb2af3e02b24907f3b5cca3571947fee3`

Diagnostics remain orthogonalized and reported separately:

- `CANONICAL`
- `ORDER_ONLY`
- `QUERY_ONLY`
- `QUERY_ORDER`

`SEALED` is measurement-only and may never determine acquisition stopping, continuation selection, LR, or whether an arm runs.

## Model / optimizer

Same Micro subject as ARK-014:

- width 128;
- 4 layers;
- 4 heads;
- FFN 512;
- `CompactVocab`;
- AdamW `betas=(0.9,0.95)`, `eps=1e-8`, `weight_decay=0.1`;
- batch 64;
- gradient clip 1.0;
- eval every 200 optimizer steps;
- HIGH = `1e-3`;
- LOW = `1e-5`.

## Fresh independent acquisition seeds

`2301, 2402, 2503`

These seeds are frozen before execution and may not be replaced after seeing outcomes.

Acquisition uses the ARK-014 deterministic `ORDER_AUGMENTED` regime.

CONTROL qualification requires 3 consecutive evaluations with:

- CANONICAL >= 0.90;
- ORDER_ONLY >= 0.85;
- QUERY_ORDER >= 0.85.

Max acquisition box: 16,000 steps.

At CONTROL qualification, snapshot exact model + optimizer + CPU RNG + CUDA RNG. Measure SEALED once for analysis only. SEALED does not gate execution.

## Frozen continuation semantic orders

For every qualified acquisition checkpoint:

`8801, 8802, 8803`

Maximum primary matched pairs: 3 acquisition checkpoints × 3 orders = 9.

Each semantic order is materialized once as semantic-example IDs and SHA256-bound. All arms for that order consume the same semantic example IDs at every optimizer step.

## Retention stress

Continuation horizon: **12,000 optimizer steps**.

The stress is **distribution narrowing**, not parameter corruption:

- during acquisition, fact order is deterministically augmented;
- during the primary stress, every continuation example is rendered in the canonical stored fact order;
- query distribution and answers remain unchanged;
- the semantic training examples are the same task family as acquisition;
- no weight noise, label noise, adversarial gradient, or artificial parameter perturbation is allowed.

### Primary matched arms

From the exact same robust acquisition checkpoint:

1. `NARROW_HIGH`
   - canonical-only continuation rendering;
   - LR `1e-3`.

2. `NARROW_LOW`
   - byte-identical canonical-only continuation rows;
   - LR `1e-5`.

These two arms are the primary causal comparison.

### Stress-specificity reference

3. `AUGMENTED_HIGH_REFERENCE`
   - same semantic-example ID stream;
   - LR `1e-3`;
   - preserves deterministic fact-order augmentation instead of narrowing to canonical order.

This arm is not part of the primary HIGH-vs-LOW risk difference. It tests whether failures are specifically associated with removal of presentation diversity rather than merely longer HIGH-LR training.

## Primary endpoint

Primary endpoint is **SEALED robust-qualification failure** among forks whose SEALED checkpoint is qualified at the acquisition fork.

A SEALED evaluation is qualified iff:

- CANONICAL >= 0.90;
- ORDER_ONLY >= 0.85;
- QUERY_ORDER >= 0.85.

Retention failure = first 3 consecutive post-fork SEALED evaluations that are not qualified.

Report:

- qualified matched NARROW_HIGH/NARROW_LOW pairs;
- HIGH failure count;
- LOW failure count;
- paired risk difference `LOW - HIGH`;
- discordance `HIGH fail / LOW stable`;
- reverse discordance;
- grouped result by independent acquisition seed;
- failure mode decomposition by CANONICAL / ORDER_ONLY / QUERY_ORDER.

## Secondary endpoints

For CONTROL and SEALED, by arm:

- robust qualification retention fraction;
- CANONICAL area/final/peak;
- ORDER_ONLY area/final/peak;
- QUERY_ONLY area/final/peak;
- QUERY_ORDER area/final/peak;
- first failure onset/confirmation;
- relative parameter displacement from the acquisition checkpoint;
- cumulative optimizer-step parameter path length;
- supervised-token count from the loss mask;
- continuation semantic-order SHA256.

For `AUGMENTED_HIGH_REFERENCE`, report the same metrics and compare its failure incidence descriptively with `NARROW_HIGH`.

## Preregistered verdicts

`SUPPORTED_NONARITHMETIC_INVARIANCE_PROTECTION` requires all:

1. at least 6 SEALED-qualified primary pairs;
2. those pairs span all 3 fresh acquisition seeds;
3. NARROW_HIGH has at least 3 SEALED robust-retention failures;
4. paired risk difference `LOW - HIGH <= -0.33`;
5. reverse discordance <= 1;
6. LOW is not worse than HIGH in a majority of acquisition-seed groups.

`INCONCLUSIVE_LOW_EVENT_RATE` if conditions 1–3 are not met.

`TRANSFER_NOT_SUPPORTED` if event rate is sufficient but the protection criterion does not fire.

`STRESS_NOT_SPECIFIC` is an additional flag if `AUGMENTED_HIGH_REFERENCE` fails at a rate within 0.15 of NARROW_HIGH, indicating the stress may mainly expose generic HIGH-LR instability rather than invariance loss from distribution narrowing.

`STRESS_SPECIFIC_TO_NARROWING` is an additional flag if NARROW_HIGH failure rate exceeds AUGMENTED_HIGH_REFERENCE by >=0.30.

No threshold may be changed after execution.

## What a positive result would mean

A positive result would demonstrate at Micro scale that the LOW-LR retention effect extends beyond arithmetic to a non-arithmetic robust binding representation under a controlled presentation-distribution narrowing stress.

It would **not** establish:

- a universal optimizer law;
- protection under arbitrary cross-task continual learning;
- production-language-model benefit;
- an optimal LR threshold;
- a Cymek scheduler promotion;
- AGI.

The main value is to determine whether the retention phenomenon survives a second cognitive task family and a realistic mixture/presentation shift.
