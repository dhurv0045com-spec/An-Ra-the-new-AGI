# CYR-GPU-013 / R1B — REPLICATED VOCABULARY RESPONSE CURVE

**Status:** PREREGISTRATION CANDIDATE / NOT EXECUTED  
**Target hardware:** Colab T4-class CUDA GPU  
**Campaign wall:** 175 minutes with 5-minute packaging reserve  
**Claim ceiling:** controlled developmental representation-mechanism evidence only.

## Why this experiment exists

CYR-GPU-012 / R1 produced a striking one-seed result at the matched 512k-row endpoint: V19 = 12.94% STANDARD, V4096 = 100%, V24576 = 0%. This falsified the simple monotonic stories `smaller vocabulary is always better` and `more classes/parameters are always better`, but it did not establish that the V4096 result replicates or reveal the shape of the response curve.

R1B asks:

> With active arithmetic tokenization held exactly fixed, is there a reproducible intermediate tied embedding/output class-space regime that forms held-out capability earlier and more reliably than both very small and very large class spaces?

## Fixed causal substrate

All arms use the same frozen CYR-GPU-011 copy of ARK-002B T2 no-carry addition:

- split SHA-256: `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`;
- 500 TRAIN rows;
- 64 DEV_CONTROLLER rows;
- 85 DEV_MEASUREMENT rows;
- previously consumed SEALED rows are not used for a new sealed claim;
- Cymek V5 RESEARCH_SMALL block geometry: 4 layers, width 128, 4 query heads, 2 KV heads, FFN 512;
- exact active compact token IDs `0..18` and identical prompt/answer segmentation;
- batch 64;
- AdamW family/scalars and LR 1e-3;
- answer/EOS semantics unchanged;
- same semantic row stream within every matched seed;
- same model seed within every matched seed;
- all non-embedding initialization and active embedding rows 0..18 copied from the V19 reference within a matched seed.

Only the **declared tied embedding/output vocabulary class count** changes.

## Frozen response-curve levels

Primary levels:

`19, 1024, 4096, 8192, 16384, 24576`

Every executed arm runs exactly:

- 2,000 optimizer updates;
- batch 64;
- **128,000 semantic row presentations**;
- fixed endpoint even if capability emerges early.

The endpoint is intentionally an early-formation screen. R1's V4096 subject reached 100% STANDARD by update 400, so 2,000 updates provides a large post-emergence margin without spending the campaign on late V19 dynamics.

## Replication

Fresh matched seeds:

- model seeds: `3611, 3612, 3613`;
- order seeds: `5901, 5902, 5903`.

Two complete six-level matched curves are mandatory. A third curve is executed only if pre-outcome hardware calibration conservatively predicts that it fits inside the wall. If two complete curves do not fit, the experiment fails closed before scientific training rather than lowering exposure or dropping levels.

Arm order is counterbalanced to reduce simple runtime/thermal ordering bias:

- seed 1: ascending vocabulary;
- seed 2: descending vocabulary;
- seed 3: rotated/interleaved order.

## Primary metric

`DEV_MEASUREMENT / STANDARD complete_exact_with_valid_stop` at exactly 128,000 row presentations.

For each seed define:

- `INTERMEDIATE_BEST = max(V1024, V4096, V8192, V16384)`;
- `EXTREME_BEST = max(V19, V24576)`;
- `INTERMEDIATE_GAP = INTERMEDIATE_BEST - EXTREME_BEST`.

## Primary decision

`REPLICATED_INTERMEDIATE_CLASS_SPACE_ADVANTAGE` requires both mandatory seeds to satisfy:

- `INTERMEDIATE_BEST >= 0.60`; and
- `INTERMEDIATE_GAP >= 0.30`.

`NO_REPLICATED_INTERMEDIATE_ADVANTAGE` if both mandatory seeds have `INTERMEDIATE_GAP <= 0.10` or no intermediate arm exceeds 0.60.

Otherwise: `MIXED_OR_SEED_SENSITIVE_RESPONSE_CURVE`.

A third seed, if run, strengthens/weakens confidence but cannot rescue failure of the two mandatory prospective curves by post-hoc selection.

## Secondary questions

Report, without changing the primary verdict:

1. Which vocabulary size is the per-seed argmax?
2. Is there a broad high-performing plateau or a narrow peak?
3. Where does performance collapse on the high-vocabulary side?
4. How seed-sensitive are V19 and V4096?
5. Does parameter displacement correlate monotonically with capability? R1 suggests no.
6. M99/G50 timing and tens/ones digit decomposition.
7. Structural transfer batteries remain diagnostics only; they are not a broad reasoning score.

## Runtime policy

Before any scientific update, benchmark all six levels at fixed batch64. Estimate training + candidate-free evaluation cost with a 1.35x safety factor. Run exactly two curves if they fit; optionally three if all three fit conservatively. Never change vocabulary levels, endpoints, thresholds, or seeds after seeing scientific outputs.

Completed arm reuse is allowed only when experiment identity, seed, vocabulary, batch, and exact 2,000-update endpoint all match. Incompatible artifacts abort.

## Evidence boundaries

A positive result would support a reproducible **intermediate tied class-space advantage on this Micro/Cymek arithmetic developmental task**. It would not prove a universal optimal vocabulary, a natural-language tokenizer rule, broad reasoning, AGI, or readiness for 500M training.

No production tokenizer change, PRE500M, 500M training, or AGI claim is authorized by this experiment alone.
