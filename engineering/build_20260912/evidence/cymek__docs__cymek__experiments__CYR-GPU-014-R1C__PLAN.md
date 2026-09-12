# CYR-GPU-014 / R1C — FIXED-MATRIX SOFTMAX-COMPETITION MECHANISM DISSECTION

**Status:** PREREGISTRATION CANDIDATE / NOT EXECUTED  
**Target hardware:** Google Colab T4-class CUDA GPU  
**Per-session wall:** 330 minutes, with 10-minute packaging reserve  
**Campaign size:** 72,000 optimizer updates / 4.608M semantic row presentations; exact-resumable across sessions  
**Claim ceiling:** controlled developmental mechanism evidence only.

## Why this experiment exists

R1 and R1B established that declared tied embedding/output class-space size can radically change early structural capability formation while active arithmetic token IDs and semantic data are fixed. R1B did not yield a universal optimum: V4096 was stable near 0.50 in both fresh seeds, V8192/V16384 were stronger in one seed and weaker in the other, and V19/V1024/V24576 were near zero at the 128k-row endpoint.

The next question is causal rather than another vocabulary sweep:

> Is the capability-formation effect driven primarily by training-time inactive-class competition / softmax-normalizer geometry, rather than by physical tied-matrix size itself?

R1C keeps the **physical model identical at vocabulary 24,576 in every arm**. Only training-logit treatment changes. Full-vocabulary candidate-free evaluation is always reported. A separately labelled active-only diagnostic is frozen prospectively to distinguish structural formation from output-calibration failure caused by inactive rows winning at inference.

## Fixed substrate

Every arm uses:

- exact CYR-GPU-011 copy of ARK-002B T2 no-carry arithmetic;
- split SHA-256 `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`;
- Cymek V5 RESEARCH_SMALL: 4 layers, width 128, 4 Q heads, 2 KV heads, FFN 512;
- **physical vocabulary / tied embedding rows = 24,576 for every arm**;
- active tokenizer IDs exactly `0..18`; no inactive token is emitted by the tokenizer;
- identical prompt/answer character segmentation;
- batch 64;
- AdamW, LR `1e-3`, betas `(0.9,0.95)`, eps `1e-8`, weight decay `0.1`, global clip `1.0`;
- answer + EOS causal CE objective;
- identical semantic-example stream within each matched seed;
- identical full model initialization bytes within each matched seed;
- fixed 3,000 optimizer updates = **192,000 semantic row presentations per arm**;
- no outcome-based early stopping.

Fresh prospective matched seeds:

- model seeds: `3711, 3712, 3713, 3714`;
- order seeds: `6001, 6002, 6003, 6004`.

Four complete matched curves are mandatory. No seed, arm, threshold, dose, or endpoint may be dropped after scientific outcomes exist.

## Training-logit arms

All arms own the same 24,576-row tied matrix. Treatment is applied only during training; ordinary evaluation always uses the full unmodified 24,576 logits.

1. `FULL_24576` — normal full softmax.
2. `MASK_19` — only logits `0..18` participate in training CE.
3. `MASK_4096` — **primary intervention**; logits `0..4095` participate.
4. `MASK_8192` — logits `0..8191` participate.
5. `MASK_16384` — logits `0..16383` participate.
6. `OFFSET_EQ4096` — all 24,576 logits remain; inactive logits `19..24575` receive fixed subtraction `log((24576-19)/(4096-19))` during training. Under equal logits this approximates the aggregate inactive partition mass of a 4096-class system while retaining gradient flow to every inactive row.

Hard masks manipulate denominator competition plus output-row gradient participation. The offset arm tests whether aggregate inactive partition mass alone is enough.

## Why MASK_4096 is primary

R1 produced V4096 = 1.00 at 512k rows. R1B produced V4096 = 0.5059 / 0.4941 in two fresh seeds at 128k rows. That makes 4096 the most stable prior intermediate condition rather than the post-hoc best of one seed. V8192/V16384 are dose-response diagnostics and cannot rescue a failed primary pair.

## Evaluation schedule

Evaluate every 150 updates and at the fixed 3,000-update endpoint. Every scheduled evaluation reports:

- DEV_CONTROLLER full-vocabulary candidate-free exact + valid EOS;
- DEV_MEASUREMENT full-vocabulary STANDARD exact + valid EOS;
- TRAIN probe full-vocabulary exact;
- `ACTIVE_ONLY_DIAGNOSTIC` on the same DEV_MEASUREMENT examples with inference candidates restricted to IDs `0..18`. This is a mechanism diagnostic, never the functional/deployment score.

Full structural batteries run only at baseline, update 1500, and update 3000. They are controlled diagnostics, not broad-reasoning scores.

## Dense mechanism diagnostics

At updates `0, 300, 600, 900, 1500, 2100, 3000`, on a frozen TRAIN diagnostic minibatch, record:

- target probability;
- active and inactive probability mass;
- max inactive probability/logit;
- target-vs-best-active-wrong and target-vs-best-inactive margins;
- full and active-only entropy;
- final hidden-state L2 statistics;
- active, participating-inactive, excluded-inactive embedding-gradient L2;
- non-embedding/core-gradient L2;
- core-gradient cosine under the arm loss versus counterfactual FULL_24576 and MASK_4096 losses on the exact same model state/batch.

The diagnostic must prove model and optimizer state are byte-identical before/after. No counterfactual diagnostic gradient may enter an optimizer step.

## Primary outcomes

R1B showed generalized states can emerge and later collapse, so endpoint-only scoring is inadequate. For **both** full-vocabulary functional performance and active-only structural performance, compute over scheduled evaluations from update 600 through 3000:

1. `FORMATION_AUC` — mean STANDARD exact.
2. `SUSTAINED_G50` — first update of 3 consecutive evaluations with STANDARD >= 0.50.
3. `ENDPOINT_STANDARD` at update 3000.
4. `PEAK_3EVAL_STANDARD` — highest 3-evaluation moving average.

### Frozen primary causal pair

`MASK_4096` versus `FULL_24576` on the same physical model and matched seed.

The same paired threshold is applied separately to structural and functional endpoints:

- at least 3/4 seeds have paired FORMATION_AUC gap >= 0.20;
- mean paired AUC gap >= 0.25;
- MASK_4096 reaches sustained G50 in >=2/4 seeds;
- FULL_24576 reaches sustained G50 in <=1/4 seeds.

Verdicts:

- `SOFTMAX_COMPETITION_FUNCTIONAL_AND_STRUCTURAL_SUPPORTED` if both tests pass;
- `SOFTMAX_COMPETITION_STRUCTURAL_SUPPORTED_OUTPUT_CALIBRATION_LIMITED` if structural passes but full-vocabulary functional does not;
- `SOFTMAX_COMPETITION_NOT_SUFFICIENT` if both mean paired gaps are <0.10 with no material G50 advantage;
- otherwise `MIXED_SOFTMAX_COMPETITION_EFFECT`.

The dual endpoint prevents two opposite errors: calling a structurally learned model a failure only because inactive rows still win at inference, or calling it functionally rescued when unrestricted inference still fails.

## Secondary mechanism decisions

These cannot alter the primary verdict.

- **Fixed-matrix dose response:** report MASK_19 -> MASK_4096 -> MASK_8192 -> MASK_16384 -> FULL_24576.
- **Inactive-partition-mass rescue:** separately for structural and functional endpoints, OFFSET_EQ4096 must beat FULL by >=0.20 AUC in >=3/4 seeds and have mean AUC within 0.15 of MASK_4096.
- **Failure decomposition:** high active-only but poor full-vocab with inactive predictions => `OUTPUT_CALIBRATION_BOTTLENECK`; both poor => `STRUCTURAL_FORMATION_BOTTLENECK`.

## Runtime and durability

R1C is intentionally **not weakened to fit one Colab session**.

- total fixed campaign = 4 × 6 × 3000 = **72,000 updates**;
- total fixed exposure = **4.608M semantic rows**;
- pre-outcome T4 calibration estimates total wall and number of sessions but may **not** alter seeds, arms, exposure, diagnostics, or thresholds;
- each execution session has a 330-minute wall with 10 minutes reserved for durable packaging;
- if the campaign is incomplete at the wall, the current arm is checkpointed and the session emits a partial evidence bundle; the next session resumes the exact frozen protocol;
- Drive checkpoint every 500 updates, plus an immediate checkpoint before a controlled timebox;
- checkpoint binds model, optimizer, counters, exact semantic stream, treatment, diagnostic trace, CPU RNG and CUDA RNG;
- completed-arm reuse is allowed only on exact identity match; incompatible artifacts abort;
- CUDA preflight requires 10 uninterrupted updates to be byte-identical to 5 + save/load + 5 at model/optimizer hashes and counters;
- no scientific result may be interpreted until all 24 arms complete.

This multi-session policy is part of the preregistration. Runtime variability changes only **how many sessions are required**, never the science.

## Evidence boundaries

A positive R1C would demonstrate that manipulating training-time output competition on a **fixed 24,576-row tied model** is sufficient to alter structural and/or functional capability formation on this controlled developmental task.

It would **not** establish a universal natural-language vocabulary rule, production tokenizer change, broad reasoning improvement, PRE500M/500M readiness, or AGI.

A negative R1C is equally useful: it would push the causal search away from training-softmax competition toward tied initialization/optimization geometry or other representation interactions.
