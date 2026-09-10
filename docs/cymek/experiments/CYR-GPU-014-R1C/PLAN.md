# CYR-GPU-014 / R1C — FIXED-MATRIX SOFTMAX-COMPETITION MECHANISM DISSECTION

**Status:** PREREGISTRATION CANDIDATE / NOT EXECUTED  
**Target hardware:** Google Colab T4-class CUDA GPU  
**Scientific campaign wall:** 420 minutes, with 10-minute packaging reserve  
**Claim ceiling:** controlled developmental mechanism evidence only.

## Why this experiment exists

R1 and R1B established that changing the declared tied embedding/output class-space size can radically change early structural capability formation even when active arithmetic token IDs and semantic data are held fixed. R1B did not yield a clean universal optimum: V4096 was stable near 0.50 across both fresh seeds, V8192/V16384 were stronger in one seed and weaker in the other, and V19/V1024/V24576 were near zero at the 128k-row endpoint.

The unresolved causal question is now narrower:

> Is the capability-formation effect driven primarily by training-time inactive-class competition / softmax-normalizer geometry, rather than by the physical size of the tied embedding matrix itself?

R1C keeps the **physical model identical at vocabulary 24,576 in every arm**. Only the training-logit treatment changes. Full-vocabulary candidate-free evaluation is always reported. A separately labelled active-only diagnostic is also frozen prospectively so we can distinguish genuine structural formation from an output-calibration failure caused by inactive rows winning at inference.

## Fixed substrate

Every arm uses:

- exact CYR-GPU-011 copy of ARK-002B T2 no-carry arithmetic;
- split SHA-256 `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`;
- Cymek V5 RESEARCH_SMALL geometry: 4 layers, width 128, 4 Q heads, 2 KV heads, FFN 512;
- **physical vocabulary / tied embedding rows = 24,576 for every arm**;
- active tokenizer IDs exactly `0..18`; no inactive token is emitted by the tokenizer;
- identical prompt/answer character segmentation;
- batch 64;
- AdamW family/scalars and LR `1e-3`;
- answer + EOS objective unchanged;
- identical semantic-example stream within a matched seed;
- identical full model initialization bytes within a matched seed;
- fixed 3,000 optimizer updates = **192,000 semantic row presentations per arm**;
- no early stop after generalization.

Fresh prospective matched seeds:

- model seeds: `3711, 3712, 3713, 3714`;
- order seeds: `6001, 6002, 6003, 6004`.

Four complete matched curves are mandatory. No seed, arm, threshold, treatment dose, or endpoint may be dropped after any scientific outcome is observed.

## Training-logit arms

All arms own the same 24,576-row tied matrix. The treatment is applied only during training; ordinary evaluation always uses the full unmodified 24,576 logits.

1. `FULL_24576`
   - normal full softmax over all 24,576 logits.

2. `MASK_19`
   - only logits `0..18` participate in training CE;
   - logits `19..24575` are replaced by a finite negative sentinel before CE.

3. `MASK_4096` — **primary intervention**
   - logits `0..4095` participate in training CE;
   - physical model remains 24,576 rows.

4. `MASK_8192`
   - logits `0..8191` participate.

5. `MASK_16384`
   - logits `0..16383` participate.

6. `OFFSET_EQ4096`
   - all 24,576 logits remain in the denominator;
   - inactive logits `19..24575` receive the fixed subtraction
     `log((24576-19)/(4096-19))` during training;
   - under equal-logit conditions this approximately reduces total inactive partition mass to that of a 4096-class system while retaining gradient flow to every inactive row.

The hard-mask arms manipulate candidate participation plus output-gradient competition. The offset arm asks whether aggregate inactive probability mass alone is sufficient.

## Why MASK_4096 is primary

R1 produced V4096 = 1.00 at 512k rows, and R1B produced 0.5059 / 0.4941 in two fresh seeds at 128k rows. That makes 4096 the most stable previously observed intermediate condition rather than a post-hoc best arm from only one seed. V8192/V16384 remain dose-response diagnostics and cannot rescue a failed primary pair.

## Evaluation schedule

Evaluate every 150 updates (9,600 semantic rows) and at the fixed 3,000-update endpoint.

Every scheduled evaluation reports:

- DEV_CONTROLLER full-vocabulary candidate-free exact + valid EOS;
- DEV_MEASUREMENT full-vocabulary STANDARD exact + valid EOS;
- TRAIN probe full-vocabulary exact;
- **ACTIVE_ONLY_DIAGNOSTIC** on the same DEV_MEASUREMENT examples, restricting inference candidates to IDs `0..18`. This is a mechanism diagnostic, never the deployment/functionality score.

Full structural batteries run at baseline, update 1500, and update 3000 only. They remain controlled diagnostics, not broad reasoning scores.

## Dense mechanism diagnostics

At updates `0, 300, 600, 900, 1500, 2100, 3000`, on a frozen TRAIN diagnostic minibatch, record:

- target-token probability;
- total active probability mass (`IDs 0..18`);
- total inactive probability mass (`IDs 19..24575`);
- max inactive probability and max inactive logit;
- target-vs-best-active-wrong logit margin;
- target-vs-best-inactive logit margin;
- full-softmax entropy;
- active-only entropy;
- final hidden-state L2 statistics;
- gradient L2 for active embedding rows;
- gradient L2 for participating inactive rows;
- gradient L2 for excluded inactive rows;
- gradient L2 for all non-embedding/core parameters;
- core-gradient cosine under the arm's loss versus counterfactual `FULL_24576` and `MASK_4096` losses at exactly the same model state and batch.

The diagnostic must mechanically verify that model and optimizer state are byte-identical before versus after the counterfactual probe. No diagnostic gradient may enter an optimizer step.

## Primary outcome metrics

R1B showed that generalized states can emerge and later collapse, so endpoint-only scoring is inadequate. For **both** the full-vocabulary functional metric and the active-only structural diagnostic, compute from updates 600..3000:

1. `FORMATION_AUC`: mean STANDARD exact across all scheduled evaluations.
2. `SUSTAINED_G50`: first update of 3 consecutive evaluations with STANDARD >= 0.50.
3. `ENDPOINT_STANDARD`: score at update 3000.
4. `PEAK_3EVAL_STANDARD`: highest 3-evaluation moving average.

### Frozen primary causal pair

`MASK_4096` versus `FULL_24576` on the same physical model and matched seed.

The same paired threshold is evaluated separately on the structural and functional metrics:

- at least 3/4 matched seeds have paired FORMATION_AUC gap >= 0.20;
- mean paired AUC gap across all 4 seeds >= 0.25;
- `MASK_4096` reaches sustained G50 in at least 2/4 seeds;
- `FULL_24576` reaches sustained G50 in at most 1/4 seeds.

Verdicts:

- `SOFTMAX_COMPETITION_FUNCTIONAL_AND_STRUCTURAL_SUPPORTED` if both structural and full-vocabulary tests pass;
- `SOFTMAX_COMPETITION_STRUCTURAL_SUPPORTED_OUTPUT_CALIBRATION_LIMITED` if the active-only structural test passes but the full-vocabulary functional test does not;
- `SOFTMAX_COMPETITION_NOT_SUFFICIENT` if both mean paired AUC gaps are <0.10 with no material sustained-G50 advantage;
- otherwise `MIXED_SOFTMAX_COMPETITION_EFFECT`.

This dual endpoint prevents a false negative where masking helps the network learn the arithmetic structure but untrained inactive output rows still win during unrestricted inference. It also prevents us from calling such a model functionally rescued when it is not.

## Secondary mechanistic decisions

These cannot alter the primary verdict.

### Fixed-matrix dose response

Report whether `MASK_19 -> MASK_4096 -> MASK_8192 -> MASK_16384 -> FULL_24576` produces a reproducible response curve when parameter count and all initialization bytes are fixed.

### Inactive-partition-mass rescue

For both structural and functional endpoints, flag partition-mass rescue if `OFFSET_EQ4096` improves AUC over FULL_24576 by >=0.20 in at least 3/4 seeds and its mean AUC lies within 0.15 of MASK_4096.

If hard MASK_4096 works but OFFSET_EQ4096 does not, literal output-row participation / gradient competition is favored over a simple aggregate-mass explanation.

### Training-vs-inference decomposition

If active-only capability is high but full-vocabulary capability is poor and inactive predictions dominate, label the failure `OUTPUT_CALIBRATION_BOTTLENECK`. If both are poor, label `STRUCTURAL_FORMATION_BOTTLENECK`.

## Runtime and durability

This is intentionally much larger than R1/R1B:

- 4 seeds × 6 arms × 3,000 updates = **72,000 optimizer updates**;
- **4.608 million semantic row presentations**;
- expected T4 runtime is several hours;
- pre-outcome CUDA calibration measures training, full-vocabulary generation, and dense diagnostic cost;
- the full fixed campaign must conservatively fit the **410-minute scientific portion** of the 420-minute wall before scientific updates begin;
- otherwise the pre-execution gate fails. Scientific exposure is never reduced to fit hardware.

Durability requirements:

- Drive-backed per-arm checkpoint every 500 updates;
- model, optimizer, update counter, real-token counter, exact semantic-stream identity, treatment identity, trace, diagnostic state, CPU RNG and CUDA RNG;
- exact resume of interrupted arms;
- completed-arm reuse only on exact identity match;
- incompatible artifacts abort rather than overwrite;
- pre-execution CUDA smoke requires 10 uninterrupted updates to be byte-identical at model/optimizer/counters to 5 + save/load + 5;
- any timebox/failure packages partial evidence instead of silently discarding it.

## Evidence boundaries

A positive R1C would demonstrate that manipulating training-time output competition on a **fixed 24,576-row tied model** is sufficient to alter structural and/or functional capability formation on this controlled developmental task.

It would **not** establish a universal natural-language vocabulary rule, production tokenizer change, broad reasoning improvement, PRE500M/500M readiness, or AGI.

A negative R1C is equally useful: it would push the causal search away from training-softmax competition toward tied initialization/optimization geometry or other representation interactions.
