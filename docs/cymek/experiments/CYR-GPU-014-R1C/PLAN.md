# CYR-GPU-014 / R1C — FIXED-MATRIX SOFTMAX-COMPETITION MECHANISM DISSECTION

**Status:** PREREGISTRATION CANDIDATE / NOT EXECUTED  
**Target hardware:** Google Colab T4-class CUDA GPU  
**Scientific campaign wall:** 360 minutes, with 10-minute packaging reserve  
**Claim ceiling:** controlled developmental mechanism evidence only.

## Why this experiment exists

R1 and R1B established that changing the declared tied embedding/output class-space size can radically change early structural capability formation even when active arithmetic token IDs and semantic data are held fixed. R1B did not yield a clean universal optimum: V4096 was stable near 0.50 across both fresh seeds, V8192/V16384 were stronger in one seed and weaker in the other, and V19/V1024/V24576 were near zero at the 128k-row endpoint.

The unresolved causal question is now narrower:

> Is the capability-formation effect driven primarily by how many inactive output classes compete in the cross-entropy normalizer / how much probability mass they absorb, rather than by the physical size of the tied embedding matrix itself?

R1C keeps the **physical model identical at vocabulary 24,576 in every arm**. Only the training-logit treatment changes. Full-vocabulary candidate-free evaluation is always unmodified.

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

Four complete matched curves are mandatory. No seed may be dropped after any scientific outcome is observed.

## Training-logit arms

All arms own the same 24,576-row tied matrix. The treatment is applied only while `model.training == True`; evaluation always uses the full unmodified 24,576 logits.

1. `FULL_24576`
   - normal production-style full softmax over all 24,576 logits.

2. `MASK_19`
   - only logits `0..18` participate in training CE;
   - logits `19..24575` are set to a numerically safe negative sentinel before CE.

3. `MASK_4096` — **primary intervention**
   - logits `0..4095` participate in training CE;
   - physical model remains 24,576 rows.

4. `MASK_8192`
   - logits `0..8191` participate.

5. `MASK_16384`
   - logits `0..16383` participate.

6. `OFFSET_EQ4096`
   - all 24,576 logits remain in the denominator;
   - inactive logits `19..24575` receive a fixed pre-registered subtraction
     `log((24576-19)/(4096-19))` during training;
   - this approximately equalizes the aggregate inactive partition contribution to a 4096-class system under equal-logit conditions while retaining gradient flow to every inactive row.

The offset arm separates a **probability-mass / normalizer hypothesis** from a literal hard candidate-count hypothesis.

## Why MASK_4096 is primary

R1 produced V4096 = 1.00 at 512k rows, and R1B produced 0.5059 / 0.4941 in two fresh seeds at 128k rows. That makes 4096 the most stable previously observed intermediate condition rather than a post-hoc best arm from only one seed. V8192 and V16384 remain dose-response diagnostics.

## Evaluation schedule

Evaluate every 150 updates (9,600 semantic rows) and at the fixed 3,000-update endpoint.

Primary candidate-free evaluation uses **full V24576 logits** and reports:

- DEV_CONTROLLER exact + valid EOS;
- DEV_MEASUREMENT STANDARD exact + valid EOS;
- train-probe exact;
- error decomposition: active wrong token vs inactive-token prediction vs stop failure.

Structural batteries remain secondary diagnostics and are not broad reasoning scores.

## Dense mechanism diagnostics

At updates `0, 300, 600, 900, 1500, 2100, 3000`, evaluate a frozen diagnostic minibatch and record before any additional optimizer update:

- target-token probability;
- total active probability mass (`IDs 0..18`);
- total inactive probability mass (`IDs 19..24575`);
- max inactive probability and max inactive logit;
- target-vs-best-active-wrong logit margin;
- target-vs-best-inactive logit margin;
- full-softmax entropy;
- active-only entropy;
- hidden-state final-position L2 statistics if exposed without changing training math;
- gradient L2 for active embedding rows;
- gradient L2 for participating inactive rows;
- gradient L2 for excluded inactive rows;
- gradient L2 for all non-embedding/core parameters;
- cosine similarity of the **core gradient** under the arm's training loss versus counterfactual `FULL_24576` and `MASK_4096` losses on the same model state and same diagnostic batch.

Counterfactual-gradient probes must snapshot/restore gradients and must not mutate optimizer state, RNG state, model parameters, or training counters. A pre-execution test verifies this.

## Primary outcome metrics

Because R1B showed that generalized states can emerge and then collapse, endpoint-only scoring is insufficient.

For each arm/seed compute:

1. `FORMATION_AUC`: mean DEV_MEASUREMENT STANDARD exact over all scheduled evaluations from update 600 through 3000.
2. `SUSTAINED_G50`: first update of 3 consecutive scheduled evaluations with STANDARD >= 0.50; null if never reached.
3. `ENDPOINT_STANDARD`: full-vocab STANDARD exact at update 3000.
4. `PEAK_3EVAL_STANDARD`: highest 3-evaluation moving average.

### Primary causal comparison

`MASK_4096` versus `FULL_24576` on the same physical model and matched seed.

`SOFTMAX_COMPETITION_CAUSALLY_SUPPORTED` requires all:

- at least 3/4 matched seeds have `FORMATION_AUC(MASK_4096) - FORMATION_AUC(FULL_24576) >= 0.20`;
- mean paired AUC gap across all 4 seeds >= 0.25;
- `MASK_4096` reaches `SUSTAINED_G50` in at least 2/4 seeds;
- `FULL_24576` reaches `SUSTAINED_G50` in at most 1/4 seeds.

`SOFTMAX_COMPETITION_NOT_SUFFICIENT` requires mean paired AUC gap < 0.10 and no material sustained-G50 advantage.

Otherwise: `MIXED_SOFTMAX_COMPETITION_EFFECT`.

The primary verdict is frozen to this pair. MASK_8192/MASK_16384 cannot rescue a failed MASK_4096 primary verdict by post-hoc selection.

## Secondary mechanistic decisions

These do not alter the primary verdict.

### Dose response

Report whether `MASK_19 -> MASK_4096 -> MASK_8192 -> MASK_16384 -> FULL_24576` yields a reproducible non-monotonic or monotonic formation curve when the physical matrix is fixed.

### Partition-mass rescue

`INACTIVE_PARTITION_MASS_RESCUE_SUPPORTED` if OFFSET_EQ4096 improves FORMATION_AUC over FULL_24576 by >=0.20 in at least 3/4 seeds and its mean AUC lies within 0.15 of MASK_4096.

If MASK_4096 works but OFFSET_EQ4096 does not, literal candidate participation / gradient-row competition is favored over a simple aggregate-mass explanation.

### Training-vs-inference failure decomposition

If an arm has high active-only accuracy but poor full-vocab accuracy because inactive classes win at inference, label `OUTPUT_CALIBRATION_BOTTLENECK`. If both active-only and full-vocab accuracy remain poor, label `STRUCTURAL_FORMATION_BOTTLENECK`.

## Runtime and durability

This is intentionally larger than R1/R1B.

- 4 seeds × 6 arms × 3,000 updates = **72,000 optimizer updates**;
- 4.608 million semantic row presentations;
- expected T4 duration is several hours, not a short screen;
- pre-outcome CUDA calibration measures representative training + full-vocab generation + diagnostic cost;
- the full mandatory campaign must conservatively fit the 360-minute scientific wall before the first scientific update;
- otherwise fail closed. Do not reduce seeds, arms, updates, diagnostics, or thresholds after seeing outcomes.

Durability requirements:

- Drive-backed per-arm checkpoint every 500 updates;
- model, optimizer, update counter, semantic-stream identity, trace, treatment identity, torch CPU/CUDA RNG state;
- exact resume from partial arms;
- completed-arm reuse only on exact identity match;
- incompatible partial output aborts rather than overwrites;
- pre-execution exact-resume smoke: uninterrupted 10 updates must match 5 + save/load + 5 at model/optimizer hash and metric level.

## Evidence boundaries

A positive R1C would demonstrate that manipulating training softmax competition on a **fixed 24,576-row tied model** is sufficient to alter capability formation on this controlled developmental task. It would still not establish a universal natural-language vocabulary rule, a production tokenizer change, broad reasoning, PRE500M/500M readiness, or AGI.

A negative R1C is equally useful: it would push the causal search away from normalizer competition toward tied initialization/optimization geometry or other representation interactions.
