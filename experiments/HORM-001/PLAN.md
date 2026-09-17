# HORM-001 — Final pre-run protocol (revision 3)

## Status and scope

Frozen before the first 1,500-update measured lane. Earlier drafts were inconsistent and are superseded; `PLAN_initial_draft.md` preserves the initial proposal. Revision 2's text-token rendering, split, and evaluator descriptions were not implemented correctly. No measured experiment used them. Five runner tests now pass, including a two-update-per-arm engineering smoke test on four probe pairs. Those smoke outputs are excluded from this experiment. This is engineering/development-scale evidence, not cognition, AGI, production readiness, or authorization to launch a frontier experiment.

## Architecture discovery and choice

Real V5: tied embedding/head; pre-norm RMSNorm; grouped-query attention with affine QK normalization then per-segment RoPE; SwiGLU; stateless tensor-in/logits-out forward. Source: `v5_model/{core,block,attention}.py`. No existing hormonal hook was found in that production package. No production file is modified.

Chosen intervention: scale queries after QK normalization and RoPE, before attention, by `exp(-log_T)` per query head, shared across layers. Pre-normalization scaling would largely be removed by normalization. Residual additive bias is a broader representation intervention; additional per-layer scalars alter the canonical inventory; interoceptive tokens alter input semantics; logit bias does not test attention. The wrapper explicitly reuses real V5 modules, not a surrogate model.

`log_T = clamp(W @ (h * active_mask), -1.5, 1.5)`. W is zero-initialized. Four channels affect forward: dopamine, cortisol, serotonin, adrenaline. Oxytocin/GABA/norepinephrine columns exist but are structurally masked from forward and gradients (not absent from the parameter inventory). W has 2 x 7 = 14 stored/trainable parameters at tiny geometry, eight functionally active. The production-sized sibling stores 98 extra parameters. Exact initial no-op, gradient reach, checkpoint snapshots, config binding, and checkpoint mismatch rejection have focused tests.

## Frozen tiny task and encoding

Modulus 97; all 9,409 ordered operand pairs. Data generator: `torch.randperm` with a local generator seed 424242, independent of model seed. First 512 shuffled pairs are evaluation-only; remaining 8,897 are training-only. Assert disjointness and exact coverage. Same split and ordered training stream for every lane; hash both lists.

Integer token fixture, not natural-language text or a new production tokenizer:
- PAD=0, BOS=1, SEP=2, value v in [0,96] maps to v+3, EOS=100.
- Input `[BOS,a+3,SEP,b+3,SEP]`.
- Training row appends `[(a+b)%97+3,EOS]`.
- Tiny vocabulary 106 (unused IDs remain eligible predictions); frozen production vocabulary 24,576 is unchanged.
- Tiny spec: 2 layers, width 64, 2 query heads, 1 KV head, head dimension 32, FFN width 128, context 16. Canonical V5 architecture modules, randomly initialized; no trained SequenceGPT checkpoint is used.

## Matched arms and schedule

Seeds: 424242 and 424243. Sequential execution order: OFF then ON for each seed. OFF is real tiny V5; ON is the composite experimental sibling. Identical seed, core initialization, data order, geometry, optimizer, loss, clipping and evaluation schedule. Hash initial shared core tensors and assert equality across arms; assert exact zero-projection forward equality before ON training.

1,500 updates per lane, batch 16. Traverse the fixed training list cyclically with sample index `((update-1)*16+i) % 8897`. AdamW: lr=.003, betas=(.9,.999), eps=1e-8, weight_decay=.01, constant LR, no warmup. All parameters participate; clip global gradient norm at 1.0, reject nonfinite norm. Production `causal_lm_loss`: all six non-BOS targets per row, including EOS. No answer-only loss or auxiliary objective. No tuning, early stopping on score, or extra variants after results.

## State update and leakage prevention

ON starts at HALState baselines. State before update t is used for its forward; after optimizer t, decay once, then optionally appraise; write this resulting state before evaluation and next update. Hold that state constant throughout evaluation. No evaluation metric or prediction enters appraisal.

Spike: current training loss > 2 x median of the preceding 25 training losses (exclude current), only after 25 prior losses, at least 50 updates since previous appraisal. Emit `VerifiedOutcome('surprise', 'train-loss/update-t')`. Nominal adrenaline delta .6 is damped by `(1-.5*serotonin)`; at baseline serotonin .5 the actual delta is .45, then clamp [0,1]. Adrenaline decay rate .2/update toward .05; cortisol receives .32 times the positive adrenaline decrease. Other rates/baselines are frozen in hash-bound `hormonal_state.py`. Log every update's loss, prior median, gradient norm, event flag, resulting hormone levels, and log-temperatures. Direct arbitrary hormone deltas exist as an API but are not used by this experiment.

## Evaluation and endpoints

Evaluate before training (descriptive) and at updates 50,100,...,1500. Chunk probe into at most 64 rows. Greedily generate exactly two tokens from the five-token prompt, with the first generated token fed back for the second. No gold answer is input. Success requires generated sequence exactly `[correct_value_token,EOS]`. No substring extraction, teacher-forced answer scoring, or constrained vocabulary. Retain all final raw generated tokens. The evaluator tests a scripted correct model, wrong answers, missing EOS, and coverage mismatch.

Primary: final held-out complete-exact-with-valid-stop fraction out of 512. Secondary: normalized trapezoidal AUC across evaluations 450..1500 divided by 1050; first scheduled evaluation >= .10 exact (null if absent); descriptive final accuracy on the first 512 training rows; spike count and numerical failures. No statistical significance or equivalence claim from two seeds.

Decision in ordered exhaustive precedence, d_s = ON exact minus OFF exact:
1. Missing/duplicate/unregistered lane, nonfinite/out-of-range score, numerical error, failed reload, changed frozen sources/protected files, or failed pairing: ENGINEERING_FAILURE.
2. Both d >= .10: HORMONAL_EFFECT_SUPPORTED_AT_DEV_SCALE.
3. Both abs(d) <= .05: NO_MEASURABLE_EFFECT_AT_THIS_SCALE (threshold label, not statistical equivalence).
4. Both d <= -.10: HORMONAL_EFFECT_HARMFUL_AT_DEV_SCALE.
5. One d >= .10 and another <= -.10: SEED_SIGN_CONFLICT.
6. Otherwise any abs(d) >= .10: MIXED_OR_SEED_SENSITIVE.
7. Otherwise: INCONCLUSIVE_SMALL_EFFECT.

Interpretation: OFF versus ON tests extra learnable temperatures PLUS the state mechanism. Nonzero uncentered hormone baselines permit learning without appraisal events; a benefit cannot isolate outcome-feedback value. No constant-state control is run. Zero events means the feedback hypothesis was not exercised. Failure at this tiny task does not prove absence of effects at other scales, nor justify unbounded retries. Hormone names are engineering analogies, not biological or subjective-experience claims.

## Resource and evidence contract

CPU-only, CUDA hidden, two intra-op/BLAS threads, one lane at a time, deterministic algorithms. Ten-minute per-lane training timeout; no full-size model test. Four lanes maximum. If resources fail, preserve ENGINEERING_FAILURE and do not fabricate measurements.

Before measured execution, write PRERUN.json with source/plan/test hashes, protected-file hashes, Python/PyTorch versions. Save each lane JSON, final tensor checkpoint with spec identity, and raw predictions. Reconstruct model, load checkpoint, require exact reload logits and predictions. Save checkpoint/source/spec/data hashes and initial/final core hashes. RESULT.json hashes all preceding run artifacts. Recheck protected files and sources after all lanes. Never overwrite evidence. Test smoke receipts are explicitly engineering_smoke_only and excluded. Final ANALYSIS.md must report negatives, confounds, and remaining limits.

Canonical `v5_contracts/model_spec.py`, production `v5_model`, tracked `launch_readiness.json` and `blueprint/` are read-only. This sibling is not a launch-gated ModelSpec. No commit/push or full/frontier training is authorized by this protocol.
