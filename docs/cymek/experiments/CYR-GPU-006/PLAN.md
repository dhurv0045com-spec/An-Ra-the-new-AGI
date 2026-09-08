# CYR-GPU-006 — PLAN

Status before Commit B: **EXECUTABLE_FREEZE_PENDING_PREREGISTRATION**.

Arkenstone authority audited before this revision: `6acd9dcbdd28d00f387ffcd004253a813aca4b66`; validated Discovery V6 bundle SHA256 `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`.

## Questions

CYR-GPU-006 asks three deliberately separate questions on the real Cymek V5 implementation:

1. **Retention policy:** on independently acquired T2 parents, which continuation policy best preserves candidate-free capability under byte-identical future data: constant HIGH LR, constant LOW LR, fixed-time HIGH→LOW, or hysteretic state-dependent HIGH↔LOW?
2. **Mechanism alternative:** when LOW/adaptive states preserve capability, is the result consistent with near-freezing rather than a stronger consolidation mechanism? Parameter displacement, gradient norms and Adam-moment norms are retained as diagnostics; no direct per-step update-norm claim is made.
3. **Plasticity after retention:** after equal same-task continuation exposure, can the prospectively chosen adaptive state learn a robust non-arithmetic binding skill as well as or better than the LOW near-freezing comparator while retaining old T2 under a fixed replay stream?

## Why the transfer stage changed after the Arkenstone audit

The live Discovery V6 evidence narrows what is worth testing:

- ARK-011 supports HIGH-for-recovery then LOW-after-recovery on Micro T2.
- ARK-012 does **not** identify an optimal universal capability threshold. CYR's threshold is therefore an operational preregistered controller setting, not a claimed law; FIXED_TIME remains a required causal control.
- ARK-013 shows LOW does not preserve T2 indefinitely under 12k pure-new-skill/no-replay updates, while its intended plasticity-frontier test was inconclusive because the new skill never acquired.
- ARK-014 shows canonical-only non-arithmetic binding is brittle to fact order, deterministic order augmentation repairs that acquisition gate, and the HIGH-vs-LOW retention screen itself had zero failures in both arms.

Therefore CYR does **not** repeat ARK-014's zero-event retention screen and does **not** compare a pre-continuation parent against an older post-continuation state. The transfer comparison is prospectively fixed to **HYSTERETIC_HIGH_LOW vs LOW_CONTINUE**, from the same acquired parent after equal continuation-token exposure.

## Model / tokenizer / data

- Real Cymek V5 only via `ModelSpec` + `v5_model.core.initialize()`.
- Frozen 24,576 tokenizer; special IDs are read from the artifact and identity receipt.
- T2 arithmetic worlds: first-operand tens 1–5 train, 6–7 evaluation; commutative closure is declared rather than hidden.
- Four T2 roles: TRAIN, DEV_CONTROLLER, DEV_MEASUREMENT, SEALED_RESERVED.
- Target = answer content + EOS; EOS carries loss; PAD/BOS do not.
- Non-arithmetic transfer uses deterministic registry binding with independent train/control/sealed latent worlds and order-augmented training examples crossing BASE, QUERY_ONLY, ORDER_ONLY and QUERY+ORDER surfaces.

## Independent arithmetic parents

Frozen seeds: 707, 808, 909. **All three are attempted.** Acquisition runs exactly once per seed at HIGH=1e-3. G90 is candidate-free complete-exact-with-valid-stop ≥0.90 on DEV_CONTROLLER for three consecutive evaluations. An unqualified parent forks nothing.

## Matched retention forks

For each qualified parent, HIGH_CONTINUE / LOW_CONTINUE / FIXED_TIME_HIGH_TO_LOW / HYSTERETIC_HIGH_LOW restore identical parent model+optimizer bytes and consume the same continuation-tail batch hashes. Arm order rotates by parent to limit wall-order bias.

Fixed-time switches at 50% of actual continuation tokens. The current hysteretic policy enters retention at ≥0.90 sustained 3 and re-enters plasticity below 0.50 sustained 3. These values are frozen experimental settings, **not** evidence that 0.90/0.50 are universal or optimal thresholds.

## Exposure

Only actual real tokens define dose. Minimum acquisition dose 2M/parent; target 4M. Minimum continuation 0.5M/arm; target 2M. The hardware-only resolver downshifts model scale before violating these floors.

## Hardware / wall

CELL 0 calibrates MIDI, MICRO and RESEARCH_SMALL where they fit, including real optimizer steps and batched candidate-free generation. The resolver uses training throughput and generation throughput only; accuracy/loss/treatment outcomes are forbidden from hardware resolution.

Desired total runtime is ≈135 min. Hard campaign ceiling is 170 min with an 8-minute packaging reserve. The final resolver emits prospective acquisition/retention/transfer stage budgets from measured hardware costs so later arms are not left to an arbitrary remainder.

## Arithmetic primary endpoint

RET90 area on DEV_MEASUREMENT. A scientific winner requires at least two independent complete, contract-valid parents; mean RET90 margin ≥0.10 against every comparator; support ≥0.05 on at least two parents; no ≥0.05 reversal. Otherwise the retention result is INCONCLUSIVE.

SEALED_RESERVED never controls optimization, switching, stopping or winner selection. Every completed arm is measured on it only after all training and the development decision are complete.

## Plasticity / non-arithmetic transfer

The transfer arms are fixed **before CYR arithmetic outcomes**:

- candidate state: `HYSTERETIC_HIGH_LOW` final retention checkpoint;
- comparator state: `LOW_CONTINUE` final retention checkpoint.

A parent enters transfer only if both states are COMPLETE, red-team clean, share the same original parent/future-tail identity, consumed the same continuation-token dose, and each remains T2-qualified at transfer start (`final_g >= 0.90`). Target is three independent source-parent pairs; at least two are required.

Both source states then:

- retain their full model + Adam optimizer state;
- are set to the same HIGH learning rate for new-skill movement;
- consume the same deterministic order-augmented registry-binding stream;
- use the same fixed training mixture of 18 binding rows + 2 old-T2 replay rows per logical batch (10% by row count; actual token fraction is measured and reported, never assumed);
- receive equal actual-token targets;
- are evaluated on binding CONTROL plus old-T2 DEV_MEASUREMENT at fixed actual-token intervals.

Binding qualification requires three consecutive CONTROL evaluations meeting canonical ≥0.90, order-only ≥0.85 and query+order ≥0.85. QUERY_ONLY is reported independently. Binding SEALED is measured at completion and never controls training.

A matched pair is plasticity-compatible only if both states complete the dose, both acquire sustained robust binding, both finish robust-qualified on SEALED, the adaptive candidate is no more than 25% slower to robust-G90, its final robust floor is no worse by >0.05, and old-T2 final exact is no worse by >0.05. At least two independent parent pairs are required for `REPLICATED_PLASTICITY_NONINFERIOR`; a preregistered speed or old-retention advantage on at least two pairs upgrades this to `REPLICATED_PLASTICITY_ADVANTAGE`.

This stage tests the plasticity of the **complete post-retention training state**. It intentionally preserves optimizer moments and therefore does not isolate parameter-state versus optimizer-state causality.

## Research-candidate rule

`research_candidate=true` requires both:

- replicated arithmetic winner = `HYSTERETIC_HIGH_LOW`; and
- replicated non-arithmetic plasticity result = NONINFERIOR or ADVANTAGE versus equal-age LOW_CONTINUE.

Even then `production_promotion_authorized=false` mechanically. A GPU research candidate is not a production scheduler change.

## Failure / resume

Completed parents, retention arms and completed transfer states are durable on Google Drive and skipped on rerun. Incomplete states restart from the same immutable source checkpoint and deterministic stream rather than resuming an ambiguous partial trajectory. Every exception packages partial evidence + FAILURE.json before surfacing the error.

## Production boundary

GPU evidence is development evidence only. It cannot certify TPU behavior and cannot authorize PRE500M or 500M. TPU/XLA remains `IMPLEMENTED_PENDING_PRE500M_TPU` until real hardware validation and an explicit later promotion decision.
