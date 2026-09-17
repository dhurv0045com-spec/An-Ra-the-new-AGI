# HORM-001 — Bounded hormonal attention-temperature modulation (development-scale)

**Status:** PREREGISTERED BEFORE ANY RUN — written before implementation results exist.
**Claim ceiling:** controlled development-scale engineering evidence only. No production, 500M, cognition, or AGI claim is authorized by this experiment.
**Resource bounds:** RTX 4050 Laptop 6 GB (idle), ≤ 2 CPU threads per lane, single lane at a time, wall budget ≤ 40 min total.

## Architecture discovery (Phase 1 note)

The real V5 core (`v5_model/core.py`, built from `v5_contracts/model_spec.ModelSpec`) is:

- pre-norm decoder: per block `hidden = hidden + attn(RMSNorm(hidden)); hidden = hidden + down(SiLU(gate(RMSNorm(hidden))) * up(RMSNorm(hidden)))` (`v5_model/block.py`);
- grouped-query attention with affine QK-norm and per-segment RoPE, then `scaled_dot_product_attention` with default scale (`v5_model/attention.py`);
- one tied embedding/output table; forward is pure tensor-in/logits-out with no persistent state (`v5_model/core.py`);
- training state (`v5_training/trainer.py`) is a frozen token-indexed state machine that owns no tensors; callers supply the backend step.

There is no existing affect/ESV/interoceptive hook anywhere in the `v5_*` tree (searched: hormon/HAL/appraise/ESV/interocept/affect/emotion). V4's HAL is not present in this checkout except as history in artifact manifests.

**Integration decision — Strategy A (attention temperature), placed post-QK-norm/post-RoPE.**

Rationale and rejections:
- **B (residual additive bias)** rejected: the residual stream is the model's only state carrier and every modification compounds across blocks; a bounded additive vector is a larger, less attributable intervention than a scalar temperature, and this project's R1/R1C evidence shows representation-adjacent interventions can dominate formation dynamics — a narrower intervention is the right first probe.
- **C (per-layer scalar bias)** rejected: the only existing per-layer scalars are the affine QK scales; extending them would change the frozen spec's parameter inventory semantics.
- **D (interoceptive input channel)** rejected: requires vocabulary/context changes and retraining from scratch to matter; explicitly disproportionate to a first causal probe.
- **E (logit bias)** rejected: weakest mechanism, acts after all computation, hardest to interpret as "attentional state".

Placement within A: query logits are scaled by `1/T_h` per head *after* QK-norm and RoPE. QK-norm output is invariant to any pre-norm query scaling (RMSNorm divides out scale), so the modulation must sit between `rope(normalize(q))` and `scaled_dot_product_attention`. `T_h = exp(clamp((W h)_h, -1.5, 1.5))` with `W` zero-initialized ⇒ `T = 1` exactly ⇒ **byte-identical forward to the baseline core at initialization** (the repo's own `tie_role_model_v1.gradient_scaled_view` / forward-patch precedent, and the no-op-when-zero invariant is tested).

## Falsifiable hypothesis (stated before implementation)

On a tiny modular-arithmetic formation task with matched seeds, data, optimizer and schedule, a **bounded, zero-initialized, learnable per-head attention-temperature modulation driven by a 7-analog heuristic hormone state** (dopamine/cortisol/serotonin/adrenaline/oxytocin/GABA/norepinephrine; appraisal tied to *measured* verified outcomes — held-out exact-match acquisitions and loss spikes — never to the model's own unverified confidence) will change **formation timing and final held-out capability**:

- Primary endpoint: 1,500 optimizer updates, batch 16.
- Primary metric: `heldout STANDARD complete_exact_with_valid_stop` at the endpoint (dev identity-style surface, exact + valid EOS, greedy).
- Secondary: formation AUC over the eligible window (updates 450–1,500), first-acquisition update, NaN count (must be zero).

Preregistered decision rule (2 matched seeds × {HAL_OFF, HAL_ON}):
- Both seeds: `HAL_ON − HAL_OFF ≥ +0.10` final and no endpoint regression → `HORMONAL_EFFECT_SUPPORTED_AT_DEV_SCALE`;
- Both seeds: `|HAL_ON − HAL_OFF| ≤ 0.05` → `NO_MEASURABLE_EFFECT_AT_THIS_SCALE` (a valid negative);
- Exactly one seed qualifies → `MIXED_OR_SEED_SENSITIVE`;
- No complete pair or NaN/Inf in any lane → `ENGINEERING_FAILURE` (no interpretation).

This connects to the standing anomaly (loss falls while exact-match stays near floor): if stress/appraisal-driven temperature widening during the pre-formation regime changes acquisition timing, that is evidence about the exploration/exploitation balance during formation, not just a demo.

## Frozen counterfactual design

Arms per seed: `HAL_OFF` (baseline core, no hormonal parameters) and `HAL_ON` (sibling spec `V5A_250M_HORMONAL_v1` geometry + zero-init projection wrapped core). Shared tensors are byte-identical at step 0 (same seed, same `initialize`); `HAL_ON` adds only a zero-initialized projection, so forward values are identical at init — receipted by SHA-256 of the shared core parameter inventory in each RESULT.

Seeds: `424242`, `424243` (fresh, nonofficial, development-only; no sealed rows exist anywhere in this experiment).

Appraisal signals (all measured, all logged per eval): train loss spike > 2× trailing median → adrenaline +0.6 (fast decay, feeds cortisol `+0.32 × prior adrenaline` on decay). **No held-out/eval signal may feed appraisal** — that would contaminate the counterfactual comparison by adapting the treatment arm to the evaluation itself. Cortisol, serotonin, dopamine are state-tracked with logged-but-inert deltas in v1 (their lawful signals are environment-reward-shaped, which this experiment does not have). Oxytocin/GABA/norepinephrine are state-tracked (decay + appraisal) with **no forward-pass effect** — per the V4 postmortem they were the least validated.

## What would falsify the hypothesis

- HAL_ON formation and endpoint statistically indistinguishable from HAL_OFF on both seeds → the mechanism, at this scale and signal wiring, does not influence formation → next action is signal-fidelity analysis, not more arms.
- HAL_ON destabilizes training (NaN, clip-fraction explosion) → the bounded-projection safety claim is wrong → fix engineering, rerun.
- HAL_ON helps only by damaging train loss (memorization aid) while held-out stays flat → the "exploration during formation" story is wrong.

## What this experiment is not

Not a TIE-ROLE-FRONTIER or CYR campaign; touches no frozen S5 surface, no sealed data, no preregistered Cymek gate. `V5A_250M` and `launch_readiness.json` are read-only here; the sibling spec is a separate constant with a different provenance label (`family="dense-decoder-transformer-hormonal-v1"`), so its hash necessarily differs and can never be mistaken for the launch-gated architecture.
