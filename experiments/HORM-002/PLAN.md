# HORM-002 — Competence-gated, feedback-isolating hormonal comparison (dev scale)

**Status:** PREREGISTERED BEFORE ANY HORM-002 RUN (2026-09-17, after HORM-001 was closed negative).
**Claim ceiling:** development-scale engineering evidence only. No production, 500M, cognition, or AGI claim. Not a launch authorization.

## Motivation

HORM-001 completed with verdict `NO_MEASURABLE_EFFECT_AT_THIS_SCALE`: all four lanes scored 3/512, loss sat at the formatting reference, and **zero loss-spike appraisal events fired** — the outcome-feedback hypothesis was never exercised, only the nonzero-baseline state mechanism. HORM-002 fixes both blockers in the preregistered order HORM-001's analysis named: (1) establish task competence first, (2) wire a measured training-side feedback event that actually fires, (3) add the constant-state control arm to isolate feedback from extra parameters.

## Phase 0 — Calibration (OFF arm only; non-causal, labeled `calibration`)

Preregistered ladder, tried in order, stop at the first pass; **no other search, no post-look changes**:

| Probe | Modulus | Updates | Eval every |
|---|---|---|---|
| P1 | 97 | 12,000 | 250 |
| P2 | 23 | 6,000 | 250 |
| P3 | 23 | 12,000 | 250 |

Pass = final held-out exact+stop ≥ **0.50** AND train-subset exact ≥ 0.50. The first passing probe's `(mod, updates)` is the frozen Phase-1 config, recorded in `results/CALIBRATION.json`. If none passes → verdict `COMPETENCE_GATE_FAILED`, causal phase not run, report honestly.

## Phase 1 — Causal comparison (only if competence gate passes)

Three arms × two seeds (424242, 424243), sequential, identical to HORM-001 rev 3 in data, geometry (2 layers, width 64, 2QH/1KVH, 81k params), optimizer (AdamW lr 3e-3, wd 0.01, clip 1.0), full-sequence loss (all six non-BOS targets), evaluation cadence, and receipts:

- **HAL_OFF**: baseline tiny core, no hormonal parameters.
- **HAL_CONST**: wrapped core + zero-init projection (same 14 extra learnable parameters), hormone state **frozen at baselines forever** — no decay, no appraisal, asserted constant in every trace row. Isolates "extra learnable per-head temperature parameters with constant input".
- **HAL_ON**: wrapped core + projection + live state machine. Isolates "measured-outcome feedback driving the state" **relative to CONST**.

**Appraisal (HAL_ON only), all measured, training-side, never from the held-out probe or evaluation code:** each update, from the same training forward used for the loss (before the optimizer step), compute measured batch answer accuracy = fraction of rows whose argmax logit at the answer position equals the true answer token. If ≥ **0.875** (≥14/16) → `VerifiedOutcome('success', f'train-batch/update-{t}')` → dopamine +0.25 (not stress-damped). The HORM-001 loss-spike rule (>2× trailing-25 median, 50-update cooldown) is retained for `surprise`. State ordering identical to HORM-001 rev 3: state_t forward → optimizer step → decay once → appraisal → state used for update t+1.

## Endpoints and decision table

Primary: final held-out exact+stop (greedy, exact answer token + EOS, no gold in prompt). Secondary: normalized AUC over evaluations with update ≥ 0.45×updates, first eval ≥ 0.10, event counts, NaN count (must be 0).

With `ΔF_s = ON_s − CONST_s` and `ΔM_s = CONST_s − OFF_s`, in precedence:

1. Missing/duplicate lane, non-finite or out-of-range score, numerical fault, failed reload, wrong purpose flag, any CONST appraisal event or state drift, any protected/source hash change → `ENGINEERING_FAILURE`.
2. Any OFF lane final < 0.50 → `COMPETENCE_GATE_FAILED` (descriptives only; no causal interpretation).
3. Either ON lane has zero appraisal events → `ENGINEERING_FAILURE` (each seed must exercise appraisal).
4. Both ΔF ≥ +0.10 → `FEEDBACK_EFFECT_SUPPORTED_AT_DEV_SCALE`.
5. Both |ΔF| ≤ 0.05 → `NO_FEEDBACK_EFFECT_AT_THIS_SCALE`.
6. Both ΔF ≤ −0.10 → `FEEDBACK_EFFECT_HARMFUL_AT_DEV_SCALE`.
7. ΔF signs conflict (one ≥ +0.10, other ≤ −0.10) → `FEEDBACK_SIGN_CONFLICT`.
8. Any |ΔF| ≥ 0.10 → `FEEDBACK_MIXED_OR_SEED_SENSITIVE`.
9. Otherwise → `FEEDBACK_INCONCLUSIVE_SMALL_EFFECT`.

ΔM is reported descriptively only (extra-parameter mechanism); it is never equated with feedback value. Two seeds support no significance or equivalence claims. Tiny toy tokens; no production model, no GPU, no frontier scope.

## Final pre-run clarifications (revision 2)

- Split: deterministic shuffle with seed 424242; probe size `min(512, mod**2//5)`, all remaining pairs train. Mod-97: 512 probe/8897 train; mod-23: 105 probe/424 train. Train-subset endpoint uses the first min(512, train size) pairs. No probe pair enters loss or appraisal.
- Calibration starts from scratch at every ladder entry, uses only seed 424242/OFF, and stores separate `calibration/mod{mod}_updates{updates}/` artifacts. Causal lanes restart from scratch in `causal/`; all measured lanes use the full probe. Smoke tests are explicitly labeled and cannot count as measured evidence.
- Local limits: CPU only, two threads, batch 16, tiny 81k-parameter geometry; 600 seconds per lane, at most 30,000 calibration updates plus six causal lanes at the first passing budget (maximum total 102,000 updates). No concurrent training, GPU allocation, production model construction, or automatic expansion after a failure.
- The calibration probe is reused to select geometry/budget. Results are exploratory and selection-conditioned, NOT independent confirmatory generalization estimates.
- ON versus CONST isolates the combined evolving-state policy (decay plus appraisal), not appraisal alone. A matched decay-only control would be needed to isolate appraisal specifically; verdict names are shorthand and do not remove this limit.
- Success and surprise events must be logged per update and each ON seed must have events. Protocol and source hashes are recorded before the first measured lane and checked afterward, including when calibration fails.

## What this is not

No cognition or subjective-state claim; hormone names are engineering analogies. `V5A_250M`, `v5_model`, tracked `blueprint/` and `artifacts/v5/launch_readiness.json` stay read-only. Calibration lanes are engineering evidence only and excluded from the causal verdict. No commit/push without explicit instruction; no scale-up is justified by any single outcome here.
