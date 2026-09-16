# TIE-ROLE-PILOT-001

## Purpose

This is a **compute gate**, not another broad mechanism campaign. It answers one operational question:

> Is TIE-ROLE-FRONTIER-001 worth spending approximately 8+ hours of T4x2 compute on?

The pilot directly tests the frontier's primary latent contrast, `T3_BALANCED_X4_X025` versus `T0_CANONICAL`, rather than merely checking whether the baseline model can learn.

## Design

- 2 fresh matched seeds: `92001`, `92002`.
- 2 arms per seed: `T0_CANONICAL`, `T3_BALANCED_X4_X025`.
- 4 total training lanes.
- 2,000 updates per lane, batch 16, canonical LR `1e-3` and global clip.
- Same frozen S5 six-family training surface; identity-only development scoring for the pilot's primary endpoint.
- No sealed rows are loaded or scored.
- Initialization must be byte-identical and forward-equivalent between T0 and T3 for each seed.
- Final checkpoints receive teacher-forced, greedy identity, and tied-gradient decomposition diagnostics.

The four lanes are expected to fit approximately 2–3 wall-hours on one T4, based on the completed S5 runtime and the same 2,000-update training geometry.

## Binary decision

The pilot prints `RUN_FULL_TIE_ROLE` only when **both fresh seeds** satisfy all of:

- T3 minus T0 formation-AUC delta >= `0.05`.
- T3 minus T0 final identity exact+valid-EOS delta >= `0.10`.
- Neither endpoint delta is negative.
- No pilot arm is at/above `0.90` exact, avoiding a ceiling regime.
- Forward-equivalence preflight passes.
- The two scored pairs cover each registered fresh seed (`92001`, `92002`) exactly once —
  duplicate or unregistered seed evidence is a NO-GO (enforced by `_decide`,
  `tools/tie_role_pilot_001_colab_v1.py`; regression-tested in
  `tests/test_tie_role_pilot_001_decision.py`).
- Post-training pipeline (checkpoint write → identity-checked reload → greedy /
  teacher-forced / tied-gradient diagnostics → pairwise decision → result and ZIP
  packaging) is integration-covered on CPU with a tiny untrained model and zero
  optimizer steps in `tests/test_tie_role_pilot_001_integration.py`. This is
  engineering coverage only; it executes no training and produces no scientific
  evidence.

Anything else is `DO_NOT_RUN_FULL_TIE_ROLE`. Ambiguous evidence is intentionally treated as NO-GO for the expensive campaign.

The thresholds are deliberately stricter than the official four-seed frontier thresholds because this pilot is only a compute-triage gate with two fresh seeds.

## What a GO means

A GO is evidence for user review, **not launch authorization**. The TIE-ROLE-FRONTIER-001 no-go gate stands until the user explicitly reviews and authorizes reopening it. If authorized, the existing preregistered frontier remains unchanged: the pilot does not alter its arms, seeds, sealed protocol, or claim ceiling.

## What a NO-GO means

Do not spend the full frontier compute. Preserve the pilot result and select a smaller bottleneck experiment. The pilot diagnostics indicate whether the failure was weak formation, inconsistent treatment response, ceiling, or a forward-equivalence/engineering issue.

## Durability

Google Drive stores exact-resume checkpoints. The final result ZIP and checkpoint ZIP are separate. The checkpoint ZIP must be preserved even if the Colab session is deleted.
