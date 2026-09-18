# HORM-001 — Bounded hormonal modulation of attention scaling

## Status
PREREGISTERED 2026-09-17T20:00:00+05:30, before implementation. No results exist yet.

## Primary question
Does a bounded, externally-updated hormonal attention-scale modulation wire into the real V5 forward pass without changing the frozen `V5A_250M` spec, the launch-gate chain, or existing training contracts?

## Hypothesis
H1 (wiring): For a tiny CPU model, `Q_scaled = Q * (1 + B * tanh(raw))` after RoPE+QK-norm, with `raw` from an external hormonal state vector and `B` bounded, produces outputs differing from the baseline by a bounded, finite margin, while zero-state output is bitwise-equal to baseline.

H2 (scope caveat): This session cannot produce training-scale capability evidence. Success means wiring + tests, not capability claims.

## Falsifiability
H1 fails if: zero-state forward differs from baseline; scale leaves [0.8, 1.2]; outputs non-finite; or the spec hash changes.

## Method
- Sibling package `v5_identity` (hormonal state, projection, config).
- No edits to `v5_model/*`, `v5_contracts/model_spec.py`, `blueprint/`, or `launch_readiness.json`.
- Frozen counterfactual pair: same seed/data/threads; treatment vs. no-op control.
- Tiny check runs on CPU, 2 threads, ~30 s; no 250M training.

## Success / failure criteria
- Pass: all new tests pass; control==treatment at zero state; bounded nonzero at nonzero state; hashes unchanged.
- Fail: any of the above violated, or protected files change.

## Verdict rules
- `WIRING_VERIFIED_PENDING_TRAINING_EVIDENCE` if pass; `WIRING_FAILED` otherwise. No capability claims either way.

# HORM-002 — Miniature-scale A/B of hormonal attention-scale modulation

## Status
CORRECTION: this HORM-002 section was written after the first diagnostic A/B
run had already produced a result in the temporary directory. It was not a
prospective preregistration. The original timestamp below is a historical
claim, not evidence of chronology.

Originally recorded: PREREGISTERED 2026-09-17T22:00:00+05:30 (before RESULT_horm002_ab.json existed).
Amends HORM-001: HORM-001 measured attention OUTPUT scaling after the attention
sublayer, not query-logit scaling. HORM-002 measures the corrected mechanism:
bounded query-logit scaling inside attention after RoPE+QK-norm.

## Primary question
At miniature scale, does a bounded (0.8-1.2), out-of-graph hormonal
query-scale modulation produce a measurable loss-trajectory difference
between matched control/treatment arms through the real ProductionTrainingBackend?

## Hypothesis (falsifiable)
H1: With identical seed, data, spec, optimizer and schedule, an inert
projection (raw_alpha=0) yields losses bitwise-equal to baseline, while an
active projection (raw_alpha=0.35, deterministic appraisal cycling) yields a
small but nonzero loss-trajectory difference (max |Δloss| > 1e-6) with all
losses finite.
H2 (scope): differences at this scale do NOT establish training benefit,
capability, or production-scale behavior.

## Method
- Real path: v5_model.core.initialize + ProductionTrainingBackend.step +
  trainer.train + CheckpointStore. No edits to v5_model/v5_contracts.
- Tiny spec: vocab 512, width 32, 2 layers, ctx 32 (spec_sha256 e4dc015a...).
- 8 updates x 256 real tokens, batch 8 x 32, two-segment packing.
- Same seed 707001 for both arms; CPU only, 2 threads; no CUDA.
- Treatment scale per update from HormonalProjection.scale() over a
  deterministic appraisal cycle (success/failure alternating), recorded.
- Integration canary: patched model differs from baseline; restore() returns
  bitwise-equal outputs (verified in probe before the A/B run).

## Success / failure / no-go
- Pass: control losses == treatment losses at inert scale; active arm differs
  by <1e-2 max; all finite; protected hashes unchanged.
- Fail: any non-finite loss, unbounded scale, or control!=baseline at inert
  state.
- No-go: any protected file (model_spec.py, launch_readiness.json) changes.

## What this cannot prove
- Production-scale training dynamics, capability formation, G90, or any
  quality claim. It is a mechanism-presence measurement at miniature scale.

# HORM-003 — Prospective multi-seed A/B with correctly-bound hormonal state

## Status
PREREGISTERED 2026-09-18T00:05:10+05:30, before any HORM-003 run and before
any HORM-003 result exists. Written against the repaired runner (commit
a76dae9c) in which attention layers provably read the logged scale.

## Primary question
With the state-binding bug fixed, does dynamic hormonal state (deterministic
alternating synthetic appraisals, NOT live verifier outcomes) produce a
consistent directional effect on mean training loss across matched seeds?

## Hypothesis (falsifiable)
H1: Across 5 fresh seeds (707011..707015), the treatment arm's mean loss
differs from control with the SAME sign in at least 4 of 5 seeds.
H0: sign is inconsistent (<=3/5), i.e. no resolvable directional effect.

## Method
- Real path (unchanged): v5_model.core.initialize + ProductionTrainingBackend
  + trainer.train + CheckpointStore; v5_identity.attention_patch applies
  bounded query-logit scaling (raw_alpha=0.35, bound=0.2, scale in [0.8,1.2]).
- Same spec as HORM-002 (sha e4dc015a...), 8 updates x 256 tokens, CPU only,
  2 threads, no CUDA.
- Seeds 707011-707015; each seed runs control then treatment with identical
  batches (seeded per seed), optimizer, and schedule.
- Per-update synthetic schedule: reset to baseline, appraise success on even
  updates / failure on odd, decay once; logged scale == scale read by layers
  (enforced by tests/test_hormonal_integration.py regression).
- Provenance: result JSON carries runner sha256, v5_identity file hashes,
  torch/platform identity, seeds, per-seed loss vectors.

## Primary readout (fixed before execution)
sign_consistent_seeds / 5 and median mean-loss difference across seeds.

## Verdict rules (no post-hoc reinterpretation)
- SUPPORTED: >=4/5 seeds share one sign AND all losses finite.
- NOT_SUPPORTED: otherwise. Result is recorded either way.
- INVALID: any non-finite loss, scale outside [0.8,1.2], or protected-file
  hash change; the run is discarded, not reinterpreted.

## What this cannot prove
- Live-verifier-driven appraisal value, capability, G90, production-scale
  behavior. A SUPPORTED verdict justifies proposing a longer protocol; it is
  not a quality claim.

# HORM-004 — Longer-horizon A/B with live-verifier appraisal (PROPOSED, GATED)

## Status
PROPOSED 2026-09-18T23:21:03+05:30. No HORM-004 code exists, no HORM-004 run
has occurred, no HORM-004 result exists. This section is a preregistration
draft awaiting explicit user authorization AND a compute budget before any
execution. It must not be executed by default, by a background process, or
as a follow-on to any other command.

## Motivation (why this, why now)
- HORM-001: wiring verified (with corrected integration-point record).
- HORM-002: mechanism measurably present at miniature scale; historical
  artifacts marked non-evidence for dynamic conditioning (state-binding bug).
- HORM-003 (prospective, repaired runner): NOT_SUPPORTED — ~1e-06-scale
  perturbations, 3/5 sign consistency, median +3.6e-07 across 5 fresh seeds.
- Open gaps HORM-004 must close: (a) 8-update horizon too short for
  per-update perturbations to compound; (b) appraisal driven by synthetic
  alternating fixtures, never by a live verifier signal.

## Primary question
Over a horizon where per-update effects can compound (64+ updates), with
appraisal driven by live evaluator-verified outcomes (gold firewall intact:
committed outputs joined to truth only after freezing), does the bounded
hormonal query-scale modulation produce a sign-consistent mean-loss
difference versus matched control?

## Hypothesis (falsifiable)
H1: Across 5 fresh seeds (707021..707025), at 64 updates x 256 tokens with
live-verifier appraisal, treatment mean loss differs from control with the
SAME sign in at least 4 of 5 seeds.
H0: sign inconsistent (<=3/5) — no resolvable directional effect even at
compounding horizon with real appraisal signal.

## Method (frozen before execution)
- Same real path and tiny spec as HORM-003 (spec sha e4dc015a...);
  v5_identity.attention_patch unchanged (raw_alpha=0.35, bound=0.2).
- Appraisal source swap ONLY: per-update outcome comes from scoring
  committed probe outputs against EvaluatorTruth via score_committed
  (v5_evaluation/firewall.py), never from model self-report; probe outputs
  frozen before truth is joined. Fixture schedule retired.
- Hormonal state persists across updates within an arm (no per-update
  reset); checkpoint/resume support for session state is a prerequisite
  task, tested before the A/B.
- CPU only, thread cap set at authorization time; no CUDA; budgets fixed.

## Primary readout (fixed before execution)
sign_consistent_seeds / 5 and median mean-loss difference, same as HORM-003.

## Verdict rules (no post-hoc reinterpretation)
- SUPPORTED: >=4/5 seeds share one sign AND all losses finite.
- NOT_SUPPORTED: otherwise. Recorded either way under the negative-results
  convention.
- INVALID: non-finite loss, scale outside [0.8,1.2], firewall leak
  (any truth field on the model-visible path), or protected-file change.

## Authorization gate (blocks execution)
1. User explicitly authorizes HORM-004 execution with a stated compute
   budget (threads, wall-clock cap, RAM ceiling).
2. Session-state checkpoint/resume is implemented AND covered by a
   regression test.
3. Live-verifier appraisal path is reviewed for firewall integrity.
Until all three hold, this plan is a document, not an instruction.

## What this cannot prove (even if SUPPORTED)
- Capability, G90, production-scale behavior, or training benefit in any
  operational sense. SUPPORTED only motivates a training-scale protocol
  proposal, itself requiring separate authorization.
