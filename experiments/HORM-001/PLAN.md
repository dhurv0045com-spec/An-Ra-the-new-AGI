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
