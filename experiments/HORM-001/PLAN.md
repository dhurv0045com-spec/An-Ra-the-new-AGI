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
