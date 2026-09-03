# REAL MODEL CAUSAL SPECIFICATION

## Central Hypothesis (falsifiable, no predetermined geometry)

> Failures may possess reproducible causal structure such that
> pre-intervention observations can predict differential responses to
> interventions better than fixed and surface-only baselines.

This does NOT assume:
- low rank (full-rank response is a valid result)
- named failure categories (routing, binding, memory, etc.)
- any particular latent geometry
- that the synthetic world's physics applies to real cognition

## What constitutes evidence

1. Interventions are legal (no evaluator truth, no gold leakage)
2. Responses are measured with matched controls
3. Predictions are committed BEFORE intervention outcomes are revealed
4. Effect is measured beyond raw accuracy (AUPRC, Brier skill, MCC)
5. Structure is tested against sparsity-matched nulls
6. Results replicate on independent cohorts

## What does NOT constitute evidence

- Raw cell accuracy under class imbalance
- Low effective rank in a sparse matrix
- Textual self-explanations
- Performance on synthetic worlds designed around the method
- Improvements that matched controls also show

## Intervention basis qualification (IBQ)

Before any self-model training:
- oracle coverage must exceed threshold (enough failures are repairable)
- no degenerate probes (never/always fire)
- response diversity must exceed matched nulls
- controls must behave as nulls
- no universal trivial solver
- legality must be mechanically verified

## Separation of concerns

- OBSERVED-ONLY: policy/predictor sees only pre-intervention state
- EVALUATOR: gold, correctness, oracle metadata — never policy input
- MECHANISM DIAGNOSTIC: evaluator-side logprob/probe analysis
- BEHAVIORAL EVIDENCE: exact generation match — the strongest claim class

## Claim hierarchy

Each level requires independent evidence. Never collapse.
