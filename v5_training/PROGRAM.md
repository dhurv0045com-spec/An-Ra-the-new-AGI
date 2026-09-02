# An-Ra V5 Training Program — cognition-first, gate-locked

This is the training program. It is not a promise of AGI; it is the
sequence of falsifiable gates that must each pass, in order, before the
next compute commitment. Every gate produces a receipt; every receipt is
commit-bound; every claim names its evidence class (RAW CORE /
CONSTRAINED / ASSISTED / ORACLE).

## Law of the program

**No gate may be skipped, and no gate may be redefined after its outcome
is known.** A failed gate is a result. The program advances on evidence
or it does not advance.

## Stage 0 — Measurement integrity (current)

| Gate | Status |
|---|---|
| Scoring fixture v2 (leakage-repaired) | DONE |
| Rotation contract executed (not declared) | DONE (this branch) |
| Powered development tournament (15 CUDA cells) | DONE: FAIL_DEVELOPMENT_POLICY (marked-prefix bias in both promotable policies) |
| CPU parity cells ×15 | PENDING (runbook ready) |
| Policy selection receipt (or null mode) | PENDING |
| 10 drifted benchmark receipts regenerated | PENDING (runbook ready) |

OUTCOME (2026-09-02): powered CUDA run executed (0.089 GPU-hours); both promotable policies FAILED the bias screen (never select marked-prefix candidates; hidden-label exactly at chance) -> production_scoring_mode = NULL per the frozen hierarchy. Next instrument: neutralize prefix-token statistics in neutral panels; no threshold edits.

Exit condition: `scoring_policy_development.json` status
`PASS_DEVELOPMENT_ELIGIBILITY` with a selected policy written to the
immutable receipt — or the honest null (`production_scoring_mode = null`),
which blocks learned E1–E3 comparisons until a better instrument exists.

## Stage 1 — P35 learned evidence (first real training)

Small random-init P35 arms, short runs, the E0 development suite as the
target. Questions, in decision-value order:

1. **Cognition-vs-loss decoupling**: does a checkpoint with *worse* LM
   loss ever show *better* E0 primitive scores? (If never, E0 needs
   redesign before any scale decision.)
2. **Query-conditioned binding**: does group-structured query-swap data
   (the core-exp result: lift 0→+0.67 nats, p=0.018) transfer to P35?
3. **Depth vs width at matched params** (26×896 vs fewer×wider): decided
   by E0 primitive deltas, not perplexity.
4. **Tokenizer 16k/24k/32k** under matched raw bytes: decided by
   candidate-free realization rates.

Budget per run: ≤ 2 GPU-hours. Promotion: raw-Core improvement +
candidate-free improvement + no assistance-only gain + bounded substrate
regression (the E3 contract, applied early).

## Stage 2 — M102 replication (~100M)

The P35 winner recipe re-trained at ~102M. Gate: every Stage-1 capability
delta **replicates in direction** (no new primitive appears that P35 did
not show). Failure here means the recipe was scale-fragile — fix before
V5-A. Also: long-context cognition at 2k vs 4k decided here.

## Stage 3 — V5-A 250M main run (UNAUTHORIZED until Stage 2 passes)

26 × 896, 14Q/7KV, SwiGLU 2368, context 4096, frozen tokenizer, ~5B
tokens, cognition-fraction curriculum (5%→15%→30%) + query-swap λ per
Stage-1 evidence. Integrity contract per training step: optimizer/
live-parameter identity, gradient finiteness, Adam moment movement,
parameter SHA per save, token ledger, sampler cursor, RNG state,
interrupted-vs-restored continuation proof. Historical rule: **metadata
is not enough — persisted parameters must demonstrate movement.**

## Stage 4 — Cognition certification (continuous, never "done")

- Causal/invariance pairs: relevant-change sensitivity AND
  irrelevant-change invariance, per primitive.
- Structural OOD + fresh sealed replications (custody external).
- Shortcut baselines must lose; assisted/RAW reported separately forever.
- The long-term question stays: can the system learn WHY it failed and
  predict WHICH intervention fixes future failures (the core-exp
  cognitive-credit line, merged once V5 cognition is measurable)?

## What this program refuses

Bigger models to hide measurement failures · assisted scores posing as
Core capability · thresholds edited after outcomes · benchmark numbers
without receipts · "AGI" as a milestone. The only success criterion is:
**uncertainty removed, gate by gate, with every negative result preserved.**
