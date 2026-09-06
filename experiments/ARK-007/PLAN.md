# ARK-007 — CAUSAL RETENTION RISK UNDER PAIRED CONTINUATION STOCHASTICITY

## Questions
Q1: Can changing ONLY the future minibatch ordering change whether the generalized solution collapses?
Q2: Does lowering LR from 1e-3 to 1e-5 reduce collapse probability when continuation order is held fixed?

## Design
Unit: ACQUIRED CHECKPOINT x CONTINUATION ORDER.
Phase 1: train seeds 707/808 on T2 to sustained G90 (3 consecutive evals >= 0.90).
Phase 2: at G90 confirmation, snapshot model+optimizer+RNG. Fork into HIGH (1e-3) and LOW (1e-5) arms.
Each fork consumes one of 8 pre-generated batch-index sequences (continuation seeds 1701-1708).
Both LR arms of a pair consume the SAME sequence. 16 paired conditions total.

## Frozen parameters
- Acquisition seeds: 707, 808
- Continuation seeds: 1701-1708
- Manifest: ARK-002B (split_sha256 0dd930569704...)
- Model: Micro 4L/128w/4H, 19-token vocab
- Acquisition LR: 1e-3; Treatment LRs: 1e-3 and 1e-5
- Batch: 64; Post-confirmation steps: 8000; Eval every 200

## G90 semantics
- g90_onset_step: first eval >= 0.90 in a >= 3-eval streak
- g90_confirmation_step: eval at which the streak becomes knowable
- Intervention applied AFTER confirmation; retention metrics exclude pre-treatment evals

## Primary endpoint
Risk difference = P(collapse | low LR) - P(collapse | high LR). Negative = protective.

## Verdict rules
- REPLICATED_PROTECTION: risk diff <= -0.30, consistent within both checkpoints, >= 4 discordant pairs
- INCONCLUSIVE_LOW_EVENT_RATE: fewer than 4 high-LR collapses total
- NOT_REPLICATED: enough high-LR collapses for power but no protection

## Determinism control
One continuation order per checkpoint rerun from the same snapshot. Trajectories must be identical.

## What will NOT be claimed
No AGI claim. No universal law. No architecture change. Micro-scale T2 only.
