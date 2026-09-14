# CYMEK 500M ALIGNMENT — ARKENSTONE READ-ONLY REVIEW

Date: 2026-09-08

Arkenstone inspected Cymek only as a read-only production reference before preregistering ARK-011.

## Exact branches inspected

- `cymek`: `28bf57a0d299a2c13a99fe0046616c00a1b8530c`
- `cymek-500m-readiness`: `3f92cf8c185379f138f3da4649ebadc6d94c31c4`

At review time `cymek-500m-readiness` was 13 commits ahead of `cymek`.

## What is demonstrated there

The readiness branch materially strengthens the production substrate: exact bucket-lane/cursor behavior, 500M 65/20/15 allocation math, resume/no-rewarm contracts, durable milestones, production-entry gates, checkpoint/mirror failure handling, evaluation scheduling/ingestion boundaries, activation-checkpointing equivalence, and a head-bound 50/50 closure receipt.

It does **not** demonstrate a 500M model-training result. The branch itself records real-corpus supply, TPU certification, and production evaluation identities as blockers/unmeasured items. Arkenstone must not use software-readiness evidence as cognition evidence.

## Consequence for Arkenstone

Arkenstone should borrow scientific/engineering discipline, not production ownership:

1. State-triggered research interventions must have exact checkpoint and continuation identity.
2. Training-control signals must be separated from sealed measurement signals.
3. Runtime failure must fail closed and preserve partial receipts.
4. Any mechanism proposed for Cymek must survive transfer and cost tests before touching Cymek's production schedule.
5. A micro result is not authorization to alter the 500M/5B production path.

ARK-011 therefore introduces an `OOD_CONTROL` / `OOD_SEALED` firewall: only CONTROL can trigger acquisition/collapse/recovery states; SEALED is measurement-only.

## Promotion path if ARK-011 is positive

A positive ARK-011 result would still be `SUPPORTED at Micro T2`, not `PROMOTED_TO_CORE`. The minimum next evidence before a Cymek scheduler proposal is:

- replication of the adaptive HIGH-recover -> LOW-retain effect on fresh Micro T2 events;
- non-arithmetic transfer with an orthogonal controller/evaluation diagnostic;
- explicit plasticity-cost measurement after LOW consolidation;
- a token-schedule compatibility design that preserves Cymek's no-rewarm/resume invariants;
- only then a small optional Cymek challenger, never a silent default change.

## Boundary

CYMEK builds and certifies the production training substrate. ARKENSTONE discovers and stress-tests candidate mechanisms. This document does not merge branches, copy Cymek production code, or transfer ownership of cognition qualification.
