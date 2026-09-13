# B2.2 integration correctness — progress

Updated: 2026-09-13. Implementing [B2.2 C1–C6](../../phase_b22_20260913/README.md) against baseline `c3662b3`, resolving chief findings R1–R7 from [the review](../B2_CHIEF_20260913/REVIEW.md). Baseline reproduced: the chief probe's observations were re-run at session start and matched `observations.json` exactly (changed prepared tokens accepted with old identity; nontrainable rows emitted; pair_group_id null; example-IDs as semantic IDs; promotion accepted without uncertainty/pair receipts; unmatched-case delta +1.0; trainability change preserved dataset identity).

## Resource ledger discipline

Build ledger preserved untouched at session start: **193/200 CPU optimizer updates, 111.234/300 s, zero GPU** (re-read live before any work). This phase consumes zero learned updates for repairs (non-learning and gradient-only checks only). The final C5 acceptance comparison is allocated at most six real updates: uninterrupted three versus interrupted one plus resumed two through the shared path. No ledger reset; no full learned-suite rerun.

## Packet ledger

| Item | Status | Notes |
|---|---|---|
| C1 data semantics | DONE | see HANDOFF R1/R2 |
| C2 optimizer boundaries | DONE | see HANDOFF R3/R5 |
| C3 checkpoint/resume integrity | DONE | see HANDOFF R4 |
| C4 protocol-bound evaluation | DONE | see HANDOFF R6 |
| C5 resource policy + final comparison | DONE | 6-update comparison AGREES; ledger honestly OVER CAP (206/200) after a recorded duplicate execution — see HANDOFF R7 |
| C6 integrate/report | DONE | HANDOFF written; commit/push recorded below |

## Final verification

- Non-learning full suite: 435 passed / 10 pre-existing baseline failures / 11 skipped (learned suite not rerun).
- Final learned acceptance (6 updates, fresh subprocesses, shared path, grad_accum=2 + pair 0.5 + replay p=0.5 + controller boundary): agrees on checksums, counters, live state and future event sequence. Executed twice (duplicate execution recorded as the over-cap cause); both agreed.

## R-findings disposition

Tracked in HANDOFF.md at completion; each gets code, command-path regression, and evidence.

## Exact checks run

- Chief probe re-run at session start (zero updates, zero model instantiations): observations match.
- Ledger re-read: 193/200 updates, 111.234/300 s, 0 GPU.
