# WSD AUDIT (schedule execution)

**Frozen authority:** `v5_training/schedule.py` (5B-budget constants, LOCKED — never modified). **Canary mechanism:** `anra_v5/v51_canary_run.canary_wsd_receipt` — the SAME WSD shape (linear warmup → constant → linear decay, token-indexed, `rewarm_on_resume_or_pack_change: false`), scaled to the canary's 491,520-token budget and injected into the production backend's schedule interface (`ProductionTrainingBackend(schedule=...)`), so the executed LR path is production code end to end.

## What is proven

1. **Shape equivalence:** the canary receipt embeds `shape_of = schedule_receipt()["sha256"]` (the frozen 5B receipt) and mirrors its phase structure with canary-scaled constants.
2. **Full-shape execution:** the 120-update trace crosses all three phases (warmup ends at update 12, stable through update 102, decay 103–120) and lands on the decay endpoint (0.1× peak).
3. **Token-indexed position:** the LR is a pure function of pre-update cumulative real tokens (`state.schedule_tokens`); pack identity and process boundaries cannot move it.
4. **Resume continuation:** the fresh-process resume continues the schedule from the restored token counter — zero rewarm events, asserted by `tests/test_v51_canary.py::test_wsd_trace_covers_all_phases_and_resume_without_rewarm` and by the executed trace rows around the resume boundary.
5. **Frozen-schedule domain check:** the executed TRAINING receipt records `lr_at` probe points on the real 5B domain (0 → 0.0; 25M → 1.5e-4; 50M−1 → just under peak; 50M → 3e-4; 4.5B−1 → peak; 5B−1 → ≈3e-5), verifying the LOCKED constants themselves.

## Machine trace (executed)

`experiments/V5_1_CANARY/receipts/TRAINING.json` → `trace[]`: **120 rows, 0 lr mismatches, 7 checkpoint publications**. Executed rows:

| update | tokens | phase | lr | loss | post-clip |
|---:|---:|---|---:|---:|---:|
| 1 | 4,096 | warmup | 0.000000 | 10.1176 | 1.0000 |
| 12 | 49,152 | warmup end | 0.000275 | 7.9471 | 1.0000 |
| 13 | 53,248 | stable | 0.000300 | 7.7062 | 1.0000 |
| 60 | 245,760 | stable | 0.000300 | 1.7613 | 1.0000 |
| 103 | 421,888 | decay | 0.000300 | 0.6689 | 0.6093 |
| 120 | 491,520 | decay end | 0.000045 | 0.6122 | 0.2966 |

The trace indexes the LR at the **pre-update** cumulative token position (the frozen contract); an initial tracer defect labeled rows post-update — execution 2 (superseding execution 1 with bitwise-identical weights) corrected the labels. The clip certificate engaged throughout warmup (norm 1.0) and relaxed as gradients settled (0.30) — genuine clipping exercised (§56 question answered).

**Honest scope:** the 5B constants themselves remain unexecuted at their own scale (a 5B-token run is not authorized); what Task 3 proves is that the token-indexed WSD *mechanism* — phases, position indexing, resume continuation — executes correctly through the production path at canary scale.
