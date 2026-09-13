# CANARY READINESS

**Pre-run verdict (prerun gate, §48): `READY_FOR_GPU_CANARY`** — issued after the full CPU qualification suite passed and the launcher contract was frozen. Machine gate: [`CANARY_READINESS.json`](../../../experiments/V5_1_CANARY/CANARY_READINESS.json).

## Pre-run gate record

| Gate | State | Evidence |
|---|---|---|
| Task-2 spec/validators | PASS | `validate_next_core_spec.py` PASSED; compute-model 8/8; reference 12/12 |
| architecture frozen | SATISFIED | Task-2 head `38925a1`; V5.1 = Candidate A |
| exact rung accounting | SATISFIED | Rung A 10,227,456; Rung B 42,092,544; instantiated == analytic |
| data instrument | SATISFIED | DATA receipt `b6e23ba3…` (amendment 1); zero collisions; worst baseline 0.267 |
| production path | SATISFIED | runner through `ProductionTrainingBackend`/`CheckpointStore`; no parallel path |
| CPU qualification suite | PASS | 15/15 `tests/test_v51_canary.py` incl. fresh-process bitwise resume |
| preregistration frozen | SATISFIED | committed before the final result; amendment 1 documented |
| output-space boundary | SATISFIED | canonical full-softmax only; EXPERIMENT_ONLY modes gated |
| corpus/production-data boundary | SATISFIED | synthetic instrument only; production corpus remains a separate blocker |

## Post-run verdict: **`CANARY_FAIL_FORMATION`**

Issued once by `receipts/FINALIZATION.json` after the substantive Rung-A execution (120 updates, 491,520 real tokens, 10,227,456-param CPU canary).

**Mechanical gates: ALL PASS.** Production path only; exact parameter accounting (10,227,456 instantiated == analytic); clean splits (zero collisions); no shortcut ≥ 0.35; finite training with real parameter updates; clip certificate valid (post-clip 1.0 → 0.30 as gradients relaxed); exact token ledger; WSD trace **0 mismatches across 120 rows**, zero rewarm events, all three phases exercised, frozen-5B domain probes recorded; 7 durable checkpoints; **fresh-process resume bitwise-identical** (model/optimizer/ledger/cursor/schedule bytes); corrupted/identity-mismatched restore rejected; EOS contract exercised; receipts valid; no EXPERIMENT_ONLY leak.

**Formation gates: FAILED.** Dev overall exact 0.1331 (< 0.15); identity 0.000 (< 0.30); binding 0.203 (< 0.25); formation_positive TRUE (termination 0.313 ≥ 0.30). Sealed mirrors dev (binding 0.206, termination 0.303, identity 0.000, state_order 0.000) — no leakage; the model generalizes exactly as poorly as dev says.

**Root cause (§50):** train final loss 0.612 = memorization-level fit while dev transfer stays near-shortcut — the documented **delayed-generalization regime** (ARK-002B; "never stop at train saturation"). The single-epoch dose (each training sequence seen once) is insufficient for induction-style generalization; identity/state_order additionally show EOS-stop collapse (0.15/0.10). **Execution mechanics are not implicated.**

**Designed, NOT executed — canary-v2:** extend to ~3 epochs (1.47M tokens / 360 updates) crossing the delayed-generalization region; fresh preregistration; identity/state_order EOS diagnostics; optional LR-1e-3 arm. `READY_FOR_500M` remains structurally impossible.

**Determinism:** two independent full executions produced **bitwise-identical model.bin** (`f9afe368…`); execution 1 is superseded solely for a trace-labeling defect (LR rows mislabeled post-update — the applied LR sequence was correct) and a verdict-mapping defect, both documented in the readiness JSON.
