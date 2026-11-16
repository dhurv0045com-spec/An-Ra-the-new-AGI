# EXP-PLANTHIR-001: Instrument Validation & Controlled Fault Calibration

- **Date (UTC):** 2026-11-15T22:29:24.812283+00:00 | **Elapsed:** `2.834s` on CPU
- **Healthy V8 Kernel Check:** `passed=True`
- **Controlled Faults Detected:** `6/6` (`all_detected=True`)

## V8 Consumed-Tool Evaluator Defect Fix Verification

Tested across `12` Template-0 tool worlds with a stateless visible-state oracle actor:
- `enforce_tool_consumption=False` (reproducing V8 defect): `6/12` correct (`50.0%`), with `6` episodes trapped repeating consumed `QUERY`/`RETRIEVE` actions.
- `enforce_tool_consumption=True` (fixed `anra_v5/elendril/v8/evaluation.py`): `12/12` correct (`100.0%`).

## Controlled Fault Injection Ledger

| Fault ID | Detected | Localized Check | Status |
| :--- | :--- | :--- | :--- |
| `v8_consumed_tool_availability_bug` | `True` | `tool_consumption_monotonicity + v8_evaluator_parity` | `INSTRUMENT_DEFECT_DETECTED` |
| `truth_firewall_leak` | `True` | `actor_truth_firewall` | `INSTRUMENT_DEFECT_DETECTED` |
| `train_eval_prompt_drift` | `True` | `train_rollout_tri_parity` | `INSTRUMENT_DEFECT_DETECTED` |
| `disconnected_policy_head_grad` | `True` | `missing_active_gradient:val_head.weight` | `INSTRUMENT_DEFECT_DETECTED` |
| `v16_epoch_aggregation_shadowing` | `True` | `validate_metric_aggregator` | `INSTRUMENT_DEFECT_DETECTED` |
| `checkpoint_rng_corruption` | `True` | `checkpoint_store_restore` | `INSTRUMENT_DEFECT_DETECTED` |

