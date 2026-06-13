# Pre-Push Audit - 2026-06-08

> **Archived evidence:** this file records what was true on June 8, 2026. It is intentionally not updated to current test counts or architecture status. Use the current engineering log and rerun the listed commands for present-day evidence.

Scope: research-roadmap implementation scaffolds, operator routing work, and documentation tracking updates currently present in the working tree.

## Verification

| Check | Result | Notes |
|---|---:|---|
| Focused roadmap/operator suite | PASS | 74 passed |
| Full test suite | PASS | 259 passed, 1 warning |
| Python compile check for touched modules | PASS | `py_compile` completed |
| Exact merge-conflict marker sweep | PASS | No `<<<<<<<`, `=======`, or `>>>>>>>` markers found |

Full suite command:

```powershell
python -m pytest tests -q
```

Focused suite command:

```powershell
python -m pytest tests\test_gepa.py tests\test_self_improvement_gepa.py tests\test_rlvr.py tests\test_rlvr_dapo.py tests\test_new_systems.py tests\test_optimizer_bakeoff.py tests\test_sparse_lora.py tests\test_eval_v2.py tests\test_unified_training_plan.py tests\test_anra_brain_unit.py tests\test_v2_stack.py tests\test_block1_architecture.py tests\test_operator_tools.py -q
```

Warning observed in the full suite:

- `tests/test_phase3_integration.py::TestSymbolicBridge::test_code_analysis_finds_issues` emitted a Windows `cp1252` subprocess stdin encoding warning from a background writer thread. Tests still passed.

## Implemented Roadmap Scaffolds

| Area | Status | Artifact |
|---|---|---|
| Golden eval baseline | Implemented scaffold | `output/v2/v2_golden_eval_baseline.json` |
| SparseLoRA estimate | Implemented logging-only | `output/v2/v2_sparse_lora_report.json` |
| Optimizer bake-off | Implemented scaffold | `output/v2/v2_optimizer_bakeoff_report.json` |
| RLVR DAPO telemetry | Implemented scaffold | `output/v2/v2_rlvr_report.json` |
| GEPA reflections | Implemented proposal-only scaffold | `output/v2/v2_gepa_report.json` |

## Push Notes

- These upgrades are measurement/proposal layers unless explicitly eval-promoted.
- No default prompt/tool-rule mutation is enabled by GEPA.
- SparseLoRA does not skip training tokens yet; it measures estimated savings.
- SCALE and GaLore are report/fallback candidates until verified implementations are installed.
- The actual golden eval baseline still needs to be produced against the checkpoint selected for promotion.

## Next Upgrade

P0-06 from the research roadmap: TurboQuant and KVarN audit.

Target artifact:

```text
output/v2/v2_turboquant_audit_report.json
```
