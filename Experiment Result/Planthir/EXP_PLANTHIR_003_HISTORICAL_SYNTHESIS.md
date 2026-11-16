# EXP-PLANTHIR-003: Historical AN-RA Research Atlas (V6-V8)

- **Date (UTC):** 2026-11-15T22:29:37.303762+00:00
- **Generations Indexed:** `7` | **Total Primary Archived Runs Indexed:** `52`

## Generation-by-Generation Summary

### V6 (runs counted: `14`)
- **Key Finding:** Only 2/15 primary diagnostic panels passed; calibration, retention, alias OOD, and 3-hop panels failed.

### V7 (runs counted: `9`)
- **Key Finding:** High query recall co-occurred with incomplete end-to-end confirmation solving; cause unresolved.
- **Curated Metrics:** confirmation mean=`0.665`, query recall mean=`1.000`, alias robustness mean=`n/a`

### V7.1 (runs counted: `9`)
- **Key Finding:** Expanded S3 mechanistic suite across uniform, blocked, and blocked-intermediate arms; surface alias brittleness and cause remained unresolved.
- **Curated Metrics:** confirmation mean=`0.742`, query recall mean=`0.979`, alias robustness mean=`0.207`

### V7.15 (runs counted: `6`)
- **Key Finding:** At S8, blocked order outperformed uniform across all 3 seeds (right_only 32/15/52 vs left_only 2/2/2), while alias robustness lagged badly.
- **Curated Metrics:** confirmation mean=`0.631`, query recall mean=`1.000`, alias robustness mean=`0.250`

### V7.16 (runs counted: `6`)
- **Key Finding:** At S20 (~26M), seed 7 flipped toward uniform (left_only=29 vs right_only=11) whereas seeds 19/41 favored blocked (31/30 vs 4/1); initial aggregator suffered epoch-shadowing bug.
- **Curated Metrics:** confirmation mean=`0.573`, query recall mean=`0.995`, alias robustness mean=`0.183`

### V7.17 (runs counted: `8`)
- **Key Finding:** Init/order crossing showed order effects depend on init; low-LR improved raw standard episodes yet reduced alias correctness (55 -> 40/336); internal mechanism left unresolved.
- **Curated Metrics:** confirmation mean=`0.669`, query recall mean=`0.977`, alias robustness mean=`0.147`

### V8 (runs counted: `0`)
- **Key Finding:** Pre-run protocol fixed evaluator consumed-tool leak via Planthir audit (enforce_tool_consumption=True); dual-T4 study awaiting Kaggle execution.

## Documented Historical Instrument Defects & Planthir Guards

| ID | Program | Location | Planthir Mitigation |
| :--- | :--- | :--- | :--- |
| `HIST-DEFECT-V716-SHADOW` | `V7.16` | `tools/elendril_v716_report.py` | `instrument.validate_metric_aggregator + run_fault_injection_suite fault #5` |
| `HIST-DEFECT-V8-UNCONSUMED-TOOL` | `V8` | `anra_v5/elendril/v8/evaluation.py` | `enforce_tool_consumption=True + instrument.validate_environment_kernel + fault #1` |
| `HIST-DEFECT-PRE-V8-METRIC-ONLY` | `V6-V7.17` | `Evaluators & reporters prior to Planthir` | `observatory.ProcessFunnelAnalyzer + observatory.InternalTelemetryCollector + causal.*` |

## Unresolved Mechanistic Questions Inherited from V6-V8

- Why does near-100% query recall (V7-V7.17) still yield <50% confirmation accuracy on novel surface aliases and 3-hop/4-hop chains?
- Does factorized typed heads + pointer attention (V8 F/P/U) overcome the readout-vs-representation bottleneck at S3/S20?
- Do recurrent step dynamics (V8 U, T=1..3) improve multi-hop variable chaining without overfitting surface token statistics?

