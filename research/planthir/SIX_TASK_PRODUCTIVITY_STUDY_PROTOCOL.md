# Planthir Six-Task Counterbalanced Paired Productivity Study Protocol

**Status:** `STUDY_OBSERVATIONS_PENDING` (Preregistered; zero empirical productivity percentages are claimed until paired observations are recorded and analyzed via `anra_v5.planthir.productivity.analyze_productivity_study`).

## 1. Objective

Evaluate whether the Planthir declarative CLI and scientific observatory (`anra-planthir-spec-v1`) reduce agent token consumption and wall-clock time without degrading scientific quality (non-inferiority margin `>= -0.05` on rubric score) compared with manual baseline scripting on AN-RA research workflows.

## 2. Counterbalanced Within-Participant Design

Each benchmark participant / fresh agent session completes **6 standardized engineering & research tasks**, randomized into counterbalanced `AB / BA` sequences (`baseline_first` vs `planthir_first`) across tasks and participants to eliminate order/learning confounds.

| Task ID | Task Description | Backend | Quality Floor |
| :--- | :--- | :--- | :--- |
| `T1_instrument_fault_triage` | Isolate and prove an evaluator state-transition fault | `v8` | `>= 0.95` |
| `T2_confounder_firewall_repair` | Diagnose an undeclared confound in an experiment spec and repair it | `v8` | `>= 0.95` |
| `T3_crash_resume_parity` | Recover an interrupted training run cleanly and verify final checkpoint parity | `x1_rd2` | `>= 0.95` |
| `T4_x1_rd2_architecture_contrast` | Author and validate a typed relational slot vs pointer architecture contrast on X1 R/D2 | `x1_rd2` | `>= 0.95` |
| `T5_bundle_replay_and_custody` | Verify custody chain and independently replay an exported result bundle | `x1_rd2` | `>= 0.95` |
| `T6_cohort_leakage_and_paired_stats` | Detect transitive cohort leakage and compute Holm-Bonferroni paired seed effects | `v8` | `>= 0.95` |

## 3. Primary Measured Endpoints

1. **Paired Token Reduction Fraction:** `(baseline_tokens - planthir_tokens) / baseline_tokens` (prompt + completion + tool tokens logged per task session).
2. **Paired Wall-Time Reduction Fraction:** `(baseline_wall_seconds - planthir_wall_seconds) / baseline_wall_seconds`.
3. **Quality Delta:** `planthir_quality_score - baseline_quality_score` (independent blinded grading rubric `[0.0, 1.0]`).

## 4. Reporting Rule

Until real counterbalanced logs are supplied to `analyze_productivity_study(observations)`, Planthir emits `"status": "STUDY_OBSERVATIONS_PENDING"` and `"claimed_efficiency_percent": null`. Legacy heuristic ratio estimates (e.g. `estimate_token_fuel_savings`) quantify handoff payload compaction only and MUST NOT be reported as empirical human/agent productivity gains.
