# Engineering Log

> **Purpose:** Dated record of every meaningful add, change, remove, and improvement — by humans or AI — tied to components and verification.  
> **Newest first.** Format: [`LOG_STANDARD.md`](LOG_STANDARD.md) · CLI: `python scripts/log_engineering_change.py`

---

## 2026-06-13 - CHANGE - `architecture` - Definitive lifecycle integration and evidence hardening

| Field | Value |
|-------|-------|
| **Date** | 2026-06-13 |
| **Author** | ai-agent |
| **Component** | `training`, `evaluation`, `runtime`, `memory`, `agent_loop`, `robotics`, `api_web`, `docs` |
| **Type** | CHANGE |
| **Summary** | Integrated the definitive architecture, then hardened migration, checkpoint failure semantics, evidence metrics, promotion verification, world-model training, and typed goal routing |
| **Files** | canonical owners plus all maintained Markdown documents |
| **Metrics** | exact frontier/3B counts; T-01 through T-26 reachability; M-01 through M-12 evidence schema |
| **Verification** | focused architecture suites and full `python -m pytest tests -q` (`312 passed`) |
| **Risk** | high |
| **Follow-up** | produce real frontier, IBS, hardware, dataset, memory and growth evidence before 3B authorization |
| **Evidence artifacts** | generated only by canonical commands; absent evidence remains an explicit blocker |
| **Rollback** | promoted artifacts remain immutable; code changes are isolated by canonical owners and tests |

### Detail

- Compact evaluation no longer impersonates IBS, RLVR, memory, sovereignty or continual-learning evidence.
- Legacy vocabulary rows remain exact; appended control rows are deterministic and provenance is checkpointed.
- Incompatible checkpoints raise a typed error instead of silently starting fresh.
- Documentation now distinguishes implemented, measured and promoted.

## 2026-06-08 - CHECK - `tests` - Pre-push roadmap audit

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `tests`, `docs` |
| **Type** | CHECK |
| **Summary** | Re-evaluated roadmap implementation scaffolds before git push and added a pre-push audit note |
| **Files** | docs/planning/PRE_PUSH_AUDIT_2026_06_08.md, docs/research/ANRA_IMPLEMENTATION_ROADMAP_BEST_OF.md, docs/planning/MASTER_GOALS.md |
| **Metrics** | Focused suite 74 passed; full suite 259 passed with 1 Windows encoding warning; compile check passed; conflict-marker sweep clean |
| **Verification** | python -m pytest tests -q |
| **Risk** | low |
| **Follow-up** | Run the actual golden eval against the promoted checkpoint before claiming model-quality improvement |

---

## 2026-06-08 - ADD - `self_improvement` - GEPA-style reflection scaffold

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `self_improvement`, `training_loop` |
| **Type** | ADD |
| **Summary** | Added trace-backed GEPA reflections and prompt/tool-policy candidate proposals; self-improvement now writes a GEPA report without auto-applying changes |
| **Files** | training/gepa.py, scripts/run_self_improvement.py, training/v2_config.py, tests/test_gepa.py, tests/test_self_improvement_gepa.py |
| **Metrics** | Reports trace count, reflections, candidates, scores, owner-review gate, and auto-apply disabled state |
| **Verification** | python -m pytest tests/test_gepa.py tests/test_self_improvement_gepa.py tests/test_rlvr.py tests/test_rlvr_dapo.py tests/test_new_systems.py tests/test_optimizer_bakeoff.py tests/test_sparse_lora.py tests/test_eval_v2.py tests/test_unified_training_plan.py tests/test_anra_brain_unit.py tests/test_v2_stack.py tests/test_block1_architecture.py -q |
| **Risk** | low |
| **Follow-up** | Score GEPA candidates against golden eval tasks before any prompt/tool-policy promotion |

---

## 2026-06-08 - ADD - `training_loop` - RLVR DAPO-style telemetry scaffold

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `training_loop`, `replay` |
| **Type** | ADD |
| **Summary** | Added RLVR overlong reward shaping, token-level policy-loss knob, dynamic sampling knob, KL/output/replay metrics, and richer replay provenance |
| **Files** | training/rlvr.py, training/replay_pipeline.py, training/v2_config.py, tests/test_rlvr_dapo.py |
| **Metrics** | Reports G, policy loss, KL loss, effective KL, output lengths, verifier pass rate, reward stats, and replay additions |
| **Verification** | python -m pytest tests/test_rlvr.py tests/test_rlvr_dapo.py tests/test_new_systems.py tests/test_optimizer_bakeoff.py tests/test_sparse_lora.py tests/test_eval_v2.py tests/test_unified_training_plan.py tests/test_anra_brain_unit.py tests/test_v2_stack.py tests/test_block1_architecture.py -q |
| **Risk** | low |
| **Follow-up** | Run RLVR A/B against golden eval before changing default reasoning-training policy |

---

## 2026-06-08 - ADD - `training_loop` - Optimizer bake-off scaffold

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `training_loop` |
| **Type** | ADD |
| **Summary** | Added named optimizer adapter/report scaffold for AdamW, Muon, SCALE, and GaLore; trainer writes the selected/fallback optimizer report |
| **Files** | training/anra_optimizer.py, scripts/build_brain.py, training/train_unified.py, training/finetune_anra.py, training/v2_config.py, tests/test_optimizer_bakeoff.py |
| **Metrics** | Reports trainable params, candidate optimizer state estimates, selected optimizer, fallback reason, and param groups |
| **Verification** | python -m pytest tests/test_optimizer_bakeoff.py tests/test_sparse_lora.py tests/test_eval_v2.py tests/test_unified_training_plan.py tests/test_anra_brain_unit.py tests/test_v2_stack.py tests/test_block1_architecture.py -q |
| **Risk** | low |
| **Follow-up** | Run same data slice with `--optimizer adamw`, `--optimizer muon`, and optional verified SCALE/GaLore packages when installed |

---

## 2026-06-08 - ADD - `training_loop` - SparseLoRA logging-only estimator

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `training_loop` |
| **Type** | ADD |
| **Summary** | Added SparseLoRA-style contextual sparsity estimator and identity fine-tune report hook; training remains unchanged until eval-gated comparison passes |
| **Files** | training/sparse_lora.py, training/finetune_anra.py, training/v2_config.py, tests/test_sparse_lora.py |
| **Metrics** | Reports active tokens, kept tokens, estimated skipped tokens, skip ratio, and measure-only decision |
| **Verification** | python -m pytest tests/test_sparse_lora.py tests/test_eval_v2.py tests/test_unified_training_plan.py tests/test_anra_brain_unit.py tests/test_v2_stack.py tests/test_block1_architecture.py -q |
| **Risk** | low |
| **Follow-up** | Compare baseline LoRA/QLoRA vs SparseLoRA logging estimates on the same data slice and checkpoint |

---

## 2026-06-08 - ADD - `evaluation` - Golden eval baseline artifact

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `evaluation`, `training_loop` |
| **Type** | ADD |
| **Summary** | Added compact-eval golden baseline schema, promotion gates, and `train_unified --mode eval` artifact hook; resolved eval/model-path merge markers |
| **Files** | anra_brain.py, training/eval_v2.py, training/train_unified.py, training/v2_config.py, training/v2_runtime.py, tests/test_eval_v2.py, tests/test_unified_training_plan.py, tests/test_block1_architecture.py |
| **Metrics** | Baseline captures suite scores, task outputs, thresholds, gates, and promotion decision |
| **Verification** | python -m pytest tests/test_eval_v2.py tests/test_unified_training_plan.py tests/test_anra_brain_unit.py tests/test_v2_stack.py tests/test_block1_architecture.py -q |
| **Risk** | low |
| **Follow-up** | Run `python -m training.train_unified --mode eval` against the promoted checkpoint to create the real baseline JSON |

---

## 2026-06-08 — ADD — `docs` — Best-of intelligence efficiency research pack

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `docs` |
| **Type** | ADD |
| **Summary** | Added a best-fit research paper and implementation roadmap for making An-Ra more intelligent and efficient without changing its core vision |
| **Files** | docs/research/ANRA_BEST_RESEARCH_FOR_INTELLIGENCE_AND_EFFICIENCY.md, docs/research/ANRA_IMPLEMENTATION_ROADMAP_BEST_OF.md, docs/planning/MASTER_GOALS.md |
| **Metrics** | 50 papers and technologies mapped to An-Ra components and priorities |
| **Verification** | manual doc audit; source links and R-07 tracking entry present |
| **Risk** | low |
| **Follow-up** | Implement P0 roadmap items after eval baseline |

---

## 2026-06-08 — CHANGE — `operator` — Natural chat routes workspace actions

| Field | Value |
|-------|-------|
| **Date** | 2026-06-08 |
| **Author** | ai-agent |
| **Component** | `operator` |
| **Type** | CHANGE |
| **Summary** | Plain chat routes concrete workspace requests to safe operator tools |
| **Files** | runtime/operator_commands.py, phase2/master_system_45m/system.py, tests/test_operator_tools.py, docs/OPERATOR.md, docs/planning/MASTER_GOALS.md |
| **Metrics** | Natural file-write UX covered by focused operator tests |
| **Verification** | python -m pytest tests/test_operator_tools.py -q - 6 passed |
| **Risk** | low |
| **Follow-up** | none |

---

## 2026-05-24 — ADD — `docs` — Documentation hub and tracking spine

| Field | Value |
|-------|-------|
| **Date** | 2026-05-24 |
| **Author** | human |
| **Component** | `docs` |
| **Type** | ADD |
| **Summary** | docs/ tree with ENGINEERING_LOG, MASTER_GOALS, LOG_STANDARD, log script |
| **Files** | docs/, scripts/log_engineering_change.py, anra/anra_paths.py |
| **Metrics** | n/a |
| **Verification** | pytest tests/test_engineering_log.py |
| **Risk** | low |
| **Follow-up** | none |

---

## 2026-05-24 — ADD — `operator` — Desktop operator pack (tools + slash commands)

| Field | Value |
|-------|-------|
| **Date** | 2026-05-24 |
| **Author** | ai-agent |
| **Component** | `agent_loop`, `operator`, `master_system` |
| **Type** | ADD |
| **Summary** | `os_action`, `cad_generate`, workspace via `get_agent_workspace()`, chat `/goal` `/write` `/open` `/cad`, audit log |
| **Files** | `phase2/agent_loop (45k)/builtin.py`, `runtime/operator_commands.py`, `anra/anra_paths.py`, `anra.py`, `OPERATOR.md`, `tests/test_operator_tools.py` |
| **Metrics** | Operator actions auditable; tool success in agent goals |
| **Verification** | `pytest tests/test_operator_tools.py`; full suite 163 passed |
| **Risk** | medium — `os_action` opens OS handlers; sandbox + allowed roots |
| **Follow-up** | Symbolic-in-loop in generate; tier gates for destructive opens |

---

## 2026-05-24 — FIX — `runtime` — Windows test + path + console fixes

| Field | Value |
|-------|-------|
| **Date** | 2026-05-24 |
| **Author** | ai-agent |
| **Component** | `runtime`, `tests` |
| **Type** | FIX |
| **Summary** | TOKENIZER lazy export; sandbox without Unix `resource`; train_oneshot path literal; symbolic Unicode console |
| **Files** | `generate.py`, `execution/sandbox.py`, `scripts/train_oneshot.py`, `anra.py` |
| **Metrics** | CI green on Windows |
| **Verification** | `pytest tests/ -q` — 156+ passed |
| **Risk** | low |
| **Follow-up** | none |

---

## 2026-05-24 — DOCS — `docs` — Major documentation refresh

| Field | Value |
|-------|-------|
| **Date** | 2026-05-24 |
| **Author** | ai-agent |
| **Component** | `docs` |
| **Type** | DOCS |
| **Summary** | Rewrote README, ARCHITECTURE, DEVELOPER, VISION, OPERATOR, phase READMEs; WALKTHROUGH §19 addendum only |
| **Files** | `README.md`, `ARCHITECTURE.md`, `DEVELOPER.md`, `VISION.md`, `OPERATOR.md`, `WALKTHROUGH.md`, `phase*/README.md` |
| **Metrics** | n/a |
| **Verification** | Human review |
| **Risk** | low |
| **Follow-up** | Keep ENGINEERING_LOG updated per change |

---

## 2026-05-17 — ADD — `engine` — Engineering spine (registry + telemetry + report)

| Field | Value |
|-------|-------|
| **Date** | 2026-05-17 |
| **Author** | owner |
| **Component** | `engine`, `runtime` |
| **Type** | ADD |
| **Summary** | Platform layer: component_base, feature_flags, telemetry, eval_harness, report; 19-component registry |
| **Files** | `engine/*`, `runtime/system_registry.py` |
| **Metrics** | `anra.py --report` scorecard axes |
| **Verification** | `python anra.py --report` — 19/19 |
| **Risk** | low |
| **Follow-up** | Fill telemetry with real workloads |

---

## 2026-06-14 - ADD - `cognition` - Cognitive extension and T4 readiness

| Field | Value |
|-------|-------|
| **Date** | 2026-06-14 |
| **Author** | codex |
| **Component** | `cognition`, `training`, `evaluation`, `service` |
| **Type** | ADD |
| **Summary** | Added C-01 through C-07 facade, exact zero-gated causal extension, consent/encrypted owner model, causal corpus/trainer, AGI evidence schemas, T4 preflight, Colab bootstrap, APIs and docs |
| **Files** | `cognition/*`, `data/causal_corpus.py`, `training/preflight.py`, `training/causal_trainer.py`, `evaluation/agi_benchmarks.py`, `app.py` |
| **Metrics** | Exact corpus total 7,500; unsupported frontier/3B T4 launches blocked before allocation |
| **Verification** | `pytest tests/test_cognitive_architecture.py`; full suite |
| **Risk** | medium |
| **Follow-up** | Run signed 25M Colab campaign and collect A-01/A-02 evidence |

---

*Append new entries above this line. Do not delete history without owner approval.*
