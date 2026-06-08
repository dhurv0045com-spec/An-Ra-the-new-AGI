# An-Ra Architecture

**Registry wins.** `runtime/system_registry.py` is the live map. If prose disagrees, regenerate:

```bash
python -m inference.full_system_connector
python scripts/status.py
python scripts/readiness.py
python scripts/verify_structure.py
```

Status line: **19/19 active** = source + imports OK. Checkpoints optional until trained/restored.

---

## Layers (how to think about the repo)

```text
┌─────────────────────────────────────────────────────────┐
│  Operator surfaces: anra.py · app.py · Colab · web UI                   │
├─────────────────────────────────────────────────────────┤
│  Governance: sovereignty · self_mod · feature_flags     │
├─────────────────────────────────────────────────────────┤
│  Cognition+: symbolic · ouroboros · ghost · identity inj  │
├─────────────────────────────────────────────────────────┤
│  Agency: goals · orchestrator · agent_loop · master_sys │
├─────────────────────────────────────────────────────────┤
│  Continuity: memory_router · phase2_memory · HAL/ESV    │
├─────────────────────────────────────────────────────────┤
│  Learning: data_mix · training_loop · eval · replay     │
├─────────────────────────────────────────────────────────┤
│  Model: brain · tokenizer · generate / inference        │
├─────────────────────────────────────────────────────────┤
│  Spine: registry · telemetry · eval_harness · report      │
└─────────────────────────────────────────────────────────┘
```

---

## Research → code

| Research line | Implementation |
|---------------|----------------|
| DFC — change → constraint → verify → update | `frontier_dfc.jsonl`, `training/rlvr.py`, falsification ledger |
| FCC — six templates | `scripts/build_frontier_dataset.py` |
| AIE — improvement as experiment | `innovation/`, `engine/eval_harness.py` |
| HAL nervous system | `identity/hal.py` |
| Proof memory | `memory/experimental_proof_graph.py` |

---

## Inference path (runtime loop)

```text
prompt
  → tokenizer_v3
  → CausalTransformerV2 (anra_brain.py)
  → identity inject / clean (45N + CIV)
  → memory_router + ghost retrieval
  → [optional] symbolic pre-check (45Q)
  → generation (generate.py)
  → [optional] Ouroboros passes
  → memory write · falsification hooks · telemetry
```

---

## Agency path (work loop)

```text
owner imperative (/goal or goal:)
  → MasterSystem.run_goal (45M)
  → Agent: plan → execute → monitor → evaluate (45K)
  → tools: file_manager · os_action · cad_generate · code · web · …
  → artifacts under workspace/ (anra_paths.get_agent_workspace)
  → operator_actions.jsonl audit
```

See [`OPERATOR.md`](docs/OPERATOR.md) and WALKTHROUGH §19.

---

## Training path

```text
anra_training.txt
  → v2_data_mix (65/15/10/5/5)
  → train_unified (session | train | eval)
  → eval_v2 · benchmark · verifier
  → hard examples → replay_pipeline
  → [milestone] identity · Ouroboros · sovereignty → promote or hold
```

---

## Component table (19)

| # | ID | Layer | Primary paths |
|---|-----|-------|----------------|
| 01 | `brain` | model | `anra_brain.py`, `training/v2_*` |
| 02 | `tokenizer` | data | `tokenizer/tokenizer_v3.json`, `tokenizer_adapter.py` |
| 03 | `data_mix` | data | `training_data/anra_training.txt`, `v2_data_mix.py` |
| 04 | `training_loop` | learning | `train_unified.py`, `build_brain.py` |
| 05 | `evaluation` | measurement | `eval_v2.py`, `benchmark.py`, `verifier.py` |
| 06 | `inference_runtime` | serving | `generate.py`, `inference/` |
| 07 | `api_web` | interface | `app.py`, `phase4/web/` |
| 08 | `identity` | alignment | `identity/`, `phase3/identity (45N)/` |
| 09 | `memory_router` | continuity | `memory/memory_router.py` |
| 10 | `phase2_memory` | continuity | `phase2/memory (45J)/` |
| 11 | `goals` | agency | `goals/goal_queue.py`, `agents/orchestrator.py` |
| 12 | `agent_loop` | agency | `phase2/agent_loop (45k)/` |
| 13 | `master_system` | autonomy | `phase2/master_system (45M)/system.py` |
| 14 | `self_improvement` | learning | `phase2/self_improvement (45l)/` |
| 15 | `self_modification` | governance | `self_modification/`, `execution/` |
| 16 | `ouroboros` | reflection | `phase3/ouroboros (45O)/` |
| 17 | `ghost_memory` | continuity | `phase3/ghost_memory (45P)/` |
| 18 | `symbolic_bridge` | verification | `phase3/symbolic_bridge (45Q)/` |
| 19 | `sovereignty` | governance | `phase3/sovereignty (45R)/` |

---

## Engineering spine (cross-cutting)

```text
component_registry()
  → feature_flags (state/feature_flags.json)
  → @trace → telemetry.jsonl
  → EvalHarness → regression artifacts
  → engine/report.py → operator scorecard
  → docs/engineering/ENGINEERING_LOG.md → human/AI change history
  → docs/planning/MASTER_GOALS.md → research / test / ship backlog
```

New capabilities must plug in here or document why not. **After shipping:** append the log (`scripts/log_engineering_change.py`) and update goal status in `MASTER_GOALS.md`.

---

## Path law

- All filesystem constants: **`anra/anra_paths.py`**
- Agent sandbox: **`get_agent_workspace()`** → default `workspace/`
- Engineering outputs: **`ENGINEERING_DIR`**
- Operator audit: **`OPERATOR_AUDIT_LOG`**

Test: `tests/test_path_registry_literals.py`

---

## Phase folders

`phase2/` and `phase3/` are **capability layers**, not separate products. Prefer mainline imports:

- `generate.py` not a forked inferencer
- `memory/memory_router.py` not ad-hoc stores
- `training.train_unified` not one-off trainers

---

## External systems (roadmap architecture)

| System | Integration pattern |
|--------|---------------------|
| Remote PC | Paired `anra_node` + TLS + tier 3 approval |
| ROS2 robot | Action server; An-Ra sends goals, not motor ticks |
| CAD pipeline | `cad_generate` → OpenSCAD; later FreeCAD/CadQuery adapters |

Fast control stays out of the LLM hot path.

---

## Rules (non-negotiable)

1. Registry truth in one place.  
2. No scattered path literals.  
3. Every upgrade moves a measurable axis.  
4. Promotion through sovereignty, not hope.  
5. Operator actions auditable.

---

## Docs

| File | Use |
|------|-----|
| [`OPERATOR.md`](docs/OPERATOR.md) | Desktop & engineering actions |
| [`WALKTHROUGH.md`](docs/WALKTHROUGH.md) | Subsystem depth + §19 operator |
| [`DEVELOPER.md`](docs/DEVELOPER.md) | Change protocol |
