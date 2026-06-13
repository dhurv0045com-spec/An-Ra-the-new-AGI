# An-Ra Developer Guide

> **Current contract (2026-06-13):** canonical lifecycles live in the ownership map below. Older phase adapters are compatibility surfaces, not permission to create a second implementation.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/An-Ra-the-new-AGI
cd An-Ra-the-new-AGI
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# 2. Verify
python -m scripts.verify_structure
python -m pytest tests/ -m "not gpu" -q

# 3. Exercise the integrated training spine
python -m training.train_unified --mode session --model-size 25m --max-steps 2

# 4. Inspect real 3B blockers
python -m training.train_unified --mode status --model-size 3b

# 5. Serve
uvicorn app:app --reload --port 8000
curl http://localhost:8000/health
```

> Training checkpoints → `output/checkpoints/`
> Metrics → `output/metrics/`
> Sessions → `state/sessions.db`

> Ship like a platform team: **thin changes, measured outcomes, owner boundaries intact.**

For humans and coding agents. An-Ra is ~70k lines of intentional systems — read before you edit.

---

## Spine (do not reimplement)

The decisive lifecycle owners are:

| Need | Canonical owner |
|------|-----------------|
| Campaign | `training/train_unified.py` |
| Training execution | `scripts/build_brain.py` |
| Model and checkpoints | `training/v2_runtime.py` |
| Data mix and replay | `training/v2_data_mix.py` |
| Model growth | `training/csii.py` |
| Evaluation | `evaluation/ibs.py` |
| Promotion and rollback | `evaluation/promotion.py` |
| Scale authorization | `training/ssg.py` |
| Agent lifecycle | `engine/agent_loop.py` |
| Robotics workflow | `robotics/workflow.py` |
| Persistent service | `app.py` |

Compatibility packages may re-export these owners, but they do not own lifecycle behavior.

| Need | Module |
|------|--------|
| Paths | `anra/anra_paths.py` |
| Config | `anra/core/config.py` — `AnRaConfig.from_yaml()` |
| Registry | `anra/core/registry.py` — `MODEL_REGISTRY`, `MEMORY_REGISTRY`, etc. |
| Protocols | `anra/core/protocols.py` — interfaces for all major components |
| Model | `anra_brain.py` → `anra/core/model.py` |
| Serving | `app.py` → `anra/serving/` |
| Inference | `generate.py` → `anra/inference/` |
| Identity | `identity/` → `anra/identity/` |
| Memory | `memory/` → `anra/memory/` |
| Training | `training/` → `anra/training/` |
| Flags | `engine/feature_flags.py` |
| Telemetry | `engine/telemetry.py` |
| Operator CLI | `runtime/operator_commands.py` |
| Audit | `state/logs/operator_actions.jsonl` |

---

## Contract per component

Must answer:

```text
What does it do?
Is it enabled?
How fast / how often does it fail?
What test proves it?
What regression protects it?
What owner boundary applies?
```

---

## Operator pack (recent)

| Piece | Location |
|-------|----------|
| Workspace sandbox | `get_agent_workspace()` |
| Slash commands | `runtime/operator_commands.py` + `_run_chat` in 45M |
| `os_action` tool | open / reveal / URL |
| `cad_generate` tool | `runtime/engineering_templates/` |
| User doc | [`OPERATOR.md`](OPERATOR.md) |

Adding tools: register in `register_all_tools()`, add dispatcher keywords, add test in `tests/test_operator_tools.py`, document in OPERATOR.md.

**Do not** put `workspace/` or `training_data/` string literals in Python — use `anra_paths`.

---

## Read before edit (by area)

| Area | Files |
|------|-------|
| CLI | `scripts/anra.py` |
| Operator | `runtime/operator_commands.py`, `OPERATOR.md` |
| Agent | `phase2/agent_loop_45k/agent_main.py`, `builtin.py`, `planner.py` |
| Master | `phase2/master_system_45m/system.py` |
| Model | `anra_brain.py`, `training/v2_runtime.py` |
| Train | `training/train_unified.py`, then `scripts/build_brain.py` |
| Verify | `phase3/symbolic_bridge_45q/` |
| Govern | `phase3/sovereignty_45r/`, `self_modification/` |

---

## AI agent workflow

1. Read relevant source.  
2. Thin adapter / decorator — no drive-by rewrites.  
3. Do not change weights, prompts, identity text, or training mix unless asked.  
4. Tests for new behavior.  
5. `python -m pytest tests/ -q`  
6. `python scripts/anra.py --report` if platform/operator touched.  
7. Report commands + residual risk.

**Good prompt:**

```text
Add tool X via register_all_tools. Tests. No model changes.
```

**Bad prompt:**

```text
Rewrite architecture for AGI.
```

---

## Rules (expanded)

### Authorship mix

65/15/10/5/5 — changing it needs drift evidence (CIV / eval).

### Paths

`anra/anra_paths.py` only. Literal ban enforced by `test_path_registry_literals.py`.

### Flags over comment-out

```python
from engine.feature_flags import is_enabled, set_flag
```

### Telemetry

One `@trace` per subsystem entrypoint — not every helper.

### Eval before boasting

```text
baseline → system_on → ablation → compare
```

### Daily vs milestone

| Mode | Command | Weight |
|------|---------|--------|
| Daily | `--mode session` | Light, reliable |
| Milestone | `--mode train` | Identity, Ouroboros, sovereignty |

### Verification stack

pytest · verifier · benchmark · symbolic · report diff · telemetry

### `engine/` imports

No torch/faiss/transformers at import time in new `engine/` modules.

### Operator safety

- `file_manager` / `cad_generate`: RESTRICTED  
- `os_action`: DANGEROUS — paths under workspace or `ANRA_ALLOWED_OPEN_ROOTS`  
- Never add silent remote shell without paired node design

---

## Commands

```bash
make install          # pip install -e ".[dev]"
make test             # full non-GPU suite
make train-tiny       # 100-step CPU smoke train
make lint             # ruff
make typecheck        # mypy anra/
python scripts/anra.py --report
python scripts/anra.py --chat
python -m pytest tests/ -q
```

---

## Definition of done

- [ ] Focused tests  
- [ ] Full suite green or explained  
- [ ] Report works if operator/platform changed  
- [ ] OPERATOR.md updated if user-facing commands changed  
- [ ] **Append [`docs/engineering/ENGINEERING_LOG.md`](engineering/ENGINEERING_LOG.md)** (use `scripts/log_engineering_change.py`)
- [ ] Update [`docs/planning/MASTER_GOALS.md`](planning/MASTER_GOALS.md) status if a goal was completed
- [ ] No path literals  
- [ ] No silent identity/prompt drift  
- [ ] Disable / trace / eval path exists  

---

## Review table

| Q | A |
|---|---|
| Component? | registry name |
| Toggle? | flag |
| Traced? | telemetry |
| Test? | file + cmd |
| Regression? | harness |
| Operator doc? | OPERATOR.md if needed |

Operate the repo weekly — admiration does not compound; loops do.
