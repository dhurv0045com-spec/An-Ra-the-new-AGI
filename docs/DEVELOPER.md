# An-Ra Developer Guide

> Ship like a platform team: **thin changes, measured outcomes, owner boundaries intact.**

For humans and coding agents. An-Ra is ~70k lines of intentional systems — read before you edit.

---

## Spine (do not reimplement)

| Need | Module |
|------|--------|
| Paths | `anra/anra_paths.py` — includes `get_agent_workspace()`, `ENGINEERING_DIR` |
| Registry | `runtime/system_registry.py` |
| Flags | `engine/feature_flags.py` |
| Telemetry | `engine/telemetry.py` — `@trace` |
| Regression | `engine/eval_harness.py` |
| Report | `engine/report.py` |
| Operator CLI | `runtime/operator_commands.py` |
| Agent tools | `phase2/agent_loop (45k)/builtin.py` |
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
| CLI | `anra.py` |
| Operator | `runtime/operator_commands.py`, `OPERATOR.md` |
| Agent | `phase2/agent_loop (45k)/agent_main.py`, `builtin.py`, `planner.py` |
| Master | `phase2/master_system (45M)/system.py` |
| Model | `anra_brain.py`, `training/v2_runtime.py` |
| Train | `training/train_unified.py` |
| Verify | `phase3/symbolic_bridge (45Q)/` |
| Govern | `phase3/sovereignty (45R)/`, `self_modification/` |

---

## AI agent workflow

1. Read relevant source.  
2. Thin adapter / decorator — no drive-by rewrites.  
3. Do not change weights, prompts, identity text, or training mix unless asked.  
4. Tests for new behavior.  
5. `python -m pytest tests/ -q`  
6. `python anra.py --report` if platform/operator touched.  
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
python anra.py --report
python anra.py --chat
python anra.py --goal "..."
python -m pytest tests/ -q
python -m training.train_unified --mode session
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
