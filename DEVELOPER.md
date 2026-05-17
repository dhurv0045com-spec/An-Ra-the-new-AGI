# An-Ra Developer Guide

> Build aggressively. Ship measurably. Do not re-implement the spine.

This is for humans and coding agents. An-Ra should feel like a **real ML platform**: registry you trust, paths you do not grep for, flags that actually disable things, telemetry you can read, reports that tell the truth.

---

## Rule zero

**Do not re-implement what already exists.**

| Need | Go here |
| --- | --- |
| Paths | `anra_paths.py` |
| Component truth | `runtime/system_registry.py` |
| Component contract | `engine/component_base.py` |
| Feature flags | `engine/feature_flags.py` |
| Telemetry | `engine/telemetry.py` |
| Regression / ablation | `engine/eval_harness.py` |
| Health snapshot | `engine/report.py` |
| Audit events | `shared_logger.py` |
| HAL telemetry | `runtime/hal_telemetry.py` |
| Startup contracts | `startup_checks.py` |

---

## Platform contract

Every registered component must be:

- listed in `component_registry()`
- toggleable via feature flags
- traceable when invoked (`@trace` on main entrypoints)
- visible in `python anra.py --report`
- covered by tests or explicit smoke checks
- comparable via eval harness when behavior changes

**If it cannot answer these, it is not ready:**

```text
What does it do?
Is it enabled?
How fast is it?
How often does it fail?
What test proves it?
What regression check protects it?
What owner-control boundary applies?
```

---

## Read before you edit

| Area | Files |
| --- | --- |
| Entry | `anra.py` |
| Paths | `anra_paths.py` |
| Registry | `runtime/system_registry.py` |
| Report | `engine/report.py`, `scripts/status.py` |
| Flags | `engine/feature_flags.py`, `agents/orchestrator.py` |
| Model | `anra_brain.py`, `training/v2_config.py`, `training/v2_runtime.py` |
| Training | `training/train_unified.py`, `scripts/build_brain.py` |
| Data | `training/v2_data_mix.py`, `scripts/setup_dataset.py` |
| Eval | `training/eval_v2.py`, `training/benchmark.py`, `training/verifier.py` |
| Memory | `memory/memory_router.py`, `phase2/memory (45J)/` |
| Agency | `goals/goal_queue.py`, `agents/orchestrator.py`, `phase2/agent_loop (45k)/` |
| Autonomy | `phase2/master_system (45M)/system.py` |
| Verify | `phase3/symbolic_bridge (45Q)/` |
| Govern | `self_modification/`, `phase3/sovereignty (45R)/` |
| Web | `phase4/web/src/App.jsx` |

---

## Working with AI agents on this repo

**Give the agent a goal and this workflow:**

1. Read relevant source first.
2. Prefer thin adapters, decorators, tables — not rewrites.
3. Do not touch model weights, prompts, identity text, or training data unless asked.
4. Add tests for every new public behavior.
5. Run focused tests, then `python -m pytest tests/ -q`.
6. Run `python anra.py --report` after platform/operator changes.
7. Report: files changed, commands run, outcomes, residual risk.

**Good prompt:**

```text
Thin adapter on registry/telemetry/flags. No model rewrite.
Focused tests. Full suite. Exact commands + results.
```

**Bad prompt:**

```text
Make it more advanced. Rewrite architecture. No tests.
```

---

## Engineering rules

### 1. Preserve authorship (65/15/10/5/5)

Changing the data mix requires evidence that identity drift does not increase.

### 2. Centralize paths

`anra_paths.py` only. The linter test `test_path_registry_literals.py` will catch you.

### 3. Centralize registry truth

Derive lists from `component_registry()` — do not fork another component inventory.

### 4. Flags over comment-out

```python
from engine.feature_flags import is_enabled, set_flag
```

### 5. Trace main operations

```python
from engine.telemetry import trace

@trace("my_module", "main_operation")
def run(...): ...
```

One trace per subsystem entrypoint. Not every helper.

### 6. Prove improvements

```text
baseline → system_on → ablation → compare → save report
```

Claims without metrics are vibes.

### 7. Daily path stays light

```bash
python -m training.train_unified --mode session
```

No mandatory Ouroboros, no hidden checkpoint deps.

### 8. Milestones earn heaviness

```bash
python -m training.train_unified --mode train
```

Identity, Ouroboros, sovereignty, promotion — milestone territory.

### 9. Verification beats fluency

Tests, `verifier.py`, `benchmark.py`, symbolic bridge, schema checks, report diffs, telemetry, eval harness.

### 10. Keep `engine/` light at import

No `torch` / `faiss` / `transformers` at module import time in new `engine/` files.

---

## Commands you will run constantly

```bash
# Operator surface
python anra.py --report
python anra.py --status
python anra.py --phase3-status
python scripts/status.py

# Training
python -m training.train_unified --mode status
python -m training.train_unified --mode session
python -m training.train_unified --mode train
python -m training.train_unified --mode eval

# Verification
python -m pytest tests/ -q
python -c "from runtime.system_registry import component_registry; print(len(component_registry()))"
python -c "from engine.telemetry import get_telemetry_bus; print(get_telemetry_bus().summary_by_module())"

# Web
cd phase4/web && npm install && npm run dev
```

---

## Definition of done

- [ ] Behavior unchanged unless the task explicitly changed it
- [ ] Focused tests for new code
- [ ] Full suite green (or documented blocker)
- [ ] `python anra.py --report` still works for platform changes
- [ ] No stray checkpoint/DB churn in diff
- [ ] No new hardcoded paths
- [ ] No silent prompt/identity edits
- [ ] Feature can be disabled, traced, or evaluated

---

## Review checklist

| Question | Expected |
| --- | --- |
| Which component? | Name from `component_registry()` |
| Toggleable? | Flag or documented exception |
| Traced? | Telemetry or documented exception |
| Tested? | File + command |
| Regression? | Eval harness / benchmark / smoke |
| Operator surface changed? | README / CLI / report / none |
| What could still fail? | Honest residual risk |

Build like you will operate this repo every week — not admire it once.
