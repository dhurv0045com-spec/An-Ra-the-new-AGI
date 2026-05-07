# An-Ra Developer Guide

> Build aggressively, but make every subsystem measurable, swappable, traceable, and safe to operate.

This guide is for humans and AI coding agents working on An-Ra. The project should feel like a professional ML platform: clear registry, clear paths, clear flags, clear telemetry, clear reports, clear tests.

## First Rule

Do not re-implement what already exists.

Use the existing spine:

| Need | Source |
| --- | --- |
| Paths | `anra_paths.py` |
| Component truth | `runtime/system_registry.py` |
| Component contract | `engine/component_base.py` |
| Feature flags | `engine/feature_flags.py` |
| Telemetry | `engine/telemetry.py` |
| Ablation/regression | `engine/eval_harness.py` |
| System report | `engine/report.py` |
| Structured audit events | `shared_logger.py` |
| HAL telemetry | `runtime/hal_telemetry.py` |
| Startup contracts | `startup_checks.py` |

## Current Platform Contract

Every registered component should be:

- described in `component_registry()`
- enabled or disabled through feature flags
- traceable through telemetry when called
- visible in `python anra.py --report`
- testable through focused tests or smoke checks
- evaluated through a baseline/system-on/ablation path when behavior changes

Do not add a subsystem that cannot answer:

```text
What does it do?
Is it enabled?
How fast is it?
How often does it fail?
What test proves it?
What regression check protects it?
What owner-control boundary applies?
```

## Files To Read Before Editing

| Area | Files |
| --- | --- |
| Entrypoint | `anra.py` |
| Paths | `anra_paths.py` |
| Registry | `runtime/system_registry.py` |
| Reporting | `engine/report.py`, `scripts/status.py` |
| Feature flags | `engine/feature_flags.py`, `agents/orchestrator.py` |
| Telemetry | `engine/telemetry.py` and traced call sites |
| Model | `anra_brain.py`, `training/v2_config.py`, `training/v2_runtime.py` |
| Training | `training/train_unified.py`, `scripts/build_brain.py`, `training/finetune_anra.py` |
| Data mix | `training/v2_data_mix.py`, `scripts/setup_dataset.py` |
| Eval | `training/eval_v2.py`, `training/benchmark.py`, `training/verifier.py` |
| Memory | `memory/memory_router.py`, `phase2/memory (45J)/` |
| Agency | `goals/goal_queue.py`, `agents/orchestrator.py`, `phase2/agent_loop (45k)/` |
| Autonomy | `phase2/master_system (45M)/system.py` |
| Verification | `phase3/symbolic_bridge (45Q)/` |
| Governance | `self_modification/`, `phase3/sovereignty (45R)/` |
| Web UI | `phase4/web/src/App.jsx` |

## AI Collaboration Process

When using an AI coding agent on this repo, give it the goal and require this workflow:

1. Read the relevant source before editing.
2. Prefer thin adapters, decorators, tables, and wrappers over rewrites.
3. Keep model logic, prompts, identity text, and training data unchanged unless explicitly requested.
4. Add tests for every new file and every changed public behavior.
5. Run focused tests first, then `python -m pytest tests/ -x -q`.
6. Run `python anra.py --report` after platform or operator-surface changes.
7. Summarize changed files, verification, and any residual risk.

Good AI instruction:

```text
Implement this as a thin adapter on the existing registry/telemetry/flag system.
Do not rewrite model logic.
Add focused tests.
Run the full suite.
Report exact commands and outcomes.
```

Bad AI instruction:

```text
Make it more advanced.
Rewrite the architecture.
Add a new agent.
Improve intelligence without tests.
```

## Engineering Rules

### 1. Preserve Authorship

Owner data stays dominant:

| Bucket | Share |
| --- | ---: |
| Own conversation/instruction | 65% |
| Own identity/selfhood | 15% |
| Teacher reasoning | 10% |
| Symbolic/code-verified samples | 5% |
| Replayed failures/corrections | 5% |

Changing this mix requires evidence that identity drift does not increase.

### 2. Keep Paths Centralized

Use `anra_paths.py`. Do not scatter Drive, dataset, tokenizer, checkpoint, workspace, or phase-folder literals.

### 3. Keep Registry Truth Centralized

Use `runtime/system_registry.py` for component facts. Do not maintain separate component lists in scripts unless they are derived from the registry.

### 4. Use Feature Flags Instead Of Call-Site Surgery

Use:

```python
from engine.feature_flags import is_enabled, set_flag
```

Disable components through `state/feature_flags.json`, not by commenting out logic.

### 5. Trace Main Operations

Use:

```python
from engine.telemetry import trace
```

Decorate one main callable per subsystem. Do not add noisy traces to tiny helper functions unless debugging a specific issue.

### 6. Evaluate Before Claiming Improvement

Use `engine/eval_harness.py` when behavior can be compared:

```text
baseline -> system_on -> ablation -> compare -> save report
```

A patch that claims better reasoning, memory, speed, or reliability needs a metric.

### 7. Keep Daily Training Reliable

Daily path:

```bash
python -m training.train_unified --mode session
```

Do not make the daily path depend on heavyweight reflection, external services, or missing local checkpoints.

### 8. Keep Milestones Selective

Milestone path:

```bash
python -m training.train_unified --mode train
```

This path may include identity tuning, Ouroboros refinement, self-improvement reporting, sovereignty audit, and milestone tests.

### 9. Verification Beats Fluency

Use deterministic or repeatable checks wherever possible:

- unit tests
- `training/verifier.py`
- `training/benchmark.py`
- symbolic bridge checks
- schema validation
- report diffs
- telemetry summaries
- ablation/regression reports

### 10. No Heavy Imports In New Platform Modules

New `engine/` modules must not import `torch`, `faiss`, `transformers`, or other heavy ML libraries at module import time.

## Main Commands

Status and report:

```bash
python anra.py --report
python anra.py --status
python anra.py --phase3-status
python scripts/status.py
python -m inference.full_system_connector
```

Training:

```bash
python -m training.train_unified --mode status
python -m training.train_unified --mode session
python -m training.train_unified --mode train
python -m training.train_unified --mode eval
```

Verification:

```bash
python -m pytest tests/ -x -q
python -c "from engine.feature_flags import load_flags; print(load_flags())"
python -c "from engine.telemetry import get_telemetry_bus; print(get_telemetry_bus().summary_by_module())"
python -c "from runtime.system_registry import component_registry; print(len(component_registry()))"
```

Web:

```bash
cd phase4/web
npm install
npm run dev
```

## Definition Of Done

A good patch leaves behind:

- source behavior preserved unless the request explicitly changes it
- focused tests for new code
- full test suite passing or a clear reason it could not run
- `python anra.py --report` still working for platform changes
- no unrelated runtime DB/checkpoint churn in the final diff
- no hidden dependency on missing local checkpoints
- no new scattered hardcoded paths
- no silent identity or prompt changes
- no feature that cannot be disabled, traced, or evaluated

## Review Checklist

Before accepting a change, ask:

| Question | Expected Answer |
| --- | --- |
| Which component changed? | Name from `component_registry()` |
| Can it be toggled? | Feature flag or clear reason not applicable |
| Is it traced? | Telemetry record or clear reason not applicable |
| How is it tested? | Test file and command |
| How is regression detected? | Eval harness, benchmark, or explicit smoke comparison |
| What user/operator surface changed? | README, notebook, CLI help, report, or none |
| What could fail? | Known residual risk |

Build like the repo will be operated repeatedly, not admired once.
