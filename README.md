# AN-RA

> A sovereign, owner-shaped AI platform built around measurable learning, memory, verification, autonomy, and controlled self-improvement.

An-Ra is not a generic chatbot wrapper. It is a from-scratch AI system and operating stack: model core, tokenizer, training loop, memory, agent dispatch, identity guardrails, symbolic verification, ghost recall, self-improvement, self-modification gates, sovereignty audits, and now a measurable engineering spine.

The rule of the project is simple:

```text
No magic subsystem.
Every component must be registered, switchable, measurable, reportable, and testable.
```

## Current Shape

The live component source of truth is `runtime/system_registry.py`.

An-Ra currently has 19 registered components:

| # | Component | Role |
| --- | --- | --- |
| 01 | `brain` | V2 transformer core: GQA, RoPE/YaRN, MoD, Flash SDP path, tied embeddings |
| 02 | `tokenizer` | Canonical 8192-token BPE tokenizer and adapter |
| 03 | `data_mix` | Owner-data-first corpus contract and dataset setup |
| 04 | `training_loop` | Daily and milestone training orchestration |
| 05 | `evaluation` | Compact eval, benchmark, verifier, and hard-example feedback |
| 06 | `runtime` | Generation, streaming, connector refresh, local inference helpers |
| 07 | `api_web` | FastAPI backend and Phase 4 web operator interface |
| 08 | `identity` | CIV/ESV guardrails, watcher checks, identity injection |
| 09 | `memory` | Unified memory router over episodic, short-term, graph, ghost, and ESV-gated writes |
| 10 | `phase2_memory` | Typed memory, retrieval, vector index, context builder, personal graph |
| 11 | `goals` | Persistent goal queue, retries, successors, orchestrator dispatch |
| 12 | `agent_loop` | Planning, execution, monitoring, evaluation |
| 13 | `master_system` | Owner-control system, persistent service, long-horizon goals, safety |
| 14 | `self_improvement` | Improvement engine, prompt/skill refinement, session learning hooks |
| 15 | `self_modification` | Type-A/Type-B patch gates, sandbox execution, filesystem actions |
| 16 | `ouroboros` | Recursive reasoning, adaptive pass selection, milestone refinement |
| 17 | `ghost_memory` | Compressed conversation memory, decay, retrieval, context injection |
| 18 | `symbolic_bridge` | Deterministic math, logic, code analysis, verified responses |
| 19 | `sovereignty` | Audit, dead-code sweep, benchmark deltas, reports, promotion gates |

Every component now carries:

- `enabled`: runtime toggle state
- `metric_hooks`: required measurement hooks such as latency, success, and error type
- source status from the registry
- telemetry summaries when it has been exercised

## Engineering Spine

The new platform layer lives in `engine/`.

| File | Purpose |
| --- | --- |
| `engine/component_base.py` | Lightweight component protocol and metrics bookkeeping |
| `engine/feature_flags.py` | Config-driven component toggles in `state/feature_flags.json` |
| `engine/telemetry.py` | Unified JSONL tracing: latency, success, errors, output size, tokens |
| `engine/eval_harness.py` | Baseline, system-on, ablation, and regression comparison harness |
| `engine/report.py` | One-command health/performance snapshot |

This means the important questions have direct answers:

| Question | Command or API |
| --- | --- |
| What exists? | `component_registry()` |
| Is it enabled? | `is_enabled("component")` |
| How fast is it? | `get_telemetry_bus().summary_by_module()` |
| Did it regress? | `EvalHarness().compare(baseline, current)` |
| What broke recently? | `python anra.py --report` |
| Can it be disabled? | `set_flag("component", False)` |

## Operator Quickstart

Create an environment, install dependencies, then run the report:

```bash
python -m pip install -r requirements.txt
python anra.py --report
```

Useful commands:

```bash
python anra.py --status
python anra.py --phase3-status
python anra.py --symbolic "solve x^2 - 9 = 0"
python anra.py --goal "summarize the current system report"
python -m training.train_unified --mode status
python -m training.train_unified --mode session
python -m training.train_unified --mode train
python -m training.train_unified --mode eval
python -m pytest tests/ -x -q
```

On CPU-only machines, `anra.py` may warn that Flash SDP/CUDA is unavailable. That is expected for local smoke checks; serious training should run on a CUDA-capable GPU environment.

## Colab Path

Use `AnRa_Master.ipynb` for the Google Colab operator workflow.

The notebook is now structured as a full An-Ra console:

1. configure session, repo, Drive, and component flags
2. mount Drive and inspect GPU/RAM
3. clone or update the repo
4. select and merge owner training data
5. restore checkpoints and run preflight checks
6. apply feature flags and print the system report
7. run training, evaluation, or smoke mode
8. inspect telemetry and scorecard metrics
9. sync reports/checkpoints back to Drive
10. launch UI or API surfaces

The notebook is for operating the platform, not editing its source. Source changes should happen in git, then Colab pulls the updated repo.

## How To Use An-Ra As An AI System

Use An-Ra in three modes:

### 1. Local Inspection

For status, routing, flags, telemetry, reports, symbolic checks, and fast tests:

```bash
python anra.py --report
python anra.py --status
python anra.py --symbolic "factor 360"
python -c "from engine.feature_flags import disabled_components; print(disabled_components())"
```

### 2. Daily Learning

For regular improvement without heavy milestone passes:

```bash
python -m training.train_unified --mode session
```

Daily sessions should keep the loop clean:

```text
restore -> validate -> train -> evaluate -> write reports -> keep failures for replay
```

### 3. Milestone Improvement

For deeper training and governance:

```bash
python -m training.train_unified --mode train
```

Milestones are allowed to be heavier because they can involve identity reinforcement, Ouroboros refinement, self-improvement analysis, sovereignty audit, and promotion gates.

## Feature Flags

Disable a component without editing call sites:

```bash
python - <<'PY'
from engine.feature_flags import set_flag, disabled_components
set_flag("ghost_memory", False)
print(disabled_components())
PY
```

The orchestrator maps task kinds to components and skips disabled components automatically:

| Task kind | Component |
| --- | --- |
| `coder` | `agent_loop` |
| `research` | `agent_loop` |
| `memory` | `memory` |
| `critic` | `evaluation` |
| `symbolic` | `symbolic_bridge` |
| `ghost` | `ghost_memory` |

## Telemetry And Scorecard

Telemetry is written to:

```text
state/logs/telemetry.jsonl
```

Each traced call records:

- module
- operation
- start/end time
- elapsed milliseconds
- success/failure
- error type/message
- token count when returned by the function
- output size/type
- confidence when present

The system report turns this into an operator scorecard:

```bash
python anra.py --report
```

Scorecard axes:

| Axis | Meaning |
| --- | --- |
| Source health | Registered files exist and required components are present |
| Import health | Importable components do not report degraded status |
| Enablement | Feature flags show which components are active |
| Latency | Average traced runtime per module |
| Reliability | Success rate and recent failures |
| Regression | Eval harness comparison between baseline and current behavior |
| Readiness | Training readiness checks and artifact state |

## Data Contract

The default training mix remains owner-centered:

| Bucket | Share | Purpose |
| --- | ---: | --- |
| Own conversation and instruction | 65% | Voice, problem style, normal behavior |
| Own identity and selfhood | 15% | Identity gravity and drift resistance |
| Teacher reasoning | 10% | Harder reasoning traces |
| Symbolic or code-verified samples | 5% | Truth-checked supervision |
| Replayed failures and corrections | 5% | Repair from real mistakes |

Teacher data is an amplifier, not the owner. An-Ra should get sharper without becoming generic.

## Artifact Layout

Canonical dataset:

```text
training_data/anra_training.txt
```

Tokenizer:

```text
tokenizer/tokenizer_v3.json
```

Main checkpoint names:

```text
anra_v2_brain.pt
anra_v2_identity.pt
anra_v2_ouroboros.pt
```

Key runtime/report paths:

```text
state/feature_flags.json
state/logs/telemetry.jsonl
output/v2/reports/
output/v2/eval/
```

Fresh clones may not include local trained checkpoints. Restore them from Drive or train them.

## Development Contract

Do not add impressive-sounding modules unless they become measurable.

Every serious change should answer:

1. What component does this improve?
2. What metric should move?
3. What verifier or test proves it?
4. What failure case becomes replay or documentation?
5. What owner-control boundary remains intact?

For implementation rules, read `DEVELOPER.md`. For the long-horizon strategy, read `VISION.md`. For the live architecture map, read `ARCHITECTURE.md`.
