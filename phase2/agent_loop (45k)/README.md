# 45K - Agent Loop

**Layer 12/19: `agent_loop`**

45K turns a plain-language goal into an interpreted objective, a plan, tool-routed execution, monitoring, recovery, and evaluation.

It is the action layer between memory/goals and the larger master system.

## Goal Flow

```text
goal text
  -> GoalInterpreter
  -> safety/risk classification
  -> Planner
  -> Executor
  -> Dispatcher/tools
  -> Monitor
  -> Evaluator
  -> memory and outcome record
```

## Main Files

| File | Purpose |
| --- | --- |
| `agent_main.py` | CLI and `Agent` facade |
| `goal.py` | Goal parsing, criteria, constraints, risk |
| `planner.py` | Step graph and dependency planning |
| `executor.py` | Step execution, retry, verification |
| `dispatcher.py` | Natural-language step to tool routing |
| `monitor.py` | Loop, stall, timeout, and call-cap detection |
| `reasoning.py` | Reasoning traces for planning decisions |
| `evaluator.py` | Outcome review and improvement notes |
| `coordinator.py` | Parallel sub-agent coordination |
| `registry.py`, `builtin.py` | Tool registry and built-in tools |

## Quick Start

From this folder:

```bash
python agent_main.py --goal "Calculate compound interest on 10000 at 7% for 20 years"
python agent_main.py --review-last
python agent_main.py --list-tools
python test_45K.py
```

From the repo root, use the unified entrypoint:

```bash
python anra.py --goal "Write a short comparison of vector memory approaches"
```

## Built-In Tool Classes

| Tool | Role | Risk |
| --- | --- | --- |
| `calculator` | Math expressions | Low |
| `file_manager` | Workspace file actions | Medium |
| `code_executor` | Sandboxed Python execution | Medium |
| `web_search` | External research when enabled | Medium |
| `memory_tool` | Store and recall step facts | Low |
| `summarizer` | Long-text compression | Low |
| `task_manager` | Track plan state | Low |

## Current Boundary

45K is source-active and useful, but the strongest integration path is through:

- `goals/goal_queue.py`
- `agents/orchestrator.py`
- `phase2/master_system (45M)/system.py`
- `anra.py --goal "..."`

Use 45K directly for isolated agent tests. Use the master system when the goal should touch memory, autonomy, safety, and owner-control layers.
