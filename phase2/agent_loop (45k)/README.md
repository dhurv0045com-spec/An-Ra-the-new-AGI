# 45K — Agent Loop

**Component 12/19 · `agent_loop`**

Plain-language goal in → interpreted objective → plan → tool execution → monitor → evaluate → outcome on disk.

This is the **action layer** between goals/memory and the master system. If the goal queue says *what*, 45K says *how*.

---

## Flow

```text
goal text
  → GoalInterpreter (safety / risk)
  → Planner (step graph)
  → Executor
  → Dispatcher → tools
  → Monitor (stall / timeout / caps)
  → Evaluator
  → memory + telemetry
```

---

## Files worth opening first

| File | Role |
| --- | --- |
| `agent_main.py` | CLI + `Agent` facade |
| `goal.py` | Parse goals, criteria, constraints |
| `planner.py` | Dependencies and step graph |
| `executor.py` | Run steps, retry, verify |
| `dispatcher.py` | NL step → tool routing |
| `monitor.py` | Loop health |
| `evaluator.py` | Score outcome, improvement notes |
| `registry.py` / `builtin.py` | Tool catalog |

---

## Quick start

**From this folder:**

```bash
python agent_main.py --goal "Compound interest on 10000 at 7% for 20 years"
python agent_main.py --review-last
python agent_main.py --list-tools
python test_45K.py
```

**From repo root (integrated path):**

```bash
python anra.py --goal "Compare vector memory approaches in 3 bullets"
```

---

## Built-in tools

| Tool | Does | Risk |
| --- | --- | --- |
| `calculator` | Math | Low |
| `file_manager` | Workspace files | Medium |
| `code_executor` | Sandboxed Python | Medium |
| `web_search` | External research (if enabled) | Medium |
| `memory_tool` | Step facts | Low |
| `summarizer` | Compress long text | Low |
| `task_manager` | Plan state | Low |

---

## Where 45K fits in the stack

| Use 45K directly | Use master system / `anra.py` |
| --- | --- |
| Isolated agent experiments | Goals + memory + safety + owner control |
| Tool routing debugging | Production-style sessions |

**Integration spine:** `goals/goal_queue.py` → `agents/orchestrator.py` → `phase2/master_system (45M)/system.py`

Flags: orchestrator skips disabled components — test with `set_flag("agent_loop", False)` from root.
