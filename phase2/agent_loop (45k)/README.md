# 45K — Agent Loop

A complete autonomous agent system built on top of the Phase 1 transformer and Phase 2 memory/alignment systems. Give it a goal in plain English and walk away. It plans, executes, recovers from failures, and reports back.

---

## What It Does

```
you: "Research the best GPU under $500 and write a comparison report"

agent:
  1. Interprets and validates the goal
  2. Checks constitutional safety rules
  3. Plans: 5 steps with dependency graph
  4. Executes: searches web, compares specs, runs calculations, writes file
  5. Recovers from any failures with automatic retry + replanning
  6. Delivers: report.md in agent_workspace/
  7. Evaluates: what worked, what failed, how to improve next time
```

---

## Quick Start

```bash
# Install
pip install pyyaml

# Give the agent a goal
python agent_main.py --goal "Calculate compound interest on $10,000 at 7% for 20 years"

# Research goal with file output
python agent_main.py \
  --goal "Research the best Python libraries for machine learning in 2024 and write a report" \
  --output-file report.md

# Review what just happened
python agent_main.py --review-last

# See all available tools
python agent_main.py --list-tools

# Require approval before each step (for high-stakes goals)
python agent_main.py --goal "..." --approve-each-step

# Run in continuous mode (interactive REPL)
python agent_main.py --continuous

# Run independent steps in parallel
python agent_main.py --goal "..." --parallel
```

---

## Architecture

```
agent_main.py          ← CLI entry point
│
├── agent/
│   ├── core/
│   │   ├── goal.py        ← Parse + validate natural language goals
│   │   ├── planner.py     ← Break goals into dependency graphs of steps
│   │   ├── executor.py    ← Run steps with retry, verification, escalation
│   │   ├── dispatcher.py  ← Route natural language → best tool
│   │   └── monitor.py     ← Detect loops, timeouts, drift → escalate
│   │
│   └── intelligence/
│       ├── reasoning.py   ← Chain-of-thought before decisions
│       ├── evaluator.py   ← Learn from outcomes across sessions
│       └── coordinator.py ← Spawn + coordinate parallel sub-agents
│
└── tools/
    ├── registry.py        ← Tool registration, safety, rate limiting
    └── builtin.py         ← 7 built-in tools (see below)
```

---

## Tools

| Tool | Description | Safety |
|------|-------------|--------|
| `web_search` | DuckDuckGo search, structured results | Restricted |
| `code_executor` | Run Python in a sandbox, capture output | Restricted |
| `file_manager` | Read/write/search files in workspace | Restricted |
| `calculator` | Safe math expressions, trig, log, sqrt | Safe |
| `memory_tool` | Store/recall/search facts across steps | Safe |
| `summarizer` | Extractive summarization of long text | Safe |
| `task_manager` | Create/track tasks, persists across sessions | Safe |

**Adding a new tool in under 10 minutes:**

```python
from tools.registry import ToolDefinition, SafetyLevel, ToolResult, get_registry

def my_tool(input_text: str, **kwargs) -> ToolResult:
    # do something
    return ToolResult(success=True, output="result here")

get_registry().register(ToolDefinition(
    name="my_tool",
    description="What it does",
    fn=my_tool,
    parameters={"input_text": "what to pass"},
    safety_level=SafetyLevel.SAFE,
    examples=["my_tool example input"],
))
```

---

## Python API

```python
from agent_main import Agent

agent = Agent(
    approve_each_step=False,   # set True to review every step
    verbose=True,
    log_level="INFO",
)

# Single goal
result = agent.run("Research the 3 best vector databases for production use")
print(result["output"])
print(result["evaluation"])   # what worked, what didn't

# Multiple goals in parallel
results = agent.run_parallel_goals([
    "Research GPU A: RTX 4090 specs",
    "Research GPU B: RTX 3090 specs",
    "Research GPU C: A100 specs",
])

# Inspect performance over time
print(agent.performance_profile())
```

---

## Goal Lifecycle

```
raw text
    ↓
GoalInterpreter
    ├── Constitutional check (unsafe → REJECTED immediately)
    ├── Extract: objective, criteria, constraints, deadline
    ├── Infer: required tools, risk level
    └── Clarify: flag ambiguities, prompt user if needed
    ↓
GoalSpec (APPROVED)
    ↓
Planner
    ├── Classify goal type: research / coding / analysis / generic
    ├── Select template + customize steps
    ├── Build dependency graph
    └── Assign tools, timeouts, retry budgets
    ↓
ExecutionPlan
    ↓
Executor
    ├── Walk dependency graph
    ├── Run each step when deps are met
    ├── Verify output after each step
    ├── On failure: replan → retry → escalate
    └── Bridge results into memory for downstream steps
    ↓
GoalEvaluator
    ├── What worked / what failed
    ├── Improvement suggestions
    └── Persist to .agent_evals.json
```

---

## Safety

Goals are checked against constitutional rules before any planning begins:

- **UNSAFE** (rejected): anything involving weapons, violence, illegal hacking, CSAM, self-harm
- **HIGH risk** (requires approval): irreversible actions, external writes, impersonation
- **MEDIUM risk** (flagged): involves external systems — proceeds with logging
- **LOW risk** (auto-approved): computation, file operations, memory access

Dangerous tools (e.g. file delete) require an explicit approval callback. Rate limits prevent runaway tool calls. The monitor detects loops and stalls and escalates to the user.

---

## Monitoring

The `AgentMonitor` runs alongside execution and detects:
- **Loop detection**: same action fingerprint repeated N times
- **Progress stall**: no step completed in the last N seconds (default: 300s)
- **Tool call cap**: max 200 tool calls per plan (configurable)
- **Periodic check-ins**: logged every 30 seconds

---

## Testing

```bash
python test_45K.py           # 65 tests, all components
pytest test_45K.py -v        # with pytest
```

Tests cover: tool registry, all 7 built-in tools, goal interpreter, planner, executor, dispatcher, monitor, reasoning engine, evaluator, full agent loop (including safety rejection and memory goals).

---

## File Structure

```
45K/
├── agent_main.py              ← CLI + Agent class
├── test_45K.py                ← Full test suite (65 tests)
├── README.md                  ← This file
│
├── agent/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── goal.py            ← GoalInterpreter, GoalSpec
│   │   ├── planner.py         ← Planner, ExecutionPlan, Step
│   │   ├── executor.py        ← Executor, StepOutcome
│   │   ├── dispatcher.py      ← Dispatcher
│   │   └── monitor.py         ← AgentMonitor
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── reasoning.py       ← ReasoningEngine, ReasoningChain
│   │   ├── evaluator.py       ← GoalEvaluator, GoalEvaluation
│   │   └── coordinator.py     ← MultiAgentCoordinator, SubAgent
│   └── memory/
│       └── __init__.py
│
└── tools/
    ├── __init__.py
    ├── registry.py            ← ToolRegistry, ToolDefinition, ToolResult
    └── builtin.py             ← All 7 built-in tools + register_all_tools()
```

---

## What Comes Next (45L Recommendations)

1. **Real LLM integration** — replace heuristic planning/reasoning with actual model calls so the agent can handle goals that don't fit templates
2. **Tool expansion** — email, calendar, GitHub API, SQL databases, REST API caller
3. **Self-improvement loop** — use evaluation data to fine-tune the planner on successful goal patterns
4. **Persistent goal queue** — multi-session goals that survive restarts, scheduled goals
5. **Streaming output** — stream step results to user in real-time rather than batching

---

## Known Limitations

- Web search requires network access (degrades gracefully without it)
- Code executor is sandboxed but not a full container — don't run untrusted goals
- Planning is heuristic-template-based — complex novel goals may need the LLM planner (45L)
- No real LLM calls yet — responses are from tools, not model generation
- Memory is in-process (memory_tool) — a full 45J integration would use vector search
