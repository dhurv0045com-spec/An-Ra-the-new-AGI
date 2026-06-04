# 45M — Master System

**Component 13/19 · `master_system`**

Owner control + autonomy runtime. Goals, safety, personalization, LLM bridge, dashboard — one process that coordinates without swallowing model internals.

```text
anra.py
  → MasterSystem (this folder)
  → LLMBridge / generate
  → autonomy engine
  → goals + decisions
  → safety gate
  → dashboard / control API
```

---

## Layout

| Area | Files |
| --- | --- |
| Entry | `system.py`, `llm_bridge.py` |
| Autonomy | `autonomy/engine.py`, `goals.py`, `proactive.py`, `decisions.py` |
| Control | `control/control.py` |
| Safety | `safety/safety.py` |
| Personalization | `personalization/models.py` |
| Memory | `memory/vector_memory.py` |
| Scale | `scale/pipeline.py` |
| Tools | `tools/tool_integrations.py` |

---

## Commands

**Repo root (preferred):**

```bash
python anra.py --status
python anra.py --briefing
python anra.py --goal "Research transformer scaling tradeoffs"
python anra.py --dashboard
python anra.py --test
```

**This folder:**

```bash
python system.py --status
python system.py --briefing
python system.py --test
```

---

## Autonomy tiers

| Tier | Behavior | Examples |
| --- | --- | --- |
| 1 | Auto-run | Read files, calc, local inspect |
| 2 | Notify + proceed | Routine writes, goals, training tasks |
| 3 | Wait for approval | Installs, deletes, deploys, external writes |
| 4 | Instruction only | Financial, credentials, irreversible |

---

## State

SQLite under `state/` for decisions, goals, safety, personalization, tools — **runtime artifacts**, not source of truth. Git should not track your local DBs.

---

## Boundary (keep 45M thin)

| Belongs elsewhere | Belongs here |
| --- | --- |
| Tokenizer paths, training loops | Coordination, owner policy, session control |
| Model architecture | Safety + autonomy decisions |
| Raw benchmark math | When to run what, at what tier |

45M coordinates. It does not become a junk drawer for one-off scripts.
