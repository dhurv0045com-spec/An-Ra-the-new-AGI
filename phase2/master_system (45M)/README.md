# 45M - Master System

**Layer 13/19: `master_system`**

45M is the owner-control and autonomy layer. It ties goals, memory, safety, personalization, control APIs, and the LLM bridge into one system-level runtime.

## Current Role

```text
anra.py
  -> 45M MasterSystem
  -> LLMBridge / generate path
  -> autonomy engine
  -> goal and decision stores
  -> owner model
  -> safety gate
  -> dashboard / JSON API
```

## Main Files

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

## Main Commands

From the repo root:

```bash
python anra.py --status
python anra.py --briefing
python anra.py --goal "Research transformer scaling techniques"
python anra.py --dashboard
python anra.py --test
```

From this folder:

```bash
python system.py --status
python system.py --test
python system.py --briefing
```

## Autonomy Tiers

| Tier | Meaning | Examples |
| --- | --- | --- |
| 1 | Auto-run | Reads, calculations, safe local inspection |
| 2 | Notify and proceed | Routine writes, goal updates, training tasks |
| 3 | Wait for approval | Installs, deletes, deploys, external writes |
| 4 | Direct instruction only | Financial, credential, irreversible actions |

## State

45M keeps local state in SQLite databases for decisions, goals, safety, scale, proactive behavior, personalization, tool registry, and self-improvement. These are runtime artifacts, not source truth.

## Boundary

45M should coordinate. It should not become a dumping ground for model internals, tokenizer paths, or one-off training logic. Keep model behavior in the model/runtime layers, keep training in `training/`, and keep owner-control decisions here.
