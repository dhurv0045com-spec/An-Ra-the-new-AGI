# Engineering Log — Entry Standard

Every human or AI change that affects a **registered component**, the **engineering spine**, or **operator behavior** must be recorded in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md).

This keeps An-Ra auditable: the repo should answer *what changed, when, why, and how we know it still works*.

---

## When to log

| Log? | Example |
|------|---------|
| **Yes** | New tool, registry entry, training path, flag, telemetry hook, test suite, operator command |
| **Yes** | Bug fix in public behavior of a component |
| **Yes** | Remove or disable a subsystem |
| **Yes** | Docs that change operator or developer contract (`OPERATOR.md`, `DEVELOPER.md`) |
| **No** | Typo in comment with zero behavior change |
| **No** | Local-only experiment not merged (log only when merged) |

---

## Entry types

| Type | Use |
|------|-----|
| `ADD` | New file, component, tool, test, doc section with new contract |
| `CHANGE` | Behavior or API change in existing code |
| `REMOVE` | Deleted or permanently disabled capability |
| `FIX` | Bug fix, no intended feature change |
| `DOCS` | Documentation-only (still log if contract changes) |
| `EVAL` | Baseline / milestone eval or promotion decision |

---

## Required fields

Copy this block for each entry (newest entries at **top** of `ENGINEERING_LOG.md`):

```markdown
## YYYY-MM-DD — TYPE — `component_id` — Short title

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD (ISO) |
| **Author** | `owner` \| `human:<name>` \| `ai-agent` |
| **Component** | registry id, e.g. `agent_loop`, `engine`, `operator` |
| **Type** | ADD \| CHANGE \| REMOVE \| FIX \| DOCS \| EVAL |
| **Summary** | One sentence: what and why |
| **Files** | Comma-separated paths touched |
| **Metrics** | What should move (latency, pass rate, eval score, …) or `n/a` |
| **Verification** | Commands run, e.g. `pytest tests/ -q`, `scripts/anra.py --report` |
| **Risk** | `low` \| `medium` \| `high` |
| **Follow-up** | Next step or `none` |

### Detail (optional)
- Bullet context for future readers.
```

---

## Component IDs

Use names from `runtime/system_registry.py` when possible:

`brain`, `tokenizer`, `data_mix`, `training_loop`, `evaluation`, `runtime`, `api_web`, `identity`, `memory`, `phase2_memory`, `goals`, `agent_loop`, `master_system`, `self_improvement`, `self_modification`, `ouroboros`, `ghost_memory`, `symbolic_bridge`, `sovereignty`

Cross-cutting spine work may use: `engine`, `operator`, `docs`, `tests`.

---

## CLI helper

```bash
python scripts/log_engineering_change.py --help
```

Prefer the script for consistent formatting. Review the diff in `ENGINEERING_LOG.md` before commit.

---

## AI agent rule

After completing a task, if the definition of done in `DEVELOPER.md` applies, **append one log entry** unless the user explicitly said not to. Include exact verification commands and outcomes.
