# An-Ra Operator Manual

> **Do work, not just talk.** Files, opens, CAD stubs, goals — measured, sandboxed, auditable.

This is the Jarvis-shaped layer: you give imperatives; An-Ra plans, calls tools, leaves artifacts on disk.

---

## Mental model

```text
You (owner)
  → chat / CLI / goal
  → Master system (45M)
  → Agent loop (45K): plan → tools → verify
  → workspace/ artifacts + audit log
```

**Chat alone** = language model conversation.  
**`/goal` or `goal:`** = full agent with tools.  
**Slash commands** = fast direct tools without full planning.

---

## Workspace (sandbox)

| Setting | Default |
|---------|---------|
| Folder | `<repo>/workspace/` |
| Override | `AGENT_FILE_ROOT` or `ANRA_AGENT_WORKSPACE` |
| Code | `anra_paths.get_agent_workspace()` |

```powershell
cd c:\Users\user\Downloads\An-Ra
$env:AGENT_FILE_ROOT = "$PWD\workspace"
python anra.py --chat
```

Everything `file_manager` touches stays inside this root unless you extend opens (below).

---

## Interactive chat commands

Start:

```bash
python anra.py --chat
```

| Command | Action |
|---------|--------|
| `/help` | Command list |
| `/workspace` | Print sandbox path |
| `/goal <text>` | Full agent loop (files, CAD, web, code…) |
| `/write <path> <content>` | Create/overwrite file |
| `/read <path>` | Read file |
| `/list [path]` | Directory listing |
| `/open <path>` | OS default app / Explorer |
| `/cad [template]` | Engineering stub (`raptor_engine` default) |
| `goal: ...` | Legacy goal prefix (same as `/goal`) |

**Audit trail:** `state/logs/operator_actions.jsonl`

---

## CLI goals (non-chat)

```bash
python anra.py --goal "Create workspace/daily.md with three priorities for training"
python anra.py --goal "Run cad_generate raptor_engine and summarize assumptions in REPORT.md"
python anra.py --goal "List all files in workspace and write inventory.txt"
```

---

## Tools (agent loop 45K)

| Tool | Risk | What it does |
|------|------|----------------|
| `file_manager` | restricted | read, write, list, search, delete in sandbox |
| `os_action` | dangerous | open / reveal / browser URL |
| `cad_generate` | restricted | OpenSCAD package + REPORT |
| `code_executor` | restricted | sandboxed Python |
| `web_search` | restricted | DuckDuckGo instant answers |
| `calculator` | safe | math |
| `memory_tool` | safe | session facts |

Planner + dispatcher pick tools from natural language in `/goal` runs.

---

## Engineering: 3D raptor engine (and beyond)

### Quick

```bash
python anra.py --chat
/cad raptor_engine
/open engineering/raptor_engine/raptor_engine.scad
```

### What you get

```text
workspace/engineering/raptor_engine/
  raptor_engine.scad    # parametric OpenSCAD cutaway (stylized)
  REPORT.md             # assumptions + falsifiers + next steps
  raptor_engine.stl     # only if openscad is on PATH
```

**Critical:** This is a **diagram scaffold**, not Pratt & Whitney OEM data. You must replace dimensions from authoritative sources. Use:

```bash
python anra.py --symbolic "verify ..."
```

for numeric claims — not model prose.

### Custom templates

Add `runtime/engineering_templates/<name>.scad`, then:

```text
/cad my_template
```

---

## Opening files outside sandbox (optional)

By default `os_action` only opens paths under the agent workspace.

To allow extra roots (e.g. Downloads):

```powershell
$env:ANRA_ALLOWED_OPEN_ROOTS = "C:\Users\you\Downloads"
```

Comma-separated list. Still blocks arbitrary system paths not listed.

---

## Autonomy tiers (45M) — what “do anything” means here

| Tier | Examples |
|------|----------|
| 1 | read, list, calc, safe writes in workspace |
| 2 | goals, training triggers, routine writes |
| 3 | delete, open outside sandbox, installs — **approve** |
| 4 | credentials, money, irreversible — **owner only** |

True superintelligence in this project = **tiered execution + verification**, not silent god-mode.

---

## Roadmap: what is NOT automatic yet

| You asked for | Requirement |
|---------------|-------------|
| Check friend’s laptop | Paired **An-Ra node** on their machine + consent |
| Robot arm / drone | **ROS2 bridge** + simulator + hardware estop |
| Perfect OEM engine CAD | **Your sourced dimensions** + professional CAD workflow |

Local operator pack (this doc) is step one.

---

## Daily operator checklist

1. `python anra.py --report` — 19/19 healthy?  
2. `python anra.py --chat` or `--goal` — one real task with a file output  
3. Check `workspace/` for artifacts  
4. Check `state/logs/operator_actions.jsonl` if something surprising happened  
5. Training days: `python -m training.train_unified --mode session`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| “Path escapes workspace” | Use paths relative to `/workspace` |
| Open does nothing | Path must exist; on Windows use forward slashes ok |
| No .stl | Install [OpenSCAD](https://openscad.org/) and ensure `openscad` on PATH |
| Agent ignores file request | Use `/goal` not plain chat |
| Unicode console error on symbolic | Fixed in `anra.py` `_safe_console` |

---

## Related docs

- [`README.md`](README.md) — project entry  
- [`WALKTHROUGH.md`](WALKTHROUGH.md) §19 — operator addendum  
- [`phase2/agent_loop (45k)/README.md`](phase2/agent_loop%20(45k)/README.md) — agent internals  
- [`VISION.md`](VISION.md) — why verification and tiers matter
