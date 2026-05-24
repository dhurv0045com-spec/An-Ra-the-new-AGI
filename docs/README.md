# An-Ra Documentation Hub

All project documentation is organized here. **Root-level files** stay as quick entry points; **detailed tracking** lives under `docs/`.

---

## Folder map

```text
docs/
├── README.md                          ← you are here
├── engineering/
│   ├── ENGINEERING_LOG.md             ← dated add/change/remove log (spine tracker)
│   └── LOG_STANDARD.md                ← required format for humans & AI
└── planning/
    └── MASTER_GOALS.md                ← every goal: research, test, ship, robotics…

Repository root (entry points — do not duplicate content):
├── README.md          → start + command cheat sheet
├── OPERATOR.md        → Jarvis / desktop work mode
├── DEVELOPER.md       → how to change code safely
├── ARCHITECTURE.md    → 19-component wiring
├── VISION.md          → why An-Ra exists
└── WALKTHROUGH.md     → deep technical tour

Subsystem READMEs (stay next to code):
├── phase2/agent_loop (45k)/README.md
├── phase2/memory (45J)/README.md
├── phase2/master_system (45M)/README.md
├── phase3/PHASE3_INTEGRATION.md
├── phase3/*/README.md
└── phase4/web/README.md
```

---

## Which file when?

| You need… | Open |
|-----------|------|
| Run the system | [`../README.md`](../README.md) |
| Log a code change | [`engineering/ENGINEERING_LOG.md`](engineering/ENGINEERING_LOG.md) + [`LOG_STANDARD.md`](engineering/LOG_STANDARD.md) |
| See what’s left to build | [`planning/MASTER_GOALS.md`](planning/MASTER_GOALS.md) |
| Do files / CAD / goals | [`../OPERATOR.md`](../OPERATOR.md) |
| Contribute a patch | [`../DEVELOPER.md`](../DEVELOPER.md) |
| Understand wiring | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) |

---

## Logging changes (mandatory for spine edits)

After any add, change, remove, or improvement that touches a registered component:

```bash
python scripts/log_engineering_change.py \
  --component agent_loop \
  --type ADD \
  --summary "os_action and cad_generate tools" \
  --files "phase2/agent_loop (45k)/builtin.py" \
  --verify "pytest tests/test_operator_tools.py"
```

Or append manually using [`engineering/LOG_STANDARD.md`](engineering/LOG_STANDARD.md).

---

## Goals tracking

[`planning/MASTER_GOALS.md`](planning/MASTER_GOALS.md) is the single backlog for research, testing, training, operator features, robotics, and governance. Update status when work completes.

---

## Engineering spine (code)

| Module | Role |
|--------|------|
| `runtime/system_registry.py` | What exists |
| `engine/feature_flags.py` | On/off |
| `engine/telemetry.py` | Speed / failures |
| `engine/eval_harness.py` | Regression |
| `engine/report.py` | Scorecard |
| `docs/engineering/ENGINEERING_LOG.md` | **Who changed what, when** |

The log is the narrative spine; telemetry is the runtime spine.
