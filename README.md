# AN-RA

> **Sovereign, owner-shaped intelligence** — it learns in your terms, verifies where truth is checkable, remembers failure, and can **do work** on your machine under measurement and gates.

You are not looking at a ChatGPT skin. This repository is a full stack:

- Custom **transformer brain** (VQA, RoPE/YaRN, MoD)
- **8192-token** owner-trained tokenizer
- **Training loop** with owner-first data law (65/15/10/5/5)
- **Memory**, **goals**, **agent loop**, **identity** (CIV / ESV / HAL)
- **Symbolic verification**, ghost recall, Ouroboros, sovereignty audits
- **Operator layer** — files, open, CAD stubs, slash commands ([`OPERATOR.md`](OPERATOR.md))

The constitution of the codebase:

```text
No magic subsystem.
Every component must be registered, switchable, measurable, reportable, and testable.
```

**First command after clone:**

```bash
# Install the package and all dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Verify structure
python scripts/verify_structure.py

# Run tests
python -m pytest tests/ -m "not gpu" -q
```

Green `19/19` = the organism’s organs are present. Trained weights may still be missing — see [Artifacts](#artifacts).

---

## Who this is for

| You are… | Start here |
|----------|------------|
| Owner / operator | This file → [`OPERATOR.md`](OPERATOR.md) → `python anra.py --chat` |
| Developer / agent | [`DEVELOPER.md`](DEVELOPER.md) → [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Deep learner | [`WALKTHROUGH.md`](WALKTHROUGH.md) (full tour; §19 = operator addendum) |
| Strategist | [`VISION.md`](VISION.md) |

---

## Two modes that matter

### Talk mode

```bash
python anra.py --chat
```

Conversation with memory and identity. Good for thinking aloud.

### Work mode

```bash
python anra.py --goal "Write workspace/status.md summarizing the last report"
```

Or inside chat:

```text
/goal Build a raptor engine CAD stub and document assumptions
/write workspace/todo.txt train session tonight
/cad raptor_engine
/open engineering/raptor_engine/raptor_engine.scad
```

**Rule:** If you want files opened or created, use **`/goal`**, **`goal:`**, or slash commands — not plain chat.

Full operator reference: **[`OPERATOR.md`](OPERATOR.md)**

---

## The 19 components

Source of truth: `runtime/system_registry.py` — never duplicate this table elsewhere by hand.

| # | ID | One-line job |
|---|-----|----------------|
| 01 | `brain` | V2 causal transformer |
| 02 | `tokenizer` | 8192 BPE, owner surface |
| 03 | `data_mix` | 65/15/10/5/5 corpus contract |
| 04 | `training_loop` | session + milestone trains |
| 05 | `evaluation` | eval, benchmark, verifier |
| 06 | `runtime` | generate, infer, stream |
| 07 | `api_web` | FastAPI + React cockpit |
| 08 | `identity` | CIV, ESV, HAL, drift resistance |
| 09 | `memory` | 4-tier router |
| 10 | `phase2_memory` | typed store, graph, vectors |
| 11 | `goals` | persistent queue |
| 12 | `agent_loop` | plan → execute → tools |
| 13 | `master_system` | owner control + autonomy tiers |
| 14 | `self_improvement` | gap-driven refinement |
| 15 | `self_modification` | Type-A/B patches + sandbox |
| 16 | `ouroboros` | recursive passes |
| 17 | `ghost_memory` | compressed long recall |
| 18 | `symbolic_bridge` | verified math/logic/code |
| 19 | `sovereignty` | audit + promote/quarantine |

Each: `enabled` flag, `metric_hooks`, registry health, telemetry when run.

---

## Engineering spine (`engine/`)

What makes An-Ra operable instead of ornamental:

| Module | Answers |
|--------|---------|
| `feature_flags.py` | Can I turn it off? |
| `telemetry.py` | How fast? How often does it fail? |
| `eval_harness.py` | Did we regress vs baseline? |
| `report.py` | What is the scorecard right now? |

```bash
python anra.py --report          # full scorecard
python anra.py --status          # master system
python anra.py --phase3-status   # identity, ghost, symbolic, sovereignty
python anra.py --symbolic "solve x^2 - 9 = 0"
```

---

## Command cheat sheet

```bash
# Health & measurement
python anra.py --report
python -m pytest tests/ -q

# Operator / Jarvis-shaped
python anra.py --chat
python anra.py --goal "your imperative here"

# Training
python -m training.train_unified --mode status
python -m training.train_unified --mode session    # daily
python -m training.train_unified --mode train      # milestone
python -m training.train_unified --mode eval

# Web UI
cd phase4/web && npm install && npm run dev
python app.py
```

**CPU warning:** Flash SDP/CUDA message on Windows CPU is expected for smoke tests. Train on GPU (Colab or local CUDA).

---

## Training rhythms

### Daily (boring = good)

```bash
python -m training.train_unified --mode session
```

```text
restore → validate → train → evaluate → reports → replay failures
```

### Milestone (judgment)

```bash
python -m training.train_unified --mode train
```

Identity reinforcement, Ouroboros, self-improvement, **sovereignty promotion gate**.

---

## Data contract

Teacher data is an **amplifier**. You are the gravitational center.

| Bucket | % | Purpose |
|--------|---:|---------|
| Your conversation / instruction | 65 | Voice, style, behavior |
| Your identity / selfhood | 15 | Anti-drift |
| Teacher reasoning | 10 | Harder traces |
| Symbolic / verified | 5 | Truth anchor |
| Replayed failures | 5 | Mistakes → supervision |

---

## Artifacts

```text
training_data/anra_training.txt
tokenizer/tokenizer_v3.json
anra_v2_brain.pt / anra_v2_identity.pt / anra_v2_ouroboros.pt
workspace/                              # operator sandbox (files, CAD)
state/feature_flags.json
state/logs/telemetry.jsonl
state/logs/operator_actions.jsonl       # slash commands + operator audit
output/v2/reports/
output/v2/eval/
```

`19/19 active` ≠ checkpoints on disk. Restore from Drive or train.

---

## Feature flags

```python
from engine.feature_flags import set_flag, disabled_components
set_flag("ghost_memory", False)
print(disabled_components())
```

| Task kind | Routes to |
|-----------|-----------|
| `coder` / `research` | `agent_loop` |
| `memory` | `memory` |
| `critic` | `evaluation` |
| `symbolic` | `symbolic_bridge` |
| `ghost` | `ghost_memory` |

---

## Colab

`AnRa_Master.ipynb` — Drive, GPU, train, eval, sync. **Edit code in git; operate in Colab.**

---

## Roadmap (honest)

| Capability | Status |
|------------|--------|
| Local files, goals, CAD stub, open | **Shipped** — [`OPERATOR.md`](OPERATOR.md) |
| Symbolic-in-every-reply | Wire in `generate.py` (next) |
| Remote paired node | Design phase |
| ROS2 / robotics bridge | Future layer |

“Superintelligence” here = **compounding verified loops**, not one bigger chat model.

---

## Before you add code

1. Which component improves?  
2. What metric moves?  
3. What test proves it?  
4. What failure becomes replay?  
5. What owner boundary stays intact?

---

## Documentation map

| Doc | Role |
|-----|------|
| [`docs/README.md`](docs/README.md) | **Hub** — folder layout for all docs |
| [`docs/engineering/ENGINEERING_LOG.md`](docs/engineering/ENGINEERING_LOG.md) | **Change tracker** — every add/change/remove (dated) |
| [`docs/planning/MASTER_GOALS.md`](docs/planning/MASTER_GOALS.md) | **Goals backlog** — research, test, train, robotics |
| [`OPERATOR.md`](OPERATOR.md) | **Do work** — slash commands, workspace, CAD |
| [`DEVELOPER.md`](DEVELOPER.md) | Contribute safely |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Wiring diagram |
| [`VISION.md`](VISION.md) | Why An-Ra exists |
| [`WALKTHROUGH.md`](WALKTHROUGH.md) | Deep technical tour |

**Log a change after you ship:**

```bash
python scripts/log_engineering_change.py --component agent_loop --type CHANGE \
  --title "Short title" --summary "What and why" --verify "pytest tests/ -q"
```

Build like you will run this every week.
