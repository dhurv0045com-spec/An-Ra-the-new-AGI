# AN-RA

> A sovereign, owner-shaped AI platform — measurable learning, real memory, verified reasoning, controlled autonomy.

**You cloned the repo. Good.** This is not a chatbot wrapper duct-taped to an API. It is a from-scratch stack: transformer brain, tokenizer, training loop, memory tiers, agent dispatch, identity guardrails, symbolic verification, ghost recall, self-improvement, self-mod gates, sovereignty audits — and an engineering spine that makes all of it **visible**.

One rule holds the whole project together:

```text
No magic subsystem.
Every component must be registered, switchable, measurable, reportable, and testable.
```

Run this first:

```bash
python -m pip install -r requirements.txt
python anra.py --report
```

If you see `19/19` components with source and import OK, you are looking at a live platform — not a README fantasy.

---

## What you are looking at

The canonical map lives in `runtime/system_registry.py`. Do not maintain component lists by hand anywhere else.

| # | Component | What it actually does |
| --- | --- | --- |
| 01 | `brain` | V2 transformer: GQA, RoPE/YaRN, MoD, Flash SDP, tied embeddings |
| 02 | `tokenizer` | 8192-token BPE, owner-trained surface |
| 03 | `data_mix` | Owner-first corpus contract (65/15/10/5/5) |
| 04 | `training_loop` | Daily sessions + milestone trains |
| 05 | `evaluation` | Eval, benchmark, verifier, hard-example capture |
| 06 | `runtime` | Generation, streaming, inference helpers |
| 07 | `api_web` | FastAPI + Phase 4 operator UI |
| 08 | `identity` | CIV / ESV / HAL + drift resistance |
| 09 | `memory` | Router across episodic, short-term, graph, ghost |
| 10 | `phase2_memory` | Typed store, vectors, graph, context builder |
| 11 | `goals` | Persistent queue, retries, successors |
| 12 | `agent_loop` | Plan → execute → monitor → evaluate |
| 13 | `master_system` | Owner control, autonomy, long-horizon goals |
| 14 | `self_improvement` | Gap-driven refinement and session learning |
| 15 | `self_modification` | Type-A/B patches, sandbox, FS agent |
| 16 | `ouroboros` | Recursive reasoning, adaptive passes |
| 17 | `ghost_memory` | Compressed long recall + decay |
| 18 | `symbolic_bridge` | Verified math, logic, code |
| 19 | `sovereignty` | Audit, benchmarks, promotion gates |

Each component exposes: `enabled`, `metric_hooks`, registry source status, and telemetry when exercised.

---

## The engineering spine (`engine/`)

This is what separates An-Ra from "impressive folder structure."

| Module | Job |
| --- | --- |
| `component_base.py` | Component protocol + metrics bookkeeping |
| `feature_flags.py` | Toggle subsystems via `state/feature_flags.json` |
| `telemetry.py` | JSONL traces: latency, success, errors, tokens |
| `eval_harness.py` | Baseline / system-on / ablation / regression |
| `report.py` | One-command operator scorecard |

**Questions you should never have to guess:**

| Question | Answer |
| --- | --- |
| What exists? | `component_registry()` |
| Is it on? | `is_enabled("component")` |
| How fast? | `get_telemetry_bus().summary_by_module()` |
| Did we regress? | `EvalHarness().compare(baseline, current)` |
| What broke? | `python anra.py --report` |
| Can I kill it? | `set_flag("component", False)` |

---

## Operator quickstart

```bash
# Health
python anra.py --report
python anra.py --status
python anra.py --phase3-status

# Reasoning with receipts
python anra.py --symbolic "solve x^2 - 9 = 0"

# Agency
python anra.py --goal "summarize the current system report"

# Training
python -m training.train_unified --mode status
python -m training.train_unified --mode session   # daily
python -m training.train_unified --mode train     # milestone
python -m training.train_unified --mode eval

# Tests (expect green on a healthy clone)
python -m pytest tests/ -q
```

**CPU-only?** You may see a Flash SDP / CUDA warning from `anra.py`. That is normal for local smoke checks. Serious training wants a GPU box.

---

## Three ways to run the system

### 1. Inspect (no training required)

Status, flags, telemetry, symbolic checks, fast tests:

```bash
python anra.py --report
python anra.py --symbolic "factor 360"
python -c "from engine.feature_flags import disabled_components; print(disabled_components())"
```

### 2. Daily learning (keep it boring)

```bash
python -m training.train_unified --mode session
```

Loop:

```text
restore → validate → train → evaluate → reports → replay failures
```

### 3. Milestone (judgment pass)

```bash
python -m training.train_unified --mode train
```

Heavier on purpose: identity reinforcement, Ouroboros, self-improvement, sovereignty audit, promotion gates.

---

## Feature flags

Turn a subsystem off without surgery:

```bash
python -c "
from engine.feature_flags import set_flag, disabled_components
set_flag('ghost_memory', False)
print(disabled_components())
"
```

Orchestrator routing (disabled components are skipped, not crashed):

| Task kind | Component |
| --- | --- |
| `coder` | `agent_loop` |
| `research` | `agent_loop` |
| `memory` | `memory` |
| `critic` | `evaluation` |
| `symbolic` | `symbolic_bridge` |
| `ghost` | `ghost_memory` |

---

## Telemetry & scorecard

Log: `state/logs/telemetry.jsonl`

Each traced call records module, operation, timing, success/failure, errors, tokens, output size, confidence.

```bash
python anra.py --report
```

| Axis | Meaning |
| --- | --- |
| Source health | Registered files exist |
| Import health | Components import clean |
| Enablement | Feature flag state |
| Latency | Avg traced ms per module |
| Reliability | Success rate + recent failures |
| Regression | Eval harness vs baseline |
| Readiness | Training artifacts + preflight |

---

## Data contract (owner stays center of gravity)

| Bucket | Share | Why |
| --- | ---: | --- |
| Own conversation / instruction | 65% | Voice, style, normal behavior |
| Own identity / selfhood | 15% | Drift resistance |
| Teacher reasoning | 10% | Harder traces — amplifier, not owner |
| Symbolic / verified samples | 5% | Truth anchor |
| Replayed failures | 5% | Mistakes become supervision |

---

## Artifacts you will actually touch

```text
training_data/anra_training.txt     # canonical dataset
tokenizer/tokenizer_v3.json           # vocab
anra_v2_brain.pt                    # main weights (may be absent on fresh clone)
anra_v2_identity.pt
anra_v2_ouroboros.pt
state/feature_flags.json
state/logs/telemetry.jsonl
output/v2/reports/
output/v2/eval/
```

Fresh clones often have **no checkpoints**. Restore from Drive or train them. `19/19 active` in the report means **source is present**, not that weights exist.

---

## Colab

`AnRa_Master.ipynb` is the operator console — mount Drive, restore checkpoints, train, eval, sync reports. **Edit source in git; run operations in Colab.**

---

## Before you add anything

Every serious change should answer:

1. Which component does this improve?
2. What metric should move?
3. What test or verifier proves it?
4. What failure becomes replay or docs?
5. What owner-control boundary stays intact?

**Read next:** [`DEVELOPER.md`](DEVELOPER.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`VISION.md`](VISION.md) · [`WALKTHROUGH.md`](WALKTHROUGH.md) (deep tour)
