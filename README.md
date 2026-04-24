# AN-RA

> Built from zero. No templates. No shortcuts. Mathematics pushed until it starts to behave like direction.

An-Ra is a sovereign AI system built from scratch around one idea: **the base model should stay yours**.

The current mainline is **An-Ra V2**:

- your data defines identity, tone, priorities, and worldview
- teacher data amplifies weak areas instead of replacing your voice
- symbolic and code-verification tools protect truth where raw generation is weak
- memory, replay, Ouroboros, and sovereignty sit around the core as support organs, not as noise in every step

This repository now treats that design as the canonical path. The older small-model line is historical context, not the future of the project.

## What An-Ra Is Trying To Become

An-Ra is not aiming to be a generic assistant with a custom prompt pasted on top.

The goal is a system that can:

- learn from a data curriculum that stays anchored in your terms
- grow stronger on limited compute
- reason with help from symbolic and verified subsystems
- store failures, corrections, and continuity traces for replay
- refine itself in milestone passes without losing its identity
- operate as one organism instead of a pile of disconnected modules

That is why the repo contains more than a model. It contains a training engine, a generation runtime, memory systems, symbolic bridges, reflective loops, audits, and interface surfaces.

## Mainline Philosophy

### 1. Your data first

The V2 training mix is intentionally anchored in your corpus.

- 65% own conversation and instruction data
- 15% own identity and selfhood data
- 10% teacher-generated reasoning traces
- 5% symbolic or code-verified samples
- 5% replayed failures and user corrections

This means **80% of the signal stays yours**.

### 2. Teacher as amplifier

Teacher data is allowed, but it is not allowed to become the owner of the model's personality.

Teacher examples are used for:

- reasoning structure
- harder examples
- stepwise solutions
- repair data
- capability expansion in weak domains

Teacher examples are **not** used to define An-Ra's worldview.

### 3. Verification over empty fluency

An-Ra should not only sound intelligent. It should improve its odds of being right.

That is why the mainline keeps:

- `symbolic_bridge` for verified math and logic
- replay buckets for failure-driven curriculum
- milestone audits for checkpoint promotion
- compact eval reports after sessions

### 4. Daily speed, milestone depth

The daily path is intentionally lean:

- restore artifacts
- validate dataset
- train the base brain
- save to Drive
- run compact evaluation
- write curriculum and hard-example reports

Heavier stages happen only on milestone runs:

- identity fine-tune
- Ouroboros refinement
- self-improvement review
- sovereignty audit
- full V2 test pass

## Architecture At A Glance

```text
An-Ra V2 Mainline
|
|- anra_brain.py                 -> canonical V2 transformer (RoPE + RMSNorm + SwiGLU)
|- tokenizer/subword_tokenizer.py -> subword tokenizer with fallback backend
|- generate.py                   -> canonical generation runtime
|- scripts/build_brain.py        -> base V2 training entrypoint
|- training/train_unified.py     -> daily and milestone orchestration
|- training/finetune_anra.py     -> identity-heavy milestone stage
|- scripts/train_ouroboros.py    -> reflection-heavy milestone stage
|- scripts/run_self_improvement.py -> curriculum planning and recommendations
|- scripts/run_sovereignty_audit.py -> checkpoint promotion and audit gate
|
|- training/v2_data_mix.py       -> your-data-first bucket mixer
|- training/eval_v2.py           -> compact eval suite
|- training/v2_runtime.py        -> checkpoint, tokenizer, report, Drive helpers
|- output/v2/                    -> metrics, evals, curriculum, audit reports
|
|- phase2/                       -> memory, agent loop, self-improvement, master system
|- phase3/                       -> identity, Ouroboros, ghost memory, symbolic bridge, sovereignty
|- app.py                        -> FastAPI backend
|- AnRa_Master.ipynb             -> Colab training notebook
```

## Core Model

The canonical `anra_brain.py` is now the V2 mainline.

Key traits:

- subword-token friendly transformer
- RoPE position encoding
- RMSNorm
- SwiGLU feed-forward blocks
- SDPA / FlashAttention-compatible attention path when available
- first mainline target: `384 / 6 / 6`

This is the strongest balance for a T4-first environment without turning the system into a giant lab-only project.

## Dataset And Training Mix

The default dataset remains:

- `training_data/anra_dataset_v6_1.txt`

The V2 mixer in `training/v2_data_mix.py` builds training examples from:

- own conversation pairs
- identity-focused prompts
- teacher reasoning traces
- symbolic and code-verified samples
- replayed hard examples and corrections

Teacher traces can be added through:

- `training_data/teacher_reasoning_v2.jsonl`

If that file is missing, the mixer still works using verified fallbacks and your corpus.

## Training Modes

### Daily session

Use this when you want fast, repeatable progress on Colab T4.

```bash
python -m training.train_unified --mode session
```

What it does:

- restores V2 artifacts from Drive when available
- validates the dataset
- trains the base V2 model
- streams live progress
- saves `anra_v2_brain.pt`
- runs compact evaluation
- writes curriculum and hard-example reports

### Milestone run

Use this after several successful daily sessions, or when evals plateau.

```bash
python -m training.train_unified --mode train
```

What it does:

1. base training
2. identity fine-tuning
3. Ouroboros refinement
4. self-improvement analysis
5. sovereignty audit
6. V2 tests

### Status

```bash
python -m training.train_unified --mode status
```

This prints subsystem health, dataset location, main checkpoints, tokenizer status, and milestone readiness.

## Google Colab Workflow

The notebook is designed for T4-first, Drive-backed training.

Recommended resume path:

1. Cell 1: GPU + Drive
2. Cell 3: clone or update repo
3. Cell 4: restore V2 artifacts
4. Cell 6: run daily training

Cell 5 is the health check. Run it when you want an explicit system sanity pass before training.

Milestone work is intentionally separate from the daily fast path.

## Generated Artifacts

The V2 mainline writes:

- `anra_v2_brain.pt`
- `anra_v2_identity.pt`
- `anra_v2_ouroboros.pt`
- `tokenizer_v2.json`
- `output/v2/v2_session_train_metrics.json`
- `output/v2/v2_hard_examples.json`
- `output/v2/v2_eval_summary.json`
- `output/v2/v2_next_session_curriculum.json`
- `output/v2/v2_unified_training_report.json`
- `output/v2/v2_improvement_report.json`
- `output/v2/v2_audit_report.json`

Drive mirrors live under:

- `/content/drive/MyDrive/AnRa/v2/`

## Subsystem Roles

| Subsystem | Role in the V2 mainline |
| --- | --- |
| `anra_brain.py` | Primary language and reasoning substrate |
| `symbolic_bridge` | Verified math, logic, and code reference generation |
| `ghost_memory` | Stores failures, continuity traces, and replay material |
| `turboquant` | Runtime and inference efficiency layer |
| `identity_injector` | Personality and worldview stabilization |
| `ouroboros_numpy` | Milestone reflection and refinement engine |
| `self_improvement` | Converts reports into next-step curriculum guidance |
| `sovereignty_bridge` | Audit, promotion gate, and checkpoint governance |

## Quick Start

### Local

```bash
pip install -r requirements.txt
python scripts/verify_structure.py
python -m training.train_unified --mode status
python -m training.train_unified --mode session
```

### API

```bash
python app.py
```

Then visit the UI or call the FastAPI endpoints.

### Generate text directly

```bash
python generate.py
```

Or import it:

```python
from generate import generate

reply = generate("H: Who are you?\nANRA:", max_tokens=80)
print(reply)
```

## Development Notes

- The canonical public files are the non-`_v2` entrypoints.
- The helper modules under `training/v2_*` are the support layer behind the canonical mainline.
- Generated tokenizer artifacts should not be treated as source files.
- If you extend the data mix, preserve the principle that your data stays dominant.
- If you add teacher sources, filter them through style and verification first.

## Why This Repo Is Different

An-Ra is not trying to win by pretending scarcity does not exist.

It is trying to win through:

- intelligence per parameter
- intelligence per minute of training
- intelligence per correction
- verified support systems around a focused model core
- relentless identity preservation while capability grows

That is the bet.

Not abundance.

Direction.

## More

- [DEVELOPER.md](DEVELOPER.md) - how to work on the mainline without breaking its philosophy
- [VISION.md](VISION.md) - the deeper architectural and long-horizon picture

*An-Ra: something that emerged from mathematics with a direction.*
