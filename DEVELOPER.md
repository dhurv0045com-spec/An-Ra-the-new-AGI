# An-Ra Developer Guide

> Build aggressively, but do not confuse chaos with ambition.

This document explains how to work on the current An-Ra mainline without losing the project's soul.

That soul matters.

An-Ra is not supposed to drift into a generic assistant trained mostly on someone else's worldview. The system is now organized so that **your data stays primary**, **teacher help stays constrained**, and **the heavy subsystems support the core instead of obscuring it**.

## The Current Canonical Truth

If you open the repository cold and want to know what is real, use this rule:

- non-`_v2` entrypoints are the public mainline
- `training/v2_*` modules are support infrastructure
- V2 is the actual active model line

The canonical entrypoints are:

- `anra_brain.py`
- `generate.py`
- `scripts/build_brain.py`
- `training/train_unified.py`
- `training/finetune_anra.py`
- `scripts/train_ouroboros.py`
- `scripts/run_self_improvement.py`
- `scripts/run_sovereignty_audit.py`
- `AnRa_Master.ipynb`

## Development Principles

### 1. The model belongs to the owner data

Do not let convenience destroy authorship.

Your corpus should continue to control:

- identity
- worldview
- tone
- preferred response behavior
- how An-Ra sounds under pressure

Teacher data can help with capability, but it must never silently replace the center of gravity.

### 2. Daily training must stay cheap and trustworthy

The daily loop should remain:

1. restore
2. validate
3. train base model
4. save once
5. evaluate
6. write next-session guidance

If a change makes the daily loop fragile, it is not an improvement.

### 3. Milestone depth belongs off the daily critical path

Ouroboros, identity reinforcement, self-improvement, and sovereignty are real components, but they are **milestone layers**.

Do not push heavy reflection into every short T4 session unless you can show that the gain is worth the time cost.

### 4. Verified support beats fake certainty

If a domain can be checked, check it.

Use:

- `symbolic_bridge` for math and logic
- code or test verification when possible
- replay on failures instead of pretending they did not happen

### 5. Preserve clear public surfaces

The repo should feel understandable.

When a new developer looks for:

- how to train
- how to generate
- how to run a milestone
- where reports go

the answer should be obvious from the canonical files, not hidden in a second copy with `_v2` in the name.

## Mainline Architecture

### Base model

`anra_brain.py` now exports the current mainline transformer.

Current characteristics:

- RoPE
- RMSNorm
- SwiGLU
- SDPA / FlashAttention-compatible attention path
- `384 / 6 / 6` first target scale

This is the primary substrate. Everything else either shapes it, tests it, or surrounds it.

### Tokenizer

`tokenizer/subword_tokenizer.py` is the V2 tokenizer implementation.

Important design choice:

- it uses `tokenizers` when available
- it falls back to an internal dependency-light backend when `tokenizers` is not installed

That fallback exists to protect Colab and local smoke runs from dying over one missing package.

### Runtime

`generate.py` is the canonical generation runtime. If the app, API, or system bridge needs model output, this is the surface that should stay stable.

Exports you should preserve unless there is a very good reason not to:

- `GenerationConfig`
- `generate`
- `generate_traced`
- `generate_stream`
- `get_model_info`
- `load_ghost_state`
- `save_ghost_state`
- `detect_repetition`
- `_check_stop`

### Training support layer

The V2 support stack lives under:

- `training/v2_config.py`
- `training/v2_data_mix.py`
- `training/v2_runtime.py`
- `training/eval_v2.py`

These files are where you should make most V2 training changes.

## Data Mix Contract

The default mix is locked for a reason:

- 65% own conversation and instruction data
- 15% own identity and selfhood data
- 10% teacher reasoning traces
- 5% symbolic or code-verified examples
- 5% replayed failures and corrections

When changing this mix, ask:

1. Does this preserve owner data dominance?
2. Does this improve capability measurably?
3. Does this change identity drift risk?
4. Does this fit T4-first training time?

If you cannot answer those, do not change the ratios casually.

## Teacher Pipeline Rules

Teacher data is allowed. Teacher control is not.

Accepted teacher roles:

- reasoning amplifier
- hard-example generator
- synthetic curriculum helper
- correction source

Rejected teacher roles:

- personality owner
- worldview owner
- permanent inference dependency

Preferred teacher flow:

1. generate candidate examples
2. verify what can be verified
3. style-filter for An-Ra voice and mission fit
4. reject off-style or low-truth samples
5. only then feed them into the teacher bucket

If you add a new teacher source, wire it through filtering before it touches the mainline mix.

## Daily Commands

### Status

```bash
python -m training.train_unified --mode status
```

Use this first when something feels off.

### Daily session

```bash
python -m training.train_unified --mode session
```

This is the normal T4 path.

### Milestone

```bash
python -m training.train_unified --mode train
```

Use this after a small run streak, or when compact evals flatten out.

## Canonical File Responsibilities

| File | Responsibility |
| --- | --- |
| `scripts/build_brain.py` | base V2 training entrypoint |
| `training/finetune_anra.py` | identity-heavy milestone stage |
| `scripts/train_ouroboros.py` | reflection-heavy milestone stage |
| `scripts/run_self_improvement.py` | report-driven curriculum recommendations |
| `scripts/run_sovereignty_audit.py` | checkpoint promotion and audit gate |
| `training/train_unified.py` | orchestration surface used by notebook and CLI |
| `scripts/verify_structure.py` | canonical repo structure sanity check |
| `scripts/status.py` | quick operator-facing artifact view |

## Reports And Artifacts

Daily and milestone artifacts should land under `output/v2/`.

Core files to watch:

- `v2_session_train_metrics.json`
- `v2_hard_examples.json`
- `v2_eval_summary.json`
- `v2_next_session_curriculum.json`
- `v2_unified_training_report.json`
- `v2_improvement_report.json`
- `v2_audit_report.json`

Primary checkpoints:

- `anra_v2_brain.pt`
- `anra_v2_identity.pt`
- `anra_v2_ouroboros.pt`

Drive mirror:

- `/content/drive/MyDrive/AnRa/v2/`

## How To Extend The System Safely

### Good changes

- better evaluation prompts
- better teacher filtering
- better replay selection
- better hard-example ranking
- symbolic verification improvements
- checkpoint promotion logic improvements
- inference/runtime efficiency improvements

### Risky but reasonable changes

- adjusting mix ratios with evidence
- modest model scale increases
- new milestone scheduling rules
- better tokenizer training heuristics

### Changes that should live behind proof

- replacing the current tokenizer again
- moving reflection into every daily session
- architecture changes that break resume assumptions
- adding more autonomous loops without stronger evals
- making teacher data dominant

## Colab Operational Truth

The project is T4-first right now.

That means:

- startup speed matters
- logs must appear early
- save behavior must be clear
- the session has to survive ordinary Colab restarts and short windows

When in doubt, optimize for:

- obvious resume behavior
- fewer silent stalls
- simpler daily commands
- stronger post-session reports

not for abstract cleverness.

## The Role Of The Larger Ecosystem

### `symbolic_bridge`

Truth layer for math, logic, and some code validation.

### `ghost_memory`

Not just chat memory. It should become the replay bank for:

- failures
- corrections
- continuity pressure cases
- future curriculum shaping

### `turboquant`

Inference and runtime efficiency support. Keep it there unless you have a very strong reason to inject it into training decisions.

### `ouroboros_numpy`

Reflection and repair engine. Strong as a milestone pass. Wasteful if forced into every short run.

### `sovereignty_bridge`

Promotion gate and audit logic. It protects the lineage of the model by forcing checkpoint judgment instead of blind replacement.

## What To Protect

Protect these three things even when making aggressive changes:

1. **identity ownership**
2. **daily loop reliability**
3. **measurable improvement**

If a patch threatens all three at once, it is almost certainly a bad patch.

## Long-Term Direction

The repo is allowed to be ambitious.

The long-term direction still includes:

- teacher-amplified capability growth
- replay-driven self-repair
- verified reasoning
- stronger memory integration
- milestone reflection
- sovereignty-controlled checkpoint promotion

But the order matters:

1. make it work
2. make it measurable
3. make it better
4. only then make it wild

That is how you keep soul and velocity at the same time.

*An-Ra does not need to become generic to become powerful.*
