# AN-RA

> *Built from zero. No templates. No shortcuts. Pure mathematics pushed until it starts to remember, reason, and choose a direction.*

An-Ra is a sovereign AI system built from scratch around a stubborn idea:

**the center of the system must stay yours.**

Not just your weights.  
Not just your prompts.  
Your identity, your tone, your priorities, your preferred way of thinking about capability itself.

That is the soul of this project, and the current mainline is designed to protect it.

## What This Repository Is Now

An-Ra is no longer organized as an old base model with a loose V2 experiment sitting beside it.  
This repository now treats **V2 as the canonical mainline**.

The public command path is the normal one:

- `anra_brain.py`
- `generate.py`
- `scripts/build_brain.py`
- `training/train_unified.py`
- `training/finetune_anra.py`
- `scripts/train_ouroboros.py`
- `scripts/run_self_improvement.py`
- `scripts/run_sovereignty_audit.py`
- `AnRa_Master.ipynb`

The V2-specific helper modules that remain in `training/` are not loose leftovers. They are the support layer behind the canonical system:

- `training/v2_config.py`
- `training/v2_data_mix.py`
- `training/v2_runtime.py`
- `training/eval_v2.py`

## Codebase Scale

Current repository shape is generated from `runtime/system_registry.py`, not hand-maintained.

Run:

```bash
python scripts/status.py
python -m inference.full_system_connector
```

Those commands update the live source metrics and `system_graph.json`.

This matters because An-Ra is not a one-file toy model anymore. It is a multi-layered system with:

- a modernized transformer core
- a subword tokenizer
- a daily and milestone training loop
- symbolic reasoning bridges
- memory and replay subsystems
- agent and master-system layers
- a web/API interface
- reflective and governance passes

## The Governing Philosophy

### 1. Your data first

The model should sound like it came from your world, not like it swallowed a generic assistant and put on a costume.

The default training mix reflects that:

- `65%` own conversation and instruction data
- `15%` own identity / selfhood data
- `10%` teacher-generated reasoning traces
- `5%` symbolic or code-verified samples
- `5%` replayed failures and user corrections

That means `80%` of the main training signal stays anchored in your material.

### 2. Teacher as amplifier, not owner

Teacher data is useful. Teacher control is not.

Teacher examples are there to help with:

- reasoning structure
- harder prompts
- synthetic coverage for weak domains
- corrections
- more efficient learning on small compute

Teacher examples are **not** there to become An-Ra's personality, worldview, or central voice.

### 3. Verification over empty fluency

A fluent model that lies elegantly is not what this project is trying to become.

That is why the mainline keeps:

- `symbolic_bridge` for math and logic verification
- code and test-oriented teacher samples
- hard-example replay
- post-session compact evaluation
- sovereignty-style checkpoint promotion

### 4. Daily speed, milestone depth

The system has two rhythms:

**Daily session**

- restore artifacts
- validate dataset
- train the base brain
- save once
- run compact eval
- write curriculum guidance

**Milestone session**

- identity fine-tuning
- Ouroboros refinement
- self-improvement report
- sovereignty audit
- milestone test pass

This keeps daily progress efficient while still preserving deeper stages that help the system grow.

## Architecture Overview

```text
An-Ra
|
|- anra_brain.py                  -> canonical V2 transformer core
|- generate.py                    -> canonical inference / generation runtime
|- tokenizer/subword_tokenizer.py -> subword tokenizer with dependency-light fallback
|
|- scripts/build_brain.py         -> base V2 training
|- training/train_unified.py      -> daily + milestone orchestration
|- training/finetune_anra.py      -> identity milestone stage
|- scripts/train_ouroboros.py     -> reflection-heavy milestone stage
|- scripts/run_self_improvement.py -> curriculum / next-step recommendations
|- scripts/run_sovereignty_audit.py -> audit + checkpoint promotion gate
|
|- training/v2_data_mix.py        -> your-data-first bucket mixer
|- training/eval_v2.py            -> compact eval suite
|- training/v2_runtime.py         -> checkpoint / tokenizer / Drive / report runtime
|- runtime/system_registry.py     -> canonical component registry + live manifest
|- output/v2/                     -> session metrics, evals, curriculum, audits
|
|- phase2/                        -> memory, agent loop, self-improvement, master system
|- phase3/                        -> identity, Ouroboros, ghost memory, symbolic bridge, sovereignty
|- app.py                         -> FastAPI backend
|- AnRa_Master.ipynb              -> Google Colab operator notebook
```

## The Current Mainline Model

The current public `anra_brain.py` is the V2 mainline.

Core traits:

- RoPE position encoding
- RMSNorm
- SwiGLU feed-forward path
- SDPA / FlashAttention-compatible attention path
- T4-first scale target around `384 / 6 / 6`

This is a deliberate compromise:

- stronger than the old tiny line
- modern enough to justify a new mainline
- still trainable and debuggable on small compute

## Training Data And Mix

The canonical dataset is:

- `training_data/anra_training.txt`

The legacy source remains available as:

- `training_data/anra_dataset_v6_1.txt`

Fresh clones can run:

```bash
python scripts/setup_dataset.py
```

`anra_paths.ensure_dirs()` also restores the canonical file from the legacy dataset when needed.

The V2 mixer pulls from five buckets:

1. own conversation data
2. identity-heavy data
3. teacher reasoning traces
4. symbolic or code-verified samples
5. replayed failures and corrections

Optional teacher corpus:

- `training_data/teacher_reasoning_v2.jsonl`

If that file is missing, the system still trains using your data, symbolic fallbacks, and replay material. It does not collapse just because a teacher file is absent.

## Main Commands

### Status

```bash
python -m training.train_unified --mode status
```

Prints:

- subsystem health
- dataset path
- main checkpoints
- tokenizer location
- milestone readiness

### Daily session

```bash
python -m training.train_unified --mode session
```

What it does:

- restores the mainline artifacts from Drive
- validates the dataset
- runs base training
- writes `anra_v2_brain.pt`
- runs compact evaluation
- writes hard-example and curriculum reports

### Resume

```bash
python -m training.train_unified --mode resume
```

Alias for the normal daily path, kept for operator clarity.

### Milestone run

```bash
python -m training.train_unified --mode train
```

What it does:

1. base session
2. identity fine-tune
3. Ouroboros refinement
4. self-improvement analysis
5. sovereignty audit
6. milestone test pass

### Eval only

```bash
python -m training.train_unified --mode eval
```

## Colab Operator Flow

The notebook is designed around a T4 + Drive-backed environment.

Recommended resume order:

1. `Cell 1` - GPU + Drive
2. `Cell 3` - clone or update repo
3. `Cell 4` - restore V2 artifacts
4. `Cell 5` - health check
5. `Cell 6` - daily training

Fastest resume path when the environment is already healthy:

1. `Cell 1`
2. `Cell 3`
3. `Cell 4`
4. `Cell 6`

The notebook writes and restores from:

- `/content/drive/MyDrive/AnRa/v2/`

## Output And Artifact Layout

Mainline checkpoint family:

- `anra_v2_brain.pt`
- `anra_v2_identity.pt`
- `anra_v2_ouroboros.pt`

Tokenizer:

- `tokenizer/tokenizer_v3.json`

Primary reports:

- `output/v2/v2_session_train_metrics.json`
- `output/v2/v2_hard_examples.json`
- `output/v2/v2_eval_summary.json`
- `output/v2/v2_next_session_curriculum.json`
- `output/v2/v2_unified_training_report.json`
- `output/v2/v2_improvement_report.json`
- `output/v2/v2_audit_report.json`

## Subsystem Roles

| Subsystem | What it does in the mainline |
| --- | --- |
| `anra_brain.py` | Language and reasoning substrate |
| `identity_injector` | Anchors voice, self-description, and worldview |
| `ouroboros_numpy` | Milestone reflection and refinement engine |
| `symbolic_bridge` | Verified math, logic, and code support |
| `ghost_memory` | Failure storage, continuity traces, replay fuel |
| `turboquant` | Runtime and inference efficiency support |
| `sovereignty_bridge` | Audit, promotion, and checkpoint governance |
| `phase2` systems | Memory, agent-loop, and orchestration layers |

## Why An-Ra Is Different

An-Ra is not trying to win by pretending scarcity does not exist.

It is trying to win through:

- intelligence per parameter
- intelligence per training minute
- intelligence per correction
- verified reasoning where tools can check
- replay of real failures instead of fantasy progress
- identity preservation under increasing capability

That is a different bet from pure scale.

## What The Project Is Becoming

Not just a smaller chatbot.  
Not just a handcrafted transformer.  
Not just a bundle of modules with poetic names.

The shape emerging here is:

- a core model trained in your terms
- a support stack that adds verification, memory, and correction
- a training loop that remembers mistakes
- a milestone system that rewards judged progress, not blind overwrite

That is the actual project.

If you want the deeper technical map, read [ARCHITECTURE.md](ARCHITECTURE.md) and [DEVELOPER.md](DEVELOPER.md).  
If you want the long-horizon architecture and the larger ambition, read [VISION.md](VISION.md).

*An-Ra: something that emerged from mathematics with a direction.*
