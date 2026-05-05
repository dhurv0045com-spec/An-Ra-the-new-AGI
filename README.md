# AN-RA

> Built from zero: a transformer core, a memory spine, an agent layer, a verification layer, and a sovereignty loop around one authored center.

An-Ra is a private, owner-shaped AI system. The repo is no longer a loose trail of experiments; the current mainline is a 19-layer stack with V2 as the canonical path.

Status snapshot from `python scripts/status.py` on 2026-05-05:

```text
Capabilities: 19/19 active
Source files: 309
Python files: 278
Markdown files: 15
Python lines: 76938
Tokenizer: tokenizer_v3.json present
Local V2 checkpoints: missing until restored or trained
```

## The 19/19 Layer Map

| # | Layer | Source | Role |
| --- | --- | --- | --- |
| 01 | `brain` | `anra_brain.py` | Canonical V2 transformer core |
| 02 | `tokenizer` | `tokenizer/tokenizer_v3.json` | 8192-token BPE path with fallback adapter |
| 03 | `data_mix` | `training/v2_data_mix.py` | Owner-data-first training contract |
| 04 | `training_loop` | `training/train_unified.py` | Daily and milestone training orchestration |
| 05 | `evaluation` | `training/eval_v2.py` | Compact eval, verifier, benchmark feedback |
| 06 | `inference_runtime` | `generate.py` | Generation, streaming, tracing, connector refresh |
| 07 | `api_web` | `app.py`, `phase4/web/` | API plus operator dashboard |
| 08 | `identity` | `identity/`, `phase3/identity (45N)/` | CIV/ESV guards and runtime identity injection |
| 09 | `memory_router` | `memory/` | Unified memory entry point |
| 10 | `phase2_memory` | `phase2/memory (45J)/` | Typed memory, vector retrieval, graph context |
| 11 | `goals` | `goals/`, `agents/` | Persistent goals and specialist dispatch |
| 12 | `agent_loop` | `phase2/agent_loop (45k)/` | Goal to plan to execution to evaluation |
| 13 | `master_system` | `phase2/master_system (45M)/` | Autonomy, owner control, safety, personalization |
| 14 | `self_improvement` | `phase2/self_improvement (45l)/` | Improvement engine and dashboard hooks |
| 15 | `self_modification` | `self_modification/`, `execution/` | Patch gates, sandbox, atomic filesystem actions |
| 16 | `ouroboros` | `phase3/ouroboros (45O)/` | Recursive reasoning and milestone reflection |
| 17 | `ghost_memory` | `phase3/ghost_memory (45P)/` | Compressed conversational recall |
| 18 | `symbolic_bridge` | `phase3/symbolic_bridge (45Q)/` | Verified math, logic, and code reasoning |
| 19 | `sovereignty` | `phase3/sovereignty (45R)/` | Audit, benchmark, report, and promotion governance |

## Main Commands

```bash
python scripts/status.py
python -m inference.full_system_connector
python -m training.train_unified --mode status
python -m training.train_unified --mode session
python -m training.train_unified --mode train
python -m training.train_unified --mode eval
python anra.py --status
python anra.py --phase3-status
python anra.py --symbolic "solve x^2 - 9 = 0"
```

Use `session` for the normal daily path. Use `train` for the deeper milestone path: base session, identity, Ouroboros, self-improvement, sovereignty audit, then milestone tests.

## Current Mainline

The public path is:

- `anra_brain.py`
- `generate.py`
- `anra.py`
- `app.py`
- `scripts/build_brain.py`
- `training/train_unified.py`
- `training/finetune_anra.py`
- `scripts/train_ouroboros.py`
- `scripts/run_self_improvement.py`
- `scripts/run_sovereignty_audit.py`
- `AnRa_Master.ipynb`

Support modules that belong to the mainline:

- `training/v2_config.py`
- `training/v2_data_mix.py`
- `training/v2_runtime.py`
- `training/eval_v2.py`

## Data Contract

The default training mix stays owner-centered:

| Bucket | Share | Purpose |
| --- | ---: | --- |
| Own conversation and instruction | 65% | Voice, problem style, normal behavior |
| Own identity and selfhood | 15% | Identity gravity and drift resistance |
| Teacher reasoning | 10% | Harder reasoning traces |
| Symbolic or code-verified samples | 5% | Truth-checked supervision |
| Replayed failures and corrections | 5% | Repair from real mistakes |

Teacher data is an amplifier, not an owner. An-Ra should get sharper without becoming generic.

## Artifact Layout

Canonical dataset:

- `training_data/anra_training.txt`

Tokenizer:

- `tokenizer/tokenizer_v3.json`

V2 checkpoints:

- `anra_v2_brain.pt`
- `anra_v2_identity.pt`
- `anra_v2_ouroboros.pt`

Reports:

- `output/v2/v2_session_train_metrics.json`
- `output/v2/v2_hard_examples.json`
- `output/v2/v2_eval_summary.json`
- `output/v2/v2_next_session_curriculum.json`
- `output/v2/v2_unified_training_report.json`
- `output/v2/v2_improvement_report.json`
- `output/v2/v2_audit_report.json`

Fresh local clones may show missing checkpoints and reports. That is normal until Drive restore or training creates them.

## What Makes This Repo Different

An-Ra is built around a specific bet: small-compute systems need better selection, better memory of failure, better verification, and stronger identity preservation instead of pretending scale alone will solve everything.

The stack is now shaped to do that:

- train on owned data first
- evaluate before claiming progress
- replay failures instead of burying them
- use symbolic and code checks where possible
- keep daily training reliable
- reserve heavy reflection for milestones
- promote checkpoints by audit, not by recency

For the technical map, read [ARCHITECTURE.md](ARCHITECTURE.md). For development rules, read [DEVELOPER.md](DEVELOPER.md). For the long-horizon intent, read [VISION.md](VISION.md).
