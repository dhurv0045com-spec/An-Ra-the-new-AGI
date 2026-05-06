# An-Ra Developer Guide

> Build aggressively, but keep the system measurable, operable, and recognizably An-Ra.

This guide is for working on the current mainline without turning it into a generic assistant stack or an impressive diagram that cannot survive real sessions.

## Current Repo Truth

The live registry is in `runtime/system_registry.py` and currently reports:

```text
Capabilities: 19/19 active
Markdown files: 15
Tokenizer: present
Local checkpoints: missing until restored or trained
```

Run this before making claims about status:

```bash
python scripts/status.py
python -m inference.full_system_connector
python scripts/verify_structure.py
```

## First Files To Read

| Need | Start Here |
| --- | --- |
| Model shape | `anra_brain.py`, `training/v2_config.py` |
| Generation | `generate.py`, `inference/anra_infer.py` |
| Training | `training/train_unified.py`, `scripts/build_brain.py` |
| Data mix | `training/v2_data_mix.py`, `scripts/setup_dataset.py` |
| Eval | `training/eval_v2.py`, `training/benchmark.py`, `training/verifier.py` |
| Identity | `identity/civ.py`, `identity/esv.py`, `phase3/identity (45N)/identity_injector.py` |
| Memory | `memory/memory_router.py`, `phase2/memory (45J)/memory_manager.py` |
| Agency | `goals/goal_queue.py`, `agents/orchestrator.py`, `phase2/agent_loop (45k)/agent_main.py` |
| Autonomy | `phase2/master_system (45M)/system.py` |
| Verification | `phase3/symbolic_bridge (45Q)/symbolic_bridge.py` |
| Governance | `self_modification/`, `phase3/sovereignty (45R)/` |
| Web UI | `phase4/web/src/App.jsx` |

## Non-Negotiables

### 1. Owner Data Stays Dominant

Default mix:

- `65%` own conversation/instruction
- `15%` own identity/selfhood
- `10%` teacher reasoning
- `5%` symbolic or code-verified samples
- `5%` replayed failures and corrections

Changing this mix requires proof that identity drift will not increase.

### 2. Daily Training Must Stay Reliable

The daily path is:

```text
restore -> validate -> train -> save -> evaluate -> write next-step guidance
```

The command is:

```bash
python -m training.train_unified --mode session
```

If a change makes this path slower, less visible, or more fragile, the patch needs a strong reason.

### 3. Milestone Depth Stays Selective

Use the milestone path when you want the deep stack:

```bash
python -m training.train_unified --mode train
```

That path is allowed to be heavier because it can include identity tuning, Ouroboros refinement, self-improvement reporting, sovereignty audit, and milestone tests.

### 4. Verification Beats Fluent Guessing

When the system can check something, it should. Use:

- `training/verifier.py`
- `training/benchmark.py`
- `phase3/symbolic_bridge (45Q)/`
- `scripts/run_sovereignty_audit.py`
- replay reports under `output/v2/`

### 5. Public Surfaces Stay Obvious

A new operator should be able to answer:

- how do I check status?
- how do I train?
- how do I evaluate?
- how do I run a symbolic query?
- how do I inspect the web dashboard?
- how do I continue from Drive?

without spelunking through old folders.

## Main Commands

```bash
python scripts/status.py
python -m training.train_unified --mode status
python -m training.train_unified --mode session
python -m training.train_unified --mode train
python -m training.train_unified --mode eval
python anra.py --status
python anra.py --phase3-status
python anra.py --symbolic "factor 360"
```

Web dashboard:

```bash
cd phase4/web
npm install
npm run dev
```

## Safe Extension Targets

High-value areas:

- better compact eval prompts
- stronger hard-example ranking
- replay prioritization
- symbolic filtering before teacher data enters the corpus
- teacher-output rejection for generic voice
- more useful sovereignty audit heuristics
- inference/runtime efficiency
- clearer web dashboard panels

Risky but valid areas:

- modest model scale increases with a T4-fit plan
- smarter milestone triggers
- stronger replay weighting
- additional verified reasoning datasets
- better memory-to-training export

Changes that require evidence:

- changing the tokenizer
- moving heavy reflection into every daily run
- making teacher data dominant
- adding subsystems without stronger evals
- increasing architecture size without a restore/train story

## Definition Of Done

A good An-Ra patch should leave behind:

- source checks still `19/19 active`
- relevant tests or smoke checks run
- docs updated if public behavior changed
- no hidden dependency on missing local checkpoints
- no silent identity drift or teacher-capture risk

The point is not just to add capability. The point is to add capability that can be measured, restored, and kept under the project's authorship.
