# An-Ra Architecture

This is the current technical map of the repo. The source of truth for the live component list is:

- `runtime/system_registry.py`
- `system_graph.json`, regenerated with `python -m inference.full_system_connector`
- `scripts/status.py`, for a quick local health and artifact view

## Canonical Layers

| Layer | Components | Purpose |
| --- | --- | --- |
| Model | `anra_brain.py`, `training/v2_config.py`, `training/v2_runtime.py` | V2 transformer core, model construction, checkpoint compatibility |
| Data | `training_data/anra_training.txt`, `tokenizer/tokenizer_v3.json`, `training/v2_data_mix.py` | Canonical corpus, BPE tokenizer, owner-data-first training mix |
| Learning | `scripts/build_brain.py`, `training/train_unified.py`, `training/benchmark.py`, `training/verifier.py` | Daily training, milestone runs, model-running benchmark, verification |
| Identity | `identity/civ.py`, `identity/esv.py` | Residual-stream identity guard and affective modulation |
| Memory | `memory/memory_router.py`, `memory/faiss_store.py` | Episodic, short-term, graph, ghost, and ESV-thresholded memory writes |
| Agency | `goals/goal_queue.py`, `agents/orchestrator.py`, `agents/specialists.py` | Goal persistence, specialist dispatch, completion/failure accounting |
| Runtime | `generate.py`, `app.py`, `scripts/session_dashboard.py`, `startup_checks.py` | Inference, API, session dashboard, GPU startup requirements |
| Governance | `execution/`, `self_modification/`, `runtime/drive_session_manager.py` | Sandboxed execution, atomic patching, Drive/session continuity |
| Phase Tools | `phase3/` | Symbolic reasoning, Ouroboros reflection, ghost memory, sovereignty audit |

## Main Runtime Loop

```text
user/API/notebook
  -> generate.py / training.train_unified
  -> anra_brain.py
  -> tokenizer_v3 + checkpoints
  -> memory router + goal queue + verifier
  -> reports, dashboard, audit, Drive/session state
```

## Training Loop

```text
anra_training.txt
  -> v2_data_mix
  -> build_brain.py
  -> eval_v2 + BenchmarkSuite
  -> hard examples + curriculum report
  -> milestone identity / Ouroboros / sovereignty when due
```

## Architectural Rules

1. Keep path constants in `anra_paths.py`; do not scatter Drive, tokenizer, dataset, or workspace literals.
2. Keep component truth in `runtime/system_registry.py`; do not hand-maintain stale file counts.
3. Prefer canonical modules (`memory_router`, `goal_queue`, `generate.py`) before reaching into historical phase folders.
4. Treat `phase2/` and `phase3/` as capability layers, not the primary source of runtime truth.
5. Every architectural upgrade should improve at least one of: measurable learning, identity stability, runtime reliability, or repair from failures.

## Regenerating The System Graph

```bash
python -m inference.full_system_connector
python scripts/status.py
python scripts/verify_structure.py
```

`system_graph.json` records live source metrics and component status. Runtime import status may be degraded in lightweight shells without `torch`, `numpy`, or optional symbolic packages; source capability status is reported separately.
