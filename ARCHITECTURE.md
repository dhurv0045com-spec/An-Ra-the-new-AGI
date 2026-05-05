# An-Ra Architecture

This is the current technical map of the repository. The source of truth is `runtime/system_registry.py`; regenerate the live manifest with:

```bash
python -m inference.full_system_connector
python scripts/status.py
python scripts/verify_structure.py
```

The current source registry is `19/19 active`.

## System Shape

```text
operator / API / notebook / web
  -> anra.py / app.py / generate.py
  -> tokenizer_v3 + anra_brain.py
  -> data mix + training loop + evaluation
  -> identity + memory + goals + agent loop
  -> Ouroboros + ghost memory + symbolic bridge
  -> self-improvement + sovereignty audit
  -> reports, checkpoints, dashboard, Drive/session state
```

## Canonical Layers

| # | Capability | Layer | Components |
| --- | --- | --- | --- |
| 01 | `brain` | model | `anra_brain.py`, `training/v2_config.py`, `training/v2_runtime.py` |
| 02 | `tokenizer` | data | `tokenizer/tokenizer_v3.json`, `tokenizer/tokenizer_adapter.py`, `scripts/train_tokenizer_v3.py` |
| 03 | `data_mix` | data | `training_data/anra_training.txt`, `training/v2_data_mix.py`, `scripts/setup_dataset.py` |
| 04 | `training_loop` | learning | `training/train_unified.py`, `scripts/build_brain.py`, `training/finetune_anra.py` |
| 05 | `evaluation` | measurement | `training/eval_v2.py`, `training/benchmark.py`, `training/verifier.py` |
| 06 | `inference_runtime` | serving | `generate.py`, `inference/full_system_connector.py`, `inference/anra_infer.py` |
| 07 | `api_web` | interface | `app.py`, `phase4/web/src/App.jsx`, `phase4/web/src/index.css` |
| 08 | `identity` | alignment | `identity/civ.py`, `identity/esv.py`, `identity/civ_watcher.py`, `phase3/identity (45N)/identity_injector.py` |
| 09 | `memory_router` | continuity | `memory/memory_router.py`, `memory/faiss_store.py` |
| 10 | `phase2_memory` | continuity | `phase2/memory (45J)/memory_manager.py`, `store.py`, `vectors.py`, `context_builder.py` |
| 11 | `goals` | agency | `goals/goal_queue.py`, `agents/orchestrator.py`, `agents/specialists.py` |
| 12 | `agent_loop` | agency | `phase2/agent_loop (45k)/agent_main.py`, `planner.py`, `executor.py`, `evaluator.py` |
| 13 | `master_system` | autonomy | `phase2/master_system (45M)/system.py`, `llm_bridge.py`, `autonomy/engine.py`, `control/control.py` |
| 14 | `self_improvement` | learning | `phase2/self_improvement (45l)/improve.py`, `self_improvement/engine.py`, `dashboard/dashboard.py` |
| 15 | `self_modification` | governance | `self_modification/type_a.py`, `self_modification/type_b.py`, `execution/sandbox.py`, `execution/fs_agent.py` |
| 16 | `ouroboros` | reflection | `phase3/ouroboros (45O)/ouroboros_numpy.py`, `adaptive.py`, `pass_gates.py` |
| 17 | `ghost_memory` | continuity | `phase3/ghost_memory (45P)/ghost_memory/memory_store.py`, `retriever.py`, `injector.py` |
| 18 | `symbolic_bridge` | verification | `phase3/symbolic_bridge (45Q)/symbolic_bridge.py`, `math_solver.py`, `logic_checker.py`, `code_verifier.py` |
| 19 | `sovereignty` | governance | `phase3/sovereignty (45R)/sovereignty_bridge.py`, `auditor.py`, `benchmarks.py`, `reporter.py` |

## Runtime Loop

```text
prompt
  -> tokenizer
  -> V2 brain
  -> generation runtime
  -> identity cleanup
  -> memory/ghost recall
  -> symbolic pre-check when applicable
  -> response
  -> memory write + failure/replay hooks
```

## Training Loop

```text
anra_training.txt
  -> owner-data-first bucket mixer
  -> base V2 training
  -> compact eval + benchmark/verifier
  -> hard-example report
  -> next-session curriculum
  -> milestone identity / Ouroboros / sovereignty when requested
```

## Architectural Rules

1. Keep path constants in `anra_paths.py`; avoid scattered Drive, tokenizer, dataset, or workspace literals.
2. Keep component truth in `runtime/system_registry.py`; do not hand-maintain source counts in prose.
3. Prefer canonical modules before historical phase folders: `generate.py`, `memory_router`, `goal_queue`, and `training.train_unified`.
4. Treat `phase2/` and `phase3/` as capability layers that feed the mainline, not as separate products.
5. Every upgrade must improve at least one measurable axis: learning, identity stability, runtime reliability, verification, autonomy, or repair from failures.

## Current Caveat

`19/19 active` means all registered source layers exist. It does not mean local trained checkpoints are present. On a fresh or Termux clone, `anra_v2_brain.pt`, `anra_v2_identity.pt`, and `anra_v2_ouroboros.pt` may be missing until restored from Drive or produced by training.
