# An-Ra Architecture

**Source of truth:** `runtime/system_registry.py`

If this doc disagrees with the registry, the registry wins. Regenerate the live manifest when in doubt:

```bash
python -m inference.full_system_connector
python scripts/status.py
python scripts/readiness.py
python scripts/verify_structure.py
```

Current registry: **19/19 active** (source present — checkpoints may still be missing on a fresh clone).

---

## Research foundation (why the shape looks like this)

Three design lines feed the implementation:

| Line | Role |
| --- | --- |
| **DFC** — Differential Falsification Cognition | Training grammar: state → constraint → hypothesis → check → update |
| **FCC** — Falsifiable Constraint Cognition | Six training templates in `frontier_dfc.jsonl` |
| **AIE** — An-Ra Innovation Engine | Self-improvement as measured experiment |

**Where that lives in code:**

| Concept | Implementation |
| --- | --- |
| HAL nervous system | `identity/hal.py` |
| Falsification ledger | `identity/falsification_ledger.py` |
| Proof graph | `memory/experimental_proof_graph.py` |
| Constraint isomorphism | `identity/constraint_isomorphism_search.py` |
| AIE scoring | `engine/eval_harness.py` |
| Verifier-shaped RL | `training/rlvr.py` (GRPO) |
| Self-taught reasoning | `training/star.py` |
| DFC corpus | `training_data/frontier_dfc.jsonl` |

---

## System shape (one picture)

```text
operator / API / notebook / web
  → anra.py · app.py · generate.py
  → tokenizer_v3 + anra_brain.py
  → data mix + training + evaluation
  → identity + memory + goals + agent_loop
  → Ouroboros + ghost + symbolic bridge
  → self-improvement + sovereignty
  → reports · checkpoints · Drive/session state
```

---

## Canonical layers (19 components)

| # | ID | Layer | Primary paths |
| --- | --- | --- | --- |
| 01 | `brain` | model | `anra_brain.py`, `training/v2_config.py`, `training/v2_runtime.py` |
| 02 | `tokenizer` | data | `tokenizer/tokenizer_v3.json`, `tokenizer/tokenizer_adapter.py` |
| 03 | `data_mix` | data | `training_data/anra_training.txt`, `training/v2_data_mix.py` |
| 04 | `training_loop` | learning | `training/train_unified.py`, `scripts/build_brain.py` |
| 05 | `evaluation` | measurement | `training/eval_v2.py`, `training/benchmark.py`, `training/verifier.py` |
| 06 | `inference_runtime` | serving | `generate.py`, `inference/full_system_connector.py` |
| 07 | `api_web` | interface | `app.py`, `phase4/web/src/App.jsx` |
| 08 | `identity` | alignment | `identity/civ.py`, `esv.py`, `hal.py`, `phase3/identity (45N)/` |
| 09 | `memory_router` | continuity | `memory/memory_router.py`, `memory/faiss_store.py` |
| 10 | `phase2_memory` | continuity | `phase2/memory (45J)/` |
| 11 | `goals` | agency | `goals/goal_queue.py`, `agents/orchestrator.py` |
| 12 | `agent_loop` | agency | `phase2/agent_loop (45k)/` |
| 13 | `master_system` | autonomy | `phase2/master_system (45M)/system.py` |
| 14 | `self_improvement` | learning | `phase2/self_improvement (45l)/` |
| 15 | `self_modification` | governance | `self_modification/`, `execution/sandbox.py` |
| 16 | `ouroboros` | reflection | `phase3/ouroboros (45O)/` |
| 17 | `ghost_memory` | continuity | `phase3/ghost_memory (45P)/` |
| 18 | `symbolic_bridge` | verification | `phase3/symbolic_bridge (45Q)/` |
| 19 | `sovereignty` | governance | `phase3/sovereignty (45R)/` |

---

## Runtime loop (inference)

```text
prompt
  → tokenizer
  → V2 brain
  → generation runtime
  → identity cleanup
  → memory / ghost recall
  → symbolic pre-check (when applicable)
  → response
  → memory write + failure/replay hooks
```

---

## Training loop

```text
anra_training.txt
  → owner-first bucket mixer (65/15/10/5/5)
  → V2 training
  → eval + benchmark + verifier
  → hard-example report
  → next-session curriculum
  → [milestone] identity · Ouroboros · sovereignty
```

---

## Engineering spine (cross-cutting)

```text
registry → feature_flags → telemetry → eval_harness → report
```

Every new capability should plug into that chain or justify why it cannot.

---

## Architectural rules (non-negotiable)

1. **Paths** live in `anra_paths.py` — no scattered `training_data/` literals.
2. **Component truth** lives in `runtime/system_registry.py` — no hand-counted "we have N modules" in prose.
3. **Prefer mainline** over historical phase folders: `generate.py`, `memory_router`, `goal_queue`, `training.train_unified`.
4. **`phase2/` and `phase3/`** are capability layers feeding the mainline — not separate products.
5. **Every upgrade** must move at least one measurable axis: learning, identity stability, reliability, verification, autonomy, or failure repair.

---

## Caveat (read this once)

`19/19 active` = registered source layers exist and import.

It does **not** guarantee `anra_v2_brain.pt`, `anra_v2_identity.pt`, or `anra_v2_ouroboros.pt` on disk. Fresh clones need Drive restore or training.

---

## Deeper reads

| Doc | When |
| --- | --- |
| [`WALKTHROUGH.md`](WALKTHROUGH.md) | Subsystem-by-subsystem deep dive |
| [`phase3/PHASE3_INTEGRATION.md`](phase3/PHASE3_INTEGRATION.md) | Phase 3 wiring |
| [`DEVELOPER.md`](DEVELOPER.md) | Contribution rules |
