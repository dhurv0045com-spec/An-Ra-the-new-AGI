# 45P — Ghost Memory

**Component 17/19 · `ghost_memory`**

Context windows are finite. Conversations are not. Ghost memory **compresses** past turns, **decays** stale state, **retrieves** what still matters, and **injects** it back into the active prompt.

Think: long-term conversational continuity without sending the entire chat history every turn.

---

## Pipeline

```text
turn ends
  → compress (summary + vector + metadata)
  → store (JSONL / sqlite per config)
  → decay older entries
new turn
  → retriever scores relevance
  → injector adds fragments to context
```

---

## Key files

| File | Role |
| --- | --- |
| `ghost_memory/memory_store.py` | Persistence |
| `ghost_memory/retriever.py` | Search + rank |
| `ghost_memory/injector.py` | Prompt injection |
| `ghost_memory/quantizer.py` | Compression |
| `ghost_memory/decay.py` | Time-based fade |
| `ghost_memory/config.py` | Limits and paths |

---

## Operator hooks

```bash
python anra.py --phase3-status    # includes ghost health
```

Orchestrator kind `ghost` → this component. Toggle: `set_flag("ghost_memory", False)`.

**Offline OK:** mock embeddings work without downloading models; real embedders optional.

---

## Boundaries

| Ghost does | Ghost does not |
| --- | --- |
| Long-horizon **conversation** recall | Replace episodic FAISS (`memory_router`) |
| Compress and inject | Store raw secrets without salience policy |
| Decay low-value ghosts | Guarantee factual correctness (pair with 45Q) |

Pair with **45N** identity and **45J** typed memory for full continuity stack.

---

## Smoke

```bash
cd "phase3/ghost_memory (45P)"
python -m ghost_memory.demo
```

Health: `ghost_memory.health_check()` when imported from integrated runtime.
