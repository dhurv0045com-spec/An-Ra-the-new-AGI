# 45J — Memory System

**Component 10/19 · `phase2_memory`**

Typed memory under the unified router: episodic turns, semantic facts, working context, vectors, personal graph, token-budgeted prompt assembly.

**Mainline entry:** `memory/memory_router.py`  
**Deep implementation:** this folder.

---

## Stack position

```text
memory_router
  → 45J typed store
  → vector retrieval
  → graph context
  → context_builder
  → enriched prompt
```

45J's job: keep what matters across sessions without stuffing the active window with noise. **Failures, corrections, and contradictions** should become retrievable evidence — not chat log landfill.

---

## Files

| File | Role |
| --- | --- |
| `memory_manager.py` | Main API |
| `store.py` | SQLite persistence |
| `vectors.py` | NumPy / TF-IDF index |
| `memory_types.py` | Episodic / semantic / working |
| `extractor.py` | Facts from turns |
| `retrieval.py` | Hybrid search |
| `memory_intelligence.py` | Score, consolidate, forget |
| `graph.py` | Personal knowledge graph |
| `context_builder.py` | Budget-aware context |

---

## Memory types

| Type | Holds | Lifetime |
| --- | --- | --- |
| Episodic | Events, conversations | Pruned by age + importance |
| Semantic | Stable facts, prefs | Long when salient |
| Working | Current session | Session-scoped |

---

## Minimal example

```python
from memory_manager import MemoryManager

mm = MemoryManager(data_dir="data/memory", user_id="owner")
mm.start_session()
mm.add_turn("user", "My project is called An-Ra.")
mm.add_turn("assistant", "Noted.")
prompt = mm.prepare_prompt("What is my project?")
mm.process_conversation([])
mm.cleanup()
```

**Smoke:**

```bash
python test_45J.py
python memory_manager.py --stats
```

---

## Integration note

- **Shipping feature?** Wire through `memory/memory_router.py`.
- **Debugging retrieval/graph?** Use 45J directly.

HAL salience gates what reaches episodic storage upstream — see `memory/memory_router.py` and identity docs in root `WALKTHROUGH.md`.
