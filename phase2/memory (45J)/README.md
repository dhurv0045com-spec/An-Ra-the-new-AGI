# 45J - Memory System

**Layer 10/19: `phase2_memory`**

45J is the typed memory layer: episodic turns, semantic facts, working context, vector retrieval, and personal graph structure. It is the detailed memory implementation behind the newer unified `memory_router` layer.

## Current Role

```text
memory_router
  -> 45J typed memory
  -> vector retrieval
  -> graph context
  -> context_builder
  -> prompt enrichment
```

45J should preserve what matters across sessions and make future prompts more informed without stuffing the active context with irrelevant history.

## Main Files

| File | Purpose |
| --- | --- |
| `memory_manager.py` | Main 45J entry point |
| `store.py` | SQLite persistence |
| `vectors.py` | NumPy/TF-IDF vector index |
| `memory_types.py` | Episodic, semantic, and working memory interfaces |
| `extractor.py` | Fact extraction from conversation turns |
| `retrieval.py` | Hybrid retrieval by semantic match, keyword, recency, and importance |
| `memory_intelligence.py` | Scoring, consolidation, forgetting |
| `graph.py` | Personal knowledge graph |
| `context_builder.py` | Token-budgeted context assembly |

## Memory Types

| Type | Stores | Lifetime |
| --- | --- | --- |
| Episodic | Conversations and specific events | Pruned by age and importance |
| Semantic | Stable facts and preferences | Long-lived when important |
| Working | Current session state | Session-scoped |

## Quick Start

```python
from memory_manager import MemoryManager

mm = MemoryManager(data_dir="data/memory", user_id="owner")
mm.start_session()
mm.add_turn("user", "My project is called An-Ra.")
mm.add_turn("assistant", "I will keep that in context.")
prompt = mm.prepare_prompt("What is my project?")
mm.process_conversation([])
mm.cleanup()
```

From this folder:

```bash
python test_45J.py
python memory_manager.py --stats
```

## Mainline Boundary

Use `memory/memory_router.py` when integrating memory into the current system. Use 45J directly when you need the lower-level typed store, graph, or retrieval details.

45J is not only "chat history." Its best use is repair: failures, corrections, contradictions, and continuity breaks should become retrievable evidence for future learning.
