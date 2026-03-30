# 45J — Memory System

Transforms the transformer from a stateless chatbot into a personal AI that
genuinely knows you. Every conversation grows its memory. Every prompt is
enriched with what it already knows.

## Architecture

```
MemoryManager (master entry point)
├── memory/
│   ├── store.py          SQLite persistence — all memory types
│   ├── vectors.py        TF-IDF + numpy vector index, <50ms retrieval
│   └── memory_types.py   Episodic / Semantic / Working typed interfaces
├── intelligence/
│   ├── extractor.py      Pattern + LLM fact extraction from conversations
│   ├── retrieval.py      Hybrid search: semantic + keyword + recency + importance
│   └── memory_intelligence.py  Importance scoring, consolidation, forgetting
└── knowledge/
    ├── graph.py           Personal knowledge graph (nodes, edges, reasoning)
    └── context_builder.py Pre-generation context assembly with token budget
```

## Three Memory Types

| Type     | What it stores                        | Expires        |
|----------|---------------------------------------|----------------|
| EPISODIC | Full conversations, specific events   | 6 months (low importance) |
| SEMANTIC | Extracted facts about the user        | Never (high importance)   |
| WORKING  | Current session state, active context | End of session |

## Quick Start

```python
from memory_manager import MemoryManager

mm = MemoryManager(data_dir="data/memory", user_id="alice")
mm.start_session()

# Every user message
mm.add_turn("user", "I'm a Rust developer working on a game engine called Vortex")
mm.add_turn("assistant", "Interesting — what renderer are you using?")

# Before every generation — inject memory context automatically
enriched_prompt = mm.prepare_prompt("What are the best ECS libraries for Rust?")
# → prompt now contains everything known about Alice + relevant memories

# After conversation
mm.process_conversation(turns)  # extracts facts, updates graph, persists
mm.cleanup()
```

## CLI

```bash
# Store a memory
python memory_manager.py --store "User prefers dark mode" --type semantic --importance high

# Retrieve
python memory_manager.py --retrieve "what does the user prefer" --limit 10

# Knowledge graph
python memory_manager.py --graph --export

# Consolidate (merge duplicates, compress old episodes)
python memory_manager.py --consolidate --confirm

# Stats
python memory_manager.py --stats

# Wipe everything (privacy)
python memory_manager.py --wipe --confirm

# End-to-end demo
python memory_manager.py --demo
```

## Integration with Inference Pipeline

The single integration point for 45K's agent loop:

```python
# 1. On each user turn:
mm.add_turn("user", user_message)          # extracts facts in real-time
enriched = mm.prepare_prompt(user_message)  # injects memory context

# 2. Feed enriched prompt to model — no other changes needed

# 3. Record response:
mm.add_turn("assistant", response)

# 4. At session end:
mm.process_conversation(session_turns)      # full extraction + graph update
```

## Performance

- Retrieval: ~10ms over 100 memories, ~50ms over 50k (approximate search)
- Extraction: ~2ms per turn (pattern-based), ~200ms with LLM extractor
- Storage: SQLite WAL mode — concurrent reads, durable writes
- Index: numpy float32 matrix — ~2MB per 1k memories

## Backends

Default: SQLite + numpy (zero dependencies beyond stdlib+numpy).

Swap to ChromaDB or Supabase pgvector without touching any other code:
```python
# In memory/backend/chroma.py (stub ready for implementation)
mm = MemoryManager(data_dir="...", backend="chroma")
```

## Tests

```bash
python test_45J.py   # 34/34 tests
```
