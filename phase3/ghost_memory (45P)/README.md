# 45P - Ghost Memory

**Layer 17/19: `ghost_memory`**

Ghost Memory stores compressed conversation traces and retrieves only the few relevant fragments that should enter the active prompt.

It is not "infinite context" in the careless sense. It is bounded prompt injection backed by an unbounded archive.

## Current Role

```text
conversation turns
  -> embedding / mock embedding
  -> compressed vector store
  -> decay and retrieval
  -> Ghost Context block
  -> generation runtime
```

## Main Files

| File | Role |
| --- | --- |
| `ghost_memory/config.py` | Defaults for storage, retrieval, decay, compression |
| `ghost_memory/quantizer.py` | TurboQuant-style vector compression |
| `ghost_memory/memory_store.py` | Persistence and `GhostMemory` facade |
| `ghost_memory/retriever.py` | Cosine retrieval and decay weighting |
| `ghost_memory/injector.py` | Ghost Context prompt construction |
| `ghost_memory/decay.py` | Half-life and pruning helpers |
| `ghost_memory/demo.py` | Compression, recall, and persistence checks |

## Quick Start

From this folder:

```bash
python -m ghost_memory.demo
```

Minimal use:

```python
from pathlib import Path
from ghost_memory import GhostMemory, default_config

cfg = default_config(storage_dir=Path.home() / ".anra_ghost_memory")
gm = GhostMemory(config=cfg)
gm.add_turn("user", "My project is called An-Ra.")
prompt = gm.build_ghost_prompt("What is my project called?")
```

The demo can use mock embeddings, so it does not need downloads. Real semantic embeddings are better when `sentence-transformers` is available locally.

## Best Memories

Ghost Memory is highest value for:

- failures
- corrections
- continuity breaks
- identity drift cases
- long-running project facts
- prompts that exposed weak reasoning

Those are the memories that should later feed replay and curriculum.
