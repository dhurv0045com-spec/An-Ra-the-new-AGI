# 45P — Ghost State Memory (`phase3`)

Ghost State Memory is a small Python library that stores conversation turns as **compressed embedding vectors** on disk (SQLite metadata + NumPy-packed bytes). At query time it **retrieves** a few relevant lines by cosine similarity and **prepends** them as a Ghost Context block before the next user message. Nothing is billed as “memory tokens” until you choose to inject retrieved text into the active prompt.

## Install

```bash
cd phase3
pip install -r requirements.txt
```

Python **3.10+** is required.

## Run the demo

```bash
cd phase3
python -m ghost_memory.demo
```

The demo uses **mock embeddings** by default (no downloads). With `sentence-transformers` installed, the library can load `all-MiniLM-L6-v2` locally for real embeddings.

## Integrate in about ten lines

```python
from pathlib import Path
from ghost_memory import GhostMemory, default_config

cfg = default_config(storage_dir=Path.home() / ".my_bot_memory")
gm = GhostMemory(config=cfg)  # loads MiniLM when you embed

gm.add_turn("user", "Remember: my project is called An-Ra.")
gm.add_turn("assistant", "I'll remember that.")

prompt_for_model = gm.build_ghost_prompt("What is my project called?")
# Send `prompt_for_model` to your chat/completions API as the user-side content
# (or merge with your system prompt policy).
```

Tune retrieval with `cfg.top_k`, `cfg.similarity_thresh`, `cfg.decay_half_life_days`, and `cfg.max_memories` on the same `GhostConfig` object.

## Files in this package

| Module            | Role                                              |
|-------------------|---------------------------------------------------|
| `config.py`       | Defaults (bits, top-k, decay, paths)               |
| `quantizer.py`    | TurboQuant-style key/value compression            |
| `memory_store.py` | Thread-safe persistence + `GhostMemory` façade    |
| `retriever.py`    | Cosine search + decay weighting                   |
| `injector.py`     | Ghost Context string + `build_prompt`             |
| `decay.py`        | Half-life decay + pruning helpers                 |
| `demo.py`         | Compression + recall + persistence checks       |

## Notes

- **“Infinite context”** here means *unbounded archival with bounded active prompt*: old text stays on disk until pruned; only retrieved snippets enter the next prompt.
- **Compression** targets roughly **6×** smaller vectors than raw `float32` for 384-dimensional MiniLM-style embeddings, with reconstruction tuned for cosine retrieval.
