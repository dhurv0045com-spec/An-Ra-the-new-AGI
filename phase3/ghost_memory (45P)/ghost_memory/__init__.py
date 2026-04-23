"""
45P — Ghost State Memory: compressed persistent embeddings for long-horizon chat.

Public exports are limited to the façade, configuration, and inject helpers so
callers do not depend on internal storage details.
"""

from .config import (
    DECAY_HALF_LIFE_DAYS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    KEY_QUANTIZE_BITS,
    MAX_MEMORIES,
    QUANTIZE_BITS,
    SIMILARITY_THRESH,
    TOP_K,
    GhostConfig,
    default_config,
)
from .injector import build_prompt, format_ghost_block, strip_ghost_for_display
from .memory_store import GhostMemory, MemoryStore

__all__ = [
    "GhostMemory",
    "MemoryStore",
    "GhostConfig",
    "default_config",
    "build_prompt",
    "format_ghost_block",
    "strip_ghost_for_display",
    "QUANTIZE_BITS",
    "KEY_QUANTIZE_BITS",
    "TOP_K",
    "DECAY_HALF_LIFE_DAYS",
    "MAX_MEMORIES",
    "SIMILARITY_THRESH",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
]


def health_check() -> dict:
    try:
        cfg = default_config()
        return {"status": "ok", "embedding_dim": cfg.embedding_dim, "max_memories": cfg.max_memories}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}
