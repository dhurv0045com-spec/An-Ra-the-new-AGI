"""Memory tier registrations for the anra package."""

from anra.core.registry import MEMORY_REGISTRY
from anra.memory.bm25 import BM25MemoryTier as _BM25MemoryTier


@MEMORY_REGISTRY.register(  # type: ignore[arg-type]
    "bm25",
    aliases=["bm25_exact", "keyword"],
)
class BM25(_BM25MemoryTier):
    """BM25 exact-match memory registered for config-driven use."""


__all__ = ["BM25"]
