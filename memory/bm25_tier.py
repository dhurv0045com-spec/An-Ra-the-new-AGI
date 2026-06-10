"""
BM25 exact-match memory tier (public import path).

Complements FAISS semantic search with keyword retrieval.
Implementation: anra.memory.bm25 (registered as MEMORY_REGISTRY "bm25").
"""

from __future__ import annotations

from anra.memory.bm25 import BM25MemoryTier

__all__ = ["BM25MemoryTier"]
