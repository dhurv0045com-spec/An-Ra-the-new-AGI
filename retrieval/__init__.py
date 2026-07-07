"""Shared retrieval substrate for memory, agents, data, and grounding."""

from retrieval.adapters import (
    BM25RetrieverAdapter,
    SkillLibraryRetrieverAdapter,
    VectorRetrieverAdapter,
)
from retrieval.corpus import CorpusDedupIndex, DuplicateDecision
from retrieval.hybrid import HybridRetriever
from retrieval.protocols import RetrievalHit, RetrievalProvenance, RetrievalQuery, Retriever

__all__ = [
    "BM25RetrieverAdapter",
    "CorpusDedupIndex",
    "DuplicateDecision",
    "HybridRetriever",
    "RetrievalHit",
    "RetrievalProvenance",
    "RetrievalQuery",
    "Retriever",
    "SkillLibraryRetrieverAdapter",
    "VectorRetrieverAdapter",
]
