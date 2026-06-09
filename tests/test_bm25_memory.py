from __future__ import annotations

import pytest
from memory.bm25_tier import BM25MemoryTier


@pytest.fixture
def store() -> BM25MemoryTier:
    return BM25MemoryTier()


def test_write_and_exact_read(store: BM25MemoryTier) -> None:
    store.write("The capital of France is Paris", {"type": "fact"})
    result = store.read("capital France", n=1)
    assert len(result) == 1
    assert "Paris" in result[0].content


def test_empty_returns_empty(store: BM25MemoryTier) -> None:
    assert store.read("anything") == []


def test_ranks_relevant_higher(store: BM25MemoryTier) -> None:
    store.write("Python is a programming language", {})
    store.write("The weather is sunny today", {})
    result = store.read("Python programming", n=2)
    assert "Python is a programming" in result[0].content


def test_delete_removes_doc(store: BM25MemoryTier) -> None:
    record_id = store.write("delete me", {})
    store.delete(record_id)
    assert not store.read("delete me")


def test_health_tracks_count(store: BM25MemoryTier) -> None:
    store.write("one", {})
    store.write("two", {})
    health = store.health()
    assert health.healthy
    assert health.details["doc_count"] == 2


def test_bm25_registered() -> None:
    import anra

    assert "bm25" in anra.MEMORY_REGISTRY
