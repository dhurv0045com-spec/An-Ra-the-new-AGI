"""Unit and integration tests for MemoryRouter."""

from __future__ import annotations

import pytest

from memory.memory_router import MemoryRouter


@pytest.fixture
def router():
    """MemoryRouter with in-memory only - no FAISS, no disk."""
    try:
        return MemoryRouter(disable_faiss=True)
    except TypeError:
        return MemoryRouter()


def test_router_instantiates(router):
    assert router is not None


def test_write_returns_id(router):
    rid = router.write("The Eiffel Tower is in Paris", metadata={"type": "fact"})
    assert rid is not None
    assert isinstance(rid, str)


def test_read_returns_list(router):
    router.write("Python was created by Guido van Rossum", metadata={})
    results = router.read("Who created Python?", n=1)
    assert isinstance(results, list)


def test_health_returns_status(router):
    h = router.health()
    assert h is not None


def test_memory_router_registered():
    import anra
    from anra.core.registry import MEMORY_REGISTRY

    assert "memory_router" in MEMORY_REGISTRY
