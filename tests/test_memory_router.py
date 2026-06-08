"""Tests for anra.memory.router - MemoryRouter interface."""

from __future__ import annotations

from anra.memory.router import MemoryRouter


def test_memory_router_instantiable() -> None:
    router = MemoryRouter()
    assert router is not None


def test_memory_router_is_type() -> None:
    assert isinstance(MemoryRouter, type)


def test_memory_router_canonical_import() -> None:
    """Canonical path: anra.memory.router, not memory.router directly."""
    from anra.memory.router import MemoryRouter  # noqa: F401


def test_memory_registry_exists() -> None:
    from anra.core.registry import MEMORY_REGISTRY

    assert MEMORY_REGISTRY is not None


def test_memory_router_all_export() -> None:
    import anra.memory.router as m

    assert hasattr(m, "MemoryRouter")
