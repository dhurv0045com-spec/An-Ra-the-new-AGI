"""Registers MemoryRouter with the memory registry."""

from __future__ import annotations

from anra.core.registry import MEMORY_REGISTRY
from memory.memory_router import MemoryRouter as _MemoryRouter


@MEMORY_REGISTRY.register("memory_router", aliases=["default"])
class MemoryRouter(_MemoryRouter):
    """MemoryRouter registered in MEMORY_REGISTRY."""


__all__ = ["MemoryRouter"]
