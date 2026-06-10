"""Registers ESV with the identity registry."""

from __future__ import annotations

from anra.core.registry import IDENTITY_REGISTRY

try:
    from identity.esv import ESVModule as _ESVModule

    @IDENTITY_REGISTRY.register("esv", aliases=["emotional_state_vector"])
    class ESVModule(_ESVModule):
        """ESV registered in IDENTITY_REGISTRY for config-driven instantiation."""

except ImportError:
    ESVModule = None  # type: ignore[misc, assignment]

__all__ = ["ESVModule"]
