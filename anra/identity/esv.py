"""Registers ESV with the identity registry."""

from __future__ import annotations

from anra.core.registry import IDENTITY_REGISTRY
from identity.esv import ESVModule as _ESVModule


@IDENTITY_REGISTRY.register("esv", aliases=["emotional_state_vector"])
class ESVModule(_ESVModule):
    """ESV registered for config-driven instantiation."""


__all__ = ["ESVModule"]
