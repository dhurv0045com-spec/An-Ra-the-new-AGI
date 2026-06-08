"""Registers HAL with the identity registry."""

from __future__ import annotations

from anra.core.registry import IDENTITY_REGISTRY
from identity.hal import HALModule as _HALModule


@IDENTITY_REGISTRY.register("hal", aliases=["default"])
class HALModule(_HALModule):
    """HAL module registered in IDENTITY_REGISTRY."""


__all__ = ["HALModule"]
