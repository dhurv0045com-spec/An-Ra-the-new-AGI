"""Registers HALModule with the identity registry for config-driven instantiation."""

from __future__ import annotations

from anra.core.registry import IDENTITY_REGISTRY
from identity.hal import HALModule as _HALModule


@IDENTITY_REGISTRY.register("hal", aliases=["hormonal_analog_layer"])
class HALModule(_HALModule):
    """HAL registered in IDENTITY_REGISTRY. Same implementation, discoverable by name."""


__all__ = ["HALModule"]
