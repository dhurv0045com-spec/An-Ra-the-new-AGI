"""Registers CIV with the identity registry."""

from __future__ import annotations

from anra.core.registry import IDENTITY_REGISTRY
from identity.civ import ConstitutionalIdentityVector as _CIVVector


@IDENTITY_REGISTRY.register("civ", aliases=["constitutional_identity_vector"])
class CIVVector(_CIVVector):
    """CIV registered for config-driven instantiation."""


__all__ = ["CIVVector"]
