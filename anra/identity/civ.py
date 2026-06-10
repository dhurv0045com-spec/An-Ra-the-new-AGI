"""Registers CIV with the identity registry."""

from __future__ import annotations

from anra.core.registry import IDENTITY_REGISTRY

try:
    from identity.civ import ConstitutionalIdentityVector as _CIVVector

    @IDENTITY_REGISTRY.register("civ", aliases=["constitutional_identity_vector"])
    class CIVVector(_CIVVector):
        """CIV registered in IDENTITY_REGISTRY."""

except ImportError:
    CIVVector = None  # type: ignore[misc, assignment]

__all__ = ["CIVVector"]
