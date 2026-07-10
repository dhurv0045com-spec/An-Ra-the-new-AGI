"""Shared verification substrate."""

# Importing the package installs builtins once through decorator registration.
from verification import builtins as _builtins  # noqa: F401, E402
from verification import formal as _formal  # noqa: F401, E402
from verification.registry import (
    DEFAULT_VERIFIER_REGISTRY,
    DuplicateVerifierError,
    InvalidVerifierResultError,
    UnknownVerifierError,
    VerifierRegistry,
    VerifierRequest,
    register_verifier,
)

__all__ = [
    "DEFAULT_VERIFIER_REGISTRY",
    "DuplicateVerifierError",
    "InvalidVerifierResultError",
    "UnknownVerifierError",
    "VerifierRegistry",
    "VerifierRequest",
    "register_verifier",
]
