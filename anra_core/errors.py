"""Typed error envelopes and machine-readable failure taxonomy for An-Ra Core."""

from __future__ import annotations

from typing import Any


class CoreError(ValueError):
    """Base exception for all An-Ra Core failures (inherits from ValueError for broad compatibility)."""

    def __init__(self, message: str, *, error_code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class CheckpointIncompatibleError(CoreError):
    """Raised when a checkpoint payload or schema does not match the model architecture."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_CHECKPOINT_INCOMPATIBLE", details=details)


class RepresentationIncompatibleError(CoreError):
    """Raised when a tokenizer or token stream does not match the required vocabulary contract."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_REPRESENTATION_INCOMPATIBLE", details=details)


class UnsupportedProfileError(CoreError):
    """Raised when an execution profile, device, or precision is not supported."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_UNSUPPORTED_PROFILE", details=details)


class UnsupportedCapabilityError(CoreError):
    """Raised when an optional capability (e.g. quantization, state serialization) is requested but unavailable."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_UNSUPPORTED_CAPABILITY", details=details)


class ContextOverflowError(CoreError):
    """Raised when sequence length exceeds the configured context window limit."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_CONTEXT_OVERFLOW", details=details)


class StateIncompatibleError(CoreError):
    """Raised when an execution state handle is incompatible with the executor, model, or checkpoint."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_STATE_INCOMPATIBLE", details=details)


class StateReleasedError(CoreError):
    """Raised when an operation is attempted on an already released execution state."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_STATE_RELEASED", details=details)


class ResourceExhaustionError(CoreError):
    """Raised when GPU/CPU memory or compute allocation is exhausted."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_RESOURCE_EXHAUSTION", details=details)


class UnexpectedExecutionFault(CoreError):
    """Raised when an unrecoverable internal execution fault occurs."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code="ERR_UNEXPECTED_EXECUTION_FAULT", details=details)
