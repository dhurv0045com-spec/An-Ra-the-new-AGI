"""The process-wide verifier registry used by every verifier consumer."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable


class DuplicateVerifierError(ValueError):
    pass


class UnknownVerifierError(KeyError):
    pass


class InvalidVerifierResultError(TypeError):
    pass


@dataclass(frozen=True)
class VerifierRequest:
    name: str
    payload: Mapping[str, Any]
    context: object | None = None


@runtime_checkable
class VerifierResult(Protocol):
    score: float
    tier: int
    reason: str


VerifierHandler = Callable[[VerifierRequest], VerifierResult]


class VerifierRegistry:
    """Thread-safe named dispatch with aliases and result conformance checks."""

    def __init__(self) -> None:
        self._handlers: dict[str, VerifierHandler] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("verifier name cannot be empty")
        return normalized

    def register(
        self,
        name: str,
        handler: VerifierHandler,
        *,
        aliases: Iterable[str] = (),
    ) -> VerifierHandler:
        canonical = self._normalize(name)
        normalized_aliases = tuple(self._normalize(alias) for alias in aliases)
        with self._lock:
            occupied = {canonical, *normalized_aliases}
            duplicates = occupied.intersection(self._handlers.keys() | self._aliases.keys())
            if duplicates:
                raise DuplicateVerifierError(
                    f"verifier names already registered: {', '.join(sorted(duplicates))}"
                )
            self._handlers[canonical] = handler
            for alias in normalized_aliases:
                self._aliases[alias] = canonical
        return handler

    def register_many(self, names: Iterable[str], handler: VerifierHandler) -> None:
        for name in names:
            self.register(name, handler)

    def resolve(self, name: str) -> tuple[str, VerifierHandler]:
        normalized = self._normalize(name)
        with self._lock:
            canonical = self._aliases.get(normalized, normalized)
            handler = self._handlers.get(canonical)
        if handler is None:
            raise UnknownVerifierError(normalized)
        return canonical, handler

    def verify(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        context: object | None = None,
    ) -> VerifierResult:
        canonical, handler = self.resolve(name)
        try:
            result = handler(VerifierRequest(canonical, dict(payload), context))
            self.assert_conforms(result, verifier=canonical)
        except Exception as exc:
            self._record_outcome(
                canonical,
                payload,
                score=0.0,
                tier=1,
                reason=f"verifier_error:{type(exc).__name__}",
                output={"error": str(exc)},
            )
            raise
        self._record_outcome(
            canonical,
            payload,
            score=float(result.score),
            tier=int(result.tier),
            reason=str(result.reason),
            output={
                "reason": str(result.reason),
                "stdout": str(getattr(result, "stdout", "")),
                "stderr": str(getattr(result, "stderr", "")),
                "return_code": int(getattr(result, "return_code", 0)),
            },
        )
        return result

    @staticmethod
    def _record_outcome(
        canonical: str,
        payload: Mapping[str, Any],
        *,
        score: float,
        tier: int,
        reason: str,
        output: Mapping[str, Any],
    ) -> None:
        try:
            from runtime.experience_ledger import record_experience

            record_experience(
                kind="verifier",
                inputs={"verifier": canonical, "payload": payload},
                output=dict(output),
                verifier_verdicts=[
                    {
                        "name": canonical,
                        "score": score,
                        "passed": score >= 0.8,
                        "tier": tier,
                        "reason": reason,
                    }
                ],
                gate_record={"allowed": score >= 0.8, "gate": "verifier_threshold"},
                source="verification.registry",
            )
        except Exception:
            # Verification remains authoritative when telemetry storage is degraded.
            pass

    @staticmethod
    def assert_conforms(result: object, *, verifier: str = "unknown") -> None:
        if not all(hasattr(result, field) for field in ("score", "tier", "reason")):
            raise InvalidVerifierResultError(
                f"{verifier} must return score, tier, and reason fields"
            )
        conforming = cast(VerifierResult, result)
        score = float(conforming.score)
        if not 0.0 <= score <= 1.0:
            raise InvalidVerifierResultError(f"{verifier} score outside [0, 1]: {score}")
        tier = conforming.tier
        if isinstance(tier, bool) or int(tier) < 1:
            raise InvalidVerifierResultError(f"{verifier} tier must be a positive integer")
        if not str(conforming.reason).strip():
            raise InvalidVerifierResultError(f"{verifier} reason cannot be empty")

    def describe(self) -> dict[str, dict[str, object]]:
        with self._lock:
            aliases_by_target: dict[str, list[str]] = {}
            for alias, target in self._aliases.items():
                aliases_by_target.setdefault(target, []).append(alias)
            return {
                name: {
                    "handler": f"{handler.__module__}:{handler.__qualname__}",
                    "aliases": sorted(aliases_by_target.get(name, [])),
                }
                for name, handler in sorted(self._handlers.items())
            }

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            self.resolve(name)
        except (KeyError, ValueError):
            return False
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._handlers)


DEFAULT_VERIFIER_REGISTRY = VerifierRegistry()


def register_verifier(
    name: str,
    *,
    aliases: Iterable[str] = (),
    registry: VerifierRegistry = DEFAULT_VERIFIER_REGISTRY,
) -> Callable[[VerifierHandler], VerifierHandler]:
    """Decorator for verifier modules; registration happens at import time."""

    def decorator(handler: VerifierHandler) -> VerifierHandler:
        return registry.register(name, handler, aliases=aliases)

    return decorator
