from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from training.verifier import VerifierHierarchy
from agents.specialists import CriticAgent
from verification.registry import (
    DEFAULT_VERIFIER_REGISTRY,
    DuplicateVerifierError,
    InvalidVerifierResultError,
    UnknownVerifierError,
    VerifierRegistry,
    VerifierRequest,
    register_verifier,
)


@dataclass
class Result:
    score: float
    tier: int
    reason: str


def test_registration_dispatch_alias_and_discovery() -> None:
    registry = VerifierRegistry()

    @register_verifier("arithmetic", aliases=("math",), registry=registry)
    def arithmetic(request: VerifierRequest) -> Result:
        return Result(1.0, 1, f"checked_{request.payload['value']}")

    result = registry.verify("MATH", {"value": 4})
    assert result.reason == "checked_4"
    assert registry.describe()["arithmetic"]["aliases"] == ["math"]
    assert "math" in registry


def test_duplicate_and_unknown_names_fail_closed() -> None:
    registry = VerifierRegistry()
    registry.register("math", lambda _request: Result(1.0, 1, "ok"))
    with pytest.raises(DuplicateVerifierError):
        registry.register("math", lambda _request: Result(1.0, 1, "again"))
    with pytest.raises(UnknownVerifierError):
        registry.verify("missing", {})


def test_verifier_exception_is_recorded_before_reraise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runtime.experience_ledger as experience_ledger
    from runtime.experience_ledger import ExperienceLedger

    ledger = ExperienceLedger(tmp_path, strict=True)
    monkeypatch.setattr(experience_ledger, "_DEFAULT_LEDGER", ledger)
    registry = VerifierRegistry()

    def explode(_request: VerifierRequest) -> Result:
        raise RuntimeError("broken verifier")

    registry.register("broken", explode)
    with pytest.raises(RuntimeError, match="broken verifier"):
        registry.verify("broken", {"claim": "x"})
    event = list(ledger.iter_events())[0]
    assert event["verifier_verdicts"][0]["passed"] is False
    assert event["verifier_verdicts"][0]["reason"] == "verifier_error:RuntimeError"


@pytest.mark.parametrize(
    "result",
    [Result(-0.1, 1, "low"), Result(1.1, 1, "high"), Result(1.0, 0, "tier"), Result(1.0, 1, "")],
)
def test_conformance_rejects_invalid_results(result: Result) -> None:
    registry = VerifierRegistry()
    registry.register("bad", lambda _request: result)
    with pytest.raises(InvalidVerifierResultError):
        registry.verify("bad", {})


def test_existing_hierarchy_routes_through_shared_registry(tmp_path) -> None:
    hierarchy = VerifierHierarchy(tmp_path)
    assert "math" in DEFAULT_VERIFIER_REGISTRY
    assert "code" in DEFAULT_VERIFIER_REGISTRY
    result = hierarchy.score("math", expression="6 * 7", expected="42")
    assert result.score == 1.0
    assert result.reason == "equivalent"


def test_builtin_math_dispatches_without_compatibility_facade() -> None:
    result = DEFAULT_VERIFIER_REGISTRY.verify(
        "math", {"expression": "9 * 9", "expected": "81"}
    )
    assert result.score == 1.0
    assert result.reason == "equivalent"


def test_critic_agent_is_a_direct_registry_consumer(tmp_path) -> None:
    critic = CriticAgent(
        "critic",
        None,
        None,
        None,
        None,
        None,
        None,
        verifier=VerifierHierarchy(tmp_path),
    )
    result = asyncio.run(
        critic.run({"task_type": "math", "expression": "10 - 3", "expected": "7"})
    )
    assert result["approved"] is True
    assert result["reason"] == "equivalent"
