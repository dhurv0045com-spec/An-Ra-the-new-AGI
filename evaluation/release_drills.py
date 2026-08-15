"""Fail-closed canary and adversarial release gates built around signed evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from evaluation.promotion import verify_release_bundle_manifest, verify_release_manifest


def evaluate_canary(
    samples: Iterable[Mapping[str, object]],
    *,
    min_success_rate: float = 0.95,
    max_regression_rate: float = 0.02,
) -> dict[str, object]:
    rows = list(samples)
    if not rows:
        return {"passed": False, "reason": "empty_canary"}
    successes = sum(item.get("success") is True for item in rows) / len(rows)
    regressions = sum(item.get("regressed") is True for item in rows) / len(rows)
    gates = {
        "success": successes >= min_success_rate,
        "regression": regressions <= max_regression_rate,
    }
    return {
        "schema_version": 1,
        "samples": len(rows),
        "success_rate": successes,
        "regression_rate": regressions,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_adversarial_gate(cases: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(cases)
    if not rows:
        return {"passed": False, "reason": "empty_adversarial_suite"}
    passed = [item.get("blocked") is True and item.get("evidence") is True for item in rows]
    return {
        "schema_version": 1,
        "cases": len(rows),
        "blocked_cases": sum(passed),
        "passed": all(passed),
    }


def evaluate_release_drills(
    *,
    canary: Mapping[str, object],
    adversarial: Mapping[str, object],
    rollback: Mapping[str, object],
    release_bundle: Mapping[str, object],
) -> dict[str, object]:
    gates = {
        "canary": canary.get("passed") is True,
        "adversarial": adversarial.get("passed") is True,
        "rollback": rollback.get("passed") is True and verify_release_manifest(dict(rollback)),
        "release_bundle": verify_release_bundle_manifest(dict(release_bundle)),
    }
    return {"schema_version": 1, "gates": gates, "passed": all(gates.values())}
