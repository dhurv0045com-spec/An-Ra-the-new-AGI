"""Cognition generator qualification against shortcut baselines (M27-M29).

A generator qualifies family by family: no inexpensive heuristic may beat
chance by more than the frozen excess on that family's development cases.
Consumes the E0 development certificate (real heuristic measurements, not
fresh runs) bound to generator source identities. Verdicts:
GENERATOR_QUALIFIED / GENERATOR_NOT_QUALIFIED. This run qualifies two
families to prove the lab: entity_value_binding (simple) and rule_induction
(held-out structures, harder). It never manufactures training data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


QUALIFICATION_SCHEMA = "anra-v5-generator-qualification/v1"
MAX_SHORTCUT_EXCESS = 0.10

# Shortcut set mirrors the E0 red-team policy: surface heuristics that could
# solve without the target computation. Excluded by construction: the
# full-truth oracle (ceiling control, scores 1.0 by design),
# deterministic_random (it DEFINES the chance proxy), and the
# direct-retrieval control (a matched diagnostic, not a shortcut; the E0
# framework contrasts against it rather than gating on it).
SHORTCUT_BASELINES = frozenset(
    {
        "bag_of_words",
        "lexical_overlap",
        "fixed_identity_rule",
        "fixed_repeat_left_rule",
        "fixed_repeat_right_rule",
        "fixed_reverse_rule",
        "first_candidate",
        "last_candidate",
        "nearest_position",
        "nearest_lexical",
        "latest_fact",
        "broken_state_tracker",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_sha256(path: Path) -> str:
    """Hash source bytes with CRLF normalized (matches launch-readiness convention)."""

    return _sha256_hex(path.read_bytes().replace(b"\r\n", b"\n"))


def _family_chance(baselines: Mapping[str, Any], family: str) -> float:
    random_baseline = baselines.get("deterministic_random") or {}
    by_family = random_baseline.get("by_family") or {}
    if family not in by_family:
        raise ValueError(f"no measured chance proxy for family: {family}")
    return float(by_family[family])


def qualify_family(
    dev_certificate: Mapping[str, Any],
    family: str,
    *,
    generator_id: str,
    generator_sha256: str,
    max_excess: float = MAX_SHORTCUT_EXCESS,
) -> dict[str, object]:
    """Judge one family: worst heuristic excess over measured chance."""

    baselines = dev_certificate.get("baselines") or {}
    if not baselines:
        raise ValueError("development certificate carries no baselines")
    suite = dev_certificate.get("suite") or {}
    histogram = suite.get("family_histogram") or {}
    if family not in histogram:
        raise ValueError(f"family absent from development suite: {family}")
    chance = _family_chance(baselines, family)
    excesses: dict[str, float] = {}
    for name, baseline in baselines.items():
        if name not in SHORTCUT_BASELINES:
            continue
        by_family = (baseline or {}).get("by_family") or {}
        if family not in by_family:
            continue
        excesses[name] = float(by_family[family]) - chance
    if not excesses:
        raise ValueError(f"no shortcut baselines cover family: {family}")
    worst = max(excesses.values())
    verdict = "GENERATOR_QUALIFIED" if worst <= max_excess else "GENERATOR_NOT_QUALIFIED"
    return {
        "schema": QUALIFICATION_SCHEMA,
        "family": family,
        "generator_id": generator_id,
        "generator_sha256": generator_sha256,
        "suite_generator_version": (suite.get("generator_version")),
        "suite_seed": suite.get("seed"),
        "suite_sha256": suite.get("sha256"),
        "cases": int(histogram[family]),
        "chance_proxy": "deterministic_random.by_family",
        "chance": chance,
        "heuristic_excesses": excesses,
        "worst_excess": worst,
        "max_excess_allowed": max_excess,
        "verdict": verdict,
    }


def qualify_suite(
    dev_certificate: Mapping[str, Any],
    families: list[str],
    *,
    generator_id: str,
    generator_sha256: str,
    max_excess: float = MAX_SHORTCUT_EXCESS,
) -> dict[str, object]:
    """Qualify several families; the suite passes only if every family does."""

    if not families:
        raise ValueError("qualification needs at least one family")
    results = [
        qualify_family(
            dev_certificate, family, generator_id=generator_id,
            generator_sha256=generator_sha256, max_excess=max_excess,
        )
        for family in families
    ]
    verdict = (
        "GENERATOR_QUALIFIED"
        if all(result["verdict"] == "GENERATOR_QUALIFIED" for result in results)
        else "GENERATOR_NOT_QUALIFIED"
    )
    receipt: dict[str, object] = {
        "schema": QUALIFICATION_SCHEMA,
        "scope": "suite",
        "generator_id": generator_id,
        "generator_sha256": generator_sha256,
        "families": results,
        "verdict": verdict,
    }
    receipt["sha256"] = _sha256_hex(_canonical_json(receipt))
    return receipt


__all__ = [
    "MAX_SHORTCUT_EXCESS",
    "QUALIFICATION_SCHEMA",
    "SHORTCUT_BASELINES",
    "qualify_family",
    "qualify_suite",
    "source_sha256",
]
