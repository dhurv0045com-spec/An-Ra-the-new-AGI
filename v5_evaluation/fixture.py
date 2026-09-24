"""Immutable task-fixture batches: the only task source a protocol may run.

A fixture binds generator identity (ID + code SHA + config SHA), the exact
generation seed, the split every case belongs to, the case count, and the
fixture SHA over canonical cases. Evaluation verifies fixture.split,
fixture.seed, fixture.generator, and case count against the protocol before
a single model call. Fresh fixtures must never enter a development protocol:
callers enforce split policy; the batch carries the evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


FIXTURE_SCHEMA = "anra-v5-task-fixture-batch/v2"

PAIR_KINDS = frozenset({
    "query_swap",
    "relevant_fact_swap",
    "irrelevant_fact_swap",
    "order_permutation",
    "state_swap",
})
SENSITIVITY_PAIR_KINDS = frozenset({"query_swap", "relevant_fact_swap", "state_swap"})
INVARIANCE_PAIR_KINDS = frozenset({"irrelevant_fact_swap", "order_permutation"})

REQUIRED_CASE_FIELDS = frozenset(
    {
        "task_id",
        "cluster_id",
        "family",
        "difficulty",
        "split",
        "prompt",
        "candidates",
        "gold",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha_of(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _assert_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_case(case: Mapping[str, Any]) -> dict[str, object]:
    if not REQUIRED_CASE_FIELDS.issubset(set(case)):
        raise ValueError(
            f"fixture case lacks required fields: {sorted(REQUIRED_CASE_FIELDS - set(case))}"
        )
    canonical = {
        "task_id": str(case["task_id"]),
        "cluster_id": str(case["cluster_id"]),
        "family": str(case["family"]),
        "difficulty": str(case["difficulty"]),
        "split": str(case["split"]),
        "prompt": str(case["prompt"]),
        "candidates": [str(candidate) for candidate in case["candidates"]],  # type: ignore[union-attr]
        "gold": str(case["gold"]),
    }
    if "causal_pairs" in case:
        pairs = case["causal_pairs"]
        if not isinstance(pairs, (list, tuple)):
            raise ValueError("causal_pairs must be a list")
        normalized: list[dict[str, str]] = []
        seen_pair_ids: set[str] = set()
        for pair in pairs:
            if not isinstance(pair, Mapping) or not {"pair_id", "pair_kind", "pair_role"}.issubset(pair):
                raise ValueError("each causal pair needs pair_id, pair_kind, and pair_role")
            pair_id = str(pair["pair_id"])
            pair_kind = str(pair["pair_kind"])
            pair_role = str(pair["pair_role"])
            if not pair_id or pair_id in seen_pair_ids or pair_kind not in PAIR_KINDS or pair_role not in {"base", "changed"}:
                raise ValueError("causal pair metadata has a duplicate or invalid id, kind, or role")
            seen_pair_ids.add(pair_id)
            normalized.append({"pair_id": pair_id, "pair_kind": pair_kind, "pair_role": pair_role})
        if normalized:
            canonical["causal_pairs"] = sorted(normalized, key=lambda pair: pair["pair_id"])
    return canonical


@dataclass(frozen=True, slots=True)
class TaskFixtureBatch:
    """One frozen, hash-bound set of evaluation cases."""

    schema: str
    generator_id: str
    generator_sha256: str
    generator_config_sha256: str
    seed: int
    split: str
    cases: tuple[dict[str, object], ...]

    def assert_valid(self) -> None:
        if self.schema != FIXTURE_SCHEMA:
            raise ValueError("unsupported fixture-batch schema")
        if not self.generator_id:
            raise ValueError("generator identity is required")
        _assert_sha256("generator", self.generator_sha256)
        _assert_sha256("generator config", self.generator_config_sha256)
        if self.seed < 0:
            raise ValueError("fixture seed cannot be negative")
        if not self.split:
            raise ValueError("fixture split is required")
        if not self.cases:
            raise ValueError("fixture holds no cases")
        task_ids = [str(case["task_id"]) for case in self.cases]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("fixture task ids are not unique")
        for case in self.cases:
            if str(case["split"]) != self.split:
                raise ValueError("fixture case split disagrees with batch split")
        pair_groups: dict[str, list[tuple[dict[str, object], dict[str, str]]]] = {}
        for case in self.cases:
            for pair in case.get("causal_pairs", []):
                pair_groups.setdefault(str(pair["pair_id"]), []).append((case, pair))
        for pair_id, members in pair_groups.items():
            if len(members) != 2 or {pair.get("pair_role") for _case, pair in members} != {"base", "changed"}:
                raise ValueError(f"causal pair {pair_id!r} must have one base and one changed member")
            base, base_pair = next(item for item in members if item[1]["pair_role"] == "base")
            changed, changed_pair = next(item for item in members if item[1]["pair_role"] == "changed")
            kind = str(base_pair["pair_kind"])
            if str(changed_pair["pair_kind"]) != kind:
                raise ValueError(f"causal pair {pair_id!r} disagrees on pair kind")
            if base["family"] != changed["family"] or base["split"] != changed["split"]:
                raise ValueError(f"causal pair {pair_id!r} crosses family or split")
            if base["candidates"] != changed["candidates"]:
                raise ValueError(f"causal pair {pair_id!r} changes its candidate set")
            if base["prompt"] == changed["prompt"]:
                raise ValueError(f"causal pair {pair_id!r} does not change the prompt")
            answers_differ = base["gold"] != changed["gold"]
            if (kind in SENSITIVITY_PAIR_KINDS) != answers_differ:
                raise ValueError(f"causal pair {pair_id!r} gold answers disagree with its effect")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha_of(
            {
                "schema": self.schema,
                "generator_id": self.generator_id,
                "generator_sha256": self.generator_sha256,
                "generator_config_sha256": self.generator_config_sha256,
                "seed": self.seed,
                "split": self.split,
                "cases": list(self.cases),
            }
        )

    @classmethod
    def freeze(
        cls,
        *,
        generator_id: str,
        generator_sha256: str,
        generator_config_sha256: str,
        seed: int,
        split: str,
        cases: list[Mapping[str, Any]],
    ) -> "TaskFixtureBatch":
        """Canonicalize raw case records into a frozen batch."""

        batch = cls(
            schema=FIXTURE_SCHEMA,
            generator_id=generator_id,
            generator_sha256=generator_sha256,
            generator_config_sha256=generator_config_sha256,
            seed=seed,
            split=split,
            cases=tuple(_canonical_case(case) for case in cases),
        )
        batch.assert_valid()
        return batch


__all__ = ["FIXTURE_SCHEMA", "REQUIRED_CASE_FIELDS", "TaskFixtureBatch"]
