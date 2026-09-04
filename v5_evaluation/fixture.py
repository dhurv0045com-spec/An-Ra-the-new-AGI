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


FIXTURE_SCHEMA = "anra-v5-task-fixture-batch/v1"

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
    return {
        "task_id": str(case["task_id"]),
        "cluster_id": str(case["cluster_id"]),
        "family": str(case["family"]),
        "difficulty": str(case["difficulty"]),
        "split": str(case["split"]),
        "prompt": str(case["prompt"]),
        "candidates": [str(candidate) for candidate in case["candidates"]],  # type: ignore[union-attr]
        "gold": str(case["gold"]),
    }


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
