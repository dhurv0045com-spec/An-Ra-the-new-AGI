"""Per-arm run manifests: every execution binds its full launch identity (M25).

A RunManifest names the experiment spec, the arm, the seed, the parent
subject, the source commit, the runtime/accelerator/topology, the data
stream manifest, the start checkpoint/state, the token budget, and the
expected evaluation schedule. Final run receipts bind the RunManifest SHA,
so a stopped process or hostile auditor reconstructs exactly what ran.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


RUN_MANIFEST_SCHEMA = "anra-v5-run-manifest/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _assert_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema: str
    experiment_spec_sha256: str
    arm_id: str
    seed: int
    parent_subject_manifest_sha256: str | None
    source_commit: str
    runtime: str
    accelerator: str
    topology: str
    data_stream_manifest_sha256: str
    start_checkpoint_sha256: str | None
    token_budget: int
    expected_evaluation_schedule: tuple[str, ...]

    def __post_init__(self) -> None:
        self.assert_valid()

    def assert_valid(self) -> None:
        if self.schema != RUN_MANIFEST_SCHEMA:
            raise ValueError("unsupported run-manifest schema")
        _assert_sha256("experiment spec", self.experiment_spec_sha256)
        if not self.arm_id:
            raise ValueError("arm identity is required")
        if self.seed < 0:
            raise ValueError("arm seed cannot be negative")
        if self.parent_subject_manifest_sha256 is not None:
            _assert_sha256("parent subject", self.parent_subject_manifest_sha256)
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("source commit must be a full lowercase git SHA-1")
        for name in ("runtime", "accelerator", "topology"):
            if not getattr(self, name):
                raise ValueError(f"{name} identity is required")
        _assert_sha256("data stream manifest", self.data_stream_manifest_sha256)
        if self.start_checkpoint_sha256 is not None:
            _assert_sha256("start checkpoint", self.start_checkpoint_sha256)
        if self.token_budget <= 0:
            raise ValueError("token budget must be positive")
        if not self.expected_evaluation_schedule:
            raise ValueError("expected evaluation schedule is required")

    def sha256(self) -> str:
        self.assert_valid()
        return hashlib.sha256(
            _canonical_json(
                {
                    "schema": self.schema,
                    "experiment_spec_sha256": self.experiment_spec_sha256,
                    "arm_id": self.arm_id,
                    "seed": self.seed,
                    "parent_subject_manifest_sha256": self.parent_subject_manifest_sha256,
                    "source_commit": self.source_commit,
                    "runtime": self.runtime,
                    "accelerator": self.accelerator,
                    "topology": self.topology,
                    "data_stream_manifest_sha256": self.data_stream_manifest_sha256,
                    "start_checkpoint_sha256": self.start_checkpoint_sha256,
                    "token_budget": self.token_budget,
                    "expected_evaluation_schedule": list(self.expected_evaluation_schedule),
                }
            )
        ).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunManifest":
        expected = {
            "schema", "experiment_spec_sha256", "arm_id", "seed",
            "parent_subject_manifest_sha256", "source_commit", "runtime",
            "accelerator", "topology", "data_stream_manifest_sha256",
            "start_checkpoint_sha256", "token_budget", "expected_evaluation_schedule",
        }
        if set(value) != expected:
            raise ValueError("run-manifest fields do not match schema")
        manifest = cls(
            **{  # type: ignore[arg-type]
                key: (tuple(item) if key == "expected_evaluation_schedule" else item)
                for key, item in value.items()
            }
        )
        manifest.assert_valid()
        return manifest


__all__ = ["RUN_MANIFEST_SCHEMA", "RunManifest"]
