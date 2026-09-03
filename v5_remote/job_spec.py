"""Hash-bound specification of one remote accelerator job.

A job pins the accelerator shape, the runtime image, the exact code commit,
the command to run, and every evidence identity the command depends on. The
remote host executes; this repository only freezes the request and later
verifies that a result binds to it bit-for-bit. Unknown fields, unpinned
images, or silent identity substitution fail closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping


JOB_SCHEMA = "anra-v5-remote-job/v1"

IDENTITY_KEYS = (
    "training_spec_sha256",
    "model_spec_sha256",
    "tokenizer_artifact_sha256",
    "data_manifest_sha256",
    "pack_manifest_sha256",
    "topology_receipt_sha256",
)


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _assert_sha256_or_null(name: str, value: str | None) -> None:
    if value is None:
        return
    _assert_sha256(name, value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class RemoteJob:
    schema: str
    job_id: str
    accelerator: str
    replicas: int
    runtime_image_sha256: str
    code_commit: str
    command: tuple[str, ...]
    seed: int
    token_budget: int
    max_wall_seconds: int
    identities: Mapping[str, str | None]

    def assert_valid(self) -> None:
        if self.schema != JOB_SCHEMA:
            raise ValueError("unsupported remote-job schema")
        for name, value in (("job_id", self.job_id), ("accelerator", self.accelerator)):
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"{name} must be a compact nonempty identity")
        if self.replicas <= 0:
            raise ValueError("replicas must be positive")
        _assert_sha256("runtime image", self.runtime_image_sha256)
        if len(self.code_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.code_commit
        ):
            raise ValueError("code commit must be a full lowercase Git SHA-1")
        if not self.command or any(not part for part in self.command):
            raise ValueError("command must be a nonempty argument vector")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.token_budget <= 0:
            raise ValueError("token budget must be positive")
        if self.max_wall_seconds <= 0:
            raise ValueError("wall-clock limit must be positive")
        if tuple(self.identities) != IDENTITY_KEYS:
            raise ValueError("job identities must carry exactly the six evidence slots")
        for key in IDENTITY_KEYS:
            _assert_sha256_or_null(key, self.identities[key])

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return {
            "schema": self.schema,
            "job_id": self.job_id,
            "accelerator": self.accelerator,
            "replicas": self.replicas,
            "runtime_image_sha256": self.runtime_image_sha256,
            "code_commit": self.code_commit,
            "command": list(self.command),
            "seed": self.seed,
            "token_budget": self.token_budget,
            "max_wall_seconds": self.max_wall_seconds,
            "identities": dict(self.identities),
        }

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RemoteJob":
        expected = {
            "schema", "job_id", "accelerator", "replicas", "runtime_image_sha256",
            "code_commit", "command", "seed", "token_budget", "max_wall_seconds",
            "identities",
        }
        if set(value) != expected:
            raise ValueError("remote-job fields do not match schema")
        command = value["command"]
        if not isinstance(command, list):
            raise ValueError("command must decode as a JSON argument vector")
        identities = value["identities"]
        if not isinstance(identities, dict):
            raise ValueError("identities must decode as a JSON object")
        job = cls(
            schema=str(value["schema"]),
            job_id=str(value["job_id"]),
            accelerator=str(value["accelerator"]),
            replicas=int(value["replicas"]),  # type: ignore[arg-type]
            runtime_image_sha256=str(value["runtime_image_sha256"]),
            code_commit=str(value["code_commit"]),
            command=tuple(str(part) for part in command),
            seed=int(value["seed"]),  # type: ignore[arg-type]
            token_budget=int(value["token_budget"]),  # type: ignore[arg-type]
            max_wall_seconds=int(value["max_wall_seconds"]),  # type: ignore[arg-type]
            identities={str(key): item for key, item in identities.items()},
        )
        job.assert_valid()
        return job


__all__ = [
    "IDENTITY_KEYS",
    "JOB_SCHEMA",
    "RemoteJob",
]
