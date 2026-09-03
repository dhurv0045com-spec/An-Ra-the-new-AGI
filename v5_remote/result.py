"""Binding of a remote host's answer back to the exact job that requested it.

A result carries no trust on its own: ``bind_result`` refuses any result whose
job hash differs from the frozen request, so a swapped, replayed, or
edited answer cannot silently satisfy a gate. Failure answers must name a
compact failure code and always carry the remote log hash, so failed jobs stay
auditable instead of vanishing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from .job_spec import RemoteJob


RESULT_SCHEMA = "anra-v5-remote-result/v1"
BINDING_SCHEMA = "anra-v5-remote-binding/v1"
STATUSES = ("succeeded", "failed")


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class RemoteResult:
    schema: str
    job_sha256: str
    status: str
    completed_update: int
    cumulative_tokens: int
    receipt_shas: Mapping[str, str]
    log_sha256: str
    failure_code: str | None

    def assert_valid(self) -> None:
        if self.schema != RESULT_SCHEMA:
            raise ValueError("unsupported remote-result schema")
        _assert_sha256("job binding", self.job_sha256)
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if self.completed_update < 0 or self.cumulative_tokens < 0:
            raise ValueError("result counters cannot be negative")
        for name, digest in self.receipt_shas.items():
            if not name:
                raise ValueError("receipt names cannot be empty")
            _assert_sha256(f"receipt {name}", digest)
        _assert_sha256("remote log", self.log_sha256)
        if self.status == "failed":
            if not self.failure_code or any(
                character.isspace() for character in self.failure_code
            ):
                raise ValueError("failed results require a compact failure code")
        elif self.failure_code is not None:
            raise ValueError("successful results must not carry a failure code")
        if self.status == "succeeded" and not self.receipt_shas:
            raise ValueError("successful results must present at least one receipt hash")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return {
            "schema": self.schema,
            "job_sha256": self.job_sha256,
            "status": self.status,
            "completed_update": self.completed_update,
            "cumulative_tokens": self.cumulative_tokens,
            "receipt_shas": dict(self.receipt_shas),
            "log_sha256": self.log_sha256,
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RemoteResult":
        expected = {
            "schema", "job_sha256", "status", "completed_update", "cumulative_tokens",
            "receipt_shas", "log_sha256", "failure_code",
        }
        if set(value) != expected:
            raise ValueError("remote-result fields do not match schema")
        receipts = value["receipt_shas"]
        if not isinstance(receipts, dict):
            raise ValueError("receipt_shas must decode as a JSON object")
        failure = value["failure_code"]
        result = cls(
            schema=str(value["schema"]),
            job_sha256=str(value["job_sha256"]),
            status=str(value["status"]),
            completed_update=int(value["completed_update"]),  # type: ignore[arg-type]
            cumulative_tokens=int(value["cumulative_tokens"]),  # type: ignore[arg-type]
            receipt_shas={str(name): str(digest) for name, digest in receipts.items()},
            log_sha256=str(value["log_sha256"]),
            failure_code=None if failure is None else str(failure),
        )
        result.assert_valid()
        return result


def submission_envelope(job: RemoteJob) -> dict[str, object]:
    """Return the transport envelope for a frozen job (spec plus its hash)."""

    job.assert_valid()
    envelope: dict[str, object] = {
        "schema": "anra-v5-remote-submission/v1",
        "job": job.canonical(),
        "job_sha256": job.sha256(),
    }
    envelope["sha256"] = hashlib.sha256(_canonical_json(envelope)).hexdigest()
    return envelope


def bind_result(*, job: RemoteJob, result: RemoteResult) -> dict[str, object]:
    """Verify a result against its job, returning a hash-bound binding receipt."""

    job.assert_valid()
    result.assert_valid()
    if result.job_sha256 != job.sha256():
        raise ValueError("result does not bind to this job; refusing silent substitution")
    binding: dict[str, object] = {
        "schema": BINDING_SCHEMA,
        "job_sha256": job.sha256(),
        "result_status": result.status,
        "completed_update": result.completed_update,
        "cumulative_tokens": result.cumulative_tokens,
        "receipt_shas": dict(result.receipt_shas),
        "log_sha256": result.log_sha256,
        "failure_code": result.failure_code,
    }
    binding["sha256"] = hashlib.sha256(_canonical_json(binding)).hexdigest()
    return binding


__all__ = [
    "BINDING_SCHEMA",
    "RESULT_SCHEMA",
    "STATUSES",
    "RemoteResult",
    "bind_result",
    "submission_envelope",
]
