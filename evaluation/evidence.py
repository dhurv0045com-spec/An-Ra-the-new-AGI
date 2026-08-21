"""Evidence identity: every promotion-grade experiment result carries full
provenance. A result missing required fields is not usable as evidence.

Also provides supersession: new results can explicitly invalidate old ones
so later agents cannot resurrect bad conclusions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

EXPERIMENT_SCHEMA = "anra-evidence/v1"

REQUIRED_FIELDS = (
    "experiment_schema",
    "source_commit",
    "checkpoint_file_sha256",
    "checkpoint_parameter_sha256",
    "global_step",
    "tokenizer_identity",
    "architecture_identity",
    "execution_profile",
    "decode_policy",
    "evaluator_version",
    "timestamp_utc",
)


@dataclass(slots=True)
class EvidenceIdentity:
    experiment_schema: str = EXPERIMENT_SCHEMA
    source_commit: str = ""
    checkpoint_file_sha256: str = ""
    checkpoint_parameter_sha256: str = ""
    global_step: int = -1
    tokenizer_identity: str = ""
    architecture_identity: str = ""
    execution_profile: str = ""
    decode_policy: dict[str, object] = field(default_factory=dict)
    evaluator_version: str = "1"
    timestamp_utc: str = ""
    supersedes: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def validate(self) -> None:
        data = asdict(self)
        missing = [name for name in REQUIRED_FIELDS if not data.get(name)]
        if missing:
            raise ValueError(
                f"evidence identity incomplete; missing {missing}. "
                "A result without identity is not promotion evidence."
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, *, decode_policy: dict[str, object], **overrides) -> "EvidenceIdentity":
        from anra_core.checkpoint import load_core_checkpoint

        try:
            _model, _payload, identity = load_core_checkpoint(checkpoint_path)
        except Exception:
            from anra_core.checkpoint import load_core_checkpoint as lc

            _model, _payload, identity = lc(checkpoint_path, legacy_unverified=True)
        digest = hashlib.sha256()
        with open(checkpoint_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        record = cls(
            source_commit=identity.source_commit or "",
            checkpoint_file_sha256=digest.hexdigest(),
            checkpoint_parameter_sha256=identity.parameter_sha256 or "",
            global_step=int(identity.global_step or -1),
            tokenizer_identity=(
                f"v4_32k:{identity.representation_id or 'unverified'}"
            ),
            architecture_identity=identity.architecture_id,
            execution_profile="cpu_exact_float32" ,
            decode_policy=dict(decode_policy),
            **overrides,
        )
        return record


def write_evidence(path: Path, identity: EvidenceIdentity, results: dict[str, object]) -> None:
    """Write a result bound to its identity. Refuses incomplete identities."""
    identity.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"identity": identity.to_dict(), "results": results}
    temporary = path.with_name(f".{path.name}.uploading")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_evidence(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    identity = payload.get("identity", {})
    missing = [name for name in REQUIRED_FIELDS if not identity.get(name)]
    if missing:
        raise ValueError(f"{path}: evidence lacks identity fields {missing}")
    return payload


def invalidated_by(new_evidence_path: Path, old_ids: list[str]) -> list[str]:
    """Record supersession on an existing evidence file."""
    payload = json.loads(Path(new_evidence_path).read_text(encoding="utf-8"))
    payload["identity"].setdefault("invalidates", []).extend(old_ids)
    Path(new_evidence_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload["identity"]["invalidates"]
