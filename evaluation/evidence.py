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
    # Checkpoint provenance: where the MODEL came from.
    checkpoint_source_commit: str = ""
    checkpoint_file_sha256: str = ""
    checkpoint_parameter_sha256: str = ""
    global_step: int = -1
    # Evaluator provenance: where the MEASUREMENT code came from. A checkpoint
    # from commit A can be measured by an evaluator from commit B - record both.
    evaluation_source_commit: str = ""
    evaluation_dirty: bool = False
    evaluation_diff_sha256: str = ""
    evaluator_file_sha256: str = ""
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
        missing = [
            name for name in
            ("experiment_schema", "checkpoint_source_commit", "checkpoint_file_sha256",
             "checkpoint_parameter_sha256", "global_step", "tokenizer_identity",
             "architecture_identity", "execution_profile", "decode_policy",
             "evaluator_version", "timestamp_utc", "evaluation_source_commit")
            if not data.get(name)
        ]
        if missing:
            raise ValueError(
                f"evidence identity incomplete; missing {missing}. "
                "A result without identity is not promotion evidence."
            )
        if self.evaluation_dirty and not self.evaluation_diff_sha256:
            raise ValueError(
                "dirty evaluation without evaluation_diff_sha256 is not "
                "reproducible - record the diff hash or commit a clean tree"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, *, decode_policy: dict[str, object], **overrides) -> "EvidenceIdentity":
        import subprocess

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

        # Evaluator provenance: current git state of the measuring repo.
        try:
            eval_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            ).stdout.strip()
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            ).stdout.strip())
            diff_sha = ""
            if dirty:
                diff = subprocess.run(
                    ["git", "diff", "HEAD"], capture_output=True, text=True,
                    cwd=str(Path(__file__).resolve().parents[1]),
                ).stdout
                diff_sha = hashlib.sha256(diff.encode()).hexdigest()
        except Exception:
            eval_commit, dirty, diff_sha = "", True, ""

        record = cls(
            checkpoint_source_commit=identity.source_commit or "",
            checkpoint_file_sha256=digest.hexdigest(),
            checkpoint_parameter_sha256=identity.parameter_sha256 or "",
            global_step=int(identity.global_step or -1),
            evaluation_source_commit=eval_commit,
            evaluation_dirty=dirty,
            evaluation_diff_sha256=diff_sha,
            tokenizer_identity=f"v4_32k:{identity.representation_id or 'unverified'}",
            architecture_identity=identity.architecture_id,
            execution_profile="cpu_exact_float32",
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
