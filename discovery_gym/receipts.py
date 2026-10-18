from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .contracts import ExperimentIdentity, canonical_json, content_hash


RECEIPT_SCHEMA = "anra-discovery-gym/result-receipt/v1"


def source_closure(root: Path) -> str:
    package_root = root / "discovery_gym"
    paths = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".json", ".py"}
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(package_root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary artifact: {temporary}")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_identity(
    *,
    source_commit: str,
    source_closure_sha256: str,
    generator_sha256: str,
    evaluator_sha256: str,
    protocol_sha256: str,
    policy_id: str,
    model_id: str,
    seed: int,
    checkpoint_sha256: str | None = None,
    resume_state_sha256: str | None = None,
) -> ExperimentIdentity:
    return ExperimentIdentity(
        schema="anra-discovery-gym/experiment-identity/v1",
        source_commit=source_commit,
        source_closure_sha256=source_closure_sha256,
        generator_sha256=generator_sha256,
        evaluator_sha256=evaluator_sha256,
        protocol_sha256=protocol_sha256,
        policy_id=policy_id,
        model_id=model_id,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        resume_state_sha256=resume_state_sha256,
    )


def result_receipt(
    *,
    identity: ExperimentIdentity,
    protocol: dict[str, Any],
    suite_manifest: dict[str, Any],
    run: dict[str, Any],
    audit: dict[str, Any],
    command: str,
) -> dict[str, Any]:
    protocol_hash = protocol.get("protocol_sha256")
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if not isinstance(protocol_hash, str) or content_hash(protocol_body) != protocol_hash:
        raise ValueError("protocol hash does not match protocol contents")
    if identity.protocol_sha256 != protocol_hash:
        raise ValueError("experiment identity does not match protocol hash")
    body = {
        "schema": RECEIPT_SCHEMA,
        "identity": identity.to_dict(),
        "protocol_sha256": protocol_hash,
        "suite_manifest_sha256": content_hash(suite_manifest),
        "run_sha256": content_hash(run),
        "audit_sha256": content_hash(audit),
        "command": command,
        "status": audit.get("status", "UNKNOWN"),
        "scientific_status": "no_model_learning_claim",
        "local_only": True,
    }
    return {**body, "receipt_sha256": content_hash(body)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    schema: str
    experiment_id: str
    model_id: str
    initialization_sha256: str
    checkpoint_sha256: str | None
    resume_state_sha256: str | None
    optimizer_sha256: str | None
    protocol_sha256: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "initialization_sha256": self.initialization_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "resume_state_sha256": self.resume_state_sha256,
            "optimizer_sha256": self.optimizer_sha256,
            "protocol_sha256": self.protocol_sha256,
            "status": self.status,
        }


def no_checkpoint_identity(*, experiment_id: str, protocol_sha256: str) -> dict[str, Any]:
    return CheckpointIdentity(
        schema="anra-discovery-gym/checkpoint-identity/v1",
        experiment_id=experiment_id,
        model_id="none",
        initialization_sha256="not-created",
        checkpoint_sha256=None,
        resume_state_sha256=None,
        optimizer_sha256=None,
        protocol_sha256=protocol_sha256,
        status="NOT_CREATED_DG0",
    ).to_dict()
