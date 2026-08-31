"""Local immutable-object durability canary.

This is a framework-neutral simulation of the remote custody contract.  It
uses content-addressed objects, exclusive publication, fsync, and verified
redownload.  It does not claim that a cloud provider or TPU filesystem has
the same guarantees; the receipt says exactly what was tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from v5_contracts.lineage import DurabilityReceipt


SCHEMA = "esoes-v5-local-durability-canary/v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _validate_hash(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ImmutableObjectStore:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, payload: bytes) -> str:
        """Publish a content-addressed object without permitting overwrite."""

        content = bytes(payload)
        identity = _sha256(content)
        destination = self.root / identity
        if destination.exists():
            if not destination.is_file() or _sha256_file(destination) != identity:
                raise ValueError("immutable object identity is already occupied by different bytes")
            return identity
        temporary = self.root / f".staging-{identity}"
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            # A hard-link publication is atomic and fails if another writer
            # already owns this content address.  os.replace would silently
            # overwrite that writer's object and violate immutability.
            os.link(temporary, destination)
        except FileExistsError as exc:
            if destination.is_file() and _sha256_file(destination) == identity:
                return identity
            raise ValueError("immutable object identity is occupied by different bytes") from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        return identity

    def get(self, identity: str) -> bytes:
        _validate_hash("object identity", identity)
        path = self.root / identity
        if not path.is_file():
            raise ValueError("immutable object is missing")
        payload = path.read_bytes()
        if _sha256(payload) != identity:
            raise ValueError("immutable object hash mismatch")
        return payload


def run_canary() -> dict[str, object]:
    manifest = _canonical_json(
        {
            "schema": "anra-v5-durability-fixture/v1",
            "checkpoint": "synthetic-checkpoint-1",
            "component_inventory": ["model.bin", "optimizer.bin", "training_state.json"],
        }
    )
    artifact = _canonical_json(
        {
            "manifest_sha256": _sha256(manifest),
            "model": "model-bytes",
            "optimizer": "optimizer-bytes",
            "training_state": {"global_update": 1, "cumulative_tokens": 8},
        }
    )
    with tempfile.TemporaryDirectory(prefix="esoes-durability-") as directory:
        store = ImmutableObjectStore(Path(directory) / "objects")
        identity = store.put(artifact)
        # Idempotent same-byte publication is permitted; a different payload
        # cannot claim the existing content address.
        if store.put(artifact) != identity:
            raise ValueError("idempotent immutable publication changed identity")
        redownload = store.get(identity)
        if redownload != artifact:
            raise ValueError("redownload bytes differ from uploaded artifact")
        clean_restore_receipt = _sha256(
            _canonical_json({"manifest_sha256": _sha256(manifest), "artifact_sha256": identity})
        )
        receipt = DurabilityReceipt(
            schema="anra-v5-durability/v1",
            checkpoint_sha256=_sha256(manifest),
            artifact_sha256=identity,
            redownload_sha256=_sha256(redownload),
            restore_receipt_sha256=clean_restore_receipt,
            byte_size=len(artifact),
            immutable=True,
            storage_provider="local-immutable-cas-simulation",
            object_identity=identity,
            independently_verified_by="local-durability-canary",
        )
        receipt.assert_valid()
        return {
            "schema": SCHEMA,
            "status": "PASS",
            "scope": "local content-addressed immutable object upload/redownload/clean-restore simulation",
            "implementation_sha256": _sha256_file(Path(__file__)),
            "manifest_sha256": _sha256(manifest),
            "artifact_sha256": identity,
            "redownload_sha256": _sha256(redownload),
            "byte_size": len(artifact),
            "receipt": {
                "schema": receipt.schema,
                "checkpoint_sha256": receipt.checkpoint_sha256,
                "artifact_sha256": receipt.artifact_sha256,
                "redownload_sha256": receipt.redownload_sha256,
                "restore_receipt_sha256": receipt.restore_receipt_sha256,
                "immutable": receipt.immutable,
                "object_identity": receipt.object_identity,
            },
            "checks": {
                "upload_redownload_equal": artifact == redownload,
                "artifact_hash_matches_redownload": identity == _sha256(redownload),
                "immutable_idempotent_publish": store.put(artifact) == identity,
                "positive_restore_receipt": bool(clean_restore_receipt),
            },
            "limitations": [
                "This is a local filesystem simulation, not cloud-provider durability or TPU storage evidence.",
                "The payload is a tiny synthetic artifact; real P35 multipart/object-store restore remains required.",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_canary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
