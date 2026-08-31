"""Atomic, content-addressed local checkpoint transaction.

Remote durability is intentionally not claimed here.  This contract proves
local publication, inventory verification, writer fencing, and recoverability.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .state import TrainingState


MANIFEST_SCHEMA = "anra-v5-checkpoint-transaction/v1"
REQUIRED_COMPONENTS = frozenset(
    {
        "model.bin",
        "optimizer.bin",
        "scheduler.json",
        "rng.bin",
        "cursor.json",
        "ledger.json",
        "training_state.json",
    }
)


class InjectedCrash(RuntimeError):
    """Test-only crash boundary after a durable transaction stage."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_sync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True, slots=True)
class Component:
    name: str
    sha256: str
    byte_size: int


class CheckpointStore:
    def __init__(self, root: Path, lineage_id: str) -> None:
        if not lineage_id or any(character in lineage_id for character in "/\\"):
            raise ValueError("lineage id must be a safe single path component")
        self.root = root.resolve()
        self.lineage_id = lineage_id
        self.lineage_root = self.root / lineage_id
        self.objects = self.lineage_root / "objects"
        self.latest = self.lineage_root / "LATEST"
        self.objects.mkdir(parents=True, exist_ok=True)

    def latest_sha256(self) -> str | None:
        if not self.latest.exists():
            return None
        value = self.latest.read_text(encoding="ascii").strip()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("LATEST pointer is corrupt")
        return value

    def publish(
        self,
        *,
        state: TrainingState,
        payloads: Mapping[str, bytes],
        expected_parent_sha256: str | None,
        inject_crash_at: str | None = None,
    ) -> str:
        state.assert_valid()
        if state.lineage_id != self.lineage_id:
            raise ValueError("state belongs to another lineage")
        if set(payloads) != REQUIRED_COMPONENTS:
            missing = sorted(REQUIRED_COMPONENTS - set(payloads))
            extra = sorted(set(payloads) - REQUIRED_COMPONENTS)
            raise ValueError(f"checkpoint inventory mismatch; missing={missing}, extra={extra}")
        current = self.latest_sha256()
        if current != expected_parent_sha256:
            raise ValueError("writer fence rejected stale parent")
        if state.parent_checkpoint_sha256 != expected_parent_sha256:
            raise ValueError("state parent does not match publication parent")
        canonical_state = _canonical_json(state.canonical())
        if payloads["training_state.json"] != canonical_state:
            raise ValueError("training-state component disagrees with manifest state")

        staging = self.lineage_root / f".staging-{state.generation}"
        if staging.exists():
            raise ValueError("staging generation already exists; recovery must resolve it")
        staging.mkdir(parents=True)
        components: list[Component] = []
        for name in sorted(payloads):
            payload = bytes(payloads[name])
            _write_sync(staging / name, payload)
            components.append(Component(name, _sha256(payload), len(payload)))
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "lineage_id": self.lineage_id,
            "state_sha256": state.sha256(),
            "state": state.canonical(),
            "components": [asdict(component) for component in components],
            "durability": "local-fsync-and-atomic-publish-only",
        }
        manifest_bytes = _canonical_json(manifest)
        checkpoint_sha256 = _sha256(manifest_bytes)
        _write_sync(staging / "manifest.json", manifest_bytes)
        self._verify_directory(staging, expected_sha256=checkpoint_sha256)
        if inject_crash_at == "after_stage":
            raise InjectedCrash("after_stage")

        destination = self.objects / checkpoint_sha256
        if destination.exists():
            self._verify_directory(destination, expected_sha256=checkpoint_sha256)
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
        else:
            os.replace(staging, destination)
        if inject_crash_at == "after_publish_before_pointer":
            raise InjectedCrash("after_publish_before_pointer")

        pointer_tmp = self.lineage_root / f".LATEST-{state.generation}"
        _write_sync(pointer_tmp, f"{checkpoint_sha256}\n".encode("ascii"))
        os.replace(pointer_tmp, self.latest)
        if inject_crash_at == "after_pointer":
            raise InjectedCrash("after_pointer")
        return checkpoint_sha256

    def restore(self, checkpoint_sha256: str | None = None) -> tuple[TrainingState, dict[str, bytes]]:
        identity = checkpoint_sha256 or self.latest_sha256()
        if identity is None:
            raise ValueError("no committed checkpoint exists")
        return self._verify_directory(self.objects / identity, expected_sha256=identity)

    def _verify_directory(
        self, directory: Path, *, expected_sha256: str
    ) -> tuple[TrainingState, dict[str, bytes]]:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("checkpoint manifest is missing")
        manifest_bytes = manifest_path.read_bytes()
        if _sha256(manifest_bytes) != expected_sha256:
            raise ValueError("checkpoint manifest hash mismatch")
        manifest = json.loads(manifest_bytes)
        if set(manifest) != {
            "schema", "lineage_id", "state_sha256", "state", "components", "durability"
        }:
            raise ValueError("checkpoint manifest fields do not match schema")
        if manifest["schema"] != MANIFEST_SCHEMA or manifest["lineage_id"] != self.lineage_id:
            raise ValueError("checkpoint schema or lineage mismatch")
        names = {item["name"] for item in manifest["components"]}
        if names != REQUIRED_COMPONENTS or len(manifest["components"]) != len(REQUIRED_COMPONENTS):
            raise ValueError("checkpoint component inventory is incomplete")
        actual_children = {path.name for path in directory.iterdir()}
        expected_children = set(REQUIRED_COMPONENTS) | {"manifest.json"}
        if actual_children != expected_children:
            raise ValueError("checkpoint directory contains missing or untracked components")
        payloads: dict[str, bytes] = {}
        for item in manifest["components"]:
            if set(item) != {"name", "sha256", "byte_size"}:
                raise ValueError("component identity fields do not match schema")
            path = directory / item["name"]
            if not path.is_file():
                raise ValueError(f"checkpoint component is missing: {item['name']}")
            payload = path.read_bytes()
            if len(payload) != item["byte_size"] or _sha256(payload) != item["sha256"]:
                raise ValueError(f"checkpoint component is corrupt: {item['name']}")
            payloads[item["name"]] = payload
        state = TrainingState.from_dict(manifest["state"])
        if state.sha256() != manifest["state_sha256"]:
            raise ValueError("training state hash mismatch")
        if payloads["training_state.json"] != _canonical_json(state.canonical()):
            raise ValueError("training state payload mismatch")
        return state, payloads
