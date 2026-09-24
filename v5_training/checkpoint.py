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

from .distributed_checkpoint import (
    DISTRIBUTED_CHECKPOINT_SCHEMA,
    DISTRIBUTED_COMPONENTS,
    DISTRIBUTED_MANIFEST_SCHEMA,
    ReplicatedTrainingCheckpoint,
    decode_rank_state_bundle,
    encode_rank_state_bundle,
)
from .state import TrainingState


MANIFEST_SCHEMA = "anra-v5-checkpoint-transaction/v1"
MILESTONES_FILENAME = "MILESTONES"
MILESTONES_SCHEMA = "anra-v5-milestone-protection/v1"
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
DISTRIBUTED_REQUIRED_COMPONENTS = (
    REQUIRED_COMPONENTS - {"rng.bin"}
) | DISTRIBUTED_COMPONENTS


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
        """Publish the established local v1 checkpoint format."""

        return self._publish_transaction(
            state=state,
            payloads=payloads,
            expected_parent_sha256=expected_parent_sha256,
            inventory=REQUIRED_COMPONENTS,
            manifest_schema=MANIFEST_SCHEMA,
            inject_crash_at=inject_crash_at,
        )

    def _publish_transaction(
        self,
        *,
        state: TrainingState,
        payloads: Mapping[str, bytes],
        expected_parent_sha256: str | None,
        inventory: frozenset[str],
        manifest_schema: str,
        inject_crash_at: str | None = None,
    ) -> str:
        state.assert_valid()
        if state.lineage_id != self.lineage_id:
            raise ValueError("state belongs to another lineage")
        names = set(payloads)
        if names != inventory:
            missing = sorted(inventory - names)
            extra = sorted(names - inventory)
            raise ValueError(f"checkpoint inventory mismatch; missing={missing}, extra={extra}")
        if manifest_schema == DISTRIBUTED_MANIFEST_SCHEMA:
            self._validate_distributed_payloads(state=state, payloads=payloads)
        current = self.latest_sha256()
        if current != expected_parent_sha256:
            raise ValueError("writer fence rejected stale parent")
        if state.parent_checkpoint_sha256 != expected_parent_sha256:
            raise ValueError("state parent does not match publication parent")
        if current is not None:
            parent_schema = self._manifest_schema_for_sha(current)
            if parent_schema == DISTRIBUTED_MANIFEST_SCHEMA and manifest_schema != DISTRIBUTED_MANIFEST_SCHEMA:
                raise ValueError("checkpoint lineage cannot downgrade from distributed v2 to local v1")
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
            "schema": manifest_schema,
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

    def publish_distributed(
        self,
        *,
        state: TrainingState,
        payloads: Mapping[str, bytes],
        expected_parent_sha256: str | None,
        inject_crash_at: str | None = None,
    ) -> str:
        """Publish a v2 replicated-state checkpoint with per-rank continuation data.

        A v1 parent is permitted only as an explicit migration boundary: callers
        must construct fresh rank-local RNG/cursor payloads at the restored state.
        """

        if set(payloads) != DISTRIBUTED_REQUIRED_COMPONENTS:
            missing = sorted(DISTRIBUTED_REQUIRED_COMPONENTS - set(payloads))
            extra = sorted(set(payloads) - DISTRIBUTED_REQUIRED_COMPONENTS)
            raise ValueError(
                f"distributed checkpoint inventory mismatch; missing={missing}, extra={extra}"
            )
        return self._publish_transaction(
            state=state,
            payloads=payloads,
            expected_parent_sha256=expected_parent_sha256,
            inventory=DISTRIBUTED_REQUIRED_COMPONENTS,
            manifest_schema=DISTRIBUTED_MANIFEST_SCHEMA,
            inject_crash_at=inject_crash_at,
        )

    def prune(self, *, keep: set[str]) -> list[str]:
        """Delete committed generations outside ``keep`` for bounded rotation.

        Refuses to remove the LATEST pointer target and every checkpoint SHA
        persisted in the lineage MILESTONES protection file: rotation never
        orphans the resumable head or an immutable milestone, across
        processes and sessions. Milestone generations stay reachable by
        recording them with ``record_milestone``. Returns removed SHAs.
        """

        import shutil

        protected = set(self.protected_shas())
        head = self.latest_sha256()
        if head is not None:
            protected.add(head)
        keep = set(keep) | protected
        removed: list[str] = []
        if not self.objects.exists():
            return removed
        for child in sorted(self.objects.iterdir()):
            if not child.is_dir() or child.name in keep:
                continue
            if len(child.name) != 64 or any(
                c not in "0123456789abcdef" for c in child.name
            ):
                continue
            shutil.rmtree(child)
            removed.append(child.name)
        return removed

    def record_milestone(self, *, threshold_tokens: int, checkpoint_sha256: str) -> dict[str, object]:
        """Persist one milestone -> checkpoint binding for cross-session protection.

        The MILESTONES file survives process destruction: later sessions and
        rotations discover the protected set mechanically instead of relying
        on in-memory state. Re-recording a threshold keeps the first binding
        (milestones are immutable); a conflicting SHA for the same threshold
        fails closed.
        """

        if threshold_tokens <= 0:
            raise ValueError("milestone threshold must be positive")
        if len(checkpoint_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in checkpoint_sha256):
            raise ValueError("milestone checkpoint must be a lowercase SHA-256")
        milestones = self._read_milestones()
        recorded = milestones["milestones"]
        if threshold_tokens in recorded:
            if recorded[threshold_tokens] != checkpoint_sha256:
                raise ValueError(
                    f"milestone {threshold_tokens} already bound to another checkpoint")
            return dict(milestones)
        recorded[threshold_tokens] = checkpoint_sha256
        milestones["milestones"] = {key: recorded[key] for key in sorted(recorded)}
        payload = _canonical_json(milestones)
        staging = self.lineage_root / f".{MILESTONES_FILENAME}.tmp"
        _write_sync(staging, payload)
        os.replace(staging, self.lineage_root / MILESTONES_FILENAME)
        return dict(milestones)

    def protected_shas(self) -> list[str]:
        """Checkpoint SHAs the rotation must never delete."""

        return list(self._read_milestones()["milestones"].values())

    def _read_milestones(self) -> dict[str, object]:
        path = self.lineage_root / MILESTONES_FILENAME
        if not path.exists():
            return {"schema": MILESTONES_SCHEMA, "lineage_id": self.lineage_id,
                    "milestones": {}}
        try:
            document = json.loads(path.read_bytes())
        except ValueError as exc:
            raise ValueError("milestone protection file is corrupt") from exc
        if (not isinstance(document, dict) or document.get("schema") != MILESTONES_SCHEMA
                or document.get("lineage_id") != self.lineage_id
                or not isinstance(document.get("milestones"), dict)):
            raise ValueError("milestone protection file is corrupt")
        milestones = {int(key): str(value)
                      for key, value in document["milestones"].items()}
        for threshold, sha in milestones.items():
            if threshold <= 0 or len(sha) != 64:
                raise ValueError("milestone protection file is corrupt")
        return {"schema": MILESTONES_SCHEMA, "lineage_id": self.lineage_id,
                "milestones": milestones}

    def restore(self, checkpoint_sha256: str | None = None) -> tuple[TrainingState, dict[str, bytes]]:
        """Restore the established local v1 format, rejecting distributed heads."""

        state, payloads = self._restore_transaction(checkpoint_sha256)
        if "distributed.json" in payloads:
            raise ValueError("distributed v2 checkpoint requires restore_distributed")
        return state, payloads

    def _restore_transaction(
        self, checkpoint_sha256: str | None = None,
    ) -> tuple[TrainingState, dict[str, bytes]]:
        identity = checkpoint_sha256 or self.latest_sha256()
        if identity is None:
            raise ValueError("no committed checkpoint exists")
        if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
            raise ValueError("checkpoint identity must be a lowercase SHA-256")
        return self._verify_directory(self.objects / identity, expected_sha256=identity)

    def restore_distributed(
        self,
        *,
        rank: int,
        expected_world_size: int,
        expected_topology: str,
        checkpoint_sha256: str | None = None,
    ) -> tuple[TrainingState, ReplicatedTrainingCheckpoint, dict[str, bytes]]:
        """Restore and select one rank's RNG/cursor bytes from a v2 checkpoint.

        The caller must restore these bytes into its runtime and compare the
        resulting next-batch fingerprint with ``metadata.assert_next_batch``
        before performing another optimizer update.
        """

        state, metadata, rank_payloads, _shared_payloads = self.restore_distributed_artifacts(
            rank=rank,
            expected_world_size=expected_world_size,
            expected_topology=expected_topology,
            checkpoint_sha256=checkpoint_sha256,
        )
        return state, metadata, rank_payloads

    def restore_distributed_artifacts(
        self,
        *,
        rank: int,
        expected_world_size: int,
        expected_topology: str,
        checkpoint_sha256: str | None = None,
    ) -> tuple[
        TrainingState,
        ReplicatedTrainingCheckpoint,
        dict[str, bytes],
        dict[str, bytes],
    ]:
        """Restore a rank's continuation bytes and verified shared train state.

        Returns the model, optimizer, and scheduler payload bytes alongside the
        selected rank RNG/cursor state. Applying those bytes to a live runtime
        remains the caller's responsibility.
        """

        if type(rank) is not int or type(expected_world_size) is not int:
            raise ValueError("distributed restore rank and world size must be integers")
        if not isinstance(expected_topology, str) or not expected_topology:
            raise ValueError("distributed restore topology is required")
        state, payloads = self._restore_transaction(checkpoint_sha256)
        if "distributed.json" not in payloads:
            raise ValueError("checkpoint is local v1; it has no per-rank resume state")
        metadata = ReplicatedTrainingCheckpoint.from_dict(
            json.loads(payloads["distributed.json"])
        )
        if metadata.world_size != expected_world_size:
            raise ValueError("distributed restore world size differs from checkpoint")
        if metadata.topology != expected_topology:
            raise ValueError("distributed restore topology differs from checkpoint")
        if not 0 <= rank < metadata.world_size:
            raise ValueError("distributed restore rank is outside the checkpoint world")
        metadata.assert_valid(
            training_state=state,
            model_payload=payloads["model.bin"],
            optimizer_payload=payloads["optimizer.bin"],
            rank_state_payload=payloads["rank_states.bin"],
        )
        rank_states = decode_rank_state_bundle(
            payloads["rank_states.bin"], world_size=metadata.world_size,
        )
        shared_payloads = {
            name: bytes(payloads[name])
            for name in ("model.bin", "optimizer.bin", "scheduler.json")
        }
        return state, metadata, rank_states[rank], shared_payloads

    def _manifest_schema_for_sha(self, checkpoint_sha256: str) -> str:
        """Read the content-addressed parent format without rereading model weights."""

        manifest_path = self.objects / checkpoint_sha256 / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("checkpoint parent manifest is missing")
        manifest_bytes = manifest_path.read_bytes()
        if _sha256(manifest_bytes) != checkpoint_sha256:
            raise ValueError("checkpoint parent manifest hash mismatch")
        try:
            manifest = json.loads(manifest_bytes)
        except ValueError as exc:
            raise ValueError("checkpoint parent manifest is corrupt") from exc
        if not isinstance(manifest, dict) or manifest.get("lineage_id") != self.lineage_id:
            raise ValueError("checkpoint parent lineage is corrupt")
        schema = manifest.get("schema")
        if schema not in {MANIFEST_SCHEMA, DISTRIBUTED_MANIFEST_SCHEMA}:
            raise ValueError("checkpoint parent schema is unsupported")
        return schema

    @staticmethod
    def _validate_distributed_payloads(
        *, state: TrainingState, payloads: Mapping[str, bytes],
    ) -> ReplicatedTrainingCheckpoint:
        try:
            document = json.loads(payloads["distributed.json"])
        except (TypeError, ValueError) as exc:
            raise ValueError("distributed checkpoint metadata is not valid JSON") from exc
        metadata = ReplicatedTrainingCheckpoint.from_dict(document)
        if bytes(payloads["distributed.json"]) != _canonical_json(metadata.canonical()):
            raise ValueError("distributed metadata must use canonical JSON")
        rank_states = decode_rank_state_bundle(
            bytes(payloads["rank_states.bin"]), world_size=metadata.world_size,
        )
        if bytes(payloads["rank_states.bin"]) != encode_rank_state_bundle(
            rank_states, world_size=metadata.world_size,
        ):
            raise ValueError("distributed rank-state bundle must use canonical bytes")
        metadata.assert_valid(
            training_state=state,
            model_payload=bytes(payloads["model.bin"]),
            optimizer_payload=bytes(payloads["optimizer.bin"]),
            rank_state_payload=bytes(payloads["rank_states.bin"]),
        )
        return metadata

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
        schema = manifest["schema"]
        if schema == MANIFEST_SCHEMA:
            required_components = REQUIRED_COMPONENTS
        elif schema == DISTRIBUTED_MANIFEST_SCHEMA:
            required_components = DISTRIBUTED_REQUIRED_COMPONENTS
        else:
            raise ValueError("unsupported checkpoint manifest schema")
        if manifest["lineage_id"] != self.lineage_id:
            raise ValueError("checkpoint schema or lineage mismatch")
        names = {item["name"] for item in manifest["components"]}
        if names != required_components or len(manifest["components"]) != len(required_components):
            raise ValueError("checkpoint component inventory is incomplete")
        actual_children = {path.name for path in directory.iterdir()}
        expected_children = set(required_components) | {"manifest.json"}
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
        try:
            cursor_payload = json.loads(payloads["cursor.json"])
            ledger_payload = json.loads(payloads["ledger.json"])
        except (TypeError, ValueError) as exc:
            raise ValueError("cursor or source ledger payload is not valid JSON") from exc
        if cursor_payload != json.loads(_canonical_json(asdict(state.cursor))):
            raise ValueError("cursor component disagrees with training state")
        if ledger_payload != dict(state.tokens_by_source):
            raise ValueError("source ledger component disagrees with training state")
        if schema == DISTRIBUTED_MANIFEST_SCHEMA:
            metadata = self._validate_distributed_payloads(state=state, payloads=payloads)
            if metadata.schema != DISTRIBUTED_CHECKPOINT_SCHEMA:
                raise ValueError("distributed metadata schema does not match transaction version")
        return state, payloads
