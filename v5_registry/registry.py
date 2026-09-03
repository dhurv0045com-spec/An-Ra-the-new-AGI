"""Content-addressed checkpoint registry with lifecycle and lineage DAG.

The registry answers: what subjects exist, which architecture, which parent,
how many training tokens, what evaluations and claims are attached, and what
promotion state each holds.  Identity is content-derived manifest SHA-256 --
never "latest checkpoint".  Lifecycle transitions are validated; negative
children remain evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .subject import CoreSubjectManifest


REGISTRY_SCHEMA = "anra-v5-checkpoint-registry/v1"

LIFECYCLE = (
    "CREATED",
    "IDENTITY_VERIFIED",
    "TRAINING_COMPLETE",
    "DEV_EVALUATED",
    "SEALED_EVALUATED",
    "PROMOTED",
    "REJECTED",
    "HISTORICAL_CONTROL",
)

_ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "CREATED": ("IDENTITY_VERIFIED", "REJECTED"),
    "IDENTITY_VERIFIED": ("TRAINING_COMPLETE", "REJECTED"),
    "TRAINING_COMPLETE": ("DEV_EVALUATED", "HISTORICAL_CONTROL", "REJECTED"),
    "DEV_EVALUATED": ("SEALED_EVALUATED", "PROMOTED", "REJECTED", "HISTORICAL_CONTROL"),
    "SEALED_EVALUATED": ("PROMOTED", "REJECTED", "HISTORICAL_CONTROL"),
    "PROMOTED": ("HISTORICAL_CONTROL",),
    "REJECTED": (),
    "HISTORICAL_CONTROL": (),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


class CheckpointRegistry:
    """Filesystem-backed, content-addressed registry of Core subjects."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.entries_dir = self.root / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)

    # -- entry management ---------------------------------------------------
    def register(self, manifest: CoreSubjectManifest) -> str:
        manifest.assert_valid()
        identity = manifest.sha256()
        path = self.entries_dir / f"{identity}.json"
        if path.exists():
            existing = json.loads(path.read_text("utf-8"))
            if existing["manifest"] != manifest.canonical():
                raise ValueError(
                    "content-address collision: a different manifest claims this identity"
                )
            return identity
        entry = {
            "schema": REGISTRY_SCHEMA,
            "manifest_sha256": identity,
            "manifest": manifest.canonical(),
            "lifecycle": "CREATED",
            "evaluations": [],
            "claims": [],
        }
        path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return identity

    def _entry_path(self, manifest_sha256: str) -> Path:
        if len(manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in manifest_sha256
        ):
            raise ValueError("registry identity must be a lowercase SHA-256")
        return self.entries_dir / f"{manifest_sha256}.json"

    def _load(self, manifest_sha256: str) -> dict:
        path = self._entry_path(manifest_sha256)
        if not path.is_file():
            raise ValueError(f"unknown subject: {manifest_sha256[:12]}")
        entry = json.loads(path.read_text("utf-8"))
        if hashlib.sha256(_canonical_json(entry["manifest"])).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("registry entry manifest hash mismatch (tampered entry)")
        return entry

    def _save(self, entry: dict) -> None:
        identity = entry["manifest_sha256"]
        path = self._entry_path(identity)
        path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # -- lifecycle -----------------------------------------------------------
    def transition(self, manifest_sha256: str, *, to: str) -> str:
        if to not in LIFECYCLE:
            raise ValueError(f"unknown lifecycle state: {to}")
        entry = self._load(manifest_sha256)
        current = entry["lifecycle"]
        if LIFECYCLE.index(to) <= LIFECYCLE.index(current):
            # already at or past the requested state: re-running a canary flow
            # must be idempotent, and regressions to earlier states are
            # handled only by explicit REJECTED/HISTORICAL_CONTROL paths
            if to != current and to not in _ALLOWED_TRANSITIONS[current]:
                return current
        if to != current and to not in _ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"invalid lifecycle transition {current} -> {to}")
        entry["lifecycle"] = to
        self._save(entry)
        return to

    def status(self, manifest_sha256: str) -> str:
        return self._load(manifest_sha256)["lifecycle"]

    def attach_evaluation(self, manifest_sha256: str, *, evaluation_receipt_sha256: str) -> str:
        if len(evaluation_receipt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in evaluation_receipt_sha256
        ):
            raise ValueError("evaluation receipt identity must be a lowercase SHA-256")
        entry = self._load(manifest_sha256)
        if evaluation_receipt_sha256 not in entry["evaluations"]:
            entry["evaluations"].append(evaluation_receipt_sha256)
            self._save(entry)
        if entry["lifecycle"] in {"CREATED", "IDENTITY_VERIFIED", "TRAINING_COMPLETE"}:
            return self.transition(manifest_sha256, to="DEV_EVALUATED")
        return entry["lifecycle"]

    # -- lineage DAG ----------------------------------------------------------
    def children_of(self, checkpoint_sha256: str) -> list[dict]:
        children = []
        for identity in self.identities():
            manifest = self._load(identity)["manifest"]
            if manifest.get("parent_checkpoint_sha256") == checkpoint_sha256:
                children.append(manifest)
        return children

    def ancestry(self, manifest_sha256: str) -> list[dict]:
        """Return the parent chain, oldest ancestor first (empty for roots)."""

        chain: list[dict] = []
        seen: set[str] = set()
        current = self._load(manifest_sha256)["manifest"]
        while current.get("parent_checkpoint_sha256"):
            parent = current["parent_checkpoint_sha256"]
            if parent in seen:
                raise ValueError("lineage cycle detected")
            seen.add(parent)
            matches = [
                self._load(identity)["manifest"]
                for identity in self.identities()
                if self._load(identity)["manifest"]["checkpoint_sha256"] == parent
            ]
            if not matches:
                break  # parent registered elsewhere; chain ends honestly
            current = matches[0]
            chain.append(current)
        chain.reverse()
        return chain

    def identities(self) -> list[str]:
        return sorted(path.stem for path in self.entries_dir.glob("*.json"))

    def all_entries(self) -> list[dict]:
        return [self._load(identity) for identity in self.identities()]


__all__ = ["LIFECYCLE", "REGISTRY_SCHEMA", "CheckpointRegistry"]
