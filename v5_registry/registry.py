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
import errno
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

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
    "DEV_EVALUATED": ("SEALED_EVALUATED", "REJECTED", "HISTORICAL_CONTROL"),
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

    @contextmanager
    def _entry_lock(self, manifest_sha256: str) -> Iterator[None]:
        """Serialize read-modify-write operations for one subject across processes.

        A leftover lock after process death blocks mutation instead of silently
        allowing a second writer; recovery requires an explicit audit.
        """

        self._entry_path(manifest_sha256)
        locks_dir = self.root / "locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_path = locks_dir / f"{manifest_sha256}.lock"
        deadline = time.monotonic() + 30.0
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "registry subject lock remains held; inspect it before recovery"
                    ) from error
                time.sleep(0.01)
        try:
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if not directory_flag:
            return
        unsupported = {
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            getattr(errno, "ENOSYS", errno.EINVAL),
        }
        try:
            descriptor = os.open(path, os.O_RDONLY | directory_flag)
        except OSError as error:
            if error.errno in unsupported:
                return
            raise
        try:
            try:
                os.fsync(descriptor)
            except OSError as error:
                if error.errno not in unsupported:
                    raise
        finally:
            os.close(descriptor)

    # -- entry management ---------------------------------------------------
    def register(self, manifest: CoreSubjectManifest) -> str:
        manifest.assert_valid()
        identity = manifest.sha256()
        path = self.entries_dir / f"{identity}.json"
        entry = {
            "schema": REGISTRY_SCHEMA,
            "manifest_sha256": identity,
            "manifest": manifest.canonical(),
            "lifecycle": "CREATED",
            "evaluations": [],
            "claims": [],
        }
        with self._entry_lock(identity):
            if path.exists():
                existing = self._load(identity)
                if existing["manifest"] != manifest.canonical():
                    raise ValueError(
                        "content-address collision: a different manifest claims this identity"
                    )
                return identity
            self._save(entry)
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
        if entry.get("schema") != REGISTRY_SCHEMA:
            raise ValueError("unsupported checkpoint-registry entry schema")
        if entry.get("manifest_sha256") != manifest_sha256:
            raise ValueError("registry entry filename disagrees with its subject identity")
        if hashlib.sha256(_canonical_json(entry["manifest"])).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("registry entry manifest hash mismatch (tampered entry)")
        if entry.get("lifecycle") not in LIFECYCLE:
            raise ValueError("registry entry has an unknown lifecycle state")
        return entry

    def _save(self, entry: dict) -> None:
        identity = entry["manifest_sha256"]
        path = self._entry_path(identity)
        temporary = self.entries_dir / f".{identity}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write((json.dumps(entry, indent=2, sort_keys=True) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_directory(self.entries_dir)
        finally:
            temporary.unlink(missing_ok=True)

    # -- lifecycle -----------------------------------------------------------
    def transition(self, manifest_sha256: str, *, to: str) -> str:
        if to not in LIFECYCLE:
            raise ValueError(f"unknown lifecycle state: {to}")
        with self._entry_lock(manifest_sha256):
            entry = self._load(manifest_sha256)
            current = entry["lifecycle"]
            if to == current:
                return current
            if to in {"SEALED_EVALUATED", "PROMOTED"}:
                raise ValueError(
                    f"{to} requires a verified Phase-One custody receipt; "
                    "generic lifecycle transitions cannot assert confirmation or promotion"
                )
            if LIFECYCLE.index(to) <= LIFECYCLE.index(current):
                # already at or past the requested state: re-running a canary flow
                # must be idempotent, and regressions to earlier states are
                # handled only by explicit REJECTED/HISTORICAL_CONTROL paths
                if to != current and to not in _ALLOWED_TRANSITIONS[current]:
                    return current
            if to not in _ALLOWED_TRANSITIONS[current]:
                raise ValueError(f"invalid lifecycle transition {current} -> {to}")
            entry["lifecycle"] = to
            self._save(entry)
            return to

    def status(self, manifest_sha256: str) -> str:
        return self._load(manifest_sha256)["lifecycle"]

    def subject_manifest(self, manifest_sha256: str) -> CoreSubjectManifest:
        """Return the verified canonical subject bound to a registry identity."""

        return CoreSubjectManifest.from_dict(self._load(manifest_sha256)["manifest"])

    def evaluation_receipts(self, manifest_sha256: str) -> tuple[str, ...]:
        """List immutable evaluation receipt identities attached to a subject."""

        return tuple(self._load(manifest_sha256)["evaluations"])

    def attach_sealed_confirmation(
        self,
        manifest_sha256: str,
        *,
        custody_receipt: Mapping[str, object],
    ) -> str:
        """Record a passing, hash-bound sealed receipt after development evaluation.

        This verifies receipt structure and subject/plan bindings.  The
        independent-review hash is an external attestation; this method does
        not verify reviewer authorship or the contents of external artifacts.
        Promotion remains a separate policy decision.
        """

        # Import lazily: Signac's custody ledger itself uses this registry.
        from signac_100m.custody import (
            CustodyPlan,
            CustodyReceipt,
            RECEIPT_SCHEMA as CUSTODY_RECEIPT_SCHEMA,
            _assert_sha256,
        )

        record = dict(custody_receipt)
        claimed_receipt_sha256 = record.pop("receipt_sha256", None)
        try:
            receipt = CustodyReceipt(**record)
        except TypeError as error:
            raise ValueError("custody receipt fields do not match the Phase-One schema") from error
        if receipt.schema != CUSTODY_RECEIPT_SCHEMA:
            raise ValueError("unsupported Phase-One custody receipt schema")
        if claimed_receipt_sha256 != receipt.sha256():
            raise ValueError("Phase-One custody receipt hash mismatch")
        if receipt.outcome != "PASS":
            raise ValueError("only a passing sealed confirmation may advance registry lifecycle")
        plan = CustodyPlan.from_dict(receipt.plan)
        if plan.split != "sealed" or plan.predecessor_receipt_sha256 is not None:
            raise ValueError("registry sealed confirmation requires a sealed-split custody plan")
        if plan.sha256() != receipt.plan_sha256:
            raise ValueError("custody receipt plan hash mismatch")
        expected_claim_id = hashlib.sha256(
            _canonical_json(
                {
                    "dataset_manifest_sha256": plan.dataset_manifest_sha256,
                    "fixture_sha256": plan.fixture_sha256,
                }
            )
        ).hexdigest()
        if receipt.claim_id != expected_claim_id:
            raise ValueError("custody receipt claim identity disagrees with its frozen surface")
        for label, digest in (
            ("evaluation receipt", receipt.evaluation_receipt_sha256),
            ("evidence artifact", receipt.evidence_artifact_sha256),
            ("custody attestation", receipt.custody_attestation_sha256),
            ("independent review", receipt.independent_review_sha256),
        ):
            _assert_sha256(label, digest)
        if receipt.evaluation_receipt_sha256 == plan.development_receipt_sha256:
            raise ValueError("sealed receipt must differ from the attached development receipt")
        if receipt.failure_receipt_sha256 is not None:
            raise ValueError("passing custody receipt also contains a failure identity")

        if plan.subject_manifest_sha256 != manifest_sha256:
            raise ValueError("sealed custody receipt is bound to a different subject")
        with self._entry_lock(manifest_sha256):
            entry = self._load(manifest_sha256)
            prior = entry.get("sealed_confirmations", [])
            if claimed_receipt_sha256 in prior:
                if entry["lifecycle"] != "SEALED_EVALUATED":
                    raise ValueError("registry confirmation receipt and lifecycle disagree")
                return entry["lifecycle"]
            if entry["lifecycle"] != "DEV_EVALUATED":
                raise ValueError("sealed confirmation requires DEV_EVALUATED lifecycle")
            if plan.development_receipt_sha256 not in entry["evaluations"]:
                raise ValueError("sealed plan does not name an attached development evaluation")
            manifest = CoreSubjectManifest.from_dict(entry["manifest"])
            if manifest.source_tree_sha256 is None or manifest.source_tree_sha256 != plan.source_tree_sha256:
                raise ValueError("sealed plan source tree disagrees with the subject manifest")
            if manifest.seed != plan.training_seed:
                raise ValueError("sealed plan training seed disagrees with the subject manifest")

            entry.setdefault("sealed_confirmations", []).append(claimed_receipt_sha256)
            entry["lifecycle"] = "SEALED_EVALUATED"
            self._save(entry)
            return "SEALED_EVALUATED"

    def attach_evaluation(self, manifest_sha256: str, *, evaluation_receipt_sha256: str) -> str:
        if len(evaluation_receipt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in evaluation_receipt_sha256
        ):
            raise ValueError("evaluation receipt identity must be a lowercase SHA-256")
        with self._entry_lock(manifest_sha256):
            entry = self._load(manifest_sha256)
            current = entry["lifecycle"]
            if current in {"CREATED", "IDENTITY_VERIFIED"}:
                raise ValueError("evaluation cannot be attached before training is complete")
            if evaluation_receipt_sha256 not in entry["evaluations"]:
                entry["evaluations"].append(evaluation_receipt_sha256)
            if current == "TRAINING_COMPLETE":
                entry["lifecycle"] = "DEV_EVALUATED"
            self._save(entry)
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
