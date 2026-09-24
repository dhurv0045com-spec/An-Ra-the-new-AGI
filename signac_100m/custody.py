"""One-shot hash ledger for externally frozen Phase-One confirmation surfaces.

This module stores identities and receipts only.  It never loads sealed cases,
chooses an evaluator, or qualifies a corpus.  A caller must obtain a claim
before opening a surface; terminal receipts make retries read-only, while an
abandoned claim remains consumed and therefore fails closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import errno
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from v5_registry.registry import CheckpointRegistry
from v5_registry.subject import CoreSubjectManifest


PLAN_SCHEMA = "anra-signac-phase1-custody-plan/v1"
CLAIM_SCHEMA = "anra-signac-phase1-custody-claim/v1"
RECEIPT_SCHEMA = "anra-signac-phase1-custody-receipt/v1"
RESOURCE_SCHEMA = "anra-signac-phase1-custody-resource/v1"


class CustodyError(ValueError):
    """A custody plan or append-only record is invalid."""


class CustodyClaimInProgress(CustodyError):
    """Another caller already claimed this surface and has no terminal result."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _assert_sha256(label: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise CustodyError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class CustodyPlan:
    """Frozen identities required before a sealed or fresh surface is opened.

    ``cluster_sha256s`` contains hashes of source-cluster identifiers, never
    prompts, answers, raw cluster names, or other evaluation data.
    """

    split: str
    dataset_manifest_sha256: str
    fixture_sha256: str
    protocol_sha256: str
    protocol_contract_sha256: str
    evaluator_sha256: str
    subject_manifest_sha256: str
    source_tree_sha256: str
    training_seed: int
    evaluation_seed: int
    development_receipt_sha256: str
    cluster_sha256s: tuple[str, ...]
    round_id: str
    predecessor_receipt_sha256: str | None = None
    schema: str = PLAN_SCHEMA

    def assert_valid(self) -> None:
        if self.schema != PLAN_SCHEMA:
            raise CustodyError("unsupported Phase-One custody plan schema")
        if self.split not in {"sealed", "fresh"}:
            raise CustodyError("custody split must be sealed or fresh")
        for label, value in (
            ("dataset manifest", self.dataset_manifest_sha256),
            ("fixture", self.fixture_sha256),
            ("protocol", self.protocol_sha256),
            ("protocol contract", self.protocol_contract_sha256),
            ("evaluator", self.evaluator_sha256),
            ("subject manifest", self.subject_manifest_sha256),
            ("source tree", self.source_tree_sha256),
            ("development receipt", self.development_receipt_sha256),
        ):
            _assert_sha256(label, value)
        if self.predecessor_receipt_sha256 is not None:
            _assert_sha256("predecessor receipt", self.predecessor_receipt_sha256)
        if self.split == "sealed" and self.predecessor_receipt_sha256 is not None:
            raise CustodyError("sealed confirmation cannot name a predecessor receipt")
        if self.split == "fresh" and self.predecessor_receipt_sha256 is None:
            raise CustodyError("fresh replication requires a sealed confirmation receipt")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise CustodyError("training seed must be a nonnegative integer")
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise CustodyError("evaluation seed must be a nonnegative integer")
        if not isinstance(self.round_id, str) or not self.round_id.strip():
            raise CustodyError("custody round identity is required")
        if not self.cluster_sha256s:
            raise CustodyError("custody plan requires source-cluster identities")
        if not isinstance(self.cluster_sha256s, tuple):
            raise CustodyError("source-cluster identities must be an immutable tuple")
        for index, cluster_sha in enumerate(self.cluster_sha256s):
            _assert_sha256(f"cluster identity {index}", cluster_sha)
        if len(set(self.cluster_sha256s)) != len(self.cluster_sha256s):
            raise CustodyError("custody plan has duplicate source-cluster identities")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        result = asdict(self)
        result["cluster_sha256s"] = sorted(self.cluster_sha256s)
        return result

    def sha256(self) -> str:
        return _sha256(self.canonical())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CustodyPlan":
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if set(value) != fields:
            raise CustodyError("custody plan fields do not match its schema")
        plan = cls(**{**value, "cluster_sha256s": tuple(value["cluster_sha256s"])})
        plan.assert_valid()
        return plan


@dataclass(frozen=True, slots=True)
class CustodyClaim:
    claim_id: str
    plan_sha256: str
    token: str


@dataclass(frozen=True, slots=True)
class CustodyReceipt:
    schema: str
    claim_id: str
    plan_sha256: str
    plan: Mapping[str, object]
    outcome: str
    evaluation_receipt_sha256: str | None
    evidence_artifact_sha256: str | None
    custody_attestation_sha256: str | None
    independent_review_sha256: str | None
    failure_receipt_sha256: str | None

    def canonical(self) -> dict[str, object]:
        return asdict(self)

    def sha256(self) -> str:
        return _sha256(self.canonical())

    def as_record(self) -> dict[str, object]:
        return {**self.canonical(), "receipt_sha256": self.sha256()}


class PhaseOneCustody:
    """Append-only claims and terminal receipts for sealed/fresh surfaces."""

    def __init__(self, root: Path, registry: CheckpointRegistry) -> None:
        self.root = Path(root)
        self.claims_dir = self.root / "claims"
        self.receipts_dir = self.root / "receipts"
        self.resources_dir = self.root / "resources"
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.resources_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry

    @staticmethod
    def _claim_id(plan: CustodyPlan) -> str:
        # Split is deliberately omitted: re-labeling one fixture cannot make
        # the same surface available for a second sealed/fresh access.
        return _sha256(
            {
                "dataset_manifest_sha256": plan.dataset_manifest_sha256,
                "fixture_sha256": plan.fixture_sha256,
            }
        )

    @staticmethod
    def _resource_id(kind: str, identity: str) -> str:
        return _sha256({"kind": kind, "identity": identity})

    def claim(self, plan: CustodyPlan) -> CustodyClaim | CustodyReceipt:
        """Atomically claim a frozen surface before reading its task contents.

        An exact retry after completion returns the existing receipt.  A
        pending, failed, or differently bound attempt cannot be reopened.
        """

        plan.assert_valid()
        claim_id = self._claim_id(plan)
        plan_sha = plan.sha256()
        receipt_path = self.receipts_dir / f"{claim_id}.json"
        claim_path = self.claims_dir / f"{claim_id}.json"
        if receipt_path.exists():
            receipt = self._read_receipt(receipt_path)
            if receipt.plan_sha256 != plan_sha:
                raise CustodyError("surface already has a receipt for a different frozen plan")
            return receipt
        if claim_path.exists():
            record = self._read_claim(claim_path)
            if record["plan_sha256"] != plan_sha:
                raise CustodyError("surface is already claimed under a different frozen plan")
            raise CustodyClaimInProgress("surface is already claimed and has no terminal receipt")

        self._verify_subject_and_development_receipt(plan)
        if plan.split == "fresh":
            self._verify_fresh_predecessor(plan)

        # Reserve both identities independently.  Rewrapping an existing
        # dataset in a new fixture manifest, or reusing a fixture with a new
        # dataset-manifest hash, still collides with one of these locks.
        resources = (
            ("dataset", plan.dataset_manifest_sha256),
            ("fixture", plan.fixture_sha256),
            *(('cluster', cluster_sha) for cluster_sha in plan.cluster_sha256s),
        )
        reserved_by_this_attempt: list[Path] = []
        for kind, identity in resources:
            resource_id = self._resource_id(kind, identity)
            resource_path = self.resources_dir / f"{resource_id}.json"
            resource_record = {
                "schema": RESOURCE_SCHEMA,
                "resource_id": resource_id,
                "kind": kind,
                "identity": identity,
                "claim_id": claim_id,
                "plan_sha256": plan_sha,
            }
            if not self._write_once(resource_path, resource_record):
                for reserved_path in reversed(reserved_by_this_attempt):
                    reserved_path.unlink(missing_ok=True)
                raise CustodyError(
                    f"{kind} identity is already reserved by a prior Phase-One attempt"
                )
            reserved_by_this_attempt.append(resource_path)

        token = secrets.token_urlsafe(32)
        record = {
            "schema": CLAIM_SCHEMA,
            "claim_id": claim_id,
            "plan": plan.canonical(),
            "plan_sha256": plan_sha,
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        }
        if not self._write_once(claim_path, record):
            # Another process won the exclusive create. Read its immutable
            # claim and fail closed instead of racing into evaluation.
            other = self._read_claim(claim_path)
            if other["plan_sha256"] != plan_sha:
                raise CustodyError("surface was concurrently claimed under a different plan")
            raise CustodyClaimInProgress("surface was concurrently claimed")
        return CustodyClaim(claim_id=claim_id, plan_sha256=plan_sha, token=token)

    def complete(
        self,
        claim: CustodyClaim,
        *,
        outcome: str,
        evaluation_receipt_sha256: str,
        evidence_artifact_sha256: str,
        custody_attestation_sha256: str,
        independent_review_sha256: str,
    ) -> CustodyReceipt:
        """Record supplied result hashes and an external review reference.

        The ledger validates hash syntax and plan bindings only.  It does not
        open the referenced artifacts or authenticate the reviewer.
        """

        if outcome not in {"PASS", "INCONCLUSIVE"}:
            raise CustodyError("complete accepts only PASS or INCONCLUSIVE outcomes")
        for label, value in (
            ("evaluation receipt", evaluation_receipt_sha256),
            ("evidence artifact", evidence_artifact_sha256),
            ("custody attestation", custody_attestation_sha256),
            ("independent review", independent_review_sha256),
        ):
            _assert_sha256(label, value)
        record = self._verified_claim(claim)
        receipt = CustodyReceipt(
            schema=RECEIPT_SCHEMA,
            claim_id=claim.claim_id,
            plan_sha256=claim.plan_sha256,
            plan=record["plan"],
            outcome=outcome,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
            evidence_artifact_sha256=evidence_artifact_sha256,
            custody_attestation_sha256=custody_attestation_sha256,
            independent_review_sha256=independent_review_sha256,
            failure_receipt_sha256=None,
        )
        return self._write_terminal(receipt)

    def fail(self, claim: CustodyClaim, *, failure_receipt_sha256: str) -> CustodyReceipt:
        """Consume a failed/partial attempt permanently without storing raw errors."""

        _assert_sha256("failure receipt", failure_receipt_sha256)
        record = self._verified_claim(claim)
        receipt = CustodyReceipt(
            schema=RECEIPT_SCHEMA,
            claim_id=claim.claim_id,
            plan_sha256=claim.plan_sha256,
            plan=record["plan"],
            outcome="FAILED",
            evaluation_receipt_sha256=None,
            evidence_artifact_sha256=None,
            custody_attestation_sha256=None,
            independent_review_sha256=None,
            failure_receipt_sha256=failure_receipt_sha256,
        )
        return self._write_terminal(receipt)

    def _verify_subject_and_development_receipt(self, plan: CustodyPlan) -> None:
        if self.registry.status(plan.subject_manifest_sha256) != "DEV_EVALUATED":
            raise CustodyError("sealed/fresh confirmation requires a DEV_EVALUATED subject")
        if plan.development_receipt_sha256 not in self.registry.evaluation_receipts(
            plan.subject_manifest_sha256
        ):
            raise CustodyError("plan does not name a development receipt attached to this subject")
        subject = self.registry.subject_manifest(plan.subject_manifest_sha256)
        if subject.source_tree_sha256 is None or subject.source_tree_sha256 != plan.source_tree_sha256:
            raise CustodyError("custody plan source tree disagrees with the subject manifest")
        if subject.seed != plan.training_seed:
            raise CustodyError("custody plan training seed disagrees with the subject manifest")

    def _verify_fresh_predecessor(self, plan: CustodyPlan) -> None:
        predecessor = self._find_receipt(plan.predecessor_receipt_sha256 or "")
        if predecessor.outcome != "PASS":
            raise CustodyError("fresh replication requires a passing sealed confirmation")
        previous = CustodyPlan.from_dict(predecessor.plan)
        if previous.split != "sealed":
            raise CustodyError("fresh replication predecessor is not a sealed confirmation")
        if plan.subject_manifest_sha256 == previous.subject_manifest_sha256:
            raise CustodyError("fresh replication requires a distinct trained subject")
        if plan.training_seed == previous.training_seed:
            raise CustodyError("fresh replication requires a distinct training seed")
        if plan.evaluation_seed == previous.evaluation_seed:
            raise CustodyError("fresh replication requires a distinct evaluation seed")
        if plan.development_receipt_sha256 == previous.development_receipt_sha256:
            raise CustodyError("fresh replication requires a subject-specific development receipt")
        if plan.round_id == previous.round_id:
            raise CustodyError("fresh replication requires a distinct round identity")
        if plan.dataset_manifest_sha256 == previous.dataset_manifest_sha256:
            raise CustodyError("fresh replication requires a distinct frozen data surface")
        if plan.fixture_sha256 == previous.fixture_sha256:
            raise CustodyError("fresh replication requires a distinct evaluation fixture")
        if plan.protocol_sha256 == previous.protocol_sha256:
            raise CustodyError("fresh replication protocol must bind its distinct evaluation fixture")
        if plan.protocol_contract_sha256 != previous.protocol_contract_sha256:
            raise CustodyError("fresh replication changed the frozen evaluation protocol contract")
        if plan.evaluator_sha256 != previous.evaluator_sha256:
            raise CustodyError("fresh replication evaluator identity changed")
        if plan.source_tree_sha256 != previous.source_tree_sha256:
            raise CustodyError("fresh replication source tree changed")
        sealed_subject = self.registry.subject_manifest(previous.subject_manifest_sha256)
        fresh_subject = self.registry.subject_manifest(plan.subject_manifest_sha256)
        matched_recipe_fields = (
            "model_spec_sha256",
            "tokenizer_artifact_sha256",
            "tokenizer_identity_sha256",
            "training_spec_sha256",
            "data_manifest_sha256",
            "pack_manifest_sha256",
            "optimizer_spec_sha256",
            "schedule_spec_sha256",
            "curriculum_spec_sha256",
            "source_commit",
            "source_tree_sha256",
            "parent_checkpoint_sha256",
            "global_update",
            "cumulative_training_tokens",
            "training_stage",
            "stage",
            "custody",
        )
        for field in matched_recipe_fields:
            if getattr(fresh_subject, field) != getattr(sealed_subject, field):
                raise CustodyError(f"fresh replication training recipe differs at {field}")
        overlap = set(plan.cluster_sha256s).intersection(previous.cluster_sha256s)
        if overlap:
            raise CustodyError("sealed and fresh surfaces share source clusters")

    def _verified_claim(self, claim: CustodyClaim) -> dict[str, Any]:
        _assert_sha256("claim identity", claim.claim_id)
        _assert_sha256("claim plan identity", claim.plan_sha256)
        path = self.claims_dir / f"{claim.claim_id}.json"
        record = self._read_claim(path)
        token_hash = hashlib.sha256(claim.token.encode("utf-8")).hexdigest()
        if record["token_sha256"] != token_hash:
            raise CustodyError("custody claim token does not match the immutable claim")
        if record["plan_sha256"] != claim.plan_sha256:
            raise CustodyError("custody claim plan identity changed")
        return record

    def _write_terminal(self, receipt: CustodyReceipt) -> CustodyReceipt:
        path = self.receipts_dir / f"{receipt.claim_id}.json"
        if not self._write_once(path, receipt.as_record()):
            existing = self._read_receipt(path)
            if existing.sha256() != receipt.sha256():
                raise CustodyError("surface already has a different terminal receipt")
            return existing
        return receipt

    def _find_receipt(self, receipt_sha256: str) -> CustodyReceipt:
        _assert_sha256("predecessor receipt", receipt_sha256)
        for path in self.receipts_dir.glob("*.json"):
            receipt = self._read_receipt(path)
            if receipt.sha256() == receipt_sha256:
                return receipt
        raise CustodyError("referenced predecessor receipt is not in this custody ledger")

    @staticmethod
    def _write_once(path: Path, value: Mapping[str, object]) -> bool:
        """Create an immutable JSON record with cross-process exclusive create."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_json(value) + b"\n"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if directory_flag:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
                getattr(errno, "ENOSYS", errno.EINVAL),
            }
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY | directory_flag)
            except OSError as error:
                if error.errno not in unsupported:
                    raise
            else:
                try:
                    try:
                        os.fsync(directory_fd)
                    except OSError as error:
                        if error.errno not in unsupported:
                            raise
                finally:
                    os.close(directory_fd)
        return True

    @staticmethod
    def _read_claim(path: Path) -> dict[str, Any]:
        try:
            record = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CustodyError("custody claim is missing or corrupt") from error
        if record.get("schema") != CLAIM_SCHEMA:
            raise CustodyError("unsupported custody claim record")
        plan = CustodyPlan.from_dict(record.get("plan", {}))
        if record.get("plan_sha256") != plan.sha256():
            raise CustodyError("custody claim plan hash mismatch")
        if record.get("claim_id") != _sha256(
            {
                "dataset_manifest_sha256": plan.dataset_manifest_sha256,
                "fixture_sha256": plan.fixture_sha256,
            }
        ):
            raise CustodyError("custody claim identity disagrees with its surface")
        if path.stem != record["claim_id"]:
            raise CustodyError("custody claim filename disagrees with its identity")
        _assert_sha256("claim token", record.get("token_sha256"))
        return record

    @staticmethod
    def _read_receipt(path: Path) -> CustodyReceipt:
        try:
            value = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CustodyError("custody receipt is missing or corrupt") from error
        if value.get("schema") != RECEIPT_SCHEMA:
            raise CustodyError("unsupported custody receipt schema")
        claimed_sha = value.pop("receipt_sha256", None)
        receipt = CustodyReceipt(**value)
        if claimed_sha != receipt.sha256():
            raise CustodyError("custody receipt hash mismatch")
        if receipt.outcome not in {"PASS", "INCONCLUSIVE", "FAILED"}:
            raise CustodyError("unknown custody receipt outcome")
        plan = CustodyPlan.from_dict(receipt.plan)
        if plan.sha256() != receipt.plan_sha256:
            raise CustodyError("custody receipt is bound to a different plan")
        expected_claim_id = _sha256(
            {
                "dataset_manifest_sha256": plan.dataset_manifest_sha256,
                "fixture_sha256": plan.fixture_sha256,
            }
        )
        if receipt.claim_id != expected_claim_id:
            raise CustodyError("custody receipt identity disagrees with its surface")
        if path.stem != receipt.claim_id:
            raise CustodyError("custody receipt filename disagrees with its identity")
        if receipt.outcome in {"PASS", "INCONCLUSIVE"}:
            for label, digest in (
                ("evaluation receipt", receipt.evaluation_receipt_sha256),
                ("evidence artifact", receipt.evidence_artifact_sha256),
                ("custody attestation", receipt.custody_attestation_sha256),
                ("independent review", receipt.independent_review_sha256),
            ):
                _assert_sha256(label, digest)
            if receipt.failure_receipt_sha256 is not None:
                raise CustodyError("completed custody receipt also contains a failure identity")
        else:
            _assert_sha256("failure receipt", receipt.failure_receipt_sha256)
            if any(
                value is not None
                for value in (
                    receipt.evaluation_receipt_sha256,
                    receipt.evidence_artifact_sha256,
                    receipt.custody_attestation_sha256,
                    receipt.independent_review_sha256,
                )
            ):
                raise CustodyError("failed custody receipt contains a completion identity")
        return receipt


__all__ = [
    "CLAIM_SCHEMA",
    "CustodyClaim",
    "CustodyClaimInProgress",
    "CustodyError",
    "CustodyPlan",
    "CustodyReceipt",
    "PhaseOneCustody",
    "PLAN_SCHEMA",
    "RECEIPT_SCHEMA",
]
