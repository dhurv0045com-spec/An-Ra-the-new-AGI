"""Capability registry and claim engine: evidence-addressed, never prose.

A capability status exists only when a receipt supports it.  Statuses are
operational labels over measured evidence -- UNKNOWN, ABSENT, PARTIAL,
NATIVE_FRAGILE, NATIVE_ROBUST, STRUCTURAL_TRANSFER -- anchored in the
REPRESENT/ADDRESS/TRANSFORM/CHOOSE/REALIZE decomposition.  Claims carry
supporting and contradicting receipts and a lifecycle status, so no README
sentence ever becomes scientific truth by aging.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


CAPABILITY_SCHEMA = "anra-v5-capability-registry/v1"
CLAIM_SCHEMA = "anra-v5-claim-registry/v1"

CAPABILITY_OPERATIONS = ("REPRESENT", "ADDRESS", "TRANSFORM", "CHOOSE", "REALIZE")

CAPABILITY_FAMILIES = (
    "identity_copy",
    "query_binding",
    "semantic_state",
    "interference_retrieval",
    "relational_composition",
    "heldout_rule_induction",
    "counterfactual_sensitivity",
    "missing_information",
    "faithful_realization",
    "long_context_memory",
)

CAPABILITY_STATUSES = (
    "UNKNOWN",
    "ABSENT",
    "PARTIAL",
    "NATIVE_FRAGILE",
    "NATIVE_ROBUST",
    "STRUCTURAL_TRANSFER",
)

CLAIM_STATUSES = (
    "HYPOTHESIS",
    "SOFTWARE_ONLY",
    "LOCAL_CANARY",
    "DEV_SUPPORTED",
    "DEV_REPLICATED",
    "STRUCTURAL_OOD_REPLICATED",
    "FRESH_REPLICATED",
    "FALSIFIED",
    "SUPERSEDED",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha_of(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


class CapabilityRegistry:
    """Per-checkpoint capability evidence, machine-readable and receipt-bound."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "capabilities.json"
        if not self.path.is_file():
            self.path.write_text("{}\n", encoding="utf-8")

    def _load(self) -> dict:
        return json.loads(self.path.read_text("utf-8"))

    def record(
        self,
        *,
        subject_manifest_sha256: str,
        family: str,
        status: str,
        operation: str,
        receipt_sha256: str,
        note: str = "",
    ) -> str:
        if family not in CAPABILITY_FAMILIES:
            raise ValueError(f"unknown capability family: {family}")
        if status not in CAPABILITY_STATUSES:
            raise ValueError(f"unknown capability status: {status}")
        if operation not in CAPABILITY_OPERATIONS:
            raise ValueError(f"unknown cognition operation: {operation}")
        if len(receipt_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in receipt_sha256
        ):
            raise ValueError("capability evidence must reference a receipt SHA-256")
        registry = self._load()
        subjects = registry.setdefault(subject_manifest_sha256, {})
        entry = {
            "status": status,
            "operation": operation,
            "receipt_sha256": receipt_sha256,
            "note": note,
        }
        subjects[family] = entry
        self.path.write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return _sha_of(entry)

    def profile(self, subject_manifest_sha256: str) -> dict:
        """The cognitive profile vector: family -> status (evidence-backed)."""

        return dict(self._load().get(subject_manifest_sha256, {}))


class ClaimRegistry:
    """Registered scientific claims with lifecycle status and receipts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "claims.json"
        if not self.path.is_file():
            self.path.write_text("[]\n", encoding="utf-8")

    def _load(self) -> list:
        return json.loads(self.path.read_text("utf-8"))

    def register(
        self,
        *,
        claim_id: str,
        text: str,
        scope: str,
        status: str,
        supporting_receipts: list[str],
        contradicting_receipts: list[str] | None = None,
        checkpoint_sha256: str | None = None,
        protocol_sha256: str | None = None,
    ) -> str:
        if not claim_id or not text or not scope:
            raise ValueError("claims need id, text, and scope")
        if status not in CLAIM_STATUSES:
            raise ValueError(f"unknown claim status: {status}")
        for receipt in supporting_receipts + (contradicting_receipts or []):
            if len(receipt) != 64 or any(
                character not in "0123456789abcdef" for character in receipt
            ):
                raise ValueError("claim receipts must be lowercase SHA-256 references")
        claims = self._load()
        if any(claim["claim_id"] == claim_id for claim in claims):
            raise ValueError(f"claim id already registered: {claim_id}")
        record = {
            "claim_id": claim_id,
            "text": text,
            "scope": scope,
            "status": status,
            "supporting_receipts": list(supporting_receipts),
            "contradicting_receipts": list(contradicting_receipts or []),
            "checkpoint_sha256": checkpoint_sha256,
            "protocol_sha256": protocol_sha256,
        }
        claims.append(record)
        self.path.write_text(
            json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return _sha_of(record)

    def all_claims(self) -> list[dict]:
        return self._load()


__all__ = [
    "CAPABILITY_FAMILIES",
    "CAPABILITY_OPERATIONS",
    "CAPABILITY_SCHEMA",
    "CAPABILITY_STATUSES",
    "CLAIM_SCHEMA",
    "CLAIM_STATUSES",
    "CapabilityRegistry",
    "ClaimRegistry",
]
