"""Signed promotion decisions by an independent promotion process.

Promotion points to an existing immutable milestone and never rewrites
weights. A decision passes only when every gate passes AND a detached
signature over the decision verifies through a caller-supplied verifier
(the signing key never enters this repository). Unsigned or
failed-verification decisions are INCONCLUSIVE or REJECTED, never PROMOTE.
Assisted results are separate columns and cannot satisfy a raw-Core gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable


DECISION_SCHEMA = "anra-v5-promotion-decision/v2"
VERDICTS = ("PROMOTE", "REJECT", "INCONCLUSIVE")


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    schema: str
    checkpoint_sha256: str
    evaluation_receipt_sha256: str
    durability_receipt_sha256: str
    gate_spec_sha256: str
    passed_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    signer_id: str | None
    detached_signature_sha256: str | None

    def assert_valid(self) -> None:
        if self.schema != DECISION_SCHEMA:
            raise ValueError("unsupported promotion-decision schema")
        for name in (
            "checkpoint_sha256", "evaluation_receipt_sha256",
            "durability_receipt_sha256", "gate_spec_sha256",
        ):
            _assert_sha256(name, getattr(self, name))
        if set(self.passed_gates) & set(self.failed_gates):
            raise ValueError("a gate cannot both pass and fail")

    def canonical(self) -> dict[str, object]:
        self.assert_valid()
        return {
            "schema": self.schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "durability_receipt_sha256": self.durability_receipt_sha256,
            "gate_spec_sha256": self.gate_spec_sha256,
            "passed_gates": list(self.passed_gates),
            "failed_gates": list(self.failed_gates),
            "signer_id": self.signer_id,
            "detached_signature_sha256": self.detached_signature_sha256,
        }

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical())).hexdigest()


def decide(
    decision: PromotionDecision,
    *,
    verifier: Callable[[str, str | None], bool] | None,
) -> str:
    """Return PROMOTE only for fully passing, signature-verified decisions."""

    decision.assert_valid()
    if decision.failed_gates:
        return "REJECT"
    if not decision.passed_gates:
        return "INCONCLUSIVE"
    if verifier is None or not decision.signer_id or not decision.detached_signature_sha256:
        return "INCONCLUSIVE"
    _assert_sha256("detached signature", decision.detached_signature_sha256)
    if not verifier(decision.sha256(), decision.detached_signature_sha256):
        return "INCONCLUSIVE"
    return "PROMOTE"


__all__ = ["DECISION_SCHEMA", "VERDICTS", "PromotionDecision", "decide"]
