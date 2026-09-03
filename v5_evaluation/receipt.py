"""Hash-bound evaluation receipt binding checkpoint, adapter, and metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping


RECEIPT_SCHEMA = "anra-v5-evaluation-receipt/v1"


def _assert_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    schema: str
    checkpoint_sha256: str
    adapter_sha256: str
    tokenizer_sha256: str
    protocol_sha256: str
    raw_metrics_sha256: str
    assisted_metrics_sha256: str
    substrate_metrics_sha256: str
    tier: str
    native_selection: Mapping[str, int]

    def assert_valid(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ValueError("unsupported evaluation-receipt schema")
        for name in (
            "checkpoint_sha256", "adapter_sha256", "tokenizer_sha256",
            "protocol_sha256", "raw_metrics_sha256", "assisted_metrics_sha256",
            "substrate_metrics_sha256",
        ):
            _assert_sha256(name, getattr(self, name))
        if self.tier not in {"tier0", "tier1", "sealed", "fresh"}:
            raise ValueError("unknown evaluation tier")
        if any(value < 0 for value in self.native_selection.values()):
            raise ValueError("selection counts cannot be negative")

    def sha256(self) -> str:
        self.assert_valid()
        payload = {
            "schema": self.schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "adapter_sha256": self.adapter_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "protocol_sha256": self.protocol_sha256,
            "raw_metrics_sha256": self.raw_metrics_sha256,
            "assisted_metrics_sha256": self.assisted_metrics_sha256,
            "substrate_metrics_sha256": self.substrate_metrics_sha256,
            "tier": self.tier,
            "native_selection": dict(self.native_selection),
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


__all__ = ["RECEIPT_SCHEMA", "EvaluationReceipt"]
