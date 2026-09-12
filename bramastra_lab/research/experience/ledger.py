"""Append-only experience ledger (B07).

Every episode collected from an environment lands here as an immutable,
hash-chained receipt with semantic provenance, quality/ambiguity flags, the
collection-policy version and per-transition costs. The chain makes any
mutation detectable; parent experience stays readable forever, including
after a candidate built on it is rejected.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from bramastra_lab.research.contracts.core import content_identity

RECEIPT_SCHEMA = "EpisodeReceipt/v1"
LEDGER_SCHEMA = "bramastra-experience-ledger/v1"
QUALITY_LEVELS = frozenset({"accepted", "ambiguous", "rejected"})


class LedgerError(ValueError):
    """A receipt or ledger file violates the ledger contract."""


class LedgerTamperedError(LedgerError):
    """The ledger file's hash chain does not verify."""


@dataclass(frozen=True)
class EpisodeReceipt:
    episode_id: str
    task_semantic_id: str
    family: str
    collection_policy: str
    success: bool
    quality: str
    quality_reason: str | None
    transition_costs: tuple[float, ...]
    episode_content_identity: str
    recorded_at_unix: float
    notes: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("episode_id", "task_semantic_id", "family", "collection_policy",
                     "episode_content_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LedgerError(f"{name} must be a nonempty string")
        if not isinstance(self.success, bool):
            raise LedgerError("success must be a boolean")
        if self.quality not in QUALITY_LEVELS:
            raise LedgerError(f"quality must be one of {sorted(QUALITY_LEVELS)}")
        if self.quality in ("ambiguous", "rejected") and not self.quality_reason:
            raise LedgerError("ambiguous/rejected receipts require quality_reason")
        if self.quality == "accepted" and self.quality_reason is not None:
            raise LedgerError("accepted receipts must not carry a quality_reason")
        for index, cost in enumerate(self.transition_costs):
            if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
                raise LedgerError(f"transition_costs[{index}] must be a nonnegative number")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA, "episode_id": self.episode_id,
            "task_semantic_id": self.task_semantic_id, "family": self.family,
            "collection_policy": self.collection_policy, "success": self.success,
            "quality": self.quality, "quality_reason": self.quality_reason,
            "transition_costs": list(self.transition_costs),
            "episode_content_identity": self.episode_content_identity,
            "recorded_at_unix": self.recorded_at_unix, "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EpisodeReceipt":
        known = {"schema", "episode_id", "task_semantic_id", "family", "collection_policy",
                 "success", "quality", "quality_reason", "transition_costs",
                 "episode_content_identity", "recorded_at_unix", "notes"}
        unknown = set(raw) - known
        if unknown:
            raise LedgerError(f"receipt has unknown fields: {sorted(unknown)}")
        if raw.get("schema") != RECEIPT_SCHEMA:
            raise LedgerError(f"unsupported receipt schema {raw.get('schema')!r}")
        return cls(
            episode_id=raw["episode_id"], task_semantic_id=raw["task_semantic_id"],
            family=raw["family"], collection_policy=raw["collection_policy"],
            success=raw["success"], quality=raw["quality"],
            quality_reason=raw["quality_reason"],
            transition_costs=tuple(raw["transition_costs"]),
            episode_content_identity=raw["episode_content_identity"],
            recorded_at_unix=float(raw["recorded_at_unix"]), notes=dict(raw.get("notes", {})),
        )

    def total_cost(self) -> float:
        return float(sum(self.transition_costs))


class ExperienceLedger:
    """Append-only JSONL ledger with a verification hash chain."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._head: str | None = None
        self._count = 0
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def append(self, receipt: EpisodeReceipt) -> str:
        """Append one receipt; returns its chained entry identity."""
        receipt.to_dict()  # validation
        entry = {"prev_head": self._head, "receipt": receipt.to_dict()}
        entry_identity = content_identity(entry)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"entry_identity": entry_identity, **entry},
                                    sort_keys=True) + "\n")
        self._head = entry_identity
        self._count += 1
        return entry_identity

    def _iter_entries(self) -> Iterator[dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerTamperedError(f"{self.path}:{line_number} is corrupt: {exc}")

    def read_all(self, *, verify_chain: bool = True) -> tuple[tuple[str, EpisodeReceipt], ...]:
        """Read every receipt, verifying the hash chain by default.

        Mutation of any historical byte invalidates every identity after it,
        which is the point: the ledger is append-only.
        """
        entries: list[tuple[str, EpisodeReceipt]] = []
        head: str | None = None
        for entry in self._iter_entries():
            if verify_chain:
                expected = content_identity({"prev_head": head, "receipt": entry["receipt"]})
                if entry.get("entry_identity") != expected:
                    raise LedgerTamperedError(
                        f"{self.path} hash chain broken at entry {len(entries)}; "
                        "the ledger was mutated after publication")
            head = entry.get("entry_identity")
            entries.append((head, EpisodeReceipt.from_dict(entry["receipt"])))
        if verify_chain and head != self._head and self._head is not None:
            raise LedgerTamperedError("ledger head does not match the session state")
        return tuple(entries)

    def identity(self) -> str:
        """Identity of the full ledger content (mutation-sensitive)."""
        return content_identity([entry_identity for entry_identity, _ in self.read_all()])

    def counts(self, *, quality: str | None = None) -> dict[str, int]:
        by_family: dict[str, int] = {}
        for _, receipt in self.read_all():
            if quality is not None and receipt.quality != quality:
                continue
            by_family[receipt.family] = by_family.get(receipt.family, 0) + 1
        return by_family

    def total_cost(self) -> float:
        return sum(receipt.total_cost() for _, receipt in self.read_all())
