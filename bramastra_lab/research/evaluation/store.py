"""Immutable outcome storage with pool separation (B09).

Measurement, confirmation and sealed pools live in separate append-only
files with separate readers. Sealed usage is logged; repeated use of the
sealed pool is reported and can never masquerade as a fresh test. The
controller-pool training data never mixes with measurement storage.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Sequence

from bramastra_lab.research.contracts.core import content_identity
from bramastra_lab.research.evaluation.scoring import EvaluationError, RawOutcome

POOLS = frozenset({"measurement", "confirmation", "sealed"})
SEALED_LEDGER = "sealed_usage.jsonl"


class StoreError(ValueError):
    """An outcome-store operation violated its contract."""


class EvaluationStore:
    """Append-only per-pool outcome files under one evaluation root."""

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _pool_path(self, pool: str) -> str:
        if pool not in POOLS:
            raise StoreError(
                f"pool {pool!r} is not a measurement pool; controller-pool training "
                "data is stored by the experience ledger, never here")
        return os.path.join(self.root, f"{pool}_outcomes.jsonl")

    def record(self, outcomes: Sequence[RawOutcome], *, pool: str, run_id: str) -> str:
        if pool not in POOLS:
            raise StoreError(f"unknown pool {pool!r}")
        if not outcomes:
            raise StoreError("no outcomes to record")
        batch_id = uuid.uuid4().hex
        payload = {
            "schema": "EvaluationBatch/v1", "batch_id": batch_id, "run_id": run_id,
            "recorded_at_unix": time.time(),
            "outcomes": [outcome.to_dict() for outcome in outcomes],
        }
        with open(self._pool_path(pool), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if pool == "sealed":
            self._log_sealed_use(run_id, [outcome.task_semantic_id for outcome in outcomes])
        return batch_id

    def read(self, pool: str) -> list[RawOutcome]:
        path = self._pool_path(pool)
        if not os.path.exists(path):
            return []
        outcomes: list[RawOutcome] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    batch = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StoreError(f"{path}:{line_number} is corrupt: {exc}")
                for raw in batch["outcomes"]:
                    raw.pop("schema", None)
                    outcomes.append(RawOutcome(**raw))
        return outcomes

    # -- sealed freshness ------------------------------------------------------

    def _log_sealed_use(self, run_id: str, item_ids: Sequence[str]) -> None:
        entry = {"run_id": run_id, "at_unix": time.time(),
                 "items": sorted(set(item_ids)), "item_count": len(set(item_ids))}
        with open(os.path.join(self.root, SEALED_LEDGER), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def sealed_freshness(self) -> dict[str, Any]:
        """Report prior sealed-pool uses. Reuse is never silently ignored."""
        path = os.path.join(self.root, SEALED_LEDGER)
        uses = 0
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                uses = sum(1 for line in handle if line.strip())
        return {"previous_uses": uses, "fresh": uses == 0,
                "note": "repeated sealed-pool use is logged and must be disclosed"}

    def identity(self, pool: str) -> str:
        path = self._pool_path(pool)
        if not os.path.exists(path):
            return content_identity({"pool": pool, "empty": True})
        with open(path, "rb") as handle:
            return content_identity({"pool": pool, "sha256_of": handle.read().hex()})
