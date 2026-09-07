"""500M campaign evaluation hooks — Cymek side of the boundary.

Triquetra owns cognition measurement, causal diagnosis, and research
interpretation. Cymek provides only the mechanical harness: a frozen
checkpoint schedule, dataset-identity bindings (hashes, never data),
an evaluator interface, receipt ingestion with verification, and a go/no-go
policy boundary: promotion decisions stay in v5_promotion (evaluate_gates /
all_pass / decide), which this module never imports — the planes stay
separate by construction. No cognition research lives here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


CAMPAIGN_EVAL_SCHEMA = "anra-v5-campaign-eval/v1"

# Frozen 500M-campaign evaluation checkpoints (cumulative real tokens).
EVAL_CHECKPOINT_TOKENS = (0, 10_000_000, 25_000_000, 50_000_000,
                          100_000_000, 200_000_000, 350_000_000,
                          500_000_000)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EvalDatasetIdentity:
    """Identity-only binding of an evaluation dataset (no data here)."""

    name: str
    split: str
    content_sha256: str
    cases: int

    def assert_valid(self) -> None:
        if not self.name or self.split not in ("sealed", "fresh", "development"):
            raise ValueError("eval dataset needs a name and a sealed/fresh/development split")
        if (len(self.content_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.content_sha256)):
            raise ValueError("eval dataset content must be a lowercase SHA-256")
        if self.cases <= 0:
            raise ValueError("eval dataset needs a positive case count")


@dataclass(frozen=True, slots=True)
class EvalManifest:
    """Frozen binding of the datasets one evaluation round may consume."""

    schema: str
    round_id: str
    datasets: tuple

    def assert_valid(self) -> None:
        if self.schema != CAMPAIGN_EVAL_SCHEMA:
            raise ValueError("unsupported campaign-eval schema")
        if not self.round_id:
            raise ValueError("eval round needs an id")
        for dataset in self.datasets:
            dataset.assert_valid()

    def sha256(self) -> str:
        self.assert_valid()
        return hashlib.sha256(_canonical_json(
            {"schema": self.schema, "round_id": self.round_id,
             "datasets": [{
                 "name": dataset.name, "split": dataset.split,
                 "content_sha256": dataset.content_sha256,
                 "cases": dataset.cases} for dataset in self.datasets]})).hexdigest()


@dataclass(frozen=True, slots=True)
class EvalCheckpointRef:
    """One schedulable evaluation point: a committed campaign checkpoint."""

    cumulative_tokens: int
    global_update: int
    checkpoint_sha256: str

    def assert_valid(self) -> None:
        if self.cumulative_tokens not in EVAL_CHECKPOINT_TOKENS:
            raise ValueError("eval checkpoint is not on the frozen campaign schedule")
        if self.global_update < 0:
            raise ValueError("eval checkpoint update cannot be negative")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("eval checkpoint must reference a SHA-256")


class CampaignEvaluator(Protocol):
    """Frozen evaluator interface (implemented by Triquetra tooling)."""

    def evaluate(self, *, checkpoint: EvalCheckpointRef,
                 manifest: EvalManifest) -> Mapping[str, Any]:
        ...  # pragma: no cover - interface only


def assert_no_train_eval_collision(*, train_source_ids,
                                   eval_source_ids) -> None:
    """Fail closed when any evaluation source appears in training data."""

    train = set(train_source_ids)
    collision = sorted(train.intersection(eval_source_ids))
    if collision:
        raise ValueError(
            f"abort TRAIN_EVAL_COLLISION: {len(collision)} evaluation sources "
            f"appear in training data")


def eval_receipt_path(receipts_dir: str | Path, digest: str) -> Path:
    return Path(receipts_dir) / f"eval-{digest}.json"


def ingest_eval_receipt(receipts_dir: str | Path,
                        receipt: Mapping[str, Any]) -> str:
    """Verify and store one evaluation receipt (content-addressed, append-only)."""

    body = dict(receipt)
    digest = body.get("sha256")
    if not isinstance(digest, str):
        raise ValueError("eval receipt carries no sha256")
    recomputed = hashlib.sha256(_canonical_json(
        {key: value for key, value in body.items() if key != "sha256"})).hexdigest()
    if recomputed != digest:
        raise ValueError("eval receipt hash mismatch")
    for key in ("checkpoint_sha256", "eval_manifest_sha256", "verdict"):
        if key not in body:
            raise ValueError(f"eval receipt lacks {key}")
    root = Path(receipts_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = eval_receipt_path(root, digest)
    if not target.exists():
        staging = root / f".eval-{digest}.tmp"
        staging.write_text(json.dumps(body, indent=2, sort_keys=True),
                           encoding="utf-8")
        staging.replace(target)
    return digest


__all__ = ["CAMPAIGN_EVAL_SCHEMA", "EVAL_CHECKPOINT_TOKENS",
           "CampaignEvaluator", "EvalCheckpointRef", "EvalDatasetIdentity",
           "EvalManifest", "assert_no_train_eval_collision",
           "eval_receipt_path", "ingest_eval_receipt"]
