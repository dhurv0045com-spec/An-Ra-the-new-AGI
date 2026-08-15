"""Durable causal proof records with counterexamples and invalidation links."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ProofRecord:
    proof_id: str
    claim: str
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    derivation: str
    verifier: str
    counterexamples: tuple[str, ...] = ()
    confidence: float = 0.0
    invalidated_by: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)


class CausalProofMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, ProofRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = ProofRecord(**json.loads(line))
                self.records[record.proof_id] = record

    def add(self, record: ProofRecord) -> None:
        if not 0.0 <= record.confidence <= 1.0:
            raise ValueError("Proof confidence must be within [0, 1].")
        self.records[record.proof_id] = record
        self._flush()

    def invalidate(self, proof_id: str, invalidating_proof_id: str) -> None:
        record = self.records[proof_id]
        record.invalidated_by = tuple(sorted(set(record.invalidated_by) | {invalidating_proof_id}))
        self._flush()

    def active(self, minimum_confidence: float = 0.0) -> list[ProofRecord]:
        return [
            record
            for record in self.records.values()
            if not record.invalidated_by and record.confidence >= minimum_confidence
        ]

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            json.dumps(asdict(record), ensure_ascii=True, sort_keys=True)
            for record in self.records.values()
        ]
        self.path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
