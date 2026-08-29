"""Preregistered tokenizer tournament plan and matched-budget checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .probes import PROBES


CANDIDATE_VOCABULARIES = (16_384, 24_576, 32_768)


def probe_manifest_sha256() -> str:
    payload = [asdict(probe) for probe in PROBES]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TournamentArm:
    name: str
    vocabulary_size: int
    artifact_sha256: str | None
    audit_receipt_sha256: str | None
    matched_raw_bytes: int
    matched_training_flops: int


@dataclass(frozen=True, slots=True)
class TournamentPlan:
    schema: str
    experiment_id: str
    probe_manifest_sha256: str
    corpus_manifest_sha256: str | None
    candidate_vocabulary_sizes: tuple[int, ...]
    raw_byte_budget: int
    training_flops_budget: int
    arms: tuple[TournamentArm, ...]

    def assert_valid(self) -> None:
        if self.schema != "esoes-e1-tokenizer-tournament/v1":
            raise ValueError("unexpected tournament schema")
        if self.probe_manifest_sha256 != probe_manifest_sha256():
            raise ValueError("committed probe identity changed")
        if self.candidate_vocabulary_sizes != CANDIDATE_VOCABULARIES:
            raise ValueError("the tournament must contain exactly 16k/24k/32k arms")
        if self.raw_byte_budget <= 0 or self.training_flops_budget <= 0:
            raise ValueError("matched budgets must be positive")
        if tuple(arm.vocabulary_size for arm in self.arms) != CANDIDATE_VOCABULARIES:
            raise ValueError("arms must be ordered by candidate vocabulary size")
        for arm in self.arms:
            if arm.matched_raw_bytes != self.raw_byte_budget:
                raise ValueError("tokenizer arms must use the same raw-byte budget")
            if arm.matched_training_flops != self.training_flops_budget:
                raise ValueError("tokenizer arms must use the same measured FLOP budget")

    def status(self) -> str:
        self.assert_valid()
        if self.corpus_manifest_sha256 is None:
            return "BLOCKED_EXTERNAL_CORPUS"
        if any(arm.artifact_sha256 is None or arm.audit_receipt_sha256 is None for arm in self.arms):
            return "WAITING_FOR_CANDIDATE_ARTIFACTS"
        return "READY_FOR_MATCHED_TRAINING"

    def as_dict(self) -> dict[str, object]:
        self.assert_valid()
        return {"status": self.status(), **asdict(self)}


def build_plan(
    *,
    raw_byte_budget: int = 10_000_000,
    training_flops_budget: int = 1,
    corpus_manifest_sha256: str | None = None,
) -> TournamentPlan:
    arms = tuple(
        TournamentArm(
            name=f"bpe-byte-fallback-{vocabulary_size}",
            vocabulary_size=vocabulary_size,
            artifact_sha256=None,
            audit_receipt_sha256=None,
            matched_raw_bytes=raw_byte_budget,
            matched_training_flops=training_flops_budget,
        )
        for vocabulary_size in CANDIDATE_VOCABULARIES
    )
    return TournamentPlan(
        schema="esoes-e1-tokenizer-tournament/v1",
        experiment_id="E1-tokenizer-tournament-v1",
        probe_manifest_sha256=probe_manifest_sha256(),
        corpus_manifest_sha256=corpus_manifest_sha256,
        candidate_vocabulary_sizes=CANDIDATE_VOCABULARIES,
        raw_byte_budget=raw_byte_budget,
        training_flops_budget=training_flops_budget,
        arms=arms,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-byte-budget", type=int, default=10_000_000)
    parser.add_argument("--training-flops-budget", type=int, default=1)
    parser.add_argument("--corpus-manifest-sha256")
    args = parser.parse_args()
    plan = build_plan(
        raw_byte_budget=args.raw_byte_budget,
        training_flops_budget=args.training_flops_budget,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan.status(), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
