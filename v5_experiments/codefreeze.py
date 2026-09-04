"""Experiment code freeze: bind real implementation identities (M24).

CODE_FREEZE binds the source commit plus the byte identities of the exact
implementation files each pipeline stage runs: model, trainer, data,
generator, evaluation, and analysis. No generic payload hash disconnected
from real code. A missing component file fails closed instead of freezing a
partial tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


FREEZE_SCHEMA = "anra-v5-code-freeze/v1"

COMPONENT_FILES: dict[str, tuple[str, ...]] = {
    "model": (
        "v5_model/config.py",
        "v5_model/embedding.py",
        "v5_model/attention.py",
        "v5_model/block.py",
        "v5_model/core.py",
        "v5_model/initialize.py",
    ),
    "trainer": (
        "v5_training/trainer.py",
        "v5_training/production_backend.py",
        "v5_training/optimizer.py",
        "v5_training/schedule.py",
        "v5_training/step.py",
        "v5_training/state.py",
        "v5_training/runner.py",
        "v5_training/checkpoint.py",
    ),
    "data": (
        "v5_data/pack.py",
        "v5_data/stream.py",
        "v5_data/manifest.py",
        "v5_data/cursor.py",
        "v5_data/sourceset.py",
        "v5_data/split.py",
    ),
    "generator": (
        "e0_cognition/training_generators.py",
        "e0_cognition/evaluation_generators.py",
    ),
    "evaluation": (
        "v5_evaluation/protocol.py",
        "v5_evaluation/firewall.py",
        "v5_evaluation/fixture.py",
        "v5_evaluation/metrics.py",
        "v5_evaluation/stats.py",
        "v5_evaluation/checkpoint_adapter.py",
    ),
    "analysis": (
        "v5_evaluation/stats.py",
        "v5_promotion/gates.py",
    ),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _component_sha(repo: Path, files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        path = repo / relative
        if not path.is_file():
            raise ValueError(f"code-freeze component missing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CodeFreezeReceipt:
    schema: str
    source_commit: str
    component_shas: tuple[tuple[str, str], ...]
    experiment_spec_sha256: str

    def assert_valid(self) -> None:
        if self.schema != FREEZE_SCHEMA:
            raise ValueError("unsupported code-freeze schema")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise ValueError("source commit must be a full lowercase git SHA-1")
        if {name for name, _ in self.component_shas} != set(COMPONENT_FILES):
            raise ValueError("code freeze must bind every pipeline component")
        for _name, digest in self.component_shas:
            if len(digest) != 64:
                raise ValueError("component identities must be SHA-256")
        if len(self.experiment_spec_sha256) != 64:
            raise ValueError("experiment spec identity must be SHA-256")

    def sha256(self) -> str:
        self.assert_valid()
        return _sha256_hex(
            _canonical_json(
                {
                    "schema": self.schema,
                    "source_commit": self.source_commit,
                    "component_shas": [list(item) for item in self.component_shas],
                    "experiment_spec_sha256": self.experiment_spec_sha256,
                }
            )
        )


def freeze_code(
    repo: Path, *, source_commit: str, experiment_spec_sha256: str
) -> CodeFreezeReceipt:
    """Hash every pipeline implementation file at the pinned tree state."""

    if len(experiment_spec_sha256) != 64:
        raise ValueError("experiment spec identity must be SHA-256")
    receipt = CodeFreezeReceipt(
        schema=FREEZE_SCHEMA,
        source_commit=source_commit,
        component_shas=tuple(
            (name, _component_sha(repo, files)) for name, files in COMPONENT_FILES.items()
        ),
        experiment_spec_sha256=experiment_spec_sha256,
    )
    receipt.assert_valid()
    return receipt


__all__ = ["COMPONENT_FILES", "FREEZE_SCHEMA", "CodeFreezeReceipt", "freeze_code"]
