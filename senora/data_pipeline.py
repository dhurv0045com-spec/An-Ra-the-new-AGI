"""Deterministic training data pipeline, curriculum mixture arithmetic, and contamination guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Sequence

from v5_contracts.data_spec import DataManifest, SourceRecord
from v5_training.state import CursorState


from v5_training.state import CURSOR_SCHEMA
DATA_PIPELINE_SCHEMA = "senora-data-pipeline/v1"
TRAIN_TEMPLATE_PREFIX = "train.causal."


class MissingCorpusArtifactError(FileNotFoundError):
    """Raised when required external corpus artifacts or pack manifests are missing."""
    pass


class ContaminationViolationError(ValueError):
    """Raised when training data violates evaluation boundary or namespace separation."""
    pass


@dataclass(frozen=True, slots=True)
class MixtureRecipe:
    """Exact arithmetic mixture for training tokens."""
    name: str
    natural_fraction: float
    code_fraction: float
    cognition_fraction: float

    def assert_valid(self) -> None:
        fractions = [self.natural_fraction, self.code_fraction, self.cognition_fraction]
        if any(f < 0.0 or f > 1.0 for f in fractions):
            raise ValueError("all mixture fractions must be in [0.0, 1.0]")
        total = sum(fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mixture fractions must sum to 1.0, got {total}")

    def token_allocation(self, total_tokens: int) -> dict[str, int]:
        self.assert_valid()
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        natural = int(round(total_tokens * self.natural_fraction))
        code = int(round(total_tokens * self.code_fraction))
        cognition = total_tokens - natural - code
        return {
            "natural": natural,
            "code": code,
            "cognition": cognition,
        }


# Standard preregistered P35 experimental mixtures
MIXTURE_CONTROL_SUBSTRATE = MixtureRecipe("control-substrate-00", 0.75, 0.25, 0.0)
MIXTURE_COGNITION_05 = MixtureRecipe("cognition-mixture-05", 0.70, 0.25, 0.05)
MIXTURE_COGNITION_15 = MixtureRecipe("cognition-mixture-15", 0.65, 0.20, 0.15)
MIXTURE_COGNITION_30 = MixtureRecipe("cognition-mixture-30", 0.50, 0.20, 0.30)


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Token batch with exact provenance and source breakdown."""
    token_ids: list[list[int]]  # [batch_size, sequence_length]
    tokens_by_source: dict[str, int]
    new_cursor: CursorState
    batch_token_count: int


class DataPipeline:
    """Deterministic, cursor-tracked data pipeline for P35 training."""

    def __init__(
        self,
        *,
        pack_manifest: Any | None,
        recipe: MixtureRecipe,
        sequence_length: int = 2048,
        batch_size: int = 64,
        allow_synthetic_mock: bool = False,
    ) -> None:
        recipe.assert_valid()
        self.recipe = recipe
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.pack_manifest = pack_manifest
        self.allow_synthetic_mock = allow_synthetic_mock

        if pack_manifest is None and not allow_synthetic_mock:
            raise MissingCorpusArtifactError(
                "A verified pack manifest is required. External data pipeline fails closed. "
                "To run software correctness unit tests, explicitly set allow_synthetic_mock=True."
            )

    def assert_no_contamination(
        self,
        training_template_ids: Sequence[str],
        evaluation_template_ids: set[str],
    ) -> None:
        """Verify that training templates never overlap with evaluation templates."""
        for template_id in training_template_ids:
            if not template_id.startswith(TRAIN_TEMPLATE_PREFIX):
                raise ContaminationViolationError(
                    f"training template {template_id!r} does not use reserved prefix {TRAIN_TEMPLATE_PREFIX!r}"
                )
            if template_id in evaluation_template_ids:
                raise ContaminationViolationError(
                    f"contamination detected: template {template_id!r} is present in evaluation suite"
                )

    def mock_stream(
        self,
        *,
        initial_cursor: CursorState,
        total_batches: int = 10,
    ) -> Iterator[TrainingBatch]:
        """Produce deterministic synthetic mock batches for unit testing and plumbing certification only."""
        if not self.allow_synthetic_mock:
            raise RuntimeError("mock_stream cannot be called unless allow_synthetic_mock=True")

        cursor = initial_cursor
        tokens_per_batch = self.batch_size * self.sequence_length
        alloc = self.recipe.token_allocation(tokens_per_batch)

        for step in range(total_batches):
            # Create deterministic mock tokens
            batch_tokens = [
                [(step * 1000 + i * self.sequence_length + j) % 24576 for j in range(self.sequence_length)]
                for i in range(self.batch_size)
            ]
            cursor = CursorState(
                schema=CURSOR_SCHEMA,
                pack_manifest_sha256=cursor.pack_manifest_sha256,
                shard_ordinal=cursor.shard_ordinal,
                sequence_ordinal=cursor.sequence_ordinal + self.batch_size,
                token_offset=cursor.token_offset + tokens_per_batch,
            )
            yield TrainingBatch(
                token_ids=batch_tokens,
                tokens_by_source=dict(alloc),
                new_cursor=cursor,
                batch_token_count=tokens_per_batch,
            )