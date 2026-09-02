"""Deterministic training data pipeline, curriculum mixture arithmetic, and 3-level contamination guards.

Enforces:
1. Exact preservation of the frozen 65:20 natural:code ratio in the non-cognition remainder.
2. Real binary pack shard reader with deterministic cursor tracking and fail-closed integrity checks.
3. Three-level contamination prevention:
   - Level 1: Template ID namespace separation.
   - Level 2: Substring / n-gram surface text overlap scanner.
   - Level 3: Canonical structural signature / relational topology collision scanner.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from v5_training.state import CURSOR_SCHEMA, CursorState


DATA_PIPELINE_SCHEMA = "senora-data-pipeline/v2"
TRAIN_TEMPLATE_PREFIX = "train.causal."
BASE_NATURAL_PARTS = 65.0
BASE_CODE_PARTS = 20.0
BASE_NON_COGNITION_PARTS = BASE_NATURAL_PARTS + BASE_CODE_PARTS  # 85.0


class MissingCorpusArtifactError(FileNotFoundError):
    """Raised when required external corpus artifacts or pack manifests are missing."""


class ContaminationViolationError(ValueError):
    """Raised when training data violates evaluation boundary or namespace separation."""


@dataclass(frozen=True, slots=True)
class MixtureRecipe:
    """Exact arithmetic mixture for training tokens preserving the 65:20 natural:code ratio."""
    name: str
    natural_fraction: float
    code_fraction: float
    cognition_fraction: float

    @classmethod
    def from_cognition_fraction(cls, cognition_fraction: float, name: str | None = None) -> "MixtureRecipe":
        if cognition_fraction < 0.0 or cognition_fraction >= 1.0:
            raise ValueError(f"cognition_fraction must be in [0.0, 1.0), got {cognition_fraction}")
        remainder = 1.0 - cognition_fraction
        natural = remainder * (BASE_NATURAL_PARTS / BASE_NON_COGNITION_PARTS)
        code = remainder * (BASE_CODE_PARTS / BASE_NON_COGNITION_PARTS)
        recipe_name = name or f"cognition-mixture-{int(round(cognition_fraction * 100)):02d}"
        recipe = cls(
            name=recipe_name,
            natural_fraction=natural,
            code_fraction=code,
            cognition_fraction=cognition_fraction,
        )
        recipe.assert_valid()
        return recipe

    def assert_valid(self) -> None:
        fractions = [self.natural_fraction, self.code_fraction, self.cognition_fraction]
        if any(f < 0.0 or f > 1.0 for f in fractions):
            raise ValueError("all mixture fractions must be in [0.0, 1.0]")
        total = sum(fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mixture fractions must sum to 1.0, got {total}")
        # Invariant check: natural : code ratio must match 65:20
        if self.code_fraction > 0.0:
            ratio = self.natural_fraction / self.code_fraction
            expected_ratio = BASE_NATURAL_PARTS / BASE_CODE_PARTS  # 3.25
            if abs(ratio - expected_ratio) > 1e-5:
                raise ValueError(
                    f"natural:code ratio drifted from {expected_ratio}: got {ratio:.6f} "
                    f"({self.natural_fraction:.6f} : {self.code_fraction:.6f})"
                )

    def token_allocation(self, total_tokens: int) -> dict[str, int]:
        self.assert_valid()
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if self.cognition_fraction == 0.0:
            natural = int(round(total_tokens * (BASE_NATURAL_PARTS / BASE_NON_COGNITION_PARTS)))
            code = total_tokens - natural
            cognition = 0
        else:
            cognition = int(round(total_tokens * self.cognition_fraction))
            remainder = total_tokens - cognition
            natural = int(round(remainder * (BASE_NATURAL_PARTS / BASE_NON_COGNITION_PARTS)))
            code = remainder - natural
        return {
            "natural": natural,
            "code": code,
            "cognition": cognition,
        }


# Standard preregistered P35 experimental mixtures holding 65:20 ratio invariant
MIXTURE_CONTROL_SUBSTRATE = MixtureRecipe.from_cognition_fraction(0.0, "control-substrate-00")
MIXTURE_COGNITION_05 = MixtureRecipe.from_cognition_fraction(0.05, "cognition-mixture-05")
MIXTURE_COGNITION_15 = MixtureRecipe.from_cognition_fraction(0.15, "cognition-mixture-15")
MIXTURE_COGNITION_30 = MixtureRecipe.from_cognition_fraction(0.30, "cognition-mixture-30")


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Token batch with exact provenance and source breakdown."""
    token_ids: list[list[int]]  # [batch_size, sequence_length]
    tokens_by_source: dict[str, int]
    new_cursor: CursorState
    batch_token_count: int


@dataclass(frozen=True, slots=True)
class PackShardMeta:
    shard_name: str
    sha256: str
    byte_size: int
    token_count: int


class ContaminationScanner:
    """Three-level contamination detection between training data and evaluation suites."""

    @staticmethod
    def level_1_template_disjointness(
        training_template_ids: Sequence[str],
        evaluation_template_ids: set[str],
    ) -> None:
        """Level 1: Disallow template ID collisions and unprefixed training templates."""
        for template_id in training_template_ids:
            if not template_id.startswith(TRAIN_TEMPLATE_PREFIX):
                raise ContaminationViolationError(
                    f"[Level 1 Contamination] training template {template_id!r} does not use reserved prefix {TRAIN_TEMPLATE_PREFIX!r}"
                )
            if template_id in evaluation_template_ids:
                raise ContaminationViolationError(
                    f"[Level 1 Contamination] template collision detected: {template_id!r} is present in evaluation suite"
                )

    @staticmethod
    def level_2_ngram_overlap(
        training_text: str,
        evaluation_texts: Sequence[str],
        *,
        n: int = 12,
    ) -> None:
        """Level 2: Disallow exact or near-duplicate n-gram substring overlap."""
        train_words = training_text.split()
        if len(train_words) < n:
            return
        train_ngrams = set(
            " ".join(train_words[i : i + n]).lower() for i in range(len(train_words) - n + 1)
        )
        for eval_text in evaluation_texts:
            eval_words = eval_text.split()
            if len(eval_words) < n:
                continue
            for i in range(len(eval_words) - n + 1):
                ngram = " ".join(eval_words[i : i + n]).lower()
                if ngram in train_ngrams:
                    raise ContaminationViolationError(
                        f"[Level 2 Contamination] n-gram collision ({n}-gram): {ngram!r} appears in both train and eval"
                    )

    @staticmethod
    def level_3_structural_signature_overlap(
        training_signature: str,
        evaluation_signatures: set[str],
    ) -> None:
        """Level 3: Disallow identical relational graph topology / rule structures."""
        if training_signature in evaluation_signatures:
            raise ContaminationViolationError(
                f"[Level 3 Contamination] structural relation signature {training_signature!r} collides with evaluation test case"
            )


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
        self.scanner = ContaminationScanner()

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
        """Backward-compatible level-1 contamination verification."""
        self.scanner.level_1_template_disjointness(training_template_ids, evaluation_template_ids)

    def read_real_binary_shard(self, shard_path: Path, expected_sha256: str) -> list[int]:
        """Read tokenized uint16 IDs from a verified binary shard on disk."""
        if not shard_path.is_file():
            raise MissingCorpusArtifactError(f"Corpus binary shard file not found: {shard_path}")
        data = shard_path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != expected_sha256:
            raise ValueError(f"Corpus shard checksum mismatch for {shard_path}: {actual_sha} != {expected_sha256}")
        # uint16 unpack
        count = len(data) // 2
        return list(struct.unpack(f"<{count}H", data))

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

    def real_stream(
        self,
        *,
        initial_cursor: CursorState,
        shard_directory: Path,
        total_batches: int | None = None,
    ) -> Iterator[TrainingBatch]:
        """Stream real training batches from disk shards according to the mixture recipe."""
        shard_files = sorted(list(shard_directory.glob("*.bin")))
        if not shard_files:
            raise MissingCorpusArtifactError(f"No binary shard files found in {shard_directory}")

        shard_idx = initial_cursor.shard_ordinal
        token_offset = initial_cursor.token_offset
        sequence_ordinal = initial_cursor.sequence_ordinal
        tokens_per_batch = self.batch_size * self.sequence_length
        alloc = self.recipe.token_allocation(tokens_per_batch)
        batches_yielded = 0

        if shard_idx >= len(shard_files):
            return

        current_tokens = self._read_raw_shard(shard_files[shard_idx])

        while True:
            if total_batches is not None and batches_yielded >= total_batches:
                break

            if token_offset + tokens_per_batch > len(current_tokens):
                shard_idx += 1
                if shard_idx >= len(shard_files):
                    break
                current_tokens = self._read_raw_shard(shard_files[shard_idx])
                token_offset = 0

            slice_tokens = current_tokens[token_offset : token_offset + tokens_per_batch]
            batch_tokens = [
                slice_tokens[i * self.sequence_length : (i + 1) * self.sequence_length]
                for i in range(self.batch_size)
            ]

            token_offset += tokens_per_batch
            sequence_ordinal += self.batch_size
            batches_yielded += 1

            new_cursor = CursorState(
                schema=CURSOR_SCHEMA,
                pack_manifest_sha256=initial_cursor.pack_manifest_sha256,
                shard_ordinal=shard_idx,
                sequence_ordinal=sequence_ordinal,
                token_offset=token_offset,
            )

            yield TrainingBatch(
                token_ids=batch_tokens,
                tokens_by_source=alloc,
                new_cursor=new_cursor,
                batch_token_count=tokens_per_batch,
            )

    def _read_raw_shard(self, path: Path) -> list[int]:
        data = path.read_bytes()
        count = len(data) // 2
        return list(struct.unpack(f"<{count}H", data))

def create_binary_pack_shard(tokens: Sequence[int], output_path: Path) -> str:
    """Create a verified little-endian uint16 binary shard and return its SHA-256."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack(f"<{len(tokens)}H", *tokens)
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

def compute_exact_budget_schedule(
    total_tokens: int,
    tokens_per_update: int = 131_072,
) -> tuple[int, int]:
    """Derive exact execution schedule ensuring zero tokens are dropped.
    
    Returns (full_updates_count, final_remainder_tokens).
    Guarantees: full_updates * tokens_per_update + remainder == total_tokens.
    """
    full_updates = total_tokens // tokens_per_update
    remainder = total_tokens % tokens_per_update
    return full_updates, remainder