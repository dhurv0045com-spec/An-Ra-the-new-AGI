"""Sequence construction and collocation (B02).

Rows carry explicit EOS, per-token supervised flags (the actual loss
denominator), episode segment IDs and a strict provenance sidecar that is
never serialized into learned tokens. Packing several episodes into one row
inserts boundary markers and renumbers segments so the model cannot attend
across protected episode boundaries. Padding and packing change no target's
supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from bramastra_lab.research.experience.codec import (
    DEFAULT_MAX_EVENT_BYTES,
    SPECIAL_BOUNDARY,
    SPECIAL_EOS,
    SPECIAL_PAD,
    EncodingError,
    encode_event,
    encode_text,
)

PROVENANCE_FIELDS = frozenset({
    "kind", "episode_id", "task_semantic_id", "split", "source", "collection_policy",
    "pair_group_id", "packed_rows",
})
FORBIDDEN_PROVENANCE_FIELDS = frozenset({
    "answer", "label", "gold", "target", "reward", "prediction", "probability",
})


class SequenceError(ValueError):
    """A sequence record violates the declared batch contract."""


@dataclass(frozen=True)
class SequenceRow:
    """One unshifted token row with supervision flags and provenance."""

    tokens: tuple[int, ...]
    supervised: tuple[bool, ...]
    provenance: Mapping[str, str]
    segment_ids: tuple[int, ...]
    pair_group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or not self.tokens:
            raise SequenceError("row tokens must be a nonempty tuple")
        if len(self.supervised) != len(self.tokens) or len(self.segment_ids) != len(self.tokens):
            raise SequenceError("supervised/segment flags must align with tokens")
        if self.tokens[-1] != SPECIAL_EOS:
            raise SequenceError("every row must terminate with an explicit EOS")
        if any(not isinstance(token, int) or isinstance(token, bool) or not 0 <= token <= SPECIAL_BOUNDARY
               for token in self.tokens):
            raise SequenceError("row tokens must be inside the 260-token vocabulary")
        unknown = set(self.provenance) - set(PROVENANCE_FIELDS)
        if unknown:
            raise SequenceError(f"provenance has undeclared fields: {sorted(unknown)}")
        leaking = set(self.provenance) & FORBIDDEN_PROVENANCE_FIELDS
        if leaking:
            raise SequenceError(f"provenance must never carry answer fields: {sorted(leaking)}")
        if not {"kind", "episode_id", "split"} <= set(self.provenance):
            raise SequenceError("provenance requires kind, episode_id and split")
        for key, value in self.provenance.items():
            if not isinstance(value, str):
                raise SequenceError(f"provenance field {key} must be a string")

    @property
    def target_count(self) -> int:
        """Supervised targets after the single causal shift."""
        return sum(1 for flag in self.supervised[1:] if flag)

    def to_sidecar(self) -> dict[str, Any]:
        sidecar = dict(self.provenance)
        sidecar["length"] = len(self.tokens)
        sidecar["target_count"] = self.target_count
        if self.pair_group_id is not None:
            sidecar["pair_group_id"] = self.pair_group_id
        return sidecar


def _validate_provenance(provenance: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(provenance, Mapping):
        raise SequenceError("provenance must be a mapping of strings")
    return dict(provenance)


def build_language_row(
    text: str,
    *,
    provenance: Mapping[str, str],
    max_tokens: int,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
) -> SequenceRow:
    """A causal language row: boundary + bytes + EOS, every token supervised."""
    _validate_provenance(provenance)
    encoded = encode_text(text)
    if len(encoded) + 2 > max_tokens:
        raise EncodingError(
            f"language row needs {len(encoded) + 2} tokens; max_tokens is {max_tokens}")
    tokens = (SPECIAL_BOUNDARY, *encoded, SPECIAL_EOS)
    return SequenceRow(
        tokens=tokens,
        supervised=(True,) * len(tokens),
        provenance={**provenance, "kind": provenance.get("kind", "language")},
        segment_ids=(1,) * len(tokens),
    )


def build_answer_row(
    prompt_events: Iterable[tuple[str, Any]],
    answer_text: str,
    *,
    provenance: Mapping[str, str],
    max_tokens: int,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
) -> SequenceRow:
    """A goal-conditioned answer row: boundary + events + answer bytes + EOS.

    Loss applies only to answer bytes and the required EOS; prompt events are
    context. The answer is a training-time teacher string and never enters
    provenance.
    """
    _validate_provenance(provenance)
    tokens: list[int] = [SPECIAL_BOUNDARY]
    supervised: list[bool] = [False]
    for role, content in prompt_events:
        event_tokens = encode_event(role, content, max_event_bytes=max_event_bytes)
        tokens.extend(event_tokens)
        supervised.extend([False] * len(event_tokens))
    answer = encode_text(answer_text)
    tokens.extend(answer)
    supervised.extend([True] * len(answer))
    tokens.append(SPECIAL_EOS)
    supervised.append(True)
    if len(tokens) > max_tokens:
        raise EncodingError(
            f"answer row needs {len(tokens)} tokens; max_tokens is {max_tokens}")
    return SequenceRow(
        tokens=tuple(tokens),
        supervised=tuple(supervised),
        provenance={**provenance, "kind": provenance.get("kind", "trajectory")},
        segment_ids=(1,) * len(tokens),
    )


def pack_rows(rows: list[SequenceRow], *, max_tokens: int) -> SequenceRow:
    """Pack rows into one sequence with per-episode segments and boundaries.

    Supervision flags are preserved exactly; the inserted boundary token is an
    unsupervised input belonging to the following episode's segment. Packing
    cannot change which targets receive loss.
    """
    if not rows:
        raise SequenceError("cannot pack an empty row list")
    boundaries_needed = len(rows) - 1
    total = sum(len(row.tokens) for row in rows) + boundaries_needed
    if total > max_tokens:
        raise SequenceError(
            f"packed row needs {total} tokens; max_tokens is {max_tokens}")
    tokens: list[int] = []
    supervised: list[bool] = []
    segment_ids: list[int] = []
    for index, row in enumerate(rows):
        if index > 0:
            tokens.append(SPECIAL_BOUNDARY)
            supervised.append(False)
            segment_ids.append(index + 1)
        tokens.extend(row.tokens)
        supervised.extend(row.supervised)
        segment_ids.extend([index + 1] * len(row.tokens))
    merged_provenance = {
        "kind": "packed",
        "episode_id": "|".join(row.provenance["episode_id"] for row in rows),
        "split": rows[0].provenance["split"],
        "packed_rows": str(len(rows)),
    }
    packed = SequenceRow(
        tokens=tuple(tokens),
        supervised=tuple(supervised),
        provenance=merged_provenance,
        segment_ids=tuple(segment_ids),
        pair_group_id=rows[0].pair_group_id if all(
            row.pair_group_id == rows[0].pair_group_id for row in rows) else None,
    )
    expected_targets = sum(row.target_count for row in rows)
    if packed.target_count != expected_targets:
        raise SequenceError("packing changed the supervised target set")
    return packed


@dataclass(frozen=True)
class CollocatedBatch:
    """Tensor batch plus provenance sidecars and exact counters."""

    input_ids: torch.Tensor            # [B, T] long
    padding_mask: torch.Tensor         # [B, T] bool, True = real token
    labels: torch.Tensor               # [B, T] long, -100 where no loss
    loss_mask: torch.Tensor            # [B, T] bool, True = supervised target
    segment_ids: torch.Tensor          # [B, T] long, 0 = padding
    target_count: int
    pair_group_ids: tuple[str | None, ...]
    provenance: tuple[Mapping[str, Any], ...]
    sidecar_identity: str

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.input_ids.shape[1]


def collocate(rows: list[SequenceRow], *, max_seq: int) -> CollocatedBatch:
    """Pad and shift rows into model-ready tensors.

    ``input_ids`` predicts ``labels`` at the same position (inputs are the row
    shifted left by one). Positions with ``loss_mask=False`` carry label -100.
    An exact-full batch has zero padding and works identically.
    """
    if not rows:
        raise SequenceError("cannot collocate an empty row list")
    # A row of n tokens occupies n-1 positions: inputs tokens[:-1] predict
    # labels tokens[1:]. Padding covers only rows shorter than the longest.
    length = max(len(row.tokens) for row in rows) - 1
    if length > max_seq:
        raise SequenceError(f"row length {length + 1} exceeds max_seq {max_seq}")
    batch = len(rows)
    input_ids = torch.full((batch, length), SPECIAL_PAD, dtype=torch.long)
    padding_mask = torch.zeros((batch, length), dtype=torch.bool)
    labels = torch.full((batch, length), -100, dtype=torch.long)
    loss_mask = torch.zeros((batch, length), dtype=torch.bool)
    segment_ids = torch.zeros((batch, length), dtype=torch.long)
    for index, row in enumerate(rows):
        size = len(row.tokens) - 1
        input_ids[index, :size] = torch.tensor(row.tokens[:-1], dtype=torch.long)
        padding_mask[index, :size] = True
        target_tokens = torch.tensor(row.tokens[1:], dtype=torch.long)
        target_flags = torch.tensor([bool(flag) for flag in row.supervised[1:]], dtype=torch.bool)
        labels[index, :size] = torch.where(target_flags, target_tokens, -100)
        loss_mask[index, :size] = target_flags
        segment_ids[index, :size] = torch.tensor(row.segment_ids[:-1], dtype=torch.long)
    target_count = int(loss_mask.sum().item())
    declared = sum(row.target_count for row in rows)
    if target_count != declared:
        raise SequenceError("collocation changed the supervised denominator")
    sidecars = [row.to_sidecar() for row in rows]
    pair_group_ids = tuple(row.pair_group_id for row in rows)
    from bramastra_lab.research.contracts.core import content_identity

    return CollocatedBatch(
        input_ids=input_ids,
        padding_mask=padding_mask,
        labels=labels,
        loss_mask=loss_mask,
        segment_ids=segment_ids,
        target_count=target_count,
        pair_group_ids=pair_group_ids,
        provenance=tuple(sidecars),
        sidecar_identity=content_identity(sidecars),
    )
