"""Canonical global-to-rank materialization for production microsteps.

The production loop and its resume fingerprint must agree on padding,
replica row assignment, source ledgers, and the update-wide denominator. This
module owns that post-sampler transformation; it does not plan lane windows or
qualify a runtime's RNG behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from v5_data.bucket_cursor import LaneWindow
from v5_training.topology_map import replica_shards


def count_eligible_targets(
    window: Any, *, bos_id: int = 2, pad_id: int = 0,
) -> int:
    """Count the shifted targets accepted by the causal-LM loss contract."""

    total = 0
    for tokens, segments, eligible in zip(
        window.tokens, window.segment_ids, window.eligible,
    ):
        for position in range(1, len(tokens)):
            if (eligible[position]
                    and segments[position] == segments[position - 1]
                    and segments[position] >= 0
                    and tokens[position] != bos_id
                    and tokens[position] != pad_id):
                total += 1
    return total


def count_rank_real_tokens(
    microsteps: Sequence["MaterializedMicrostep"], *, pad_id: int = 0,
) -> int:
    """Count cursor-selected tokens on one rank, including BOS/EOS tokens.

    The distributed training ledger counts all real packed input tokens, not
    only shifted loss targets. The sampler's eligibility mask distinguishes
    newly consumed positions from context carried in a packed row; segment IDs
    distinguish real packed positions from padding. This count partitions
    exactly across replica row shards without charging reused context twice.
    """

    if not microsteps:
        raise ValueError("rank token accounting requires at least one microstep")
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("rank token accounting pad id must be nonnegative")
    total = 0
    for microstep in microsteps:
        if not isinstance(microstep, MaterializedMicrostep):
            raise ValueError("rank token accounting requires materialized microsteps")
        if not (len(microstep.tokens) == len(microstep.segment_ids)
                == len(microstep.eligible)):
            raise ValueError("rank token, segment, and eligibility rows disagree")
        for tokens, segments, eligible in zip(
            microstep.tokens, microstep.segment_ids, microstep.eligible,
        ):
            if not (len(tokens) == len(segments) == len(eligible)):
                raise ValueError("rank token, segment, and eligibility widths disagree")
            for token, segment, is_eligible in zip(tokens, segments, eligible):
                if type(token) is not int or token < 0:
                    raise ValueError("rank token IDs must be nonnegative integers")
                if type(segment) is not int or segment < -1:
                    raise ValueError("rank segment IDs must be integers >= -1")
                if type(is_eligible) is not bool:
                    raise ValueError("rank eligibility values must be booleans")
                if segment == -1:
                    if token != pad_id or is_eligible:
                        raise ValueError("rank padding must use pad ID and be ineligible")
                else:
                    if token == pad_id:
                        raise ValueError("rank real-token segment cannot contain padding")
                    if is_eligible:
                        total += 1
    return total


@dataclass(frozen=True, slots=True)
class MaterializedMicrostep:
    """One padded global or rank-local microstep and its shared loss ledger."""

    bucket: int
    family: str
    subfamily: str
    tokens: tuple[tuple[int, ...], ...]
    segment_ids: tuple[tuple[int, ...], ...]
    eligible: tuple[tuple[bool, ...], ...]
    tokens_by_source: Mapping[str, int]
    planned_total: int

    def fingerprint_mapping(self) -> dict[str, object]:
        """Return the canonical mapping consumed by the resume fingerprint."""

        return {
            "bucket": self.bucket,
            "family": self.family,
            "subfamily": self.subfamily,
            "tokens": [list(row) for row in self.tokens],
            "segment_ids": [list(row) for row in self.segment_ids],
            "eligible": [list(row) for row in self.eligible],
            "tokens_by_source": dict(self.tokens_by_source),
            "planned_total": self.planned_total,
        }


def materialize_rank_microsteps(
    windows: Sequence[tuple[int, str, str, LaneWindow]],
    *,
    replicas: int,
    rank: int | None,
    planned_total: int,
    pad_id: int = 0,
) -> tuple[MaterializedMicrostep, ...]:
    """Pad planned lane windows and optionally select one physical replica.

    ``rank=None`` is the CPU/CUDA logical-global path: it receives every row
    as one tensor. A concrete rank uses the same deterministic contiguous
    sharding contract as XLA. The denominator remains global and identical on
    all ranks for every accumulation microstep.
    """

    if type(replicas) is not int or replicas <= 0:
        raise ValueError("microstep replica count must be a positive integer")
    if rank is not None and (type(rank) is not int or not 0 <= rank < replicas):
        raise ValueError("microstep rank is outside the replica world")
    if type(planned_total) is not int or planned_total <= 0:
        raise ValueError("microstep update-wide denominator must be positive")
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("microstep pad token id must be a nonnegative integer")
    if not windows:
        raise ValueError("an optimizer update must contain at least one microstep")

    result: list[MaterializedMicrostep] = []
    for item in windows:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            raise ValueError("planned window must be (bucket, family, subfamily, window)")
        bucket, family, subfamily, window = item
        if type(bucket) is not int or bucket <= 0:
            raise ValueError("microstep bucket must be a positive integer")
        if not isinstance(family, str) or not isinstance(subfamily, str):
            raise ValueError("microstep family identities must be strings")
        if not isinstance(window, LaneWindow):
            raise ValueError("planned microsteps must contain production LaneWindow values")
        rows = window.tokens
        segments = window.segment_ids
        eligibility = window.eligible
        source_ledger = window.tokens_by_source
        if not isinstance(rows, (tuple, list)) or not rows:
            raise ValueError("lane window must contain at least one row")
        if not isinstance(segments, (tuple, list)) or not isinstance(eligibility, (tuple, list)):
            raise ValueError("lane window segment and eligibility rows are required")
        if not (len(rows) == len(segments) == len(eligibility)):
            raise ValueError("lane window token, segment, and eligibility rows disagree")
        if not isinstance(source_ledger, Mapping) or any(
            not isinstance(source, str) or not source
            or type(count) is not int or count < 0
            for source, count in source_ledger.items()
        ):
            raise ValueError("lane window source-token ledger is invalid")
        if (type(window.real_tokens) is not int or window.real_tokens <= 0
                or sum(source_ledger.values()) != window.real_tokens):
            raise ValueError("lane window source-token ledger disagrees with real-token count")
        if tuple(map(len, rows)) != tuple(window.row_widths):
            raise ValueError("lane window row-width receipt disagrees with token rows")

        padded_rows: list[tuple[list[int], list[int], list[bool]]] = []
        for token_row, segment_row, eligible_row in zip(rows, segments, eligibility):
            if not all(isinstance(row, (tuple, list)) for row in
                       (token_row, segment_row, eligible_row)):
                raise ValueError("lane window rows must be sequences")
            if not (len(token_row) == len(segment_row) == len(eligible_row)):
                raise ValueError("lane window row tensors have different lengths")
            if len(token_row) > bucket:
                raise ValueError("lane window row exceeds its declared bucket")
            if any(type(token) is not int or token < 0 for token in token_row):
                raise ValueError("lane window token ids must be nonnegative integers")
            if any(type(segment) is not int or segment < -1 for segment in segment_row):
                raise ValueError("lane window segment ids must be integers >= -1")
            if any(type(flag) is not bool for flag in eligible_row):
                raise ValueError("lane window eligibility values must be booleans")
            if any(flag and segment < 0 for flag, segment in zip(eligible_row, segment_row)):
                raise ValueError("lane window padding positions cannot be eligible targets")
            saw_padding = False
            previous_segment = -1
            for token, segment in zip(token_row, segment_row):
                if segment < 0:
                    if token != pad_id:
                        raise ValueError("lane window padding must use the pad token id")
                    saw_padding = True
                else:
                    if saw_padding:
                        raise ValueError("lane window padding must be trailing")
                    if token == pad_id:
                        raise ValueError("real-token segments cannot contain the pad token id")
                    if segment < previous_segment:
                        raise ValueError(
                            "lane window segment ids must be nondecreasing; segments cannot reappear"
                        )
                    previous_segment = segment
            width = bucket - len(token_row)
            padded_rows.append((
                [*token_row, *([pad_id] * width)],
                [*segment_row, *([-1] * width)],
                [*eligible_row, *([False] * width)],
            ))

        if rank is None:
            selected = padded_rows
        else:
            # Every replica must receive the same physical row count. Preserve
            # contiguous source-row order and append only inert rows, so short
            # tails shard even when they contain fewer rows than replicas.
            # Eligibility keeps them out of loss and real-token accounting.
            replica_padding = (-len(padded_rows)) % replicas
            if replica_padding:
                empty_row = (
                    [pad_id] * bucket,
                    [-1] * bucket,
                    [False] * bucket,
                )
                padded_rows.extend(
                    ([*empty_row[0]], [*empty_row[1]], [*empty_row[2]])
                    for _ in range(replica_padding)
                )
            selected = replica_shards(padded_rows, replicas=replicas)[rank]
        result.append(MaterializedMicrostep(
            bucket=bucket,
            family=family,
            subfamily=subfamily,
            tokens=tuple(tuple(row[0]) for row in selected),
            segment_ids=tuple(tuple(row[1]) for row in selected),
            eligible=tuple(tuple(row[2]) for row in selected),
            tokens_by_source=dict(sorted(source_ledger.items())),
            planned_total=planned_total,
        ))
    return tuple(result)


__all__ = [
    "MaterializedMicrostep",
    "count_eligible_targets",
    "count_rank_real_tokens",
    "materialize_rank_microsteps",
]
