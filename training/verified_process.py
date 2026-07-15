"""Verifier-grounded token weighting for the canonical V4 objective.

Only rows from the immutable ``verified_dfc`` source are eligible.  The
objective increases the next-token weight inside complete, explicitly tagged
reasoning/correction spans; malformed or window-truncated spans receive no
extra weight.  It adds no parameters and no inference-time path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch

VERIFIED_PROCESS_OBJECTIVE = "verified_dfc_process_spans_v1"
PROCESS_SPAN_TAGS = ("hyp", "verify", "err", "upd")


@dataclass(frozen=True)
class VerifiedProcessReport:
    eligible_rows: int
    rows_with_complete_spans: int
    complete_spans: int
    weighted_tokens: int
    malformed_or_truncated_spans: int
    multiplier: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _tag_pairs(special_token_ids: Mapping[str, int]) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for name in PROCESS_SPAN_TAGS:
        opening = f"<{name}>"
        closing = f"</{name}>"
        if opening not in special_token_ids or closing not in special_token_ids:
            raise ValueError(f"canonical tokenizer is missing {opening}/{closing}")
        pairs.append((int(special_token_ids[opening]), int(special_token_ids[closing])))
    return tuple(pairs)


def apply_verified_process_weights(
    target_ids: torch.Tensor,
    base_weights: torch.Tensor,
    *,
    verified_rows: torch.Tensor,
    special_token_ids: Mapping[str, int],
    multiplier: float,
) -> tuple[torch.Tensor, VerifiedProcessReport]:
    """Return token weights with extra mass on complete verified process spans."""

    if target_ids.ndim != 2 or base_weights.shape != target_ids.shape:
        raise ValueError("target IDs and base weights must have the same [batch, time] shape")
    if verified_rows.ndim != 1 or verified_rows.shape[0] != target_ids.shape[0]:
        raise ValueError("verified_rows must contain one boolean per batch row")
    factor = float(multiplier)
    if not 1.0 <= factor <= 2.0:
        raise ValueError("verified process multiplier must be in [1.0, 2.0]")

    pairs = _tag_pairs(special_token_ids)
    output = base_weights.clone()
    eligible_rows = int(verified_rows.to(dtype=torch.int64).sum().item())
    rows_with_spans = 0
    span_count = 0
    weighted_tokens = 0
    incomplete = 0

    for row_index in range(target_ids.shape[0]):
        if not bool(verified_rows[row_index].item()):
            continue
        row = target_ids[row_index]
        row_had_span = False
        for opening, closing in pairs:
            open_position: int | None = None
            for position, raw_token in enumerate(row.tolist()):
                token = int(raw_token)
                if token == opening:
                    if open_position is not None:
                        incomplete += 1
                    open_position = position
                    continue
                if token != closing:
                    continue
                if open_position is None:
                    incomplete += 1
                    continue
                start = open_position + 1
                stop = position
                if stop > start:
                    output[row_index, start:stop] *= factor
                    count = stop - start
                    weighted_tokens += count
                    span_count += 1
                    row_had_span = True
                open_position = None
            if open_position is not None:
                incomplete += 1
        rows_with_spans += int(row_had_span)

    return output, VerifiedProcessReport(
        eligible_rows=eligible_rows,
        rows_with_complete_spans=rows_with_spans,
        complete_spans=span_count,
        weighted_tokens=weighted_tokens,
        malformed_or_truncated_spans=incomplete,
        multiplier=factor,
    )
