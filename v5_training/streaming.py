"""Window-to-batch conversion on the canonical training spine.

The single adapter between the canonical data plane (``v5_data.stream``) and
the production backend: converts one ``UpdateWindow`` into a validated
``PackedBatch`` with the real sampler cursor coordinates the training state
machine advances.  Production drivers, canaries, and miniatures all consume
this so there is no parallel batch-construction path.
"""

from __future__ import annotations

from typing import Any

from .production_backend import PackedBatch
from .state import CURSOR_SCHEMA, CursorState
from v5_data.stream import UpdateWindow


def batch_from_window(
    window: UpdateWindow,
    *,
    pack_manifest_sha256: str,
    update_ordinal: int,
    device: Any | None = None,
    torch_module: Any = None,
    rng_state_sha256: str = "0" * 64,
) -> PackedBatch:
    """Materialize one update window as a backend batch."""

    if torch_module is None:
        import torch as torch_module
    per_source: dict[str, int] = {}
    for sequence in window.sequences:
        for index, source in enumerate(sequence.sources):
            count = sum(1 for segment in sequence.segment_ids if segment == index)
            per_source[source] = per_source.get(source, 0) + count
    total = sum(per_source.values())
    if total != window.real_tokens:
        raise ValueError(
            f"window ledger claims {total} real tokens but the window is sized for "
            f"{window.real_tokens}"
        )
    tokens = torch_module.tensor(
        [sequence.tokens for sequence in window.sequences], dtype=torch_module.long
    )
    segment_ids = torch_module.tensor(
        [sequence.segment_ids for sequence in window.sequences],
        dtype=torch_module.int32,
    )
    if device is not None:
        tokens = tokens.to(device)
        segment_ids = segment_ids.to(device)
    return PackedBatch(
        tokens=tokens,
        segment_ids=segment_ids,
        tokens_by_source=dict(sorted(per_source.items())),
        cursor=CursorState(
            CURSOR_SCHEMA,
            pack_manifest_sha256,
            window.shard_ordinal,
            window.sequence_ordinal,
            window.real_tokens * (update_ordinal + 1),
        ),
        rng_state_sha256=rng_state_sha256,
    )


__all__ = ["batch_from_window"]
