"""Causal cross-entropy: the V5 launch objective.

Replica-global mean over eligible target tokens; label smoothing and z-loss
are zero. BOS and PAD never enter the loss; content and EOS do; positions
that cross a pack-segment boundary are excluded.
"""

from __future__ import annotations

from typing import Any


def causal_lm_loss(
    logits: Any,
    tokens: Any,
    segment_ids: Any,
    *,
    bos_id: int = 2,
    pad_id: int = 0,
    torch_module: Any = None,
) -> tuple[Any, int]:
    """Shift once; exclude BOS/PAD and segment transitions; include EOS targets."""

    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if logits.ndim != 3 or logits.shape[:2] != tokens.shape or tokens.shape != segment_ids.shape:
        raise ValueError("incompatible logits/tokens/segments")
    targets = tokens[:, 1:]
    keep = (segment_ids[:, 1:] == segment_ids[:, :-1]) & (segment_ids[:, 1:] >= 0)
    keep = keep & (targets != bos_id) & (targets != pad_id)
    count = int(keep.sum().item())
    if count == 0:
        raise ValueError("batch has no supervised targets")
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    return (losses * keep.reshape(-1)).sum() / count, count


__all__ = ["causal_lm_loss"]
