"""Causal cross-entropy: the V5 launch objective.

Replica-global mean over eligible target tokens; label smoothing and z-loss
are zero. BOS and PAD never enter the loss; content and EOS do; positions
that cross a pack-segment boundary are excluded. An optional boolean
``eligible`` mask over [batch, length] restricts the loss to budgeted real
tokens for exact token accounting; padding is never eligible. Empty-rank
training may opt into a graph-connected zero contribution with ``allow_empty``;
ordinary callers remain fail-closed when no target is supervised.
"""

from __future__ import annotations

from typing import Any

from v5_checkpointing import activation_checkpoint


def _target_mask(
    tokens: Any,
    segment_ids: Any,
    *,
    bos_id: int,
    pad_id: int,
    eligible: Any | None,
) -> tuple[Any, Any]:
    if tokens.ndim != 2 or tokens.shape != segment_ids.shape:
        raise ValueError("tokens and segment IDs must be equal rank-two tensors")
    targets = tokens[:, 1:]
    keep = (segment_ids[:, 1:] == segment_ids[:, :-1]) & (segment_ids[:, 1:] >= 0)
    keep = keep & (targets != bos_id) & (targets != pad_id)
    if eligible is not None:
        if eligible.shape != tokens.shape:
            raise ValueError("eligibility mask must match token shape")
        keep = keep & eligible[:, 1:]
    return targets, keep


def causal_lm_loss(
    logits: Any,
    tokens: Any,
    segment_ids: Any,
    *,
    bos_id: int = 2,
    pad_id: int = 0,
    eligible: Any | None = None,
    return_numerator: bool = False,
    allow_empty: bool = False,
    torch_module: Any = None,
) -> tuple[Any, int] | tuple[Any, int, Any]:
    """Shift once; exclude BOS/PAD and segment transitions; include EOS targets."""

    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if logits.ndim != 3 or logits.shape[:2] != tokens.shape:
        raise ValueError("incompatible logits/tokens/segments")
    targets, keep = _target_mask(
        tokens, segment_ids, bos_id=bos_id, pad_id=pad_id, eligible=eligible,
    )
    if type(return_numerator) is not bool:
        raise ValueError("return_numerator must be boolean")
    if type(allow_empty) is not bool:
        raise ValueError("allow_empty must be boolean")
    count = int(keep.sum().item())
    if count == 0:
        if not allow_empty:
            raise ValueError("batch has no supervised targets")
        # An empty replica still joins the same gradient collective as ranks
        # with targets. Attach zero to every output so backward materializes
        # zero gradients for the model. Multiplying before reduction avoids
        # overflowing a sum of large finite logits; NaN/Inf still propagate
        # into this value and are rejected by production loss certification.
        numerator = logits.float().mul(0.0).sum()
        if return_numerator:
            return numerator, 0, numerator
        return numerator, 0
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    numerator = (losses * keep.reshape(-1)).sum()
    mean = numerator / count
    if return_numerator:
        return mean, count, numerator
    return mean, count


def causal_lm_loss_from_hidden(
    hidden: Any,
    output_weight: Any,
    tokens: Any,
    segment_ids: Any,
    *,
    bos_id: int = 2,
    pad_id: int = 0,
    eligible: Any | None = None,
    return_numerator: bool = False,
    allow_empty: bool = False,
    chunk_tokens: int = 512,
    torch_module: Any = None,
) -> tuple[Any, int] | tuple[Any, int, Any]:
    """Exact causal loss with checkpointed, token-chunked tied projection.

    The full-vocabulary output matrix is recomputed chunk by chunk in backward
    so training does not retain a ``[batch, sequence, vocabulary]`` logits
    activation. The loss, eligible-token mask, and tied output weights are
    identical to :func:`causal_lm_loss`.
    """

    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if (hidden.ndim != 3 or hidden.shape[:2] != tokens.shape
            or tokens.shape != segment_ids.shape):
        raise ValueError("incompatible hidden/tokens/segments")
    if (output_weight.ndim != 2
            or output_weight.shape[1] != hidden.shape[-1]):
        raise ValueError("output projection weight does not match hidden width")
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("output-projection chunk size must be a positive integer")
    if type(return_numerator) is not bool:
        raise ValueError("return_numerator must be boolean")
    if type(allow_empty) is not bool:
        raise ValueError("allow_empty must be boolean")

    targets, keep = _target_mask(
        tokens, segment_ids, bos_id=bos_id, pad_id=pad_id, eligible=eligible,
    )
    count = int(keep.sum().item())
    if count == 0:
        if not allow_empty:
            raise ValueError("batch has no supervised targets")
        numerator = hidden.float().mul(0.0).sum() + output_weight.float().mul(0.0).sum()
        if return_numerator:
            return numerator, 0, numerator
        return numerator, 0

    def chunk_loss(hidden_chunk, target_chunk, keep_chunk, weight):
        logits = torch.nn.functional.linear(hidden_chunk, weight)
        losses = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, weight.shape[0]),
            target_chunk.reshape(-1),
            reduction="none",
        )
        return (losses * keep_chunk.reshape(-1)).sum()

    numerator = None
    predictions = hidden[:, :-1, :]
    for start in range(0, predictions.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, predictions.shape[1])
        hidden_chunk = predictions[:, start:stop, :]
        targets_chunk = targets[:, start:stop]
        keep_chunk = keep[:, start:stop]
        if torch.is_grad_enabled() and (
            hidden_chunk.requires_grad or output_weight.requires_grad
        ):
            chunk_numerator = activation_checkpoint(
                chunk_loss,
                hidden_chunk,
                targets_chunk,
                keep_chunk,
                output_weight,
                torch_module=torch,
            )
        else:
            chunk_numerator = chunk_loss(
                hidden_chunk, targets_chunk, keep_chunk, output_weight,
            )
        numerator = chunk_numerator if numerator is None else numerator + chunk_numerator

    assert numerator is not None
    mean = numerator / count
    if return_numerator:
        return mean, count, numerator
    return mean, count


__all__ = ["causal_lm_loss", "causal_lm_loss_from_hidden"]
