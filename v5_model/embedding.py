"""Tied input/output embedding with exact initialization."""

from __future__ import annotations

from typing import Any


def build_embedding(config: Any, *, torch_module: Any) -> Any:
    """Create the token embedding; the output head reuses its storage."""

    torch = torch_module
    return torch.nn.Embedding(config.vocabulary_size, config.width)


def assert_tied_storage(embedding: Any, output_weight: Any) -> None:
    """Require input and output projections to share one storage identity."""

    if embedding.weight.data_ptr() != output_weight.data_ptr():
        raise ValueError("embedding and output weights are not tied storage")
    if tuple(embedding.weight.shape) != tuple(output_weight.shape):
        raise ValueError("tied embedding/output shapes disagree")


__all__ = ["assert_tied_storage", "build_embedding"]
