"""Pure tensor-in/logits-out V5 core with receipt certification.

The constructor takes a validated ``ModelSpec`` and exposes only tensor
behavior plus a parameter receipt. It never constructs an optimizer, loads
checkpoints, inspects task families, or decides generation policy. Packed
segments use block-diagonal causal attention with per-segment RoPE resets.
The output projection reuses embedding storage, so no separate head exists.
"""

from __future__ import annotations

from typing import Any

from v5_contracts.model_spec import QK_NORM_EPSILON, ModelSpec

from .block import build_block, build_rmsnorm
from .config import from_spec
from .embedding import build_embedding
from .initialize import initialize_module


def initialize(spec: ModelSpec, seed: int, *, torch_module: Any = None) -> Any:
    """Build the core for ``spec`` under a seeded generator."""

    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    nn = torch.nn
    functional = torch.nn.functional
    config = from_spec(spec, qk_norm_epsilon=QK_NORM_EPSILON)

    class Core(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.config = config
            self.embedding = build_embedding(config, torch_module=torch)
            self.blocks = nn.ModuleList(
                build_block(config, torch_module=torch) for _ in range(config.layers))
            self.final_norm = build_rmsnorm(
                config.width, epsilon=config.norm_epsilon, torch_module=torch)

        def forward(self, token_ids: Any, positions: Any, mask: Any) -> Any:
            if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= config.context_length:
                raise ValueError("token ids must be [batch, length] within native context")
            hidden = self.embedding(token_ids)
            for block in self.blocks:
                hidden = block(hidden, positions, mask)
            return functional.linear(self.final_norm(hidden), self.embedding.weight)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Core()
        initialize_module(model, layers=config.layers, torch_module=torch)
    assert_single_embedding(model)
    assert_receipt(model, spec)
    return model


def assert_single_embedding(model: Any) -> None:
    """Require exactly one embedding table and no separate output head."""

    names = [name for name, _ in model.named_parameters()]
    embedding_names = [name for name in names if "embedding.weight" in name]
    if len(embedding_names) != 1:
        raise ValueError("core must own exactly one tied embedding table")
    for name in names:
        leaf = name.rsplit(".", 1)[-1]
        head = name.rsplit(".", 2)
        if leaf == "weight" and len(head) == 3 and head[-2] in {"lm_head", "output_head"}:
            raise ValueError("core must not carry a separate output head")


def parameter_receipt(model: Any) -> dict[str, int]:
    """Return executable tensor names and counts for the core."""

    return {name: int(parameter.numel()) for name, parameter in model.named_parameters()}


def assert_receipt(model: Any, spec: ModelSpec) -> None:
    """Require the executable inventory to equal the pure ModelSpec receipt."""

    spec.assert_valid()
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    if total != spec.parameter_receipt().total:
        raise ValueError("model parameter inventory does not match ModelSpec")


def packed_layout(segment_ids: Any, *, torch_module: Any) -> tuple[Any, Any]:
    """Nondecreasing segment IDs; -1 means trailing padding; reset RoPE per segment."""

    torch = torch_module
    if segment_ids.ndim != 2 or segment_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("segment IDs must be a rank-two integer tensor")
    valid = segment_ids >= 0
    if (segment_ids < -1).any().item() or ((~valid[:, :-1]) & valid[:, 1:]).any().item():
        raise ValueError("padding must be -1 and trailing")
    if ((segment_ids[:, 1:] < segment_ids[:, :-1]) & valid[:, 1:]).any().item():
        raise ValueError("segment IDs must be nondecreasing; segments cannot reappear")
    length = segment_ids.shape[1]
    indices = torch.arange(length, device=segment_ids.device)
    starts = torch.ones_like(valid)
    starts[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    offsets = torch.where(starts, indices[None, :], 0).cummax(dim=1).values
    positions = torch.where(valid, indices[None, :] - offsets, 0)
    causal = indices[None, :] <= indices[:, None]
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    mask = causal & same_segment & valid[:, :, None] & valid[:, None, :]
    mask = mask | ((~valid)[:, :, None] & torch.eye(length, device=segment_ids.device, dtype=torch.bool))
    return positions, mask[:, None, :, :]


__all__ = [
    "assert_receipt",
    "assert_single_embedding",
    "initialize",
    "packed_layout",
    "parameter_receipt",
]
