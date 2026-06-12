"""Blockwise fake-quantization experiments with FP master weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def blockwise_fake_quantize(
    weight: torch.Tensor,
    *,
    bits: int = 8,
    block_size: int = 64,
) -> torch.Tensor:
    if bits not in {4, 8}:
        raise ValueError("QAT supports 4-bit or 8-bit experiments.")
    original_shape = weight.shape
    flat = weight.reshape(-1)
    padding = (-flat.numel()) % block_size
    if padding:
        flat = F.pad(flat, (0, padding))
    blocks = flat.view(-1, block_size)
    qmax = float(2 ** (bits - 1) - 1)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(blocks / scale).clamp(-qmax, qmax) * scale
    quantized = quantized.view(-1)[: weight.numel()].view(original_shape)
    return weight + (quantized - weight).detach()


class QATLinear(nn.Module):
    """Linear layer retaining trainable floating-point master weights."""

    def __init__(self, base: nn.Linear, bits: int = 8, block_size: int = 64) -> None:
        super().__init__()
        self.base = base
        self.bits = int(bits)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_weight = blockwise_fake_quantize(
            self.base.weight, bits=self.bits, block_size=self.block_size
        )
        return F.linear(x, quantized_weight, self.base.bias)


def attach_qat(
    model: nn.Module,
    *,
    bits: int = 8,
    block_size: int = 64,
    protected_terms: tuple[str, ...] = (
        "esv",
        "hal",
        "civ",
        "rim",
        "embedding",
        "norm",
    ),
) -> list[str]:
    attached: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(term in module_name.lower() for term in protected_terms):
            continue
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, QATLinear(module, bits=bits, block_size=block_size))
        attached.append(module_name)
    return attached
