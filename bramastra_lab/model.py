"""Dense causal Transformer used as the BRAMASTRA B0 control model."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings for the from-scratch causal decoder."""

    vocab: int = 260
    width: int = 256
    layers: int = 8
    heads: int = 4
    ffn: int = 704
    max_seq: int = 256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {"rms_norm_eps", "rope_theta"}:
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (self.width // self.heads) % 2:
            raise ValueError("the per-head width must be even for RoPE")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")


def parameter_count(config: ModelConfig) -> int:
    """Compute the exact learned-parameter count without constructing a model."""

    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    d, layers, ffn = config.width, config.layers, config.ffn
    return config.vocab * d + layers * (4 * d * d + 3 * d * ffn + 2 * d) + d


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, inputs: Tensor) -> Tensor:
        source_dtype = inputs.dtype
        values = inputs.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(source_dtype)


def _apply_rope(inputs: Tensor, inv_freq: Tensor) -> Tensor:
    length = inputs.shape[-2]
    positions = torch.arange(length, device=inputs.device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq.float())
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    source_dtype = inputs.dtype
    pairs = inputs.float().reshape(*inputs.shape[:-1], -1, 2)
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(source_dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_width, 2, dtype=torch.float32) / self.head_width)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.max_seq, config.max_seq, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, inputs: Tensor, padding_mask: Tensor | None) -> Tensor:
        batch, length, width = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)

        query = _apply_rope(split_heads(query), self.inv_freq)
        key = _apply_rope(split_heads(key), self.inv_freq)
        value = split_heads(value)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        allowed = self.causal_mask[:length, :length][None, None, :, :]
        if padding_mask is not None:
            allowed = allowed & padding_mask[:, None, None, :]

        # Softmax is deliberately reduced in FP32. Multiplication and normalization
        # after masking also keep a fully padded row finite on accelerator graphs.
        probabilities = torch.softmax(
            scores.float().masked_fill(~allowed, torch.finfo(torch.float32).min), dim=-1
        )
        probabilities = probabilities * allowed
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        attended = torch.matmul(probabilities.to(value.dtype), value)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.width, config.ffn, bias=False)
        self.up = nn.Linear(config.width, config.ffn, bias=False)
        self.down = nn.Linear(config.ffn, config.width, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.width, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, inputs: Tensor, padding_mask: Tensor | None) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs), padding_mask)
        return inputs + self.feed_forward(self.ffn_norm(inputs))


class TransformerDecoder(nn.Module):
    """Bias-free pre-norm decoder with RoPE and a tied output projection.

    ``padding_mask`` is optional and has shape ``[batch, sequence]``. ``True``
    denotes a real token. Padding must be a single trailing suffix per example.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        if not isinstance(self.config, ModelConfig):
            raise TypeError("config must be a ModelConfig")

        self.embedding = nn.Embedding(self.config.vocab, self.config.width)
        self.blocks = nn.ModuleList(DecoderBlock(self.config) for _ in range(self.config.layers))
        self.final_norm = RMSNorm(self.config.width, self.config.rms_norm_eps)
        self.reset_parameters()

        actual = sum(parameter.numel() for parameter in self.parameters())
        expected = parameter_count(self.config)
        if actual != expected:
            raise RuntimeError(f"model has {actual:,} parameters; expected {expected:,}")

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        residual_std = 0.02 / math.sqrt(2 * self.config.layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down.weight, mean=0.0, std=residual_std)

    def forward(self, tokens: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if not isinstance(tokens, Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("tokens must have non-empty batch and sequence dimensions")
        if tokens.shape[1] > self.config.max_seq:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_seq {self.config.max_seq}"
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("tokens must use torch.int32 or torch.int64")

        if padding_mask is not None:
            if not isinstance(padding_mask, Tensor):
                raise TypeError("padding_mask must be a torch.Tensor")
            if padding_mask.shape != tokens.shape:
                raise ValueError("padding_mask must have the same shape as tokens")
            if padding_mask.dtype != torch.bool:
                raise TypeError("padding_mask must have boolean dtype")
            if padding_mask.device != tokens.device:
                raise ValueError("padding_mask and tokens must be on the same device")
            # This semantic check is cheap for local tests and avoids a host sync on
            # XLA. The producer owns the same trailing-padding invariant on devices.
            if padding_mask.device.type == "cpu":
                invalid_transition = padding_mask[:, 1:] & ~padding_mask[:, :-1]
                if bool(invalid_transition.any()):
                    raise ValueError("padding_mask may contain only trailing padding")

        hidden = self.embedding(tokens)
        for block in self.blocks:
            hidden = block(hidden, padding_mask)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.embedding.weight)


# The experiment-facing name keeps receipts readable while TransformerDecoder
# states the architecture precisely for callers that prefer that spelling.
BramastraModel = TransformerDecoder

