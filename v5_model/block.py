"""Pre-norm attention + SwiGLU residual block."""

from __future__ import annotations

from typing import Any

from .attention import build_attention


def build_rmsnorm(width: int, *, epsilon: float, torch_module: Any) -> Any:
    torch = torch_module
    nn = torch.nn

    class RMSNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))

        def forward(self, value: Any) -> Any:
            inverse = torch.rsqrt(
                value.float().square().mean(-1, keepdim=True) + epsilon)
            return (value.float() * inverse * self.weight.float()).to(value.dtype)

    return RMSNorm()


def build_block(config: Any, *, torch_module: Any) -> Any:
    """Build one residual block: attention sublayer then SwiGLU sublayer."""

    torch = torch_module
    nn = torch.nn
    functional = torch.nn.functional

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = build_rmsnorm(
                config.width, epsilon=config.norm_epsilon, torch_module=torch)
            self.ffn_norm = build_rmsnorm(
                config.width, epsilon=config.norm_epsilon, torch_module=torch)
            self.attention = build_attention(config, torch_module=torch)
            self.gate = nn.Linear(config.width, config.ffn_width, bias=False)
            self.up = nn.Linear(config.width, config.ffn_width, bias=False)
            self.down = nn.Linear(config.ffn_width, config.width, bias=False)

        def forward(self, hidden: Any, positions: Any, mask: Any) -> Any:
            hidden = hidden + self.attention(self.attention_norm(hidden), positions, mask)
            normalized = self.ffn_norm(hidden)
            return hidden + self.down(
                functional.silu(self.gate(normalized)) * self.up(normalized))

    return Block()


__all__ = ["build_block", "build_rmsnorm"]
