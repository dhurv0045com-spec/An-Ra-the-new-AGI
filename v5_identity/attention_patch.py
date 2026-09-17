"""Runtime attention patch for HORM-002 experiments only.

Patches an initialized V5 model's attention forward to scale queries by
1 + B*tanh(alpha * w.h) after RoPE/QK-norm. Does NOT modify v5_model source.
"""

from __future__ import annotations

from typing import Any

from .hormonal_state import HormonalState


class HormonalAttentionPatch:
    def __init__(self, model: Any, *, projection: Any | None) -> None:
        self.model = model
        self.projection = projection
        self.state = HormonalState.baseline()
        self._originals = []
        self._install()

    def _install(self) -> None:
        import torch

        for block in self.model.blocks:
            attention = block.attention
            self._originals.append((attention, attention.forward))
            config = self.model.config
            projection = self.projection
            hormone_state = self.state
            original = attention.forward

            def patched(hidden, positions, mask, *, module=attention,
                        cfg=config, proj=projection, state=hormone_state,
                        base_forward=original):
                if proj is None:
                    return base_forward(hidden, positions, mask)
                batch, length, _ = hidden.shape
                q = module.query(hidden).view(
                    batch, length, cfg.query_heads, cfg.head_dimension).transpose(1, 2)
                k = module.key(hidden).view(
                    batch, length, cfg.kv_heads, cfg.head_dimension).transpose(1, 2)
                v = module.value(hidden).view(
                    batch, length, cfg.kv_heads, cfg.head_dimension).transpose(1, 2)
                q = module.rope(module.normalize(q, module.query_scale), positions)
                k = module.rope(module.normalize(k, module.key_scale), positions)
                repeats = cfg.query_heads // cfg.kv_heads
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
                scale = proj.scale(state.vector())
                attended = torch.nn.functional.scaled_dot_product_attention(
                    q * scale, k, v, attn_mask=mask, dropout_p=0.0)
                return module.output(attended.transpose(1, 2).contiguous().view(
                    batch, length, cfg.width))

            attention.forward = patched

    def restore(self) -> None:
        for attention, original in self._originals:
            attention.forward = original
        self._originals.clear()
