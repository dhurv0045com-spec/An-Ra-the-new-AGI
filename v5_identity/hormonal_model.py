"""Experimental V5 forward with explicit, checkpoint-safe hormonal input.

Uses the real core's weights, normalization, RoPE and FFN modules without
patching them. The only numerical intervention is post-RoPE query scaling.
The caller owns appraisal/decay; forward snapshots state and never updates it.
"""
from __future__ import annotations

import math
from typing import Any

from v5_identity.hormonal_projection import hal_to_log_temperature, scale_queries_for_temperature
from v5_identity.hormonal_state import HORMONE_NAMES, HALState


def wrap_core_with_hal(core, *, torch, shared_projection, max_log_temperature=None):
    """Own the core and a copy of initial projection weights; no monkeypatches.

    Hormone levels are serialized as non-gradient buffers. Optimizers must be
    constructed from the returned model, not from the supplied weight tensor.
    One state is shared across the batch (session-level v1, not per-token).
    """
    from torch.utils.checkpoint import checkpoint

    from v5_identity.hormonal_config import MAX_LOG_T

    config = core.config
    functional = torch.nn.functional
    max_log_temperature = MAX_LOG_T if max_log_temperature is None else float(max_log_temperature)
    if not math.isfinite(max_log_temperature) or not 0 < max_log_temperature <= MAX_LOG_T:
        raise ValueError("log-temperature bound must be finite in (0, 1.5]")
    if tuple(shared_projection.shape) != (config.query_heads, len(HORMONE_NAMES)):
        raise ValueError("projection must have shape [query_heads, 7]")
    if not torch.isfinite(shared_projection).all():
        raise ValueError("projection must be finite")

    def block_forward(block, hidden, positions, mask, log_t):
        residual = hidden
        hidden = block.attention_norm(hidden)
        attention = block.attention
        batch, length, _ = hidden.shape
        query = attention.query(hidden).view(
            batch, length, config.query_heads, config.head_dimension).transpose(1, 2)
        key = attention.key(hidden).view(
            batch, length, config.kv_heads, config.head_dimension).transpose(1, 2)
        value = attention.value(hidden).view(
            batch, length, config.kv_heads, config.head_dimension).transpose(1, 2)
        query = attention.rope(attention.normalize(query, attention.query_scale), positions)
        key = attention.rope(attention.normalize(key, attention.key_scale), positions)
        repeats = config.query_heads // config.kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        query = scale_queries_for_temperature(query, log_t, torch=torch)
        attended = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0)
        hidden = residual + attention.output(attended.transpose(1, 2).contiguous().view(
            batch, length, config.width))
        normalized = block.ffn_norm(hidden)
        return hidden + block.down(functional.silu(block.gate(normalized)) * block.up(normalized))

    class HalCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core
            self.projection = torch.nn.Parameter(shared_projection.detach().clone().to(
                device=core.embedding.weight.device, dtype=torch.float32))
            self.register_buffer('hormones', torch.zeros(7, device=self.projection.device))

        def get_extra_state(self):
            return {
                'schema': 'anra-hal/v1',
                'base_sha256': self.core.spec.sha256(),
                'max_log_temperature': max_log_temperature,
                'spec_sha256': self.spec.sha256() if hasattr(self, 'spec') else None,
            }

        def set_extra_state(self, state):
            if state != self.get_extra_state():
                raise ValueError('incompatible hormonal checkpoint contract')

        def load_state_dict(self, state_dict, strict=True, assign=False):
            # Validate identity before copying any tensors (including non-strict loads).
            self.set_extra_state(state_dict.get('_extra_state'))
            values = state_dict.get('hormones')
            projection = state_dict.get('projection')
            if values is None or not torch.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError('invalid checkpoint hormone levels')
            if projection is None or not torch.isfinite(projection).all():
                raise ValueError('invalid checkpoint projection')
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        def set_hormones(self, state: HALState):
            self.set_hormone_values(state.as_dict())

        def set_hormone_values(self, values):
            unknown = set(values) - set(HORMONE_NAMES)
            if unknown:
                raise KeyError(f"unknown hormones: {sorted(unknown)}")
            for name in HORMONE_NAMES:
                value = float(values.get(name, 0.0))
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"hormone {name} must be finite and within [0, 1], got {value}")
            state = HALState(**{name: values.get(name, 0.0) for name in HORMONE_NAMES})
            # Replace rather than mutate: an outstanding backward owns its snapshot.
            self.hormones = self.projection.new_tensor([getattr(state, n) for n in HORMONE_NAMES])

        def hormone_state(self):
            return dict(zip(HORMONE_NAMES, self.hormones.detach().cpu().tolist()))

        def log_temperature(self):
            return hal_to_log_temperature(
                self.hormones.detach().clone(), self.projection, torch=torch,
                max_log_temperature=max_log_temperature)

        def forward(self, token_ids, positions, mask, use_activation_checkpointing=False):
            if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= config.context_length:
                raise ValueError("token ids must be [batch, length] within native context")
            log_t = self.log_temperature()
            hidden = self.core.embedding(token_ids)
            for block in self.core.blocks:
                if use_activation_checkpointing and self.training:
                    # Bind this block, and pass the captured temperature as a tensor
                    # argument; recomputation must not read mutable session state.
                    def run(h, p, m, t, block=block):
                        return block_forward(block, h, p, m, t)
                    hidden = checkpoint(run, hidden, positions, mask, log_t, use_reentrant=False)
                else:
                    hidden = block_forward(block, hidden, positions, mask, log_t)
            return functional.linear(self.core.final_norm(hidden), self.core.embedding.weight)

    return HalCore()


def initialize_hormonal(spec, seed, *, torch_module=None):
    """Build an explicitly experimental spec with accurate parameter accounting."""
    if torch_module is None:
        import torch as torch_module
    from v5_contracts.model_spec_hormonal import HormonalModelSpec
    from v5_model.core import initialize

    if not isinstance(spec, HormonalModelSpec):
        raise ValueError("requires a HormonalModelSpec, never the launch candidate")
    core = initialize(spec.base, seed, torch_module=torch_module)
    model = wrap_core_with_hal(
        core, torch=torch_module,
        shared_projection=torch_module.zeros(spec.base.query_heads, len(HORMONE_NAMES)),
        max_log_temperature=spec.hormonal.max_log_temperature)
    model.spec = spec
    assert sum(p.numel() for p in model.parameters()) == spec.parameter_receipt()['total']
    return model



__all__ = ['wrap_core_with_hal', 'initialize_hormonal']
