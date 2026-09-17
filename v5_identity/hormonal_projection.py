"""Zero-initialized, bounded per-head log-temperature projection."""
from __future__ import annotations

from collections.abc import Mapping

from v5_identity.hormonal_config import ACTIVE_HORMONES, MAX_LOG_T
from v5_identity.hormonal_state import HORMONE_NAMES


def hal_to_log_temperature(hormones, weights, *, torch, max_log_temperature=MAX_LOG_T):
    if isinstance(hormones, Mapping):
        vector = weights.new_tensor([float(hormones[n]) for n in HORMONE_NAMES])
    else:
        vector = hormones.detach().to(device=weights.device, dtype=torch.float32)
    if tuple(vector.shape) != (7,) or weights.ndim != 2 or weights.shape[1] != 7:
        raise ValueError('expected seven hormone channels')
    active = weights.new_tensor([float(n in ACTIVE_HORMONES) for n in HORMONE_NAMES])
    return (weights.float() @ (vector * active)).clamp(-max_log_temperature, max_log_temperature)


def scale_queries_for_temperature(queries, log_temperature, *, torch):
    return queries * torch.exp(-log_temperature).to(queries.dtype)[None, :, None, None]
