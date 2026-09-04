"""Minimal PyTorch/XLA execution shim. Hardware isolation lives HERE only.

Architecture (v5_model), objectives (v5_objectives), optimizer semantics
(v5_training.optimizer) and checkpoint format (v5_training.checkpoint) are
reused unchanged. This module owns only: device placement, mark_step /
optimizer_step ordering, host-vs-device split, and fail-closed CPU detection.
"""

from __future__ import annotations

from typing import Any


def require_xla():
    """Import torch_xla or raise. Never silently fall back to CPU/CUDA."""
    try:
        import torch_xla.core.xla_model as xm
    except Exception as exc:
        raise RuntimeError("ABORT_NO_TPU: torch_xla unavailable; refusing fallback.") from exc
    return xm


def xla_device(*, index: int = 0):
    xm = require_xla()
    return xm.xla_device(index)


def assert_tpu_active(*, min_devices: int = 1) -> int:
    """Verify XLA TPU devices are actually active. Returns device count."""
    xm = require_xla()
    n = int(xm.xrt_world_size())
    try:
        hw = xm.xla_device_hw(str(xm.xla_device()))
    except Exception:
        hw = "unknown"
    if n < min_devices or "TPU" not in str(hw).upper():
        raise RuntimeError(f"ABORT_NO_TPU: hw={hw} devices={n}; refusing CPU fallback.")
    return n


def mark_step() -> None:
    require_xla().mark_step()


def optimizer_step(optimizer: Any) -> None:
    """Single XLA-safe optimizer step (xm.optimizer_step, not optimizer.step)."""
    xm = require_xla()
    xm.optimizer_step(optimizer)


def to_xla(tensor: Any, device: Any = None):
    return tensor.to(device or xla_device())


def barrier(tag: str = "citadel-tpu-barrier") -> None:
    """Cross-replica rendezvous for the multi-device path (no-op on 1 device)."""
    xm = require_xla()
    if int(xm.xrt_world_size()) > 1:
        import torch_xla.runtime as xr

        xr.rendezvous(tag)


__all__ = ["assert_tpu_active", "barrier", "mark_step", "optimizer_step", "require_xla", "to_xla", "xla_device"]
