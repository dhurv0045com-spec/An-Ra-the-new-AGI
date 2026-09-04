"""Minimal PyTorch/XLA execution shim. Hardware isolation lives HERE only.

Architecture (v5_model), objectives (v5_objectives), optimizer semantics
(v5_training.optimizer) and checkpoint format (v5_training.checkpoint) are
reused unchanged. This module owns only: device placement, mark_step /
optimizer_step ordering, host-vs-device split, and fail-closed CPU detection.

Compatibility policy: prefer current PJRT APIs, with legacy fallbacks only
where needed. In particular, xrt_world_size() is not required.
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


def _runtime_world_size() -> int:
    """Return PJRT world size across current and legacy torch-xla releases."""
    xm = require_xla()
    try:
        import torch_xla.runtime as xr
        return int(xr.world_size())
    except Exception:
        pass
    for name in ("get_world_size", "xrt_world_size"):
        fn = getattr(xm, name, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                pass
    return 0


def xla_device(*, index: int = 0):
    """Return an XLA device, preferring torch_xla.device() on modern releases."""
    require_xla()
    try:
        import torch_xla
        fn = getattr(torch_xla, "device", None)
        if callable(fn):
            return fn(index)
    except Exception:
        pass
    xm = require_xla()
    fn = getattr(xm, "xla_device", None)
    if callable(fn):
        return fn(index)
    raise RuntimeError("ABORT_NO_TPU: no usable XLA device API is available.")


def _hardware_type() -> str:
    """Return runtime hardware type without assuming a TPU generation."""
    xm = require_xla()
    try:
        import torch_xla.runtime as xr
        fn = getattr(xr, "device_type", None)
        if callable(fn):
            value = fn()
            if value:
                return str(value)
    except Exception:
        pass
    try:
        fn = getattr(xm, "xla_device_hw", None)
        if callable(fn):
            return str(fn(str(xla_device())))
    except Exception:
        pass
    return "unknown"


def assert_tpu_active(*, min_devices: int = 1) -> int:
    """Verify XLA TPU devices are actually active. Returns device count."""
    n = world_size()
    hw = device_hardware()
    if n < min_devices or "TPU" not in str(hw).upper():
        raise RuntimeError(f"ABORT_NO_TPU: hw={hw} devices={n}; refusing CPU fallback.")
    return n


def mark_step() -> None:
    """Materialize queued XLA work."""
    xm = require_xla()
    fn = getattr(xm, "mark_step", None)
    if callable(fn):
        fn()
        return
    try:
        import torch_xla
        sync = getattr(torch_xla, "sync", None)
        if callable(sync):
            sync()
            return
    except Exception:
        pass
    raise RuntimeError("torch-xla exposes neither mark_step() nor sync().")


def optimizer_step(optimizer: Any) -> None:
    """Single XLA-safe optimizer step; fail-closed if the API is absent."""
    xm = require_xla()
    fn = getattr(xm, "optimizer_step", None)
    if callable(fn):
        fn(optimizer)
        return
    raise RuntimeError(
        "UNSUPPORTED_OPERATION: torch-xla exposes no optimizer_step(); "
        "refusing a silent plain-optimizer fallback on XLA."
    )


def to_xla(tensor: Any, device: Any = None):
    return tensor.to(device or xla_device())


def barrier(tag: str = "citadel-tpu-barrier") -> None:
    """Cross-replica rendezvous for the multi-device path (no-op on 1 device)."""
    if _runtime_world_size() <= 1:
        return
    try:
        import torch_xla.runtime as xr
        fn = getattr(xr, "rendezvous", None)
        if callable(fn):
            fn(tag)
            return
    except Exception:
        pass
    xm = require_xla()
    fn = getattr(xm, "rendezvous", None)
    if callable(fn):
        fn(tag)
        return
    raise RuntimeError("multi-device XLA runtime exposes no rendezvous API")


def get_device(*, index: int = 0):
    """Stable helper: prefer torch_xla.device(), legacy fallback only if needed."""
    return xla_device(index=index)


def world_size() -> int:
    """Stable helper: PJRT world size, legacy fallback only if needed."""
    return _runtime_world_size()


def device_hardware() -> str:
    """Stable helper: TPU/CPU/GPU kind actually active (never the torch build tag)."""
    return _hardware_type()


__all__ = [
    "assert_tpu_active",
    "barrier",
    "device_hardware",
    "get_device",
    "mark_step",
    "optimizer_step",
    "require_xla",
    "to_xla",
    "world_size",
    "xla_device",
]
