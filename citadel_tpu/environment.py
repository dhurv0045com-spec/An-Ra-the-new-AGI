"""TPU runtime detection probe. Fail-closed. No model, no training.

Produces docs/citadel/tpu_receipts/TPU_ENVIRONMENT.json (path overridable).
Every later TPU receipt must embed this environment block; a TPU receipt
without one is invalid. CPU fallback when TPU was requested is ABORT, not a
degraded run.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class NoTpuError(RuntimeError):
    """Raised when TPU was requested but no XLA TPU device is active."""


def _cmd(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip() or None


def _detect_platform(*, explicit: str | None = None) -> str:
    """Identify colab/kaggle/other from runtime signals — never from TPU generation."""
    for candidate in (explicit, os.environ.get("CITADEL_PLATFORM", "").strip().lower()):
        if candidate in ("colab", "kaggle", "other"):
            return candidate
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("TBE_RUNTIME_ADDR"):
        return "colab"
    return "other"


def _runtime_world_size(xr: Any, xm: Any) -> int:
    """Return PJRT world size with compatibility fallbacks for older torch-xla."""
    try:
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


def _runtime_device(torch_xla: Any, xm: Any):
    """Return an XLA device using the modern API when available."""
    fn = getattr(torch_xla, "device", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    fn = getattr(xm, "xla_device", None)
    if callable(fn):
        return fn()
    raise RuntimeError("torch-xla exposes no usable XLA device API")


def _runtime_hardware(xr: Any, xm: Any, device: Any) -> str:
    """Return TPU/CPU/GPU runtime kind across current and legacy torch-xla APIs."""
    fn = getattr(xr, "device_type", None)
    if callable(fn):
        try:
            value = fn()
            if value:
                return str(value)
        except Exception:
            pass
    fn = getattr(xm, "xla_device_hw", None)
    if callable(fn):
        try:
            return str(fn(str(device)))
        except Exception:
            pass
    return "unknown"


def probe(*, require_tpu: bool = True, platform_override: str | None = None,
          accelerator_requested: str = "TPU") -> dict[str, Any]:
    """Detect the actual runtime. Never assumes a TPU generation or device count."""
    if require_tpu and accelerator_requested.upper() == "TPU":
        os.environ.setdefault("PJRT_DEVICE", "TPU")

    try:
        import torch
        torch_version = getattr(torch, "__version__", "unknown")
        python_version = platform.python_version()
    except Exception:
        torch_version, python_version = "unavailable", platform.python_version()

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        device = _runtime_device(torch_xla, xm)
        n_devices = _runtime_world_size(xr, xm)
        device_type = _runtime_hardware(xr, xm, device)
        torch_xla_version = getattr(torch_xla, "__version__", "unknown")
        xla_available = True
        xla_runtime = "PJRT"
    except Exception:
        xla_available, device_type, n_devices = False, "none", 0
        torch_xla_version, xla_runtime = "unavailable", "none"

    tpu_present = bool(xla_available and n_devices >= 1 and "TPU" in str(device_type).upper())
    total_ram = (
        round((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9), 1)
        if hasattr(os, "sysconf")
        else "unknown"
    )
    disk_free = shutil.disk_usage(Path.cwd().anchor).free if Path.cwd().anchor else 0
    env: dict[str, Any] = {
        "schema": "citadel-tpu-environment/v1",
        "probe_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _cmd(["git", "rev-parse", "HEAD"]),
        "platform": _detect_platform(explicit=platform_override),
        "accelerator_requested": accelerator_requested,
        "accelerator_detected": str(device_type) if xla_available else "none",
        "pjrt_device_env": os.environ.get("PJRT_DEVICE", "unset"),
        "xla_device_count": n_devices,
        "python_version": python_version,
        "torch_version": torch_version,
        "torch_xla_version": torch_xla_version,
        "xla_runtime": xla_runtime,
        "xla_device_type": device_type,
        "xla_devices": n_devices,
        "tpu_present": tpu_present,
        "tpu_generation": str(device_type) if tpu_present else "none",
        "host_cpu": platform.processor() or platform.machine(),
        "host_ram_gb": total_ram,
        "local_disk_free_bytes": int(disk_free),
        "kaggle_session_limits": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown"),
        "sys_argv0": sys.argv[0] if sys.argv else "unknown",
    }
    env["probe_pass"] = bool(tpu_present) if require_tpu else True
    return env


def main(*, out: str = "docs/citadel/tpu_receipts/TPU_ENVIRONMENT.json", require_tpu: bool = True,
         platform_override: str | None = None) -> dict[str, Any]:
    env = probe(require_tpu=require_tpu, platform_override=platform_override)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")
    if require_tpu and not env["tpu_present"]:
        raise NoTpuError(f"ABORT_NO_TPU: {path} records tpu_present=false; refusing CPU fallback.")
    return env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/citadel/tpu_receipts/TPU_ENVIRONMENT.json")
    parser.add_argument("--allow-no-tpu", action="store_true")
    parser.add_argument("--platform", default=None, choices=["colab", "kaggle", "other"])
    args = parser.parse_args()
    main(out=args.out, require_tpu=not args.allow_no_tpu, platform_override=args.platform)
    print("probe_pass; see", args.out)


__all__ = ["NoTpuError", "main", "probe"]
