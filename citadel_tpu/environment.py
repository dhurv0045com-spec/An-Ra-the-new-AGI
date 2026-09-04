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


def probe(*, require_tpu: bool = True) -> dict[str, Any]:
    """Detect the actual runtime. Never assumes v5e-8 or any generation."""
    try:
        import torch
        torch_version = getattr(torch, "__version__", "unknown")
        python_version = platform.python_version()
    except Exception:
        torch_version, python_version = "unavailable", platform.python_version()
    try:
        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm

        xla_available = True
        try:
            device_type = xm.xla_device_hw(str(xm.xla_device())) if hasattr(xm, "xla_device_hw") else "unknown"
        except Exception:
            device_type = "unknown"
        try:
            n_devices = int(xm.xrt_world_size())
        except Exception:
            n_devices = 0
        try:
            import torch_xla.runtime as xr

            device_names: Any = getattr(xr, "get_master_ip", None)
            _ = device_names
        except Exception:
            pass
        try:
            import torch_xla.version as xv

            torch_xla_version: Any = getattr(xv, "__version__", "unknown")
        except Exception:
            torch_xla_version = "unknown"
        xla_runtime = "PJRT"
    except Exception:
        xla_available, device_type, n_devices = False, "none", 0
        torch_xla_version, xla_runtime = "unavailable", "none"
    tpu_present = bool(xla_available and n_devices >= 1 and "TPU" in str(device_type).upper())
    total_ram = round((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9), 1) if hasattr(os, "sysconf") else "unknown"
    disk_free = shutil.disk_usage(Path.cwd().anchor).free if Path.cwd().anchor else 0
    env: dict[str, Any] = {
        "schema": "citadel-tpu-environment/v1",
        "probe_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _cmd(["git", "rev-parse", "HEAD"]),
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


def main(*, out: str = "docs/citadel/tpu_receipts/TPU_ENVIRONMENT.json", require_tpu: bool = True) -> dict[str, Any]:
    env = probe(require_tpu=require_tpu)
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
    args = parser.parse_args()
    main(out=args.out, require_tpu=not args.allow_no_tpu)
    print("probe_pass; see", args.out)


__all__ = ["NoTpuError", "main", "probe"]
