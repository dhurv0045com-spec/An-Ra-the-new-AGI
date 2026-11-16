"""Measured wall-clock, memory, and disk resource accounting and fail-closed quota controls."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import ctypes
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

import torch

from .execution import _atomic_write, _sync_dir


class ResourceQuotaExceededError(RuntimeError):
    """Raised when a measured wall-clock, memory, or disk quota is violated."""

    def __init__(self, *, quota_name: str, limit: float | int, observed: float | int, phase: str) -> None:
        self.quota_name = str(quota_name)
        self.limit = limit
        self.observed = observed
        self.phase = str(phase)
        super().__init__(
            f"resource quota exceeded ({self.quota_name}) in phase {self.phase!r}: "
            f"observed={self.observed}, limit={self.limit}"
        )


@dataclass(frozen=True, slots=True)
class ResourceQuota:
    """Explicit fail-closed bounds enforced during Planthir execution and bundle publication."""

    max_wall_seconds: float | None = None
    max_total_disk_bytes: int | None = None
    min_free_disk_bytes: int | None = None
    max_cpu_rss_bytes: int | None = None
    max_cuda_allocated_bytes: int | None = None
    schema: str = "anra-planthir-resource-quota-v1"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema != "anra-planthir-resource-quota-v1":
            raise ValueError("unsupported ResourceQuota schema")
        if self.max_wall_seconds is not None and (not isinstance(self.max_wall_seconds, (int, float)) or self.max_wall_seconds <= 0):
            raise ValueError("max_wall_seconds must be positive when set")
        for int_field in (
            "max_total_disk_bytes",
            "min_free_disk_bytes",
            "max_cpu_rss_bytes",
            "max_cuda_allocated_bytes",
        ):
            val = getattr(self, int_field)
            if val is not None and (type(val) is not int or val < 0):
                raise ValueError(f"{int_field} must be a non-negative integer when set")


def measure_process_memory_bytes() -> tuple[int, int, str]:
    """Return `(current_rss_bytes, peak_rss_bytes, source_label)` using OS/process instrumentation."""

    try:
        import psutil  # type: ignore

        info = psutil.Process(os.getpid()).memory_info()
        cur = int(info.rss)
        peak = int(getattr(info, "peak_wset", cur))
        return cur, max(cur, peak), "psutil"
    except Exception:
        pass

    if os.name == "nt":
        try:
            from ctypes import wintypes

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                cur = int(counters.WorkingSetSize)
                peak = int(counters.PeakWorkingSetSize)
                return cur, max(cur, peak), "win32_get_process_memory_info"
        except Exception:
            pass

    try:
        import resource  # type: ignore

        max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KB; macOS reports bytes; normalize conservatively
        peak = max_rss if os.uname().sysname == "Darwin" else max_rss * 1024
        return peak, peak, "posix_rusage"
    except Exception:
        return 0, 0, "unavailable"


def measure_directory_bytes(path: Path | str) -> int:
    """Measure exact allocated file bytes under `path`, returning 0 if missing."""

    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return int(root.stat().st_size)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)
        for fn in filenames:
            fp = dpath / fn
            try:
                if fp.is_file() and not fp.is_symlink():
                    total += int(fp.stat().st_size)
            except OSError:
                pass
    return total


def measure_free_disk_bytes(path: Path | str) -> int:
    """Return free bytes available on the filesystem containing `path` (or its closest existing parent)."""

    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    return int(shutil.disk_usage(p).free)


def cleanup_unreferenced_staging(work_root: Path | str) -> list[str]:
    """Delete unreferenced temporary/staging publication artifacts without altering committed checkpoints or CURRENT.json."""

    wroot = Path(work_root)
    removed: list[str] = []
    if not wroot.exists():
        return removed
    for parent in (wroot, wroot / "result_bundle", wroot / "checkpoints"):
        if not parent.is_dir():
            continue
        for item in sorted(parent.iterdir()):
            if item.name.startswith((".staging-", ".tmp-", ".planthir-final-", ".backup-")):
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                removed.append(str(item))
        # Also remove uncommitted staging inside checkpoint lineages
        if parent.name == "checkpoints":
            for ldir in sorted(parent.iterdir()):
                if ldir.is_dir():
                    for item in sorted(ldir.iterdir()):
                        if item.name.startswith((".staging-", ".tmp-")):
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                            else:
                                item.unlink(missing_ok=True)
                            removed.append(str(item))
    return removed


class ResourceMonitor:
    """Per-phase measured runtime telemetry and quota enforcement for Planthir runs."""

    def __init__(
        self,
        *,
        work_root: Path | str,
        device: str = "cpu",
        quota: ResourceQuota | None = None,
        disk_usage_fn: Any = None,
    ) -> None:
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.device_str = str(device)
        self.dev_obj = torch.device(self.device_str)
        self.quota = quota if quota is not None else ResourceQuota()
        self.quota.validate()
        self._free_disk_fn = disk_usage_fn or measure_free_disk_bytes

        self._t0 = time.perf_counter()
        cur_rss, peak_rss, rss_src = measure_process_memory_bytes()
        self._rss_source = rss_src
        self._peak_rss_bytes = max(cur_rss, peak_rss)
        self._peak_cuda_alloc = 0
        self._peak_cuda_reserved = 0
        self._cuda_measured = bool(self.dev_obj.type == "cuda" and torch.cuda.is_available())
        if self._cuda_measured:
            try:
                torch.cuda.synchronize(self.dev_obj)
                torch.cuda.reset_peak_memory_stats(self.dev_obj)
            except Exception:
                self._cuda_measured = False

        self.phases: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []

    def _sample_cuda(self) -> tuple[int | None, int | None]:
        if not self._cuda_measured:
            return None, None
        torch.cuda.synchronize(self.dev_obj)
        alloc = int(torch.cuda.max_memory_allocated(self.dev_obj))
        reserved = int(torch.cuda.max_memory_reserved(self.dev_obj))
        self._peak_cuda_alloc = max(self._peak_cuda_alloc, alloc)
        self._peak_cuda_reserved = max(self._peak_cuda_reserved, reserved)
        return alloc, reserved

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(0.0, time.perf_counter() - self._t0)
        cur_rss, peak_rss, rss_src = measure_process_memory_bytes()
        self._rss_source = rss_src
        self._peak_rss_bytes = max(self._peak_rss_bytes, cur_rss, peak_rss)
        cuda_alloc, cuda_res = self._sample_cuda()
        ckpt_bytes = measure_directory_bytes(self.work_root / "checkpoints")
        bundle_bytes = measure_directory_bytes(self.work_root / "result_bundle")
        total_bytes = measure_directory_bytes(self.work_root)
        free_bytes = int(self._free_disk_fn(self.work_root))
        return {
            "wall_seconds": round(float(elapsed), 6),
            "current_cpu_rss_bytes": cur_rss,
            "peak_cpu_rss_bytes": self._peak_rss_bytes,
            "cpu_rss_source": self._rss_source,
            "cuda_measured": self._cuda_measured,
            "cuda_allocated_bytes": cuda_alloc,
            "cuda_reserved_bytes": cuda_res,
            "checkpoints_bytes": ckpt_bytes,
            "result_bundle_bytes": bundle_bytes,
            "work_total_bytes": total_bytes,
            "free_disk_bytes": free_bytes,
        }

    def check_quotas(self, *, phase: str) -> dict[str, Any]:
        snap = self.snapshot()
        q = self.quota
        checks: list[tuple[str, float | int | None, float | int | None, str]] = [
            ("max_wall_seconds", q.max_wall_seconds, snap["wall_seconds"], "gt"),
            ("max_total_disk_bytes", q.max_total_disk_bytes, snap["work_total_bytes"], "gt"),
            ("min_free_disk_bytes", q.min_free_disk_bytes, snap["free_disk_bytes"], "lt"),
            ("max_cpu_rss_bytes", q.max_cpu_rss_bytes, snap["current_cpu_rss_bytes"], "gt"),
            (
                "max_cuda_allocated_bytes",
                q.max_cuda_allocated_bytes,
                snap["cuda_allocated_bytes"],
                "gt",
            ),
        ]
        for name, limit, observed, op in checks:
            if limit is None or observed is None:
                continue
            breached = (observed > limit) if op == "gt" else (observed < limit)
            if breached:
                self.violations.append(
                    {"quota_name": name, "limit": limit, "observed": observed, "phase": phase}
                )
                cleanup_unreferenced_staging(self.work_root)
                raise ResourceQuotaExceededError(quota_name=name, limit=limit, observed=observed, phase=phase)
        return snap

    def enter_phase(self, name: str) -> dict[str, Any]:
        self.check_quotas(phase=f"{name}:enter")
        cur_start, _, _ = measure_process_memory_bytes()
        return {"phase": name, "_t_start": time.perf_counter(), "start_cpu_rss_bytes": cur_start}

    def exit_phase(self, token: dict[str, Any], *, enforce: bool = True) -> dict[str, Any]:
        t_start = float(token.pop("_t_start", self._t0))
        dt = max(0.0, time.perf_counter() - t_start)
        snap = self.snapshot()
        token.update(
            {
                "wall_seconds": round(float(dt), 6),
                "end_cpu_rss_bytes": snap["current_cpu_rss_bytes"],
                "peak_cpu_rss_bytes": snap["peak_cpu_rss_bytes"],
                "work_total_bytes": snap["work_total_bytes"],
                "free_disk_bytes": snap["free_disk_bytes"],
                "cuda_allocated_bytes": snap["cuda_allocated_bytes"],
            }
        )
        self.phases.append(token)
        if enforce:
            self.check_quotas(phase=f"{token['phase']}:exit")
        return token

    @contextmanager
    def phase(self, name: str) -> Iterator[dict[str, Any]]:
        entry = self.enter_phase(name)
        ok = False
        try:
            yield entry
            ok = True
        finally:
            self.exit_phase(entry, enforce=ok)

    def receipt(self, *, status: str = "COMPLETE") -> dict[str, Any]:
        snap = self.snapshot()
        return {
            "schema": "anra-planthir-resource-receipt-v1",
            "status": status,
            "work_root": str(self.work_root),
            "device": self.device_str,
            "quota": asdict(self.quota),
            "total_wall_seconds": snap["wall_seconds"],
            "peak_cpu_rss_bytes": snap["peak_cpu_rss_bytes"],
            "cpu_rss_source": snap["cpu_rss_source"],
            "cuda_measured": snap["cuda_measured"],
            "peak_cuda_allocated_bytes": self._peak_cuda_alloc if self._cuda_measured else None,
            "peak_cuda_reserved_bytes": self._peak_cuda_reserved if self._cuda_measured else None,
            "checkpoints_bytes": snap["checkpoints_bytes"],
            "result_bundle_bytes": snap["result_bundle_bytes"],
            "work_total_bytes": snap["work_total_bytes"],
            "free_disk_bytes": snap["free_disk_bytes"],
            "phases": list(self.phases),
            "violations": list(self.violations),
        }

    def write_receipt(self, *, status: str = "COMPLETE", path: Path | None = None) -> Path:
        out_path = path if path is not None else (self.work_root / "RESOURCE_RECEIPT.json")
        payload = self.receipt(status=status)
        raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        _atomic_write(out_path, raw)
        _sync_dir(out_path.parent)
        return out_path


__all__ = [
    "ResourceMonitor",
    "ResourceQuota",
    "ResourceQuotaExceededError",
    "cleanup_unreferenced_staging",
    "measure_directory_bytes",
    "measure_free_disk_bytes",
    "measure_process_memory_bytes",
]
