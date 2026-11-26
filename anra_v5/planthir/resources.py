"""Measured wall-clock, memory, and disk resource accounting and fail-closed quota controls."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

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


def measure_tensor_container_bytes(
    obj: Any,
    *,
    max_depth: int = 32,
    top_k: int = 8,
) -> dict[str, Any]:
    """Recursively measure physical tensor/ndarray byte footprint in a prepacked or model container."""

    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore[assignment]

    seen_containers: set[int] = set()
    seen_torch_storages: set[tuple[Any, ...]] = set()
    seen_numpy_bases: set[tuple[int, int]] = set()
    tensor_count = 0
    ndarray_count = 0
    total_bytes = 0
    logical_bytes = 0
    cpu_bytes = 0
    cuda_bytes = 0
    entries: list[dict[str, Any]] = []

    def _torch_storage_key(t: torch.Tensor) -> tuple[Any, ...]:
        try:
            st = t.untyped_storage()
            ptr = int(st.data_ptr())
            st_bytes = int(st.nbytes())
            if ptr != 0 and st_bytes > 0:
                return ("torch", str(t.device), ptr, st_bytes)
        except Exception:
            try:
                st_old = t.storage()
                ptr = int(st_old.data_ptr())
                st_bytes = int(t.element_size()) * int(t.nelement())
                if ptr != 0 and st_bytes > 0:
                    return ("torch", str(t.device), ptr, st_bytes)
            except Exception:
                pass
        return ("torch_obj", id(t), int(t.element_size()) * int(t.nelement()))

    def _walk(val: Any, path: str, depth: int) -> None:
        nonlocal tensor_count, ndarray_count, total_bytes, logical_bytes, cpu_bytes, cuda_bytes
        if val is None or isinstance(val, (bool, int, float, str, bytes, bytearray, Path, type)) or callable(val):
            return
        if depth > max_depth:
            return

        if isinstance(val, torch.Tensor):
            tensor_count += 1
            nbytes = int(val.element_size()) * int(val.nelement())
            logical_bytes += nbytes
            dev_str = str(val.device)
            s_key = _torch_storage_key(val)
            unique_bytes = 0
            if s_key not in seen_torch_storages:
                seen_torch_storages.add(s_key)
                if len(s_key) == 4 and int(s_key[3]) > 0:
                    unique_bytes = int(s_key[3])
                else:
                    unique_bytes = nbytes
                total_bytes += unique_bytes
                if val.device.type == "cuda":
                    cuda_bytes += unique_bytes
                else:
                    cpu_bytes += unique_bytes
            entries.append(
                {
                    "path": path,
                    "kind": "torch.Tensor",
                    "shape": [int(d) for d in val.shape],
                    "dtype": str(val.dtype),
                    "device": dev_str,
                    "bytes": nbytes,
                    "unique_storage_bytes": unique_bytes,
                }
            )
            return

        if np is not None and isinstance(val, np.ndarray):
            ndarray_count += 1
            nbytes = int(val.nbytes)
            logical_bytes += nbytes
            root = val
            while getattr(root, "base", None) is not None and isinstance(root.base, np.ndarray):
                root = root.base
            np_key = (id(root), int(root.nbytes))
            unique_bytes = 0
            if np_key not in seen_numpy_bases:
                seen_numpy_bases.add(np_key)
                unique_bytes = int(root.nbytes)
                total_bytes += unique_bytes
                cpu_bytes += unique_bytes
            entries.append(
                {
                    "path": path,
                    "kind": "numpy.ndarray",
                    "shape": [int(d) for d in val.shape],
                    "dtype": str(val.dtype),
                    "device": "cpu",
                    "bytes": nbytes,
                    "unique_storage_bytes": unique_bytes,
                }
            )
            return

        oid = id(val)
        if oid in seen_containers:
            return

        if isinstance(val, Mapping):
            seen_containers.add(oid)
            for k, v in val.items():
                child_path = f"{path}.{k}" if path else str(k)
                _walk(v, child_path, depth + 1)
            return

        if isinstance(val, (list, tuple, set, frozenset)):
            seen_containers.add(oid)
            for idx, item in enumerate(val):
                _walk(item, f"{path}[{idx}]", depth + 1)
            return

        if is_dataclass(val) and not isinstance(val, type):
            seen_containers.add(oid)
            for k, v in asdict(val).items():
                child_path = f"{path}.{k}" if path else str(k)
                _walk(v, child_path, depth + 1)
            return

        if hasattr(val, "__dict__"):
            seen_containers.add(oid)
            for k, v in vars(val).items():
                child_path = f"{path}.{k}" if path else str(k)
                _walk(v, child_path, depth + 1)

    _walk(obj, "$", 0)
    entries.sort(key=lambda item: (-int(item["bytes"]), str(item["path"])))
    return {
        "schema": "anra-planthir-tensor-footprint-v1",
        "total_bytes": int(total_bytes),
        "logical_bytes": int(logical_bytes),
        "cpu_bytes": int(cpu_bytes),
        "cuda_bytes": int(cuda_bytes),
        "tensor_count": int(tensor_count),
        "ndarray_count": int(ndarray_count),
        "largest_tensors": entries[: int(top_k)],
    }


def audit_generator_scaling(
    generator_fn: Callable[[int], Any] | None = None,
    *,
    sample_sizes: Sequence[int] = (64, 256),
    target_size: int = 40000,
    max_scaling_exponent: float = 1.65,
    max_projected_seconds: float = 120.0,
    min_measurable_seconds: float = 1e-4,
    regime_name: str = "default",
    clock_fn: Callable[[], float] | None = None,
    measured_timings: Mapping[int, float] | None = None,
    fail_closed: bool = False,
) -> dict[str, Any]:
    """Probe empirical generator scaling across sizes to catch O(N^2) bottlenecks before fixed_40000 launch."""

    if isinstance(target_size, bool) or int(target_size) < 1:
        raise ValueError(f"target_size must be a positive integer, got {target_size!r}")
    tgt = int(target_size)

    measurements: list[tuple[int, float]] = []
    if measured_timings is not None:
        is_synthetic = True
        for k, v in sorted((int(kk), float(vv)) for kk, vv in measured_timings.items()):
            if k <= 0 or not math.isfinite(v) or v < 0.0:
                raise ValueError(f"invalid measured_timings entry ({k!r}: {v!r})")
            measurements.append((k, v))
        if len(measurements) < 2:
            raise ValueError("measured_timings requires at least 2 distinct positive sizes")
    else:
        is_synthetic = False
        if generator_fn is None:
            raise ValueError("audit_generator_scaling requires generator_fn or measured_timings")
        sizes = [int(s) for s in sample_sizes]
        if len(sizes) < 2 or any(s <= 0 for s in sizes) or any(b <= a for a, b in zip(sizes, sizes[1:])):
            raise ValueError(f"sample_sizes must contain >= 2 strictly increasing positive integers, got {sizes!r}")
        clk = clock_fn or time.perf_counter
        for sz in sizes:
            t0 = float(clk())
            _ = generator_fn(sz)
            dt = max(0.0, float(clk()) - t0)
            measurements.append((sz, dt))

    xs = [math.log(float(n)) for n, _ in measurements]
    ys = [math.log(max(float(t), 1e-9)) for _, t in measurements]
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom <= 0.0:
        raise ValueError("sample sizes must vary to estimate scaling exponent")
    exponent = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom

    n_last, t_last = measurements[-1]
    if tgt <= n_last:
        projected_seconds = float(t_last)
    else:
        clamped_exp = min(6.0, max(1.0, float(exponent)))
        projected_seconds = float(t_last) * ((float(tgt) / float(n_last)) ** clamped_exp)

    defects: list[dict[str, Any]] = []
    if exponent > float(max_scaling_exponent) and (
        is_synthetic or t_last >= float(min_measurable_seconds) or projected_seconds > float(max_projected_seconds)
    ):
        defects.append(
            {
                "code": "GENERATOR_SUPERLINEAR_SCALING",
                "regime": str(regime_name),
                "detail": (
                    f"regime {regime_name!r} exhibits empirical scaling exponent O(N^{exponent:.2f}) > "
                    f"{float(max_scaling_exponent):.2f} (projected {projected_seconds:.2f}s at N={tgt})"
                ),
            }
        )
    if projected_seconds > float(max_projected_seconds):
        defects.append(
            {
                "code": "GENERATOR_WALL_BUDGET_EXCEEDED",
                "regime": str(regime_name),
                "detail": (
                    f"regime {regime_name!r} projected {projected_seconds:.2f}s at N={tgt} exceeds "
                    f"max_projected_seconds={float(max_projected_seconds):.2f}s"
                ),
            }
        )

    passed = len(defects) == 0
    defect_codes = [d["code"] for d in defects]
    remediation = (
        []
        if passed
        else [
            "Replace O(N^2) duplicate scans or full-cohort pair checks in the world generator with O(N) hash-set indexing.",
            "Re-run `audit_generator_scaling(..., target_size=40000)` before exporting or launching full production campaigns.",
        ]
    )
    payload = {
        "schema": "anra-planthir-generator-scaling-audit-v1",
        "passed": passed,
        "status": "SCALING_PASS" if passed else "SCALING_DEFECT",
        "regime_name": str(regime_name),
        "target_size": tgt,
        "measurements": [{"size": int(n), "seconds": round(float(t), 6)} for n, t in measurements],
        "scaling_exponent": round(float(exponent), 4),
        "projected_seconds": round(float(projected_seconds), 4),
        "max_scaling_exponent": float(max_scaling_exponent),
        "max_projected_seconds": float(max_projected_seconds),
        "defects": defects,
        "defect_codes": defect_codes,
        "remediation": remediation,
    }
    if fail_closed and not passed:
        raise ValueError(f"GENERATOR_SCALING_FAILED:{defect_codes}:{defects[0]['detail']}")
    return payload


def estimate_production_preflight(
    *,
    total_updates: int,
    num_jobs: int = 1,
    parallel_workers: int = 1,
    model_params: int = 0,
    batch_size: int = 1,
    seq_len: int = 0,
    hidden_dim: int = 0,
    num_layers: int = 0,
    precision: str = "fp16",
    seconds_per_update: float | None = None,
    step_fn: Callable[[], Any] | None = None,
    step_warmup: int = 1,
    step_repeats: int = 3,
    prepacked_payload: Any = None,
    prepack_sample_updates: int | None = None,
    prepack_bytes_per_worker: int | None = None,
    per_job_disk_bytes: int = 128 * 1024 * 1024,
    startup_overhead_seconds_per_job: float = 0.0,
    generator_fn: Callable[[int], Any] | None = None,
    generator_target_size: int = 40000,
    generator_sample_sizes: Sequence[int] = (64, 256),
    generator_max_exponent: float = 1.65,
    generator_max_projected_seconds: float = 120.0,
    generator_regimes: Mapping[str, Mapping[str, Any]] | None = None,
    max_vram_bytes_per_worker: int = 14 * 1024**3,
    max_prepack_bytes_per_worker: int = 1024**3,
    max_session_minutes: float = 660.0,
    reserve_minutes: float = 45.0,
    max_disk_bytes: int = 18 * 1024**3,
    fail_closed: bool = False,
) -> dict[str, Any]:
    """Estimate full total_updates VRAM/prepack bloat, O(N^2) generator scaling, wall-clock & disk budgets before launch."""

    if isinstance(total_updates, bool) or int(total_updates) <= 0:
        raise ValueError(f"total_updates must be a positive integer, got {total_updates!r}")
    if isinstance(num_jobs, bool) or int(num_jobs) <= 0:
        raise ValueError(f"num_jobs must be a positive integer, got {num_jobs!r}")
    if isinstance(parallel_workers, bool) or int(parallel_workers) <= 0:
        raise ValueError(f"parallel_workers must be a positive integer, got {parallel_workers!r}")
    if int(model_params) < 0 or int(batch_size) < 0 or int(seq_len) < 0 or int(per_job_disk_bytes) < 0:
        raise ValueError("model_params, batch_size, seq_len, and per_job_disk_bytes must be non-negative")
    if float(max_session_minutes) <= 0.0 or float(reserve_minutes) < 0.0:
        raise ValueError("max_session_minutes must be positive and reserve_minutes non-negative")

    tot_u = int(total_updates)
    n_jobs = int(num_jobs)
    p_workers = int(parallel_workers)
    n_params = int(model_params)
    bs = int(batch_size)
    sl = int(seq_len)
    h_dim = int(hidden_dim)
    n_layers = int(num_layers)
    prec = str(precision).lower().strip()
    is_amp = prec in ("fp16", "amp", "float16", "bf16", "bfloat16")

    defects: list[dict[str, Any]] = []

    # 1. Per-worker VRAM & prepack mask bloat analysis
    bytes_per_param = 18 if is_amp else 16
    model_state_bytes = n_params * bytes_per_param
    if sl > 0 and bs > 0:
        if h_dim > 0 and n_layers > 0:
            activation_bytes = bs * sl * h_dim * n_layers * (2 if is_amp else 4) * 4
        else:
            activation_bytes = bs * sl * sl * (2 if is_amp else 4)
    else:
        activation_bytes = 0

    prepack_footprint: dict[str, Any] | None = None
    payload_projected_bytes = 0
    if prepacked_payload is not None:
        prepack_footprint = measure_tensor_container_bytes(prepacked_payload)
        raw_bytes = int(prepack_footprint["total_bytes"])
        if prepack_sample_updates is not None:
            if int(prepack_sample_updates) <= 0:
                raise ValueError("prepack_sample_updates must be positive when set")
            payload_projected_bytes = int(round(raw_bytes * (float(tot_u) / float(int(prepack_sample_updates)))))
        else:
            payload_projected_bytes = raw_bytes

    direct_prepack_bytes = int(prepack_bytes_per_worker) if prepack_bytes_per_worker is not None else 0
    if direct_prepack_bytes < 0:
        raise ValueError("prepack_bytes_per_worker must be non-negative")
    projected_prepack_bytes = max(payload_projected_bytes, direct_prepack_bytes)

    cuda_overhead_bytes = (
        512 * 1024 * 1024 if (n_params > 0 or sl > 0 or projected_prepack_bytes > 0) else 0
    )
    estimated_total_vram_bytes = (
        model_state_bytes + activation_bytes + projected_prepack_bytes + cuda_overhead_bytes
    )

    if projected_prepack_bytes > int(max_prepack_bytes_per_worker):
        top_tensor_hint = ""
        if prepack_footprint and prepack_footprint.get("largest_tensors"):
            top_item = prepack_footprint["largest_tensors"][0]
            top_tensor_hint = f" largest={top_item['path']} shape={top_item['shape']}"
        defects.append(
            {
                "code": "PREPACK_VRAM_BLOAT",
                "detail": (
                    f"projected per-worker prepack bytes {projected_prepack_bytes / (1024**3):.2f} GiB > "
                    f"limit {int(max_prepack_bytes_per_worker) / (1024**3):.2f} GiB{top_tensor_hint}"
                ),
            }
        )
    if estimated_total_vram_bytes > int(max_vram_bytes_per_worker):
        defects.append(
            {
                "code": "WORKER_VRAM_BUDGET_EXCEEDED",
                "detail": (
                    f"estimated per-worker VRAM {estimated_total_vram_bytes / (1024**3):.2f} GiB > "
                    f"max_vram_bytes_per_worker {int(max_vram_bytes_per_worker) / (1024**3):.2f} GiB"
                ),
            }
        )

    # 2. Generator scaling audit across regimes
    generator_audits: dict[str, dict[str, Any]] = {}
    if generator_fn is not None:
        ga = audit_generator_scaling(
            generator_fn,
            sample_sizes=generator_sample_sizes,
            target_size=int(generator_target_size),
            max_scaling_exponent=float(generator_max_exponent),
            max_projected_seconds=float(generator_max_projected_seconds),
            regime_name="default",
        )
        generator_audits["default"] = ga
        defects.extend(ga["defects"])
    if generator_regimes:
        for r_name, r_cfg in generator_regimes.items():
            ga = audit_generator_scaling(
                r_cfg.get("generator_fn"),
                sample_sizes=tuple(r_cfg.get("sample_sizes", generator_sample_sizes)),
                target_size=int(r_cfg.get("target_size", generator_target_size)),
                max_scaling_exponent=float(r_cfg.get("max_scaling_exponent", generator_max_exponent)),
                max_projected_seconds=float(r_cfg.get("max_projected_seconds", generator_max_projected_seconds)),
                min_measurable_seconds=float(r_cfg.get("min_measurable_seconds", 1e-4)),
                regime_name=str(r_name),
                clock_fn=r_cfg.get("clock_fn"),
                measured_timings=r_cfg.get("measured_timings"),
            )
            generator_audits[str(r_name)] = ga
            defects.extend(ga["defects"])

    total_generator_seconds = sum(float(g["projected_seconds"]) for g in generator_audits.values())

    # 3. Full total_updates wall-clock estimation
    measured_spu: float | None = None
    spu_source = "not_provided"
    if seconds_per_update is not None:
        measured_spu = float(seconds_per_update)
        if not math.isfinite(measured_spu) or measured_spu < 0.0:
            raise ValueError(f"seconds_per_update must be finite and non-negative, got {seconds_per_update!r}")
        spu_source = "provided"
    elif step_fn is not None:
        if int(step_warmup) < 0 or int(step_repeats) <= 0:
            raise ValueError("step_warmup must be >= 0 and step_repeats >= 1")
        for _ in range(int(step_warmup)):
            _ = step_fn()
        use_cuda_sync = bool(torch.cuda.is_available())
        if use_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(int(step_repeats)):
            _ = step_fn()
        if use_cuda_sync:
            torch.cuda.synchronize()
        measured_spu = max(1e-9, (time.perf_counter() - t0) / float(int(step_repeats)))
        spu_source = "measured_step_fn"

    waves = math.ceil(float(n_jobs) / float(p_workers))
    usable_session_minutes = max(0.0, float(max_session_minutes) - float(reserve_minutes))
    if measured_spu is not None:
        training_seconds = float(waves) * (
            float(tot_u) * measured_spu + max(0.0, float(startup_overhead_seconds_per_job))
        )
        total_projected_seconds = training_seconds + total_generator_seconds
    else:
        training_seconds = None
        total_projected_seconds = total_generator_seconds
    total_projected_minutes = float(total_projected_seconds) / 60.0
    estimated_sessions_required = (
        max(1, math.ceil(total_projected_minutes / usable_session_minutes))
        if usable_session_minutes > 0.0
        else None
    )

    if total_projected_minutes > usable_session_minutes:
        defects.append(
            {
                "code": "WALL_CLOCK_BUDGET_EXCEEDED",
                "detail": (
                    f"projected wall-clock {total_projected_minutes:.2f}m exceeds usable session budget "
                    f"{usable_session_minutes:.2f}m (sessions_required={estimated_sessions_required})"
                ),
            }
        )

    # 4. Disk budget
    projected_disk_bytes = n_jobs * int(per_job_disk_bytes)
    if projected_disk_bytes > int(max_disk_bytes):
        defects.append(
            {
                "code": "DISK_BUDGET_EXCEEDED",
                "detail": (
                    f"projected disk usage {projected_disk_bytes / (1024**3):.2f} GiB > "
                    f"max_disk_bytes {int(max_disk_bytes) / (1024**3):.2f} GiB"
                ),
            }
        )

    passed = len(defects) == 0
    defect_codes = list(dict.fromkeys(str(d["code"]) for d in defects))
    remediation: list[str] = []
    if "PREPACK_VRAM_BLOAT" in defect_codes or "WORKER_VRAM_BUDGET_EXCEEDED" in defect_codes:
        remediation.append(
            "Build attention/causal masks lazily per micro-batch or keep prepacked tensors compact in uint8/bool on CPU before micro-batch slice upload."
        )
    if "GENERATOR_SUPERLINEAR_SCALING" in defect_codes or "GENERATOR_WALL_BUDGET_EXCEEDED" in defect_codes:
        remediation.append(
            "Replace O(N^2) generator uniqueness/pair scans with O(N) hash-set lookups and re-verify with audit_generator_scaling()."
        )
    if "WALL_CLOCK_BUDGET_EXCEEDED" in defect_codes:
        remediation.append(
            "Enable multi-session CampaignOrchestrator resume (`PENDING_QUEUE.json` + `RESUME_INPUT`) or reduce waves / raise usable session budget."
        )
    if "DISK_BUDGET_EXCEEDED" in defect_codes:
        remediation.append(
            "Enable `prune_completed_checkpoints=True`, coarse `ckpt_every`, and `LATEST`-only checkpoint archiving (`collect_minimal_checkpoint_files`)."
        )

    report = {
        "schema": "anra-planthir-production-preflight-v1",
        "passed": passed,
        "status": "PREFLIGHT_PASS" if passed else "PREFLIGHT_BUDGET_EXCEEDED",
        "schedule": {
            "total_updates": tot_u,
            "num_jobs": n_jobs,
            "parallel_workers": p_workers,
            "waves": waves,
            "precision": prec,
            "model_params": n_params,
            "batch_size": bs,
            "seq_len": sl,
        },
        "vram_per_worker": {
            "model_state_bytes": int(model_state_bytes),
            "activation_bytes": int(activation_bytes),
            "prepack_bytes": int(projected_prepack_bytes),
            "cuda_overhead_bytes": int(cuda_overhead_bytes),
            "estimated_total_bytes": int(estimated_total_vram_bytes),
            "estimated_total_gib": round(float(estimated_total_vram_bytes) / float(1024**3), 4),
            "max_prepack_bytes_per_worker": int(max_prepack_bytes_per_worker),
            "max_vram_bytes_per_worker": int(max_vram_bytes_per_worker),
            "prepack_footprint": prepack_footprint,
        },
        "wall_clock": {
            "seconds_per_update": round(measured_spu, 6) if measured_spu is not None else None,
            "seconds_per_update_source": spu_source,
            "training_seconds": round(float(training_seconds), 3) if training_seconds is not None else None,
            "generator_projected_seconds": round(float(total_generator_seconds), 3),
            "total_projected_seconds": round(float(total_projected_seconds), 3),
            "total_projected_minutes": round(float(total_projected_minutes), 3),
            "max_session_minutes": float(max_session_minutes),
            "reserve_minutes": float(reserve_minutes),
            "usable_session_minutes": round(float(usable_session_minutes), 3),
            "estimated_sessions_required": estimated_sessions_required,
        },
        "disk": {
            "per_job_disk_bytes": int(per_job_disk_bytes),
            "projected_disk_bytes": int(projected_disk_bytes),
            "projected_disk_gib": round(float(projected_disk_bytes) / float(1024**3), 4),
            "max_disk_bytes": int(max_disk_bytes),
        },
        "generator_audits": generator_audits,
        "defects": defects,
        "defect_codes": defect_codes,
        "remediation": remediation,
    }
    if fail_closed and not passed:
        raise ValueError(f"PRODUCTION_PREFLIGHT_FAILED:{defect_codes}:{defects[0]['detail']}")
    return report


__all__ = [
    "ResourceMonitor",
    "ResourceQuota",
    "ResourceQuotaExceededError",
    "audit_generator_scaling",
    "cleanup_unreferenced_staging",
    "estimate_production_preflight",
    "measure_directory_bytes",
    "measure_free_disk_bytes",
    "measure_process_memory_bytes",
    "measure_tensor_container_bytes",
]
