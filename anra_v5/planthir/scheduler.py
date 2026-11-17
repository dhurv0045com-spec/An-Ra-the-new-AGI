"""Independent multi-worker scheduler with isolated work_roots, device pinning, and partial-failure containment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .execution import _atomic_write, _sync_dir


def _remap_cuda_command_args(cmd: Sequence[str], *, child_device: str = "cuda:0") -> list[str]:
    """Rewrite child CLI `--device cuda:N` flags to `cuda:0` once CUDA_VISIBLE_DEVICES pins one GPU."""

    out: list[str] = []
    i = 0
    items = [str(x) for x in cmd]
    while i < len(items):
        tok = items[i]
        if tok == "--device" and i + 1 < len(items) and items[i + 1].startswith("cuda:"):
            out.extend(["--device", child_device])
            i += 2
            continue
        if tok.startswith("--device=cuda:"):
            out.append(f"--device={child_device}")
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


@dataclass(frozen=True, slots=True)
class WorkerJobSpec:
    """Declarative specification for one isolated Planthir worker subprocess."""

    worker_id: str
    work_root: Path | str
    command: tuple[str, ...] = ()
    spec_path: Path | str | None = None
    device: str = "cpu"
    canary: bool = True
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def resolved_command(self) -> tuple[str, ...]:
        if self.command:
            return tuple(str(x) for x in self.command)
        if self.spec_path is not None:
            cmd = [
                sys.executable,
                "-m",
                "anra_v5.planthir",
                "run",
                "--spec",
                str(self.spec_path),
                "--work-root",
                str(self.work_root),
                "--device",
                str(self.device),
            ]
            if self.canary:
                cmd.append("--canary")
            return tuple(cmd)
        raise ValueError(f"worker {self.worker_id!r} requires either command or spec_path")


class _WorkerDirectoryLock:
    """Fail-closed per-work_root lock ensuring independent workers never share a mutable directory."""

    def __init__(self, work_root: Path, worker_id: str) -> None:
        self.lock_path = work_root / ".planthir_worker.lock"
        self.worker_id = worker_id
        self._fd: int | None = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = json.dumps({"worker_id": self.worker_id, "coordinator_pid": os.getpid()}).encode("utf-8")
            os.write(self._fd, payload)
            os.close(self._fd)
            self._fd = None
        except FileExistsError as exc:
            raise RuntimeError(f"work_root lock already held at {self.lock_path}") from exc

    def release(self) -> None:
        try:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        finally:
            self.lock_path.unlink(missing_ok=True)


def validate_worker_batch(
    jobs: Sequence[WorkerJobSpec],
    *,
    allow_shared_cuda_device: bool = False,
) -> list[WorkerJobSpec]:
    if not jobs:
        raise ValueError("IndependentWorkerScheduler requires at least one WorkerJobSpec")
    seen_ids: set[str] = set()
    seen_roots: set[Path] = set()
    seen_cuda: set[str] = set()
    normalized: list[WorkerJobSpec] = []
    for job in jobs:
        if not job.worker_id or "/" in job.worker_id or "\\" in job.worker_id:
            raise ValueError(f"invalid worker_id: {job.worker_id!r}")
        if job.worker_id in seen_ids:
            raise ValueError(f"duplicate worker_id in batch: {job.worker_id!r}")
        seen_ids.add(job.worker_id)
        root = Path(job.work_root).resolve()
        if root in seen_roots:
            raise ValueError(f"two workers cannot share work_root: {root}")
        seen_roots.add(root)
        # Ensure neither work_root is nested inside another worker's root
        for prev in seen_roots:
            if prev != root and (prev in root.parents or root in prev.parents):
                raise ValueError(f"nested worker work_roots rejected: {root} vs {prev}")
        dev_norm = str(job.device).strip().lower()
        if dev_norm.startswith("cuda") and not allow_shared_gpu(dev_norm, seen_cuda, allow_shared_cuda_device):
            raise ValueError(f"duplicate CUDA device assignment {job.device!r} without allow_shared_cuda_device=True")
        if dev_norm.startswith("cuda"):
            seen_cuda.add(dev_norm)
        job.resolved_command()
        normalized.append(job)
    return normalized


def allow_shared_gpu(dev_norm: str, seen_cuda: set[str], allow_shared_cuda_device: bool) -> bool:
    return allow_shared_cuda_device or dev_norm not in seen_cuda


class IndependentWorkerScheduler:
    """Runs multiple isolated Planthir worker subprocesses concurrently with fail-contained accounting."""

    def __init__(
        self,
        *,
        coordinator_root: Path | str,
        max_parallel: int | None = None,
        allow_shared_cuda_device: bool = False,
        remap_child_cuda_device: bool = True,
        in_process_runner: Callable[[WorkerJobSpec, Mapping[str, str], Sequence[str]], int | Mapping[str, Any]]
        | Callable[[WorkerJobSpec], int | Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.coordinator_root = Path(coordinator_root)
        self.coordinator_root.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel
        self.allow_shared_cuda_device = allow_shared_cuda_device
        self.remap_child_cuda_device = bool(remap_child_cuda_device)
        self.in_process_runner = in_process_runner

    def _run_one(self, job: WorkerJobSpec, *, log_dir: Path) -> dict[str, Any]:
        wroot = Path(job.work_root).resolve()
        lock = _WorkerDirectoryLock(wroot, job.worker_id)
        lock.acquire()
        t0 = time.perf_counter()
        log_path = log_dir / f"{job.worker_id}.log"
        cmd = list(job.resolved_command())
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in job.env.items()})
        env["PLANTHIR_WORKER_ID"] = job.worker_id
        env["PLANTHIR_ASSIGNED_DEVICE"] = str(job.device)
        env["PLANTHIR_WORKER_LOCK_PATH"] = str(lock.lock_path)
        if str(job.device).startswith("cuda:"):
            env.setdefault("CUDA_VISIBLE_DEVICES", str(job.device).split(":", 1)[1])
            env["PLANTHIR_VISIBLE_CUDA_DEVICE"] = "cuda:0"
            if self.remap_child_cuda_device:
                cmd = _remap_cuda_command_args(cmd, child_device="cuda:0")

        try:
            if self.in_process_runner is not None:
                import inspect

                sig = inspect.signature(self.in_process_runner)
                if len(sig.parameters) >= 3:
                    raw_res = self.in_process_runner(job, env, cmd)  # type: ignore[misc]
                else:
                    raw_res = self.in_process_runner(job)  # type: ignore[call-arg]
                if isinstance(raw_res, Mapping):
                    exit_code = int(raw_res.get("exit_code", 0))
                    stdout_text = str(raw_res.get("stdout", ""))
                    stderr_text = str(raw_res.get("stderr", ""))
                else:
                    exit_code = int(raw_res)
                    stdout_text = ""
                    stderr_text = ""
            else:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=job.timeout_seconds,
                    check=False,
                )
                exit_code = int(proc.returncode)
                stdout_text = proc.stdout
                stderr_text = proc.stderr
            dt = round(float(time.perf_counter() - t0), 6)
            log_text = (
                f"$ {' '.join(cmd)}\n"
                f"--- STDOUT ---\n{stdout_text}\n"
                f"--- STDERR ---\n{stderr_text}\n"
            )
            log_path.write_text(log_text, encoding="utf-8", errors="replace")
            status = "SUCCEEDED" if exit_code == 0 else "FAILED"
            resource_receipt = wroot / "RESOURCE_RECEIPT.json"
            res_summary = None
            if resource_receipt.is_file():
                try:
                    res_summary = json.loads(resource_receipt.read_text(encoding="utf-8"))
                except Exception:
                    res_summary = None
            return {
                "worker_id": job.worker_id,
                "status": status,
                "exit_code": exit_code,
                "work_root": str(wroot),
                "device": str(job.device),
                "visible_cuda_device": env.get("PLANTHIR_VISIBLE_CUDA_DEVICE", str(job.device)),
                "command": cmd,
                "wall_seconds": dt,
                "log_path": str(log_path),
                "stdout_tail": stdout_text[-1000:],
                "stderr_tail": stderr_text[-1000:],
                "resource_receipt_path": str(resource_receipt) if resource_receipt.is_file() else None,
                "resource_summary": (
                    {
                        "total_wall_seconds": res_summary.get("total_wall_seconds"),
                        "peak_cpu_rss_bytes": res_summary.get("peak_cpu_rss_bytes"),
                        "work_total_bytes": res_summary.get("work_total_bytes"),
                    }
                    if isinstance(res_summary, dict)
                    else None
                ),
            }
        except subprocess.TimeoutExpired as exc:
            dt = round(float(time.perf_counter() - t0), 6)
            log_path.write_text(
                f"$ {' '.join(cmd)}\nTIMEOUT after {job.timeout_seconds}s\n",
                encoding="utf-8",
            )
            return {
                "worker_id": job.worker_id,
                "status": "TIMED_OUT",
                "exit_code": -1,
                "work_root": str(wroot),
                "device": str(job.device),
                "command": cmd,
                "wall_seconds": dt,
                "log_path": str(log_path),
                "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
                "resource_receipt_path": None,
                "resource_summary": None,
            }
        finally:
            lock.release()

    def run(self, jobs: Sequence[WorkerJobSpec]) -> dict[str, Any]:
        validated = validate_worker_batch(
            jobs,
            allow_shared_cuda_device=self.allow_shared_cuda_device,
        )
        log_dir = self.coordinator_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        workers_n = self.max_parallel if self.max_parallel is not None else len(validated)
        workers_n = max(1, min(int(workers_n), len(validated)))

        t0 = time.perf_counter()
        results_by_id: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers_n) as pool:
            futs = {pool.submit(self._run_one, job, log_dir=log_dir): job for job in validated}
            for fut in as_completed(futs):
                res = fut.result()
                results_by_id[res["worker_id"]] = res

        ordered = [results_by_id[j.worker_id] for j in validated]
        succeeded = [r["worker_id"] for r in ordered if r["status"] == "SUCCEEDED"]
        failed = [r["worker_id"] for r in ordered if r["status"] != "SUCCEEDED"]
        if len(succeeded) == len(ordered):
            overall = "COMPLETE"
        elif succeeded:
            overall = "PARTIAL_FAILURE"
        else:
            overall = "FAILED"

        receipt = {
            "schema": "anra-planthir-coordinator-receipt-v1",
            "status": overall,
            "coordinator_root": str(self.coordinator_root),
            "total_workers": len(ordered),
            "max_parallel": workers_n,
            "succeeded_count": len(succeeded),
            "failed_count": len(failed),
            "succeeded_workers": succeeded,
            "failed_workers": failed,
            "total_wall_seconds": round(float(time.perf_counter() - t0), 6),
            "workers": ordered,
        }
        receipt_path = self.coordinator_root / "COORDINATOR_RECEIPT.json"
        _atomic_write(
            receipt_path,
            (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
        )
        _sync_dir(self.coordinator_root)
        return receipt


def run_scheduled_workers(
    jobs: Sequence[WorkerJobSpec],
    *,
    coordinator_root: Path | str,
    max_parallel: int | None = None,
    allow_shared_cuda_device: bool = False,
) -> dict[str, Any]:
    return IndependentWorkerScheduler(
        coordinator_root=coordinator_root,
        max_parallel=max_parallel,
        allow_shared_cuda_device=allow_shared_cuda_device,
    ).run(jobs)


__all__ = [
    "IndependentWorkerScheduler",
    "WorkerJobSpec",
    "run_scheduled_workers",
    "validate_worker_batch",
]
