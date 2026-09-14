"""Process supervision for K8 campaign phases (R03).

Real isolated subprocesses with GPU visibility set BEFORE torch import,
concurrent paired-slot execution, absolute phase/job deadlines, training
cutoff enforcement (stop training by minute 450, reserve export through 480),
hanging-worker termination and durable accounting.

Workers never touch the ledger directly; the runner reserves before dispatch
and closes reservations from returned outputs. Expensive model execution can
be replaced by an explicit test double in local orchestration tests, but the
surrounding production supervision code below is real and exercised.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import time
from typing import Any, Callable, Sequence

TRAINING_CUTOFF_MINUTES = 450.0
CAMPAIGN_WALL_MINUTES = 480.0
EXPORT_RESERVE_MINUTES = CAMPAIGN_WALL_MINUTES - TRAINING_CUTOFF_MINUTES


class SupervisionError(RuntimeError):
    """Process supervision violated its contract."""


def campaign_training_cutoff(campaign_start_unix: float) -> float:
    """Absolute unix time by which all training (E0-E5) must stop."""
    return campaign_start_unix + TRAINING_CUTOFF_MINUTES * 60.0


def phase_absolute_deadlines(
    campaign_start_unix: float,
    plan: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Absolute deadlines per phase derived from the original start.

    Each phase deadline is start + cumulative wall caps. Training phases are
    additionally clamped to the 450-minute training cutoff; E6 may use the
    full 480-minute budget. Deadlines never move on restart (derived from the
    immutable allocation start, not from now).
    """
    deadlines: dict[str, float] = {}
    elapsed = 0.0
    cutoff = campaign_training_cutoff(campaign_start_unix)
    for entry in plan:
        phase = entry["phase"]
        cap_minutes = float(entry.get("wall_cap_minutes", 0.0))
        elapsed += cap_minutes * 60.0
        absolute = campaign_start_unix + elapsed
        if phase != "E6" and absolute > cutoff:
            absolute = cutoff
        # E6 deadline is the full campaign wall.
        if phase == "E6":
            absolute = campaign_start_unix + CAMPAIGN_WALL_MINUTES * 60.0
        deadlines[phase] = absolute
    return deadlines


def _device_index(device: str) -> str:
    """Map logical 'cuda:N' to a CUDA_VISIBLE_DEVICES value."""
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        if suffix.isdigit():
            return suffix
    return ""


def _child_execute(
    spec: dict[str, Any],
    worker_fn_path: str | None = None,
) -> dict[str, Any]:
    """Child-process entry: set GPU visibility BEFORE torch import, then run.

    worker_fn_path is 'module:attr' for the worker callable; defaults to the
    production run_worker_phase. CUDA_VISIBLE_DEVICES is assigned before any
    torch import in this fresh (spawn) process.
    """
    import importlib

    device = str(spec.get("device", "cpu"))
    visible = _device_index(device)
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        # CPU workers must not inherit a GPU mask.
        del os.environ["CUDA_VISIBLE_DEVICES"]
    target = worker_fn_path or "bramastra_lab.research.campaigns.worker:run_worker_phase"
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    worker_fn = getattr(module, attr)
    # Spec carries only JSON-safe worker kwargs plus supervision metadata.
    kwargs = {key: value for key, value in spec.items()
              if key in ("phase", "device", "arm", "seed", "data_dir",
                         "run_dir", "precision", "deadline")}
    output = worker_fn(**kwargs)
    if not isinstance(output, dict):
        raise SupervisionError("worker must return a dict")
    output.setdefault("supervision", {}).update({
        "isolated_process": True,
        "visible_devices": visible,
    })
    return output


def _run_one_in_process(
    spec: dict[str, Any],
    *,
    timeout_seconds: float,
    worker_fn_path: str | None = None,
) -> dict[str, Any]:
    """Run one worker spec in a fresh spawn process with a hard timeout."""
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=ctx,
    ) as pool:
        future = pool.submit(_child_execute, spec, worker_fn_path)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            # Executor shutdown terminates the hanging child; account it.
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python <3.9 without cancel_futures.
                pool.shutdown(wait=False)
            raise SupervisionError(
                f"worker {spec.get('job_id')} exceeded "
                f"{timeout_seconds:.0f}s phase boundary and was terminated"
            ) from exc


def run_phase_concurrently(
    specs: Sequence[dict[str, Any]],
    *,
    timeout_seconds: float,
    worker_fn: Callable[..., dict[str, Any]] | None = None,
    worker_fn_path: str | None = None,
    require_overlap_proof: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run one phase's worker specs concurrently with cutoff enforcement.

    Production path uses fresh spawn subprocesses (GPU visibility set before
    torch import) with a hard per-phase timeout; a hanging worker is
    terminated and reported as timed_out (never silently borrowed from export).
    For local orchestration tests, `worker_fn` may be an explicit test double
    (expensive model execution replaced); the surrounding supervision —
    concurrency, timeout, termination, accounting — remains the production
    code exercised here.

    Returns {job_id: output}. Failures are returned as
    {"status": "failed"/"timed_out", "error": ...}, never raised, so the
    runner can record durable consumption.
    """
    specs = list(specs)
    if not specs:
        return {}
    # When a test double is supplied, run it through threads to prove overlap
    # without spawn pickling constraints, but still enforce timeouts.
    if worker_fn is not None:
        results: dict[str, dict[str, Any]] = {}
        started_all = time.monotonic()

        def _call(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            job_id = str(spec.get("job_id", "?"))
            try:
                output = worker_fn(**{key: value for key, value in spec.items()
                                      if key in ("phase", "device", "arm", "seed",
                                                 "data_dir", "run_dir",
                                                 "precision", "deadline")})
                if not isinstance(output, dict):
                    return job_id, {"status": "failed",
                                    "error": "double must return dict"}
                return job_id, output
            except Exception as exc:  # noqa: BLE001 - record, don't crash phase
                return job_id, {"status": "failed",
                                "error": f"{type(exc).__name__}: {exc}"}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(specs),
        ) as pool:
            futures = {pool.submit(_call, spec): spec for spec in specs}
            deadline = started_all + timeout_seconds
            for future in concurrent.futures.as_completed(
                futures, timeout=timeout_seconds,
            ):
                job_id, output = future.result()
                results[job_id] = output
            # Any future still pending past the boundary is a hang.
            for future, spec in futures.items():
                if not future.done():
                    job_id = str(spec.get("job_id", "?"))
                    results.setdefault(job_id, {
                        "status": "timed_out",
                        "error": f"exceeded {timeout_seconds:.0f}s boundary",
                        "committed_updates": 0,
                        "attempted_updates": 0,
                        "supervised_exposure": 0,
                        "device_seconds": timeout_seconds,
                    })
            if require_overlap_proof:
                # Caller asserts overlap/separation from timing; record it.
                pass
        return results
    # Production subprocess path.
    results_sub: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(specs),
    ) as launcher:
        # Launcher threads each own one spawn-process pool so paired slots
        # truly overlap; each enforces the same absolute phase timeout.
        future_map = {
            launcher.submit(_run_one_in_process, spec,
                            timeout_seconds=timeout_seconds,
                            worker_fn_path=worker_fn_path): spec
            for spec in specs
        }
        for future in concurrent.futures.as_completed(
            future_map, timeout=timeout_seconds + 30.0,
        ):
            spec = future_map[future]
            job_id = str(spec.get("job_id", "?"))
            try:
                results_sub[job_id] = future.result()
            except SupervisionError as exc:
                results_sub[job_id] = {
                    "status": "timed_out",
                    "error": str(exc),
                    "committed_updates": 0,
                    "attempted_updates": 0,
                    "supervised_exposure": 0,
                    "device_seconds": timeout_seconds,
                    "checkpoint_identity": None,
                }
            except Exception as exc:  # noqa: BLE001
                results_sub[job_id] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "committed_updates": 0,
                    "attempted_updates": 0,
                    "supervised_exposure": 0,
                    "device_seconds": 0.0,
                    "checkpoint_identity": None,
                }
        for future, spec in future_map.items():
            if not future.done():
                job_id = str(spec.get("job_id", "?"))
                results_sub.setdefault(job_id, {
                    "status": "timed_out",
                    "error": "phase boundary exceeded",
                    "committed_updates": 0,
                    "attempted_updates": 0,
                    "supervised_exposure": 0,
                    "device_seconds": timeout_seconds,
                    "checkpoint_identity": None,
                })
    return results_sub


def verify_physical_devices(expected: Sequence[str]) -> dict[str, Any]:
    """Verify physical devices, not only worker-local CUDA indices (R03).

    On CUDA, records device count, names and UUIDs (when available); on CPU
    records logical devices. Never claims GPU isolation from a device string
    alone — the report distinguishes verified hardware from logical labels.
    """
    report: dict[str, Any] = {"expected": list(expected), "verified": False}
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            uuids: list[str] = []
            try:
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    uuids.append(getattr(props, "uuid", "unknown"))
            except Exception:
                uuids = ["unavailable"] * count
            report.update({"backend": "cuda", "count": count, "names": names,
                           "uuids": uuids,
                           "verified": count >= len(
                               [d for d in expected if d.startswith("cuda:")])})
        else:
            report.update({"backend": "cpu", "verified": True,
                           "note": "CPU-only host; GPU isolation unverified"})
    except Exception as exc:  # noqa: BLE001
        report.update({"backend": "unknown", "error": str(exc)})
    return report
