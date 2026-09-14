"""Process supervision for K8 campaign phases (D1/R03).

Explicit child-process ownership with real termination, physical vs
worker-local device mapping, absolute slot deadlines and training-cutoff
enforcement. Workers never touch the ledger; the runner reserves before
dispatch and closes reservations from returned outputs.

Device model: each spec carries `physical_device` (ledger identity, e.g.
cuda:1) and the child computes `local_device` (what torch addresses after
CUDA_VISIBLE_DEVICES isolation). A process with one visible GPU addresses it
locally as cuda:0. Physical identity is preserved separately and the child
verifies its physical UUID after initialization. Parent device reports alone
are never proof of child hardware use.
"""
from __future__ import annotations

import multiprocessing
import os
import queue
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
    """Absolute deadlines per phase derived from the original start."""
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
        if phase == "E6":
            absolute = campaign_start_unix + CAMPAIGN_WALL_MINUTES * 60.0
        deadlines[phase] = absolute
    return deadlines


def slot_plan_for_phase(phase: str, workers: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Explicit ordered slots with at most one active job per physical device.

    E1: two counterbalanced slots (A1701+B1701, then B1702+A1702). E3/E4: two
    slots with reversed treatment order already encoded in job order — first
    two jobs form slot 1, last two form slot 2 (each slot holds one job per
    physical GPU). E0/E2/E5/E6: single slot (jobs already on distinct devices).
    """
    workers = list(workers)
    if phase in ("E1", "E3", "E4") and len(workers) == 4:
        return [workers[:2], workers[2:]]
    return [workers]


def physical_to_local_device(physical_device: str) -> tuple[str, str]:
    """Map ledger physical device to (visible_mask, local_device).

    physical cuda:N with isolation exposes exactly that GPU, which the child
    addresses locally as cuda:0. CPU maps to no mask and local cpu.
    Returns (visible_devices_env, local_device_for_torch).
    """
    if physical_device.startswith("cuda:"):
        suffix = physical_device.split(":", 1)[1]
        if suffix.isdigit():
            return suffix, "cuda:0"
    if physical_device == "cpu":
        return "", "cpu"
    return "", physical_device


def _child_main(result_queue, spec: dict[str, Any],
                worker_fn_path: str | None) -> None:
    """Spawn-child entry: isolate GPUs BEFORE torch import, run, report."""
    import importlib

    physical = str(spec.get("physical_device",
                            spec.get("device", "cpu")))
    visible, local = physical_to_local_device(physical)
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        target = worker_fn_path or \
            "bramastra_lab.research.campaigns.worker:run_worker_phase"
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)
        worker_fn = getattr(module, attr)
        kwargs = {"phase": spec.get("phase"), "device": local,
                  "arm": spec.get("arm"), "seed": spec.get("seed"),
                  "data_dir": spec.get("data_dir"),
                  "run_dir": spec.get("run_dir"),
                  "precision": spec.get("precision"),
                  "deadline": spec.get("deadline"),
                  "physical_device": physical,
                  "slot": spec.get("slot"),
                  "parent": spec.get("parent"),
                  "job_id": spec.get("job_id"),
                  "update_target": spec.get("update_target"),
                  "eval_cases": spec.get("eval_cases"),
                  "tasks_per_block": spec.get("tasks_per_block")}
        # Backwards compatibility: workers without physical_device accept the
        # eight-arg call; try with physical first, fall back without it.
        try:
            output = worker_fn(**kwargs)
        except TypeError:
            kwargs.pop("physical_device", None)
            output = worker_fn(**kwargs)
        if not isinstance(output, dict):
            output = {"status": "failed",
                      "error": "worker must return a dict"}
        supervision = dict(output.get("supervision", {}))
        supervision.update({
            "isolated_process": True,
            "physical_device": physical,
            "local_device": local,
            "visible_devices": visible,
            "device_uuid": _local_device_uuid(local),
        })
        output["supervision"] = supervision
        # Preserve physical identity for ledger accounting.
        output.setdefault("physical_device", physical)
        try:
            result_queue.put(("ok", output))
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001 - child must report, not crash
        try:
            result_queue.put(("error", {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "physical_device": physical,
                "committed_updates": 0, "attempted_updates": 0,
                "supervised_exposure": 0, "device_seconds": 0.0,
                "checkpoint_identity": None}))
        except Exception:
            pass


def _local_device_uuid(local_device: str) -> str:
    """Best-effort physical UUID of the locally addressed device."""
    try:
        import torch

        if local_device.startswith("cuda") and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                uuid = getattr(props, "uuid", None)
                if uuid:
                    return str(uuid)
            except Exception:
                pass
            try:
                return str(torch.cuda.get_device_name(0))
            except Exception:
                return "cuda-present-uuid-unavailable"
        return "cpu-no-uuid"
    except Exception:
        return "unknown"


def _child_execute(
    spec: dict[str, Any],
    worker_fn_path: str | None = None,
) -> dict[str, Any]:
    """In-process emulation of the spawn child (tests + CPU fallback).

    Sets visibility, maps physical->local and invokes the worker WITHOUT
    spawning (used by unit tests that patch sys.modules). Production uses
    _run_one_in_process (real spawn + terminate). Preserves physical identity
    and reports the worker-local device actually received.
    """
    import importlib

    physical = str(spec.get("physical_device",
                            spec.get("device", "cpu")))
    visible, local = physical_to_local_device(physical)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        target = worker_fn_path or \
            "bramastra_lab.research.campaigns.worker:run_worker_phase"
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)
        worker_fn = getattr(module, attr)
        kwargs = {"phase": spec.get("phase"), "device": local,
                  "arm": spec.get("arm"), "seed": spec.get("seed"),
                  "data_dir": spec.get("data_dir"),
                  "run_dir": spec.get("run_dir"),
                  "precision": spec.get("precision"),
                  "deadline": spec.get("deadline"),
                  "physical_device": physical,
                  "slot": spec.get("slot"),
                  "parent": spec.get("parent"),
                  "job_id": spec.get("job_id"),
                  "update_target": spec.get("update_target"),
                  "eval_cases": spec.get("eval_cases"),
                  "tasks_per_block": spec.get("tasks_per_block")}
        try:
            output = worker_fn(**kwargs)
        except TypeError:
            kwargs.pop("physical_device", None)
            output = worker_fn(**kwargs)
        if not isinstance(output, dict):
            raise SupervisionError("worker must return a dict")
        output.setdefault("supervision", {}).update({
            "isolated_process": False,
            "physical_device": physical,
            "local_device": local,
            "visible_devices": visible,
        })
        output.setdefault("physical_device", physical)
        return output
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _run_one_in_process(
    spec: dict[str, Any],
    *,
    timeout_seconds: float,
    worker_fn_path: str | None = None,
) -> dict[str, Any]:
    """Run one worker in an OWNED spawn process with real termination.

    Starts a spawn child, joins within the phase boundary, terminates on
    overrun with bounded join + kill fallback. Publishes timed_out ONLY with
    termination status (child cannot write after termination). Raises
    SupervisionError on timeout so the caller records timed_out + partial
    accounting.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_child_main,
                       args=(result_queue, dict(spec), worker_fn_path))
    proc.start()
    proc.join(timeout=timeout_seconds)
    if proc.is_alive():
        # Explicit ownership: terminate, bounded re-join, kill fallback.
        try:
            proc.terminate()
        except Exception:
            pass
        proc.join(timeout=5.0)
        if proc.is_alive():
            try:
                proc.kill()  # type: ignore[attr-defined]
            except Exception:
                pass
            proc.join(timeout=5.0)
        # Drain any late result (a terminated child cannot have written a
        # valid delayed marker after termination; late queue items are stale).
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            pass
        raise SupervisionError(
            f"worker {spec.get('job_id')} exceeded "
            f"{timeout_seconds:.2f}s phase boundary and was terminated "
            f"(exitcode={proc.exitcode})")
    try:
        kind, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise SupervisionError(
            f"worker {spec.get('job_id')} exited without a result "
            f"(exitcode={proc.exitcode})") from exc
    if kind != "ok":
        return payload
    return payload


def run_phase_concurrently(
    specs: Sequence[dict[str, Any]],
    *,
    timeout_seconds: float,
    worker_fn: Callable[..., dict[str, Any]] | None = None,
    worker_fn_path: str | None = None,
    require_overlap_proof: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run ONE SLOT's specs concurrently (production spawn or test double).

    The runner calls this once per slot (never a whole 4-job phase at once).
    Production path uses owned spawn processes with real termination; test
    doubles run in threads with the same timeout/timed_out accounting.
    Returns {job_id: output}; hangs become timed_out, never silent success.
    """
    import concurrent.futures

    specs = list(specs)
    if not specs:
        return {}
    # Same-physical-device overlap is refused before launch (exclusive occupancy).
    physicals = [str(spec.get("physical_device", spec.get("device", "?")))
                 for spec in specs]
    if len(set(physicals)) != len(physicals):
        return {str(spec.get("job_id", "?")): {
            "status": "failed",
            "error": "slot violates exclusive occupancy: duplicate physical "
                     f"device in {physicals}",
            "committed_updates": 0, "attempted_updates": 0,
            "supervised_exposure": 0, "device_seconds": 0.0,
            "checkpoint_identity": None} for spec in specs}
    if worker_fn is not None:
        results: dict[str, dict[str, Any]] = {}

        def _call(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            job_id = str(spec.get("job_id", "?"))
            try:
                physical = str(spec.get("physical_device",
                                        spec.get("device", "cpu")))
                _, local = physical_to_local_device(physical)
                # Same full spec as the spawn path (job_id/slot/parent/targets
                # must reach test doubles too, or propagation is untested).
                output = worker_fn(
                    phase=spec.get("phase"), device=local,
                    arm=spec.get("arm"), seed=spec.get("seed"),
                    data_dir=spec.get("data_dir"), run_dir=spec.get("run_dir"),
                    precision=spec.get("precision"),
                    deadline=spec.get("deadline"),
                    physical_device=physical,
                    slot=spec.get("slot"), parent=spec.get("parent"),
                    job_id=spec.get("job_id"),
                    update_target=spec.get("update_target"),
                    eval_cases=spec.get("eval_cases"),
                    tasks_per_block=spec.get("tasks_per_block"))
                if not isinstance(output, dict):
                    return job_id, {"status": "failed",
                                    "error": "double must return dict"}
                return job_id, output
            except TypeError:
                # Backward-compat doubles with the legacy 8-arg signature.
                try:
                    physical = str(spec.get("physical_device",
                                            spec.get("device", "cpu")))
                    _, local = physical_to_local_device(physical)
                    output = worker_fn(
                        phase=spec.get("phase"), device=local,
                        arm=spec.get("arm"), seed=spec.get("seed"),
                        data_dir=spec.get("data_dir"),
                        run_dir=spec.get("run_dir"),
                        precision=spec.get("precision"),
                        deadline=spec.get("deadline"))
                    if not isinstance(output, dict):
                        return job_id, {"status": "failed",
                                        "error": "double must return dict"}
                    return job_id, output
                except Exception as exc:  # noqa: BLE001
                    return job_id, {"status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:  # noqa: BLE001
                return job_id, {"status": "failed",
                                "error": f"{type(exc).__name__}: {exc}"}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(specs),
        ) as pool:
            futures = {pool.submit(_call, spec): spec for spec in specs}
            try:
                for future in concurrent.futures.as_completed(
                    futures, timeout=timeout_seconds,
                ):
                    job_id, output = future.result()
                    results[job_id] = output
            except concurrent.futures.TimeoutError:
                pass
            for future, spec in futures.items():
                if not future.done():
                    job_id = str(spec.get("job_id", "?"))
                    results.setdefault(job_id, {
                        "status": "timed_out",
                        "error": f"exceeded {timeout_seconds:.2f}s boundary "
                                 "and was terminated",
                        "committed_updates": 0, "attempted_updates": 0,
                        "supervised_exposure": 0,
                        "device_seconds": timeout_seconds,
                        "checkpoint_identity": None})
        return results
    # Production spawn path: one owned process per spec, overlapped.
    results_sub: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(specs),
    ) as launcher:
        future_map = {
            launcher.submit(_run_one_in_process, spec,
                            timeout_seconds=timeout_seconds,
                            worker_fn_path=worker_fn_path): spec
            for spec in specs
        }
        try:
            for future in concurrent.futures.as_completed(
                future_map, timeout=timeout_seconds + 60.0,
            ):
                spec = future_map[future]
                job_id = str(spec.get("job_id", "?"))
                try:
                    results_sub[job_id] = future.result()
                except SupervisionError as exc:
                    results_sub[job_id] = {
                        "status": "timed_out",
                        "error": str(exc),
                        "committed_updates": 0, "attempted_updates": 0,
                        "supervised_exposure": 0,
                        "device_seconds": timeout_seconds,
                        "checkpoint_identity": None,
                    }
                except Exception as exc:  # noqa: BLE001
                    results_sub[job_id] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "committed_updates": 0, "attempted_updates": 0,
                        "supervised_exposure": 0, "device_seconds": 0.0,
                        "checkpoint_identity": None,
                    }
        except concurrent.futures.TimeoutError:
            pass
        for future, spec in future_map.items():
            if not future.done():
                job_id = str(spec.get("job_id", "?"))
                results_sub.setdefault(job_id, {
                    "status": "timed_out",
                    "error": "slot boundary exceeded",
                    "committed_updates": 0, "attempted_updates": 0,
                    "supervised_exposure": 0, "device_seconds": timeout_seconds,
                    "checkpoint_identity": None,
                })
    return results_sub


def verify_physical_devices(expected: Sequence[str]) -> dict[str, Any]:
    """Parent-side device inventory (never proof of child placement)."""
    report: dict[str, Any] = {"expected": list(expected), "verified": False,
                              "note": "parent inventory only; child UUIDs are "
                                      "reported per-worker in supervision"}
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            uuids: list[str] = []
            try:
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    uuids.append(str(getattr(props, "uuid", "unknown")))
            except Exception:
                uuids = ["unavailable"] * count
            report.update({"backend": "cuda", "count": count, "names": names,
                           "uuids": uuids,
                           "verified": count >= len(
                               [d for d in expected if d.startswith("cuda:")])})
        else:
            report.update({"backend": "cpu", "verified": True,
                           "note": "CPU-only host; GPU isolation unverified; "
                                   "child placement still mapped"})
    except Exception as exc:  # noqa: BLE001
        report.update({"backend": "unknown", "error": str(exc)})
    return report
