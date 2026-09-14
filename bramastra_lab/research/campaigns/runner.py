"""Campaign runner (I05): one supervisor, two workers, one persistent
allowance. ``run --mode e0`` runs only E0; ``run --mode full`` executes the
gated campaign. The runner refuses to start E1+ without qualified successful
E0 receipts for both workers (distinct devices, actual outcomes).

Fail-closed: a blocked or failed required job yields a non-successful
campaign result and nonzero exit code. Resume is per-job (successful
individual job receipts), never per-phase names alone. Training stops by
minute 450; export reserves 450-480. Workers run in isolated spawn
subprocesses with GPU visibility set before torch import (R03).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Sequence

from bramastra_lab.research.campaigns.supervisor import (
    CampaignLedger,
    SupervisorError,
    SupervisorLease,
)


def run_campaign(*, run_dir: str, mode: str, data_dir: str,
                 max_wall_minutes: float = 480.0,
                 devices: Sequence[str] = ("cuda:0", "cuda:1"),
                 precision: str = "fp16_autocast") -> int:
    from bramastra_lab.research.contracts.core import content_identity

    if mode not in ("e0", "full"):
        raise SupervisorError(f"unknown mode {mode!r}")
    os.makedirs(run_dir, exist_ok=True)
    # Fail before expensive work when requirements are absent (R08).
    manifest_path = os.path.join(data_dir, "manifest.json")
    if mode in ("e0", "full") and not os.path.exists(manifest_path):
        print(json.dumps({"status": "DATA_NOT_READY",
                          "message": f"prepared bundle manifest missing: {manifest_path}; "
                                     "run prepare/validate first"}))
        return 2
    source_hash = _source_closure_hash()
    data_hash = _hash_dir(data_dir)
    allocation_id = content_identity({"data": data_hash,
                                      "source": source_hash,
                                      "max_wall": max_wall_minutes})
    lease = SupervisorLease(run_dir)
    token = lease.acquire()
    ledger: CampaignLedger | None = None
    try:
        ledger = CampaignLedger(run_dir)
        try:
            deadline = ledger.record_allocation(allocation_id, source_hash,
                                                data_hash, max_wall_minutes)
        except SupervisorError as exc:
            # Incompatible reentry into a bound run directory (R02).
            print(json.dumps({"status": "ALLOCATION_REJECTED",
                              "error": str(exc)}))
            return 2
        campaign_start = deadline - max_wall_minutes * 60.0
        # Per-job resume: successful individual job receipts, not phase names.
        completed_jobs = {row[0] for row in ledger.conn.execute(
            "SELECT job_id FROM reservations WHERE status='completed'"
        ).fetchall()}
        qualified_e0 = ledger.qualified_phase_receipts("E0")
        if mode == "e0" and len(qualified_e0) >= 2 and len(
                {row.device for row in qualified_e0}) >= 2:
            print(json.dumps({"status": "E0_ALREADY_RUN",
                              "message": "qualified E0 receipts exist; full mode uses the "
                                         "same allocation without duplicating E0"}))
            return 0
        # E0 admission gate: full mode requires QUALIFIED E0 on distinct devices.
        if mode == "full" and not ledger.phase_success("E0", required_workers=2):
            print(json.dumps({"status": "E0_GATE_BLOCKED",
                              "message": "full campaign requires qualified successful E0 "
                                         "receipts on both workers (distinct devices, "
                                         "checkpoint + updates); run --mode e0 first"}))
            return 1
        campaign_plan = _phase_plan(mode, deadline, devices)
        ledger.append_event("campaign_started", {"mode": mode,
                                                 "plan": campaign_plan,
                                                 "allocation_id": allocation_id})
        from bramastra_lab.research.campaigns import process_supervision as ps
        from bramastra_lab.research.campaigns.worker import run_worker_phase  # noqa: F401 (path ref)

        phase_deadlines = ps.phase_absolute_deadlines(campaign_start, campaign_plan)
        training_cutoff = ps.campaign_training_cutoff(campaign_start)
        device_report = ps.verify_physical_devices(list(devices))
        ledger.append_event("device_verification", device_report)

        results: dict[str, Any] = {}
        campaign_failed = False
        for phase_entry in campaign_plan:
            phase = phase_entry["phase"]
            if campaign_failed and phase != "E6":
                ledger.append_event("phase_skipped", {"phase": phase,
                                                       "reason": "prior_phase_failed"})
                continue
            # Training cutoff: E0-E5 must not borrow export time (R03).
            if phase != "E6" and time.time() > training_cutoff:
                ledger.append_event("phase_skipped", {
                    "phase": phase, "reason": "training_cutoff_exceeded",
                    "cutoff_unix": training_cutoff})
                campaign_failed = True
                continue
            # Per-job filtering: skip only already-completed jobs.
            pending_workers = [entry for entry in phase_entry["workers"]
                               if entry["job_id"] not in completed_jobs]
            if not pending_workers:
                continue
            # Reserve all pending jobs in this phase first (transactional
            # capacity is enforced per-device inside reserve).
            reservations: dict[str, Any] = {}
            reserve_failed = False
            for worker_entry in pending_workers:
                try:
                    reservations[worker_entry["job_id"]] = ledger.reserve(
                        job_id=worker_entry["job_id"], worker=worker_entry["worker"],
                        device=worker_entry["device"], phase=phase,
                        arm=worker_entry.get("arm"), seed=worker_entry.get("seed"),
                        reserved_seconds=worker_entry["wall_seconds"])
                except SupervisorError as exc:
                    ledger.append_event("worker_failed", {
                        "job_id": worker_entry["job_id"],
                        "error": f"SupervisorError: {exc}"})
                    results[worker_entry["job_id"]] = {"status": "failed",
                                                       "error": str(exc)}
                    reserve_failed = True
                    campaign_failed = True
            if reserve_failed:
                ledger.append_event("phase_failed", {"phase": phase,
                                                     "reason": "reservation_refused"})
                continue
            # Absolute phase timeout: min(wall cap, time to phase deadline,
            # time to training cutoff for training phases, time to campaign deadline).
            now = time.time()
            phase_deadline = phase_deadlines.get(phase, deadline)
            cap_seconds = float(phase_entry.get("wall_cap_minutes", 30.0)) * 60.0
            timeout_seconds = min(cap_seconds, max(1.0, phase_deadline - now),
                                  max(1.0, deadline - now))
            if phase != "E6":
                timeout_seconds = min(timeout_seconds,
                                      max(1.0, training_cutoff - now))
            if timeout_seconds <= 0:
                ledger.append_event("phase_failed", {
                    "phase": phase, "reason": "deadline_already_exceeded"})
                campaign_failed = True
                continue
            specs = []
            for worker_entry in pending_workers:
                specs.append({
                    "job_id": worker_entry["job_id"],
                    "phase": phase, "device": worker_entry["device"],
                    "arm": worker_entry.get("arm"), "seed": worker_entry.get("seed"),
                    "data_dir": data_dir, "run_dir": run_dir,
                    "precision": precision,
                    "deadline": min(deadline, phase_deadline),
                })
            outputs = ps.run_phase_concurrently(
                specs, timeout_seconds=timeout_seconds,
                worker_fn_path="bramastra_lab.research.campaigns.worker:run_worker_phase",
            )
            phase_failed = False
            for worker_entry in pending_workers:
                job_id = worker_entry["job_id"]
                reservation = reservations[job_id]
                reserve_started = reservation.reserved_at_unix
                output = outputs.get(job_id, {"status": "failed",
                                              "error": "missing worker output"})
                status = output.get("status")
                # E0 requires resume agreement; other learned phases require
                # completed + real consumption; refused/pending/blocked fail.
                if phase == "E0":
                    success = status == "completed" and output.get(
                        "resume_agrees", False) is True
                elif phase == "E6":
                    success = status == "completed"
                else:
                    success = status == "completed" and int(
                        output.get("committed_updates", 0)) > 0
                # Durable failure consumption: never erase work already
                # attempted; on exception record elapsed wall time (R02).
                if status in ("failed", "timed_out") and not output.get(
                        "device_seconds"):
                    output["device_seconds"] = max(
                        0.0, time.time() - reserve_started)
                try:
                    ledger.close_reservation(
                        reservation.reservation_id,
                        status="completed" if success else "failed",
                        committed_updates=int(output.get("committed_updates", 0)),
                        attempted_updates=int(output.get("attempted_updates", 0)),
                        supervised_exposure=int(output.get("supervised_exposure", 0)),
                        device_seconds=float(output.get("device_seconds", 0.0)),
                        checkpoint_identity=output.get("checkpoint_identity"))
                except SupervisorError as exc:
                    ledger.append_event("worker_failed", {
                        "job_id": job_id,
                        "error": f"close_reservation refused: {exc}"})
                    success = False
                results[job_id] = output
                if success:
                    completed_jobs.add(job_id)
                else:
                    phase_failed = True
                    campaign_failed = True
                    if status not in ("failed", "timed_out"):
                        ledger.append_event("worker_failed", {
                            "job_id": job_id,
                            "error": f"unsuccessful status {status!r}: "
                                     f"{output.get('error', output.get('reason', ''))}"})
                    else:
                        ledger.append_event("worker_failed", {
                            "job_id": job_id,
                            "error": str(output.get("error", status))})
            if phase_failed:
                ledger.append_event("phase_failed", {"phase": phase})
        remaining = (ledger.deadline() or time.time()) - time.time()
        final_status = "CAMPAIGN_FAILED" if campaign_failed else "CAMPAIGN_PHASE_COMPLETE"
        print(json.dumps({"status": final_status, "mode": mode,
                          "remaining_minutes": round(remaining / 60.0, 1),
                          "results": {key: value.get("status")
                                      for key, value in results.items()}}, indent=2))
        # Failed or blocked campaigns return nonzero (R01).
        return 1 if campaign_failed else 0
    finally:
        try:
            if ledger is not None:
                ledger.close()
        except Exception:
            pass
        lease.release()


def _phase_plan(mode: str, deadline: float,
                devices: Sequence[str]) -> list[dict[str, Any]]:
    """The phase plan from experiment.md §3, as worker assignments."""
    device_list = list(devices)
    plan = []
    e0_workers = []
    for index, device in enumerate(device_list[:2]):
        e0_workers.append({"job_id": f"E0-w{index}", "worker": f"w{index}",
                           "device": device, "phase": "E0",
                           "wall_seconds": 25 * 60.0})
    plan.append({"phase": "E0", "wall_cap_minutes": 30, "workers": e0_workers})
    if mode == "full":
        plan.extend([
            {"phase": "E1", "wall_cap_minutes": 120, "workers": [
                {"job_id": "E1-A-1701", "worker": "w0", "device": device_list[0],
                 "phase": "E1", "arm": "A", "seed": 1701, "wall_seconds": 60 * 60.0},
                {"job_id": "E1-B-1701", "worker": "w1", "device": device_list[1],
                 "phase": "E1", "arm": "B", "seed": 1701, "wall_seconds": 60 * 60.0},
                {"job_id": "E1-B-1702", "worker": "w0", "device": device_list[0],
                 "phase": "E1", "arm": "B", "seed": 1702, "wall_seconds": 60 * 60.0},
                {"job_id": "E1-A-1702", "worker": "w1", "device": device_list[1],
                 "phase": "E1", "arm": "A", "seed": 1702, "wall_seconds": 60 * 60.0},
            ]},
            {"phase": "E2", "wall_cap_minutes": 45, "workers": [
                {"job_id": f"E2-w{i}", "worker": f"w{i}", "device": device,
                 "phase": "E2", "wall_seconds": 45 * 60.0}
                for i, device in enumerate(device_list[:2])]},
            {"phase": "E3", "wall_cap_minutes": 60, "workers": [
                {"job_id": f"E3-T0-w{i}", "worker": f"w{i}", "device": device,
                 "phase": "E3", "arm": "T0", "wall_seconds": 30 * 60.0}
                for i, device in enumerate(device_list[:2])] + [
                {"job_id": f"E3-T1-w{i}", "worker": f"w{i}", "device": device,
                 "phase": "E3", "arm": "T1", "wall_seconds": 30 * 60.0}
                for i, device in enumerate(device_list[:2])]},
            {"phase": "E4", "wall_cap_minutes": 60, "workers": [
                {"job_id": f"E4-S0-w{i}", "worker": f"w{i}", "device": device,
                 "phase": "E4", "arm": "S0", "wall_seconds": 30 * 60.0}
                for i, device in enumerate(device_list[:2])] + [
                {"job_id": f"E4-S1-w{i}", "worker": f"w{i}", "device": device,
                 "phase": "E4", "arm": "S1", "wall_seconds": 30 * 60.0}
                for i, device in enumerate(device_list[:2])]},
            {"phase": "E5", "wall_cap_minutes": 135, "workers": [
                {"job_id": f"E5-w{i}", "worker": f"w{i}", "device": device,
                 "phase": "E5", "wall_seconds": 135 * 60.0}
                for i, device in enumerate(device_list[:2])]},
            {"phase": "E6", "wall_cap_minutes": 30, "workers": [
                {"job_id": "E6-export", "worker": "supervisor",
                 "device": device_list[0] if device_list else "cpu",
                 "phase": "E6", "wall_seconds": 30 * 60.0}]},
        ])
    return plan


def _source_closure_hash() -> str:
    from bramastra_lab.research.runtime.provenance import source_closure_sha256

    return source_closure_sha256()


def _hash_dir(directory: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    if not os.path.isdir(directory):
        return digest.hexdigest()
    for base, _dirs, names in sorted(os.walk(directory)):
        for name in sorted(names):
            path = os.path.join(base, name)
            try:
                digest.update(hashlib.sha256(open(path, "rb").read()).hexdigest().encode())
            except OSError:
                continue
    return digest.hexdigest()
