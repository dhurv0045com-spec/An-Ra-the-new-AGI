"""Campaign runner (I05): one supervisor, two workers, one persistent
allowance. ``run --mode e0`` runs only E0; ``run --mode full`` executes the
gated campaign. The runner refuses to start E1+ without a successful E0
receipt for both workers.
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
    source_hash = _source_closure_hash()
    data_hash = _hash_dir(data_dir)
    allocation_id = content_identity({"data": data_hash,
                                      "source": source_hash,
                                      "max_wall": max_wall_minutes})
    lease = SupervisorLease(run_dir)
    token = lease.acquire()
    try:
        ledger = CampaignLedger(run_dir)
        deadline = ledger.record_allocation(allocation_id, source_hash,
                                            data_hash, max_wall_minutes)
        existing_phases = {row[0] for row in ledger.conn.execute(
            "SELECT DISTINCT phase FROM reservations WHERE status='completed'"
        ).fetchall()}
        if "E0" in existing_phases and mode == "e0":
            print(json.dumps({"status": "E0_ALREADY_RUN",
                              "message": "E0 receipts exist; full mode uses the "
                                         "same allocation without duplicating E0"}))
            return 0
        # E0 admission gate: full mode requires successful E0 on both devices.
        if mode == "full" and not ledger.phase_success("E0", required_workers=2):
            print(json.dumps({"status": "E0_GATE_BLOCKED",
                              "message": "full campaign requires successful E0 "
                                         "receipts on both workers; run --mode e0 first"}))
            return 1
        campaign_plan = _phase_plan(mode, deadline, devices)
        ledger.append_event("campaign_started", {"mode": mode,
                                                 "plan": campaign_plan})
        from bramastra_lab.research.campaigns.worker import run_worker_phase

        results: dict[str, Any] = {}
        campaign_failed = False
        for phase_entry in campaign_plan:
            phase = phase_entry["phase"]
            if campaign_failed and phase != "E6":
                ledger.append_event("phase_skipped", {"phase": phase,
                                                       "reason": "prior_phase_failed"})
                continue
            if phase in existing_phases:
                continue
            phase_failed = False
            for worker_entry in phase_entry["workers"]:
                device = worker_entry["device"]
                reservation = ledger.reserve(
                    job_id=worker_entry["job_id"], worker=worker_entry["worker"],
                    device=device, phase=phase, arm=worker_entry.get("arm"),
                    seed=worker_entry.get("seed"),
                    reserved_seconds=worker_entry["wall_seconds"])
                try:
                    output = run_worker_phase(
                        phase=phase, device=device, arm=worker_entry.get("arm"),
                        seed=worker_entry.get("seed"), data_dir=data_dir,
                        run_dir=run_dir, precision=precision,
                        deadline=deadline)
                    success = output.get("status") == "completed" and                         output.get("resume_agrees", True)
                    ledger.close_reservation(
                        reservation.reservation_id,
                        status="completed" if success else "failed",
                        committed_updates=output.get("committed_updates", 0),
                        attempted_updates=output.get("attempted_updates", 0),
                        supervised_exposure=output.get("supervised_exposure", 0),
                        device_seconds=output.get("device_seconds", 0.0),
                        checkpoint_identity=output.get("checkpoint_identity"))
                    results[worker_entry["job_id"]] = output
                    if not success:
                        phase_failed = True
                        campaign_failed = True
                except Exception as exc:
                    ledger.close_reservation(
                        reservation.reservation_id, status="failed",
                        committed_updates=0, attempted_updates=0,
                        supervised_exposure=0, device_seconds=0.0)
                    ledger.append_event("worker_failed", {
                        "job_id": worker_entry["job_id"],
                        "error": f"{type(exc).__name__}: {exc}"})
                    results[worker_entry["job_id"]] = {"status": "failed",
                                                       "error": str(exc)}
                    phase_failed = True
                    campaign_failed = True
            if phase_failed:
                ledger.append_event("phase_failed", {"phase": phase})
        remaining = ledger.deadline() - time.time()
        print(json.dumps({"status": "CAMPAIGN_PHASE_COMPLETE", "mode": mode,
                          "remaining_minutes": round(remaining / 60.0, 1),
                          "results": {key: value.get("status")
                                      for key, value in results.items()}}, indent=2))
        return 0
    finally:
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
    for base, _dirs, names in sorted(os.walk(directory)):
        for name in sorted(names):
            path = os.path.join(base, name)
            digest.update(hashlib.sha256(open(path, "rb").read()).hexdigest().encode())
    return digest.hexdigest()

