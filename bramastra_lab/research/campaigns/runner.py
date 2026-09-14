"""Campaign runner (D1/D4/D5): slot-ordered execution with exclusive devices.

Slots: E1 has two counterbalanced slots; E3/E4 have two slots with reversed
treatment order between seeds (GPU0 owns seed1701, GPU1 owns seed1702). Only
one slot's jobs are reserved and launched at a time (at most one active job
per physical GPU). Slot progress persists via individual job receipts; restart
never repeats accepted outcomes. Leases are ownership-token fenced. E2 frozen
evaluation binds cases + checkpoint (no positive-update gate); E6 is export.
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


def _preflight(data_dir: str, mode: str) -> str | None:
    """Validate bundle + handler availability before E0 (D5).

    Returns an error string when preflight fails, else None. Fails before
    expensive work: missing manifest, invalid bundle, or missing phase
    executors refuse with exit 2 (never silent campaign success).
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return f"prepared bundle manifest missing: {manifest_path}"
    try:
        from bramastra_lab.research.data.k8_bundle import validate_bundle

        report = validate_bundle(data_dir, min_confirmation=1)
        if not report.get("valid"):
            return f"bundle invalid: {report.get('issues', [])[:3]}"
    except Exception as exc:  # noqa: BLE001
        return f"bundle validation error: {exc}"
    try:
        from bramastra_lab.research.campaigns import phases  # noqa: F401
        from bramastra_lab.research.campaigns.phases import e1, e2, e3, e4, e5, e6  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"phase executors unavailable: {exc}"
    if mode not in ("e0", "full"):
        return f"unknown mode {mode!r}"
    return None


def _phase_success_for_output(phase: str, output: dict[str, Any]) -> bool:
    """Per-phase success predicate (D4).

    E0: completed + resume agreement + positive committed updates.
    E1/E3/E4: completed + positive committed updates.
    E2 (frozen eval): completed + evaluated_cases>0 + checkpoint bound;
      optimizer updates MUST be zero (no optimizer path in frozen eval).
    E5: completed + trials>0 + archive bound (attempted>0).
    E6: completed (export verification inside the executor).
    """
    status = output.get("status")
    if status != "completed":
        return False
    if phase == "E0":
        return bool(output.get("resume_agrees") is True) and int(
            output.get("committed_updates", 0)) > 0
    if phase in ("E1", "E3", "E4"):
        return int(output.get("committed_updates", 0)) > 0
    if phase == "E2":
        evaluated = int(output.get("evaluated_cases", 0))
        checkpoint = output.get("checkpoint_identity")
        optimizer_updates = int(output.get("optimizer_updates", 0))
        return evaluated > 0 and bool(checkpoint) and optimizer_updates == 0
    if phase == "E5":
        trials = int(output.get("trials", output.get("attempted_updates", 0)))
        return trials > 0 and bool(output.get("archive_identity") or
                                   output.get("checkpoint_identity"))
    if phase == "E6":
        return True
    return False


def run_campaign(*, run_dir: str, mode: str, data_dir: str,
                 max_wall_minutes: float = 480.0,
                 devices: Sequence[str] = ("cuda:0", "cuda:1"),
                 precision: str = "fp16_autocast") -> int:
    from bramastra_lab.research.contracts.core import content_identity

    if mode not in ("e0", "full"):
        raise SupervisorError(f"unknown mode {mode!r}")
    os.makedirs(run_dir, exist_ok=True)
    preflight_error = _preflight(data_dir, mode)
    if preflight_error:
        print(json.dumps({"status": "DATA_NOT_READY" if "bundle" in preflight_error
                          or "manifest" in preflight_error else "PREFLIGHT_REFUSED",
                          "message": preflight_error}))
        return 2
    source_hash = _source_closure_hash()
    data_hash = _hash_dir(data_dir)
    allocation_id = content_identity({"data": data_hash,
                                      "source": source_hash,
                                      "max_wall": max_wall_minutes})
    lease = SupervisorLease(run_dir)
    lease_token = lease.acquire()
    ledger: CampaignLedger | None = None
    try:
        ledger = CampaignLedger(run_dir)
        try:
            deadline = ledger.record_allocation(allocation_id, source_hash,
                                                data_hash, max_wall_minutes)
        except SupervisorError as exc:
            print(json.dumps({"status": "ALLOCATION_REJECTED",
                              "error": str(exc)}))
            return 2
        campaign_start = deadline - max_wall_minutes * 60.0
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
            if phase != "E6" and time.time() > training_cutoff:
                ledger.append_event("phase_skipped", {
                    "phase": phase, "reason": "training_cutoff_exceeded",
                    "cutoff_unix": training_cutoff})
                campaign_failed = True
                continue
            slots: list[list[dict[str, Any]]] = phase_entry.get(
                "slots") or [phase_entry["workers"]]
            for slot_index, slot_workers in enumerate(slots):
                pending = [entry for entry in slot_workers
                           if entry["job_id"] not in completed_jobs]
                if not pending:
                    ledger.append_event("slot_skipped_completed", {
                        "phase": phase, "slot": slot_index})
                    continue
                if campaign_failed and phase != "E6":
                    ledger.append_event("phase_skipped", {
                        "phase": phase, "slot": slot_index,
                        "reason": "prior_phase_failed"})
                    continue
                # Exclusive occupancy within the slot: at most one job per
                # physical device (D1). Slots are constructed to satisfy this;
                # refuse rather than oversubscribe.
                physicals = [entry["device"] for entry in pending]
                if len(set(physicals)) != len(physicals):
                    ledger.append_event("phase_failed", {
                        "phase": phase, "slot": slot_index,
                        "reason": "slot violates exclusive occupancy"})
                    for entry in pending:
                        results[entry["job_id"]] = {
                            "status": "failed",
                            "error": "slot violates exclusive occupancy"}
                    campaign_failed = True
                    continue
                reservations: dict[str, Any] = {}
                reserve_failed = False
                for worker_entry in pending:
                    try:
                        reservations[worker_entry["job_id"]] = ledger.reserve(
                            job_id=worker_entry["job_id"],
                            worker=worker_entry["worker"],
                            device=worker_entry["device"], phase=phase,
                            arm=worker_entry.get("arm"),
                            seed=worker_entry.get("seed"),
                            reserved_seconds=worker_entry["wall_seconds"])
                    except SupervisorError as exc:
                        ledger.append_event("worker_failed", {
                            "job_id": worker_entry["job_id"],
                            "phase": phase, "slot": slot_index,
                            "error": f"SupervisorError: {exc}"})
                        results[worker_entry["job_id"]] = {
                            "status": "failed", "error": str(exc)}
                        reserve_failed = True
                        campaign_failed = True
                if reserve_failed:
                    ledger.append_event("phase_failed", {
                        "phase": phase, "slot": slot_index,
                        "reason": "reservation_refused"})
                    continue
                now = time.time()
                phase_deadline = phase_deadlines.get(phase, deadline)
                slot_cap = float(phase_entry.get("wall_cap_minutes", 30.0)) * 60.0
                # Slot timeout: proportional share of the phase cap (one slot
                # at a time), bounded by absolute deadlines.
                slot_cap_share = slot_cap / max(1, len(slots))
                timeout_seconds = min(slot_cap_share,
                                      max(1.0, phase_deadline - now),
                                      max(1.0, deadline - now))
                if phase != "E6":
                    timeout_seconds = min(timeout_seconds,
                                          max(1.0, training_cutoff - now))
                if timeout_seconds <= 0:
                    ledger.append_event("phase_failed", {
                        "phase": phase, "slot": slot_index,
                        "reason": "deadline_already_exceeded"})
                    campaign_failed = True
                    continue
                specs = []
                for worker_entry in pending:
                    specs.append({
                        "job_id": worker_entry["job_id"],
                        "phase": phase,
                        "physical_device": worker_entry["device"],
                        "device": worker_entry["device"],
                        "arm": worker_entry.get("arm"),
                        "seed": worker_entry.get("seed"),
                        "slot": slot_index,
                        "parent": worker_entry.get("parent"),
                        "data_dir": data_dir, "run_dir": run_dir,
                        "precision": precision,
                        "deadline": min(deadline, phase_deadline),
                    })
                outputs = ps.run_phase_concurrently(
                    specs, timeout_seconds=timeout_seconds,
                    worker_fn_path="bramastra_lab.research.campaigns.worker:run_worker_phase",
                )
                slot_failed = False
                for worker_entry in pending:
                    job_id = worker_entry["job_id"]
                    reservation = reservations[job_id]
                    reserve_started = reservation.reserved_at_unix
                    output = outputs.get(job_id, {"status": "failed",
                                                  "error": "missing worker output"})
                    success = _phase_success_for_output(phase, output)
                    if output.get("status") in ("failed", "timed_out") \
                            and not output.get("device_seconds"):
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
                            "job_id": job_id, "phase": phase,
                            "slot": slot_index,
                            "error": f"close_reservation refused: {exc}"})
                        success = False
                    results[job_id] = output
                    if success:
                        completed_jobs.add(job_id)
                    else:
                        slot_failed = True
                        campaign_failed = True
                        ledger.append_event("worker_failed", {
                            "job_id": job_id, "phase": phase,
                            "slot": slot_index,
                            "error": str(output.get("error", output.get(
                                "reason", output.get("status"))))})
                ledger.append_event(
                    "slot_completed" if not slot_failed else "slot_failed",
                    {"phase": phase, "slot": slot_index,
                     "jobs": [entry["job_id"] for entry in pending]})
                if slot_failed:
                    ledger.append_event("phase_failed", {
                        "phase": phase, "slot": slot_index})
                    break
        remaining = (ledger.deadline() or time.time()) - time.time()
        final_status = "CAMPAIGN_FAILED" if campaign_failed else "CAMPAIGN_PHASE_COMPLETE"
        print(json.dumps({"status": final_status, "mode": mode,
                          "remaining_minutes": round(remaining / 60.0, 1),
                          "results": {key: value.get("status")
                                      for key, value in results.items()}}, indent=2))
        return 1 if campaign_failed else 0
    finally:
        try:
            if ledger is not None:
                ledger.close()
        except Exception:
            pass
        try:
            lease.release(lease_token)
        except Exception:
            try:
                lease.release()
            except Exception:
                pass


def _phase_plan(mode: str, deadline: float,
                devices: Sequence[str]) -> list[dict[str, Any]]:
    """Phase plan with explicit ordered slots (D1).

    E1: slot1 A1701/gpu0 + B1701/gpu1; slot2 B1702/gpu0 + A1702/gpu1.
    E3: slot1 T0/seed1701/gpu0 + T1/seed1702/gpu1; slot2 T1/seed1701/gpu0 +
      T0/seed1702/gpu1 (reversed treatment order between seeds).
    E4: slot1 S0/seed1701/gpu0 + S1/seed1702/gpu1; slot2 S1/seed1701/gpu0 +
      S0/seed1702/gpu1. GPU0 owns seed1701, GPU1 owns seed1702.
    """
    device_list = list(devices)
    gpu0 = device_list[0] if len(device_list) > 0 else "cuda:0"
    gpu1 = device_list[1] if len(device_list) > 1 else "cuda:1"
    plan = []
    e0_workers = [
        {"job_id": "E0-w0", "worker": "w0", "device": gpu0, "phase": "E0",
         "wall_seconds": 25 * 60.0},
        {"job_id": "E0-w1", "worker": "w1", "device": gpu1, "phase": "E0",
         "wall_seconds": 25 * 60.0},
    ]
    plan.append({"phase": "E0", "wall_cap_minutes": 30, "workers": e0_workers,
                 "slots": [e0_workers]})
    if mode == "full":
        e1_slot1 = [
            {"job_id": "E1-A-1701", "worker": "w0", "device": gpu0,
             "phase": "E1", "arm": "A", "seed": 1701, "wall_seconds": 60 * 60.0},
            {"job_id": "E1-B-1701", "worker": "w1", "device": gpu1,
             "phase": "E1", "arm": "B", "seed": 1701, "wall_seconds": 60 * 60.0},
        ]
        e1_slot2 = [
            {"job_id": "E1-B-1702", "worker": "w0", "device": gpu0,
             "phase": "E1", "arm": "B", "seed": 1702, "wall_seconds": 60 * 60.0},
            {"job_id": "E1-A-1702", "worker": "w1", "device": gpu1,
             "phase": "E1", "arm": "A", "seed": 1702, "wall_seconds": 60 * 60.0},
        ]
        e1_workers = e1_slot1 + e1_slot2
        plan.append({"phase": "E1", "wall_cap_minutes": 120,
                     "workers": e1_workers, "slots": [e1_slot1, e1_slot2]})
        e2_workers = [
            {"job_id": "E2-w0", "worker": "w0", "device": gpu0,
             "phase": "E2", "seed": 1701, "wall_seconds": 45 * 60.0,
             "parent": "E1-A-1701/E1-B-1701"},
            {"job_id": "E2-w1", "worker": "w1", "device": gpu1,
             "phase": "E2", "seed": 1702, "wall_seconds": 45 * 60.0,
             "parent": "E1-B-1702/E1-A-1702"},
        ]
        plan.append({"phase": "E2", "wall_cap_minutes": 45,
                     "workers": e2_workers, "slots": [e2_workers]})
        e3_slot1 = [
            {"job_id": "E3-T0-1701", "worker": "w0", "device": gpu0,
             "phase": "E3", "arm": "T0", "seed": 1701,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1701"},
            {"job_id": "E3-T1-1702", "worker": "w1", "device": gpu1,
             "phase": "E3", "arm": "T1", "seed": 1702,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1702"},
        ]
        e3_slot2 = [
            {"job_id": "E3-T1-1701", "worker": "w0", "device": gpu0,
             "phase": "E3", "arm": "T1", "seed": 1701,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1701"},
            {"job_id": "E3-T0-1702", "worker": "w1", "device": gpu1,
             "phase": "E3", "arm": "T0", "seed": 1702,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1702"},
        ]
        e3_workers = e3_slot1 + e3_slot2
        plan.append({"phase": "E3", "wall_cap_minutes": 60,
                     "workers": e3_workers, "slots": [e3_slot1, e3_slot2]})
        e4_slot1 = [
            {"job_id": "E4-S0-1701", "worker": "w0", "device": gpu0,
             "phase": "E4", "arm": "S0", "seed": 1701,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1701"},
            {"job_id": "E4-S1-1702", "worker": "w1", "device": gpu1,
             "phase": "E4", "arm": "S1", "seed": 1702,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1702"},
        ]
        e4_slot2 = [
            {"job_id": "E4-S1-1701", "worker": "w0", "device": gpu0,
             "phase": "E4", "arm": "S1", "seed": 1701,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1701"},
            {"job_id": "E4-S0-1702", "worker": "w1", "device": gpu1,
             "phase": "E4", "arm": "S0", "seed": 1702,
             "wall_seconds": 30 * 60.0, "parent": "E1-B-1702"},
        ]
        e4_workers = e4_slot1 + e4_slot2
        plan.append({"phase": "E4", "wall_cap_minutes": 60,
                     "workers": e4_workers, "slots": [e4_slot1, e4_slot2]})
        e5_workers = [
            {"job_id": "E5-w0", "worker": "w0", "device": gpu0,
             "phase": "E5", "seed": 1701, "wall_seconds": 135 * 60.0},
            {"job_id": "E5-w1", "worker": "w1", "device": gpu1,
             "phase": "E5", "seed": 1702, "wall_seconds": 135 * 60.0},
        ]
        plan.append({"phase": "E5", "wall_cap_minutes": 135,
                     "workers": e5_workers, "slots": [e5_workers]})
        e6_workers = [
            {"job_id": "E6-export", "worker": "supervisor",
             "device": gpu0, "phase": "E6", "wall_seconds": 30 * 60.0},
        ]
        plan.append({"phase": "E6", "wall_cap_minutes": 30,
                     "workers": e6_workers, "slots": [e6_workers]})
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
