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


def _preflight(data_dir: str, mode: str, *,
               build_report: str | None = None) -> str | None:
    """Validate bundle + handler availability before E0 (D5).

    Returns an error string when preflight fails, else None. Fails before
    expensive work: missing manifest, invalid bundle, or missing phase
    executors refuse with exit 2 (never silent campaign success).
    """
    if mode not in ("e0", "full"):
        return f"unknown mode {mode!r}"
    # Importable handlers are not proof that they implement the experiment.
    # Refuse known unqualified implementations before allocating GPU time.
    from bramastra_lab.research.campaigns.readiness import implementation_readiness

    # The gate consumes the build report the operator just generated with
    # verify-build (owner notebook passes --build-report); the repository
    # default is only a fallback for non-notebook callers.
    readiness = implementation_readiness(report_path=build_report)
    if not readiness["ready"]:
        return "IMPLEMENTATION_NOT_READY: " + json.dumps(readiness, sort_keys=True)
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
    """Per-phase success predicate (D4 + contracts S1).

    E0: completed + resume agreement + positive committed updates.
    E1/E3/E4: completed + positive committed updates + non-fixture evidence.
    E2 (frozen eval): completed + evaluated_cases>0 + checkpoint bound;
      optimizer updates MUST be zero (no optimizer path in frozen eval) +
      non-fixture evidence.
    E5: completed + trials>0 + archive bound + non-fixture evidence.
    E6: completed + non-fixture evidence (export verification inside executor).
    Fixture receipts never enter accepted campaign aggregates (contracts S1).
    """
    status = output.get("status")
    if status != "completed":
        return False
    # Fixture receipts (doubles, metadata-only) never qualify for the campaign.
    if output.get("evidence_kind") == "fixture":
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
                 max_wall_minutes: float = 600.0,
                 devices: Sequence[str] = ("cuda:0", "cuda:1"),
                 precision: str = "fp32",
                 build_report: str | None = None) -> int:
    from bramastra_lab.research.contracts.core import content_identity

    if mode not in ("e0", "full"):
        raise SupervisorError(f"unknown mode {mode!r}")
    os.makedirs(run_dir, exist_ok=True)
    preflight_error = _preflight(data_dir, mode, build_report=build_report)
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
                              "error": str(exc),
                              "hint": "keep --data bundle bytes, source revision "
                                      "and --max-wall-minutes identical between "
                                      "e0 and full cells; regenerate nothing "
                                      "mid-campaign (new BRAMASTRA_RUN_ID = new campaign)"}))
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
        campaign_plan = _phase_plan(mode, deadline, devices,
                                    max_wall_minutes=max_wall_minutes)
        ledger.append_event("campaign_started", {"mode": mode,
                                                 "plan": campaign_plan,
                                                 "allocation_id": allocation_id})
        from bramastra_lab.research.campaigns import process_supervision as ps

        phase_deadlines = ps.phase_absolute_deadlines(campaign_start, campaign_plan)
        training_cutoff = ps.campaign_training_cutoff(campaign_start, max_wall_minutes)
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
            if phase != "E6" and (deadline - time.time()) < 15 * 60.0:
                # Last-window guard: under 15 minutes of wall remain. Never
                # start new training that the session cap would kill mid-write;
                # snapshot completed work and stop gracefully instead.
                _safety_sweep(ledger, run_dir, reason=f"last-window-before-{phase}")
                ledger.append_event("phase_skipped", {
                    "phase": phase, "reason": "last_window_exceeded",
                    "remaining_seconds": round(deadline - time.time(), 1)})
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
                # Fresh-connection verification: re-read every just-reserved
                # row through a NEW connection (the way spawned workers
                # will read it). Catches path/visibility divergence here,
                # loudly, instead of 100 silent worker noops later.
                # Transient cloud-disk hiccups get a few spaced retries;
                # a persistent refusal closes the slot reservations (so no
                # leaked open row poisons later slots) and fails the slot.
                visibility_error: str | None = None
                for _attempt in range(3):
                    try:
                        _verify_reservations_readable(
                            ledger, run_dir,
                            [entry["job_id"] for entry in pending])
                        visibility_error = None
                        break
                    except SupervisorError as exc:
                        visibility_error = str(exc)
                        time.sleep(5.0)
                if visibility_error is not None:
                    ledger.append_event("phase_failed", {
                        "phase": phase, "slot": slot_index,
                        "reason": f"reservation_visibility_refused: {visibility_error}"})
                    for entry in pending:
                        job_id = entry["job_id"]
                        results[job_id] = {
                            "status": "failed", "error": visibility_error}
                        try:
                            ledger.close_reservation(
                                reservations[job_id].reservation_id,
                                status="failed")
                        except Exception:
                            pass
                    campaign_failed = True
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
                frozen = _load_usable_protocol(run_dir, list(devices))
                for worker_entry in pending:
                    # Targets come from the frozen E0-calibrated protocol when
                    # present; otherwise campaign minima labeled uncalibrated
                    # (never masquerading as calibrated).
                    spec_update_target = None
                    spec_eval_cases: int | None = None
                    spec_tasks: int | None = None
                    calibration_source = "uncalibrated-minima"
                    if phase == "E1":
                        spec_update_target = 200
                    elif phase in ("E3", "E4"):
                        spec_update_target = 80
                    elif phase == "E2":
                        spec_eval_cases = 32
                    elif phase == "E5":
                        spec_tasks = 12
                    # E5 learning boundary is explicit: production on the
                    # campaign path (real steps under ledger-gated allocation),
                    # test_substitute only in direct local calls (default).
                    spec_learning_boundary = "production" \
                        if phase == "E5" else None
                    if frozen is not None:
                        selection = frozen.get("selection", {})
                        targets = selection.get("update_targets", {})
                        if phase in targets:
                            spec_update_target = int(targets[phase])
                        if phase == "E2" and selection.get(
                                "confirmation_clusters_per_family"):
                            spec_eval_cases = int(selection[
                                "confirmation_clusters_per_family"])
                        calibration_source = "e0-calibrated"
                    reservation = reservations[worker_entry["job_id"]]
                    specs.append({
                        "job_id": worker_entry["job_id"],
                        "phase": phase,
                        "physical_device": worker_entry["device"],
                        "device": worker_entry["device"],
                        "arm": worker_entry.get("arm"),
                        "seed": worker_entry.get("seed"),
                        "slot": slot_index,
                        "parent": worker_entry.get("parent"),
                        "update_target": spec_update_target,
                        "eval_cases": spec_eval_cases,
                        "tasks_per_block": spec_tasks,
                        "learning_boundary": spec_learning_boundary,
                        "reservation_id": reservation.reservation_id,
                        "allocation_id": allocation_id,
                        "deadline_unix": reservation.reserved_at_unix
                        + reservation.reserved_seconds,
                        "remaining_updates": spec_update_target,
                        "data_dir": data_dir, "run_dir": run_dir,
                        "precision": precision,
                        "deadline": min(deadline, phase_deadline),
                    })
                ledger.append_event("phase_targets", {
                    "phase": phase, "slot": slot_index,
                    "calibration_source": calibration_source,
                    "update_target": spec_update_target,
                    "eval_cases": spec_eval_cases,
                    "tasks_per_block": spec_tasks})
                outputs = ps.run_phase_concurrently(
                    specs, timeout_seconds=timeout_seconds,
                    worker_fn_path="bramastra_lab.research.campaigns.worker:run_worker_phase",
                    require_overlap_proof=len(specs) == 2 and phase != "E6",
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
                if phase == "E0" and not slot_failed:
                    # Freeze the calibrated protocol from measured E0 pilot
                    # samples before any learning phase consumes targets.
                    try:
                        _maybe_freeze_protocol(
                            ledger, run_dir, results, pending,
                            source_hash=source_hash, data_hash=data_hash,
                            devices=list(devices))
                    except Exception as exc:
                        ledger.append_event("calibration_failed", {
                            "phase": phase, "error": str(exc)[:300]})
                if slot_failed:
                    ledger.append_event("phase_failed", {
                        "phase": phase, "slot": slot_index})
                    break
            _safety_sweep(ledger, run_dir, reason=f"phase-{phase}-done")
        remaining = (ledger.deadline() or time.time()) - time.time()
        final_status = "CAMPAIGN_FAILED" if campaign_failed else "CAMPAIGN_PHASE_COMPLETE"
        final_report: dict[str, Any] = {
            "status": final_status, "mode": mode,
            "remaining_minutes": round(remaining / 60.0, 1),
            "results": {key: value.get("status")
                        for key, value in results.items()},
            "failures": _failure_summary(results)}
        if campaign_failed:
            # Recovery contract: completed jobs are reusable (E0 skip works),
            # but failed jobs never silently rebind. Recovery is a fresh
            # RUN_ID reusing the same bundle; completed artifacts survive via
            # the safety snapshot written below.
            final_report["recovery"] = (
                "completed jobs are kept; failed jobs cannot retry in this "
                "run directory (closed reservations never rebind). Recover "
                "with a fresh BRAMASTRA_RUN_ID reusing the same bundle, then "
                "rerun verify (if needed), e0 and full in order.")
        print(json.dumps(final_report, indent=2))
        return 1 if campaign_failed else 0
    finally:
        try:
            _safety_sweep(ledger, run_dir, reason="campaign-exit")
        except Exception:
            pass
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


def _load_usable_protocol(run_dir: str,
                          devices: list[str]) -> dict[str, Any] | None:
    """Frozen protocol when present and hardware-matched, else None."""
    from bramastra_lab.research.campaigns import calibration as cal

    try:
        protocol = cal.load_frozen_protocol(run_dir)
    except Exception:
        return None
    if protocol is None:
        return None
    try:
        cal.check_hardware_match(protocol, device_ids=list(devices))
    except Exception:
        return None
    return protocol


def _verify_reservations_readable(ledger: Any, run_dir: str,
                                  job_ids: list[str]) -> None:
    """Fresh-connection re-read of just-reserved rows (spawn visibility).

    Opens a NEW sqlite connection to the ledger file — exactly like a
    spawned worker would — and requires every job_id to resolve to an
    open row. Raises SupervisorError naming the file, so a path or
    visibility divergence fails the slot here instead of stranding
    workers in silent no-op boundaries.
    """
    import os as _os
    import sqlite3 as _sqlite

    ledger_path = _os.path.join(run_dir, "campaign_ledger.sqlite")
    try:
        conn = _sqlite.connect(ledger_path, timeout=60.0)
    except Exception as exc:
        raise SupervisorError(
            f"reservation re-read refused: cannot open {ledger_path}: {exc}")
    try:
        try:
            missing = []
            for job_id in job_ids:
                row = conn.execute(
                    "SELECT status FROM reservations WHERE job_id=? "
                    "ORDER BY rowid DESC LIMIT 1", (str(job_id),)).fetchone()
                if row is None:
                    missing.append(str(job_id))
                elif row[0] != "open":
                    raise SupervisorError(
                        f"reservation for job {job_id!r} is {row[0]!r}, "
                        f"not 'open' (ledger {ledger_path})")
            if missing:
                raise SupervisorError(
                    f"just-reserved jobs have no ledger rows: {missing} "
                    f"(ledger {ledger_path}); refusing to spawn workers "
                    "that could never step")
        finally:
            conn.close()
    except SupervisorError:
        raise
    except Exception as exc:
        raise SupervisorError(
            f"reservation re-read refused ({ledger_path}): {exc}")


def _safety_sweep(ledger: Any, run_dir: str, reason: str) -> None:
    """Best-effort safety snapshot; never fails the campaign.

    Writes a results-only snapshot of completed work into
    <run_dir>/safety/ and records a ledger event. All errors are
    swallowed: a snapshot is a survival net, never a gate.
    """
    try:
        from bramastra_lab.research.campaigns.results_pack import snapshot_run_dir
        receipt = snapshot_run_dir(run_dir, reason)
        try:
            ledger.append_event("safety_snapshot", {
                "reason": reason, "archive": receipt["archive"],
                "files": receipt["files"], "bytes": receipt["bytes"]})
        except Exception:
            pass
    except Exception:
        pass


def _failure_summary(results: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Return actionable worker failures in the parent CLI result.

    Spawned workers correctly keep their complete process state private, but
    hiding even their returned exception forces an operator to inspect SQLite
    after an E0 refusal. Preserve the full ledger evidence while exposing a
    bounded, plain-text cause in the command that failed.
    """
    failures: dict[str, str] = {}
    for job_id, output in sorted(results.items()):
        if str(output.get("status")) == "completed":
            continue
        reason = output.get("error", output.get("reason", output.get("status")))
        failures[str(job_id)] = str(reason)[:2000]
    return failures


def _maybe_freeze_protocol(ledger: Any, run_dir: str,
                           results: dict[str, Any],
                           pending: list[dict[str, Any]], *,
                           source_hash: str, data_hash: str,
                           devices: list[str]) -> None:
    """Freeze E0-measured calibration exactly once (O03)."""
    from bramastra_lab.research.campaigns import calibration as cal

    if cal.load_frozen_protocol(run_dir) is not None:
        return
    worst_update: float | None = None
    worst_eval_rate: float | None = None
    for entry in pending:
        samples = (results.get(entry["job_id"], {}) or {}).get(
            "calibration_samples") or {}
        update_seconds = samples.get("worst_update_seconds")
        if isinstance(update_seconds, (int, float)):
            worst_update = update_seconds if worst_update is None \
                else max(worst_update, float(update_seconds))
        rate = samples.get("eval_cases_per_second")
        if isinstance(rate, (int, float)):
            worst_eval_rate = rate if worst_eval_rate is None \
                else min(worst_eval_rate, float(rate))
    if worst_update is None or worst_eval_rate is None:
        raise ValueError(
            "E0 outputs carry no full-profile pilot samples; refusing to "
            "freeze unmeasured targets")
    selection = {"update_targets": {
        phase: cal.select_update_target(
            phase=phase, worst_update_seconds=worst_update)
        for phase in ("E1", "E3", "E4")},
        "confirmation_clusters_per_family":
        cal.select_confirmation_inventory(
            eval_cases_per_second=worst_eval_rate)}
    protocol = cal.freeze_protocol(
        run_dir, selection=selection,
        identities={"source_hash": source_hash, "data_hash": data_hash,
                    "device_ids": sorted(devices)})
    ledger.append_event("protocol_frozen", {
        "selection": selection,
        "protocol": {key: protocol[key] for key in ("schema",)}})


# Registered training-phase cap proportions over the wall-minus-reserve
# pool (from the 480-minute plan: 30/120/45/60/60/135 training + 30 export).
# The caps scale proportionally to the declared wall budget and must sum
# EXACTLY to wall minus export reserve.
TRAINING_CAP_PARTS = {"E0": 30.0, "E1": 120.0, "E2": 45.0,
                      "E3": 60.0, "E4": 60.0, "E5": 135.0}


def _phase_caps(max_wall_minutes: float,
                export_reserve_minutes: float = 30.0) -> dict[str, float]:
    """Scale the registered phase proportions to the declared wall budget.

    The training pool is wall minus the export reserve; each phase takes
    its proportional share with the largest-remainder rounding so the caps
    sum exactly to the pool (never a silent minute short or over).
    """
    pool = float(max_wall_minutes) - float(export_reserve_minutes)
    if pool <= 0:
        raise ValueError(f"wall budget {max_wall_minutes} leaves no training "
                         "pool after the export reserve")
    total_parts = sum(TRAINING_CAP_PARTS.values())
    raw = {phase: pool * part / total_parts
           for phase, part in TRAINING_CAP_PARTS.items()}
    floors = {phase: int(value) for phase, value in raw.items()}
    remainder = int(round(pool)) - sum(floors.values())
    # Largest-remainder distribution of the leftover whole minutes.
    order = sorted(raw, key=lambda p: (raw[p] - floors[p], p), reverse=True)
    for phase in order[:max(0, remainder)]:
        floors[phase] += 1
    return floors


def _phase_plan(mode: str, deadline: float,
                devices: Sequence[str], *,
                max_wall_minutes: float = 600.0) -> list[dict[str, Any]]:
    """Phase plan with explicit ordered slots (D1).

    Phase caps scale proportionally to the declared wall budget (exact sum
    = wall minus the 30-minute export reserve).

    E1: slot1 A1701/gpu0 + B1701/gpu1; slot2 B1702/gpu0 + A1702/gpu1.
    E3: slot1 T0/seed1701/gpu0 + T1/seed1702/gpu1; slot2 T1/seed1701/gpu0 +
      T0/seed1702/gpu1 (reversed treatment order between seeds).
    E4: slot1 S0/seed1701/gpu0 + S1/seed1702/gpu1; slot2 S1/seed1701/gpu0 +
      S0/seed1702/gpu1. GPU0 owns seed1701, GPU1 owns seed1702.
    """
    from bramastra_lab.research.campaigns.process_supervision import (
        EXPORT_RESERVE_MINUTES)

    from bramastra_lab.research.campaigns.kaggle_env import (
        K8EnvironmentError,
        normalize_devices,
    )
    try:
        gpu0, gpu1 = normalize_devices(list(devices))
    except K8EnvironmentError as exc:
        raise ValueError(str(exc)) from exc
    device_list = [gpu0, gpu1]
    caps = _phase_caps(max_wall_minutes, EXPORT_RESERVE_MINUTES)
    plan_sum = sum(caps.values()) + EXPORT_RESERVE_MINUTES
    if abs(plan_sum - float(max_wall_minutes)) > 1e-6:
        raise ValueError(
            f"phase caps sum to {plan_sum} but the declared wall is "
            f"{max_wall_minutes}; refusing an unbalanced schedule")

    def per_slot(phase: str) -> float:
        # Each phase runs its slots sequentially on both devices; a
        # worker's wall share is its slot's cap (cap / slot count).
        slots_for_phase = 2 if phase in ("E1", "E3", "E4") else 1
        return caps[phase] / slots_for_phase * 60.0

    gpu0 = device_list[0]
    gpu1 = device_list[1]
    plan = []
    e0_wall = max(60.0, (caps["E0"] - 5) * 60.0)
    e0_workers = [
        {"job_id": "E0-w0", "worker": "w0", "device": gpu0, "phase": "E0",
         "wall_seconds": e0_wall},
        {"job_id": "E0-w1", "worker": "w1", "device": gpu1, "phase": "E0",
         "wall_seconds": e0_wall},
    ]
    plan.append({"phase": "E0", "wall_cap_minutes": caps["E0"],
                 "workers": e0_workers, "slots": [e0_workers]})
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
        e1_wall = per_slot("E1")
        for worker in e1_workers:
            worker["wall_seconds"] = e1_wall
        plan.append({"phase": "E1", "wall_cap_minutes": caps["E1"],
                     "workers": e1_workers, "slots": [e1_slot1, e1_slot2]})
        e2_workers = [
            {"job_id": "E2-w0", "worker": "w0", "device": gpu0,
             "phase": "E2", "seed": 1701, "wall_seconds": 45 * 60.0,
             "parent": "E1-A-1701/E1-B-1701"},
            {"job_id": "E2-w1", "worker": "w1", "device": gpu1,
             "phase": "E2", "seed": 1702, "wall_seconds": 45 * 60.0,
             "parent": "E1-B-1702/E1-A-1702"},
        ]
        e2_wall = per_slot("E2")
        for worker in e2_workers:
            worker["wall_seconds"] = e2_wall
        plan.append({"phase": "E2", "wall_cap_minutes": caps["E2"],
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
        e3_wall = per_slot("E3")
        for worker in e3_workers:
            worker["wall_seconds"] = e3_wall
        plan.append({"phase": "E3", "wall_cap_minutes": caps["E3"],
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
        e4_wall = per_slot("E4")
        for worker in e4_workers:
            worker["wall_seconds"] = e4_wall
        plan.append({"phase": "E4", "wall_cap_minutes": caps["E4"],
                     "workers": e4_workers, "slots": [e4_slot1, e4_slot2]})
        e5_wall = per_slot("E5")
        e5_workers = [
            {"job_id": "E5-w0", "worker": "w0", "device": gpu0,
             "phase": "E5", "seed": 1701, "wall_seconds": e5_wall,
             "parent": "E1-B-1701"},
            {"job_id": "E5-w1", "worker": "w1", "device": gpu1,
             "phase": "E5", "seed": 1702, "wall_seconds": e5_wall,
             "parent": "E1-B-1702"},
        ]
        plan.append({"phase": "E5", "wall_cap_minutes": caps["E5"],
                     "workers": e5_workers, "slots": [e5_workers]})
        e6_workers = [
            {"job_id": "E6-export", "worker": "supervisor",
             "device": gpu0, "phase": "E6",
             "wall_seconds": EXPORT_RESERVE_MINUTES * 60.0},
        ]
        plan.append({"phase": "E6", "wall_cap_minutes": EXPORT_RESERVE_MINUTES,
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
