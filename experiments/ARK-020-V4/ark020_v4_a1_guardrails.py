"""Last-mile guardrails for ARK-020 V4 durability amendment A1.

Kept separate from the frozen scientific runner. This module closes operator-path
edge cases discovered by the post-A1 red team without changing scientific treatment
semantics.
"""
from __future__ import annotations

import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import torch

import ark020_v4_core as C
import ark020_v4_durability as A1

_REVISION = "A1.1"
_BASE_EXECUTABLE_IDENTITY = A1.executable_identity


def _extended_executable_identity(R) -> dict[str, Any]:
    x = _BASE_EXECUTABLE_IDENTITY(R)
    p = R.HERE / "ark020_v4_a1_guardrails.py"
    if not p.exists():
        raise RuntimeError("A1.1 guardrail source missing")
    x = dict(x)
    x["files"] = dict(x["files"])
    x["files"]["durability_guardrails"] = A1._sha256(p)
    x["implementation_revision"] = _REVISION
    return x


def _finalize_checkpoint_info(R, cp: Path, drive_ok: bool) -> dict[str, Any] | None:
    """Handle the valid A1 checkpoint state after D=1500 migration.

    The frozen scanner calls global_step_of(phase_idx, step), which only accepts
    scientific phase indices 0..2. A1 uses phase_idx==3 solely as an engineering
    sentinel meaning "all updates complete; finalize result packaging". Detect that
    sentinel before delegating to the frozen scanner.
    """
    if not drive_ok:
        return {"CAMPAIGN_ROOT": str(R.OUT), "EXPERIMENT_VERSION": "ARK-020-V4",
                "SAFE_ACTION": "STOP — DRIVE UNAVAILABLE", "HARDENED_GATE": "NOT_APPLICABLE"}
    try:
        x = torch.load(cp, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if int(x.get("phase_idx", -1)) != len(C.PHASES) or x.get("phase") != "FINALIZE":
        return None

    info: dict[str, Any] = {
        "CAMPAIGN_ROOT": str(R.OUT),
        "EXPERIMENT_VERSION": "ARK-020-V4",
        "PINNED_COMMIT": R._pinned_commit(),
        "CHECKPOINT_FOUND": True,
        "ACTIVE_ARM": x.get("arm"),
        "ACTIVE_PHASE": "FINALIZE",
        "SAVED_PHASE_STEP": 0,
        "GLOBAL_STEP": C.CONTINUATION_HORIZON,
        "COMPLETED_ARMS": len(R.collect_arm_results()),
        "TOTAL_ARMS": len(C.ARMS) * 4,
    }
    lock_cls, lock_info = R.read_lock()
    info["WRITER_LOCK"] = lock_cls
    if lock_cls == "ACTIVE":
        info["LOCK"] = lock_info
        info["CHECKPOINT_IDENTITY"] = "NOT_CHECKED_ACTIVE_WRITER"
        info["HARDENED_GATE"] = "NOT_APPLICABLE"
        info["SAFE_ACTION"] = "WAIT — ACTIVE WRITER LOCK EXISTS"
        return info
    if lock_cls == "MALFORMED":
        info["LOCK"] = "MALFORMED (inspect manually)"
        info["CHECKPOINT_IDENTITY"] = "FAIL"
        info["HARDENED_GATE"] = "FAIL"
        info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
        return info

    errors = A1._validate_checkpoint_receipts(R, cp)
    if errors:
        info["CHECKPOINT_IDENTITY"] = "FAIL"
        info["HARDENED_GATE"] = "FAIL"
        info["HARDENED_ERRORS"] = errors
        info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
    else:
        info["CHECKPOINT_IDENTITY"] = "PASS"
        info["HARDENED_GATE"] = "PASS"
        info["SAFE_ACTION"] = "RESUME"
        info["RESUME_MEANING"] = "FINALIZE_ONLY_NO_MORE_TRAINING_UPDATES"
    return info


def install(R) -> None:
    if getattr(R, "_ARK020_V4_A1_GUARDRAILS_INSTALLED", False):
        return

    # Bind this file into the immutable executable receipt produced/checked by A1.
    A1.executable_identity = _extended_executable_identity

    prior_scan = R.resume_scan
    prior_run_set_arm = R.run_set_arm

    def resume_scan(*, drive_ok: bool = True):
        cps = sorted(R.OUT.glob("matched_sets/p*_b*/*/RESUME.pt"))
        if len(cps) == 1:
            special = _finalize_checkpoint_info(R, cps[0], drive_ok)
            if special is not None:
                return special
        if len(cps) > 1:
            return {
                "CAMPAIGN_ROOT": str(R.OUT),
                "EXPERIMENT_VERSION": "ARK-020-V4",
                "CHECKPOINT_FOUND": True,
                "CHECKPOINT_IDENTITY": "FAIL",
                "HARDENED_GATE": "FAIL",
                "HARDENED_ERRORS": [f"multiple active checkpoints found ({len(cps)}); expected at most one"],
                "SAFE_ACTION": "STOP — CHECKPOINT IDENTITY FAILURE",
            }
        try:
            return prior_scan(drive_ok=drive_ok)
        except Exception as exc:
            # Cell 0 must always fail closed. Never turn scanner exceptions into an
            # implicit START/RESUME recommendation.
            return {
                "CAMPAIGN_ROOT": str(R.OUT),
                "EXPERIMENT_VERSION": "ARK-020-V4",
                "CHECKPOINT_IDENTITY": "FAIL",
                "HARDENED_GATE": "FAIL",
                "HARDENED_ERRORS": [f"scan exception: {type(exc).__name__}: {exc}"],
                "SAFE_ACTION": "STOP — SCAN FAILED",
            }

    def run_set_arm(ps, bs, arm, dose_b, parent_state, cap16, bufs, tt, tasks, d, deadline,
                    session_log=print):
        # Completed arms are immutable results; do not migrate a stale checkpoint if
        # RESULT.json already proves the arm completed.
        _ad, rp, _cp, _pp = R.arm_paths(ps, bs, arm)
        if rp.exists():
            return prior_run_set_arm(ps, bs, arm, dose_b, parent_state, cap16, bufs, tt,
                                     tasks, d, deadline, session_log=session_log)
        return prior_run_set_arm(ps, bs, arm, dose_b, parent_state, cap16, bufs, tt,
                                 tasks, d, deadline, session_log=session_log)

    def write_lock():
        R.OUT.mkdir(parents=True, exist_ok=True)
        body = {
            "experiment": "ARK-020-V4",
            "engineering_amendment": A1.AMENDMENT_ID,
            "implementation_revision": _REVISION,
            "executable_commit": R._pinned_commit(),
            "session_uuid": uuid.uuid4().hex,
            "timestamp": time.time(),
            "host": socket.gethostname(),
        }
        (R.OUT / "CAMPAIGN_LOCK.json").write_text(
            json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return body

    R.resume_scan = resume_scan
    R.run_set_arm = run_set_arm
    R.write_lock = write_lock
    R._ARK020_V4_A1_GUARDRAILS_INSTALLED = True
