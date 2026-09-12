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


def _load_verified_json(R, path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read one savej()-style receipt and verify its self hash."""
    try:
        x = json.loads(path.read_text())
    except Exception as exc:
        return None, f"{path.name} unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(x, dict):
        return None, f"{path.name} is not a JSON object"
    got = x.get("receipt_sha256")
    if got is None:
        return None, f"{path.name} missing receipt_sha256"
    body = dict(x)
    body.pop("receipt_sha256", None)
    if got != R.hjson(body):
        return None, f"{path.name} receipt_sha256 mismatch"
    return x, None


def _campaign_state_artifacts(root: Path) -> list[str]:
    """Recognized durable state that makes an output root non-new."""
    names = (
        "ENTRY_RECEIPT.json",
        "PARENT_IDENTITIES.json",
        "DOSE_SELECTION_IMPORTED.json",
        "PREEXECUTION_GATE.json",
        "EXACT_RESUME_SMOKE_V4.json",
        "EXACT_RESUME_SMOKE_V4_A1.json",
        "RUNTIME_CALIBRATION.json",
        "SESSION_STATE.json",
        "ARK-020_V4_RESULT.json",
        "ARK-020_V4_FAILURE.json",
    )
    found = [n for n in names if (root / n).exists()]
    for d in ("parents", "v4_reuse", "matched_sets"):
        p = root / d
        if p.exists() and any(p.rglob("*")):
            found.append(d + "/")
    return sorted(set(found))


def _validate_completed_results(R) -> list[str]:
    """Validate all immutable completed arm results before trusting RESUME.

    Result files do not contain task_hash, so task identity remains bound by the
    immutable executable receipt + ENTRY_RECEIPT. Here we independently bind each
    completed result to its path, matched-set seeds, acquired parent, selected dose,
    and cap calibration, and verify savej receipt integrity.
    """
    paths = sorted(R.OUT.glob("matched_sets/p*_b*/*/RESULT.json"))
    if not paths:
        return []
    errors: list[str] = []

    entry, e = _load_verified_json(R, R.OUT / "ENTRY_RECEIPT.json")
    parents, p_err = _load_verified_json(R, R.OUT / "PARENT_IDENTITIES.json")
    dose, d_err = _load_verified_json(R, R.OUT / "DOSE_SELECTION_IMPORTED.json")
    for err in (e, p_err, d_err):
        if err:
            errors.append(err)
    if errors:
        return errors
    assert entry is not None and parents is not None and dose is not None
    if entry.get("schema") != "arkenstone-ark020-v4-entry/v1":
        errors.append("ENTRY_RECEIPT schema mismatch")
    if parents.get("schema") != "arkenstone-ark020-v4-parent-identities/v1":
        errors.append("PARENT_IDENTITIES schema mismatch")
    if dose.get("verification_status") != "verified":
        errors.append("dose receipt is not verified")
    try:
        selected_dose = int(dose.get("v4_local_selected_slots", dose.get("selected_b_slots", -1)))
    except Exception:
        selected_dose = -1
        errors.append("dose receipt selected slots invalid")

    seen: set[tuple[int, int, str]] = set()
    parent_table = parents.get("parents", {})
    for path in paths:
        r, err = _load_verified_json(R, path)
        rel = str(path.relative_to(R.OUT))
        if err:
            errors.append(f"{rel}: {err}")
            continue
        assert r is not None
        try:
            ps = int(r["parent_seed"])
            bs = int(r["b_order_seed"])
            arm = str(r["arm"])
        except Exception as exc:
            errors.append(f"{rel}: result identity unreadable: {exc}")
            continue
        ident = (ps, bs, arm)
        if ident in seen:
            errors.append(f"{rel}: duplicate completed result identity {ident}")
        seen.add(ident)
        expected_dir = R.OUT / "matched_sets" / f"p{ps}_b{bs}" / arm / "RESULT.json"
        if path != expected_dir:
            errors.append(f"{rel}: path does not match embedded identity")
        if r.get("schema") != "arkenstone-ark020-v4-arm/v1" or r.get("status") != "COMPLETE":
            errors.append(f"{rel}: result schema/status mismatch")
        if ps not in C.PARENT_SEEDS or bs not in C.PHASE_ORDER_SEEDS["B"] or arm not in C.ARMS:
            errors.append(f"{rel}: identity outside preregistered matched set")
            continue
        oi = C.PHASE_ORDER_SEEDS["B"].index(bs)
        if int(r.get("c_order_seed", -1)) != C.PHASE_ORDER_SEEDS["C"][oi]:
            errors.append(f"{rel}: C order seed mismatch")
        if int(r.get("d_order_seed", -1)) != C.PHASE_ORDER_SEEDS["D"][oi]:
            errors.append(f"{rel}: D order seed mismatch")
        if int(r.get("dose_b", -1)) != selected_dose:
            errors.append(f"{rel}: dose mismatch")
        try:
            acquired = parent_table[str(ps)]["acquired_parent_model_sha256"]
            if r.get("parent_sha") != acquired:
                errors.append(f"{rel}: acquired-parent hash mismatch")
        except Exception as exc:
            errors.append(f"{rel}: acquired-parent receipt missing: {exc}")
        cap_path = R.OUT / "matched_sets" / f"p{ps}_b{bs}" / "CAP_CALIBRATION.json"
        if cap_path.exists():
            caprec, caperr = _load_verified_json(R, cap_path)
            if caperr:
                errors.append(f"{rel}: {caperr}")
            elif caprec is not None and float(r.get("cap16x", float("nan"))) != float(caprec.get("cap16x", float("nan"))):
                errors.append(f"{rel}: cap16x != calibration receipt")
        else:
            errors.append(f"{rel}: CAP_CALIBRATION.json absent")
    return errors


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

    try:
        completed = len(R.collect_arm_results())
    except Exception as exc:
        return {
            "CAMPAIGN_ROOT": str(R.OUT), "EXPERIMENT_VERSION": "ARK-020-V4",
            "CHECKPOINT_FOUND": True, "CHECKPOINT_IDENTITY": "FAIL", "HARDENED_GATE": "FAIL",
            "HARDENED_ERRORS": [f"completed result scan failed: {type(exc).__name__}: {exc}"],
            "SAFE_ACTION": "STOP — SCAN FAILED",
        }
    info: dict[str, Any] = {
        "CAMPAIGN_ROOT": str(R.OUT),
        "EXPERIMENT_VERSION": "ARK-020-V4",
        "PINNED_COMMIT": R._pinned_commit(),
        "CHECKPOINT_FOUND": True,
        "ACTIVE_ARM": x.get("arm"),
        "ACTIVE_PHASE": "FINALIZE",
        "SAVED_PHASE_STEP": 0,
        "GLOBAL_STEP": C.CONTINUATION_HORIZON,
        "COMPLETED_ARMS": completed,
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
    errors.extend(_validate_completed_results(R))
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
            info = prior_scan(drive_ok=drive_ok)
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

        if info.get("SAFE_ACTION") == "START NEW CAMPAIGN":
            artifacts = _campaign_state_artifacts(R.OUT)
            if artifacts:
                # The frozen scan only counts arm results/checkpoints. A crash during
                # parent/dose/preexecution leaves durable state but would otherwise be
                # misclassified as a fresh campaign. Same-executable evidence makes
                # this a pre-main resume; missing/mismatched evidence fails closed.
                try:
                    A1.ensure_executable_receipt(R, allow_create=False)
                except Exception as exc:
                    info["CHECKPOINT_IDENTITY"] = "FAIL"
                    info["HARDENED_GATE"] = "FAIL"
                    info["HARDENED_ERRORS"] = [
                        f"non-empty campaign root lacks matching executable identity: {exc}"
                    ]
                    info["PREEXISTING_ARTIFACTS"] = artifacts
                    info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
                    return info
                info["CHECKPOINT_IDENTITY"] = "PASS"
                info["HARDENED_GATE"] = "PASS"
                info["PREEXISTING_ARTIFACTS"] = artifacts
                info["SAFE_ACTION"] = "RESUME"
                info["RESUME_MEANING"] = "PRE_MAIN_SAME_EXECUTABLE"
                return info

        if info.get("SAFE_ACTION") == "RESUME" and not cps:
            errors = _validate_completed_results(R)
            if errors:
                info["CHECKPOINT_IDENTITY"] = "FAIL"
                info["HARDENED_GATE"] = "FAIL"
                info["HARDENED_ERRORS"] = errors
                info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
        return info

    def run_set_arm(ps, bs, arm, dose_b, parent_state, cap16, bufs, tt, tasks, d, deadline,
                    session_log=print):
        # Completed arms are immutable results. Crucially, do NOT invoke A1's boundary
        # migrator on a stale RESUME.pt after RESULT.json already proves completion.
        _ad, rp, _cp, _pp = R.arm_paths(ps, bs, arm)
        if rp.exists():
            r, err = _load_verified_json(R, rp)
            if err or r is None:
                raise RuntimeError(f"corrupt ARK-020 V4 arm result {rp}: {err}")
            parent_sha = R.V3.state_hash(parent_state["model"])
            oi = C.PHASE_ORDER_SEEDS["B"].index(bs)
            expected_c = C.PHASE_ORDER_SEEDS["C"][oi]
            expected_d = C.PHASE_ORDER_SEEDS["D"][oi]
            compatible = (
                r.get("schema") == "arkenstone-ark020-v4-arm/v1"
                and r.get("status") == "COMPLETE"
                and int(r.get("parent_seed", -1)) == int(ps)
                and int(r.get("b_order_seed", -1)) == int(bs)
                and int(r.get("c_order_seed", -1)) == int(expected_c)
                and int(r.get("d_order_seed", -1)) == int(expected_d)
                and r.get("arm") == arm
                and int(r.get("dose_b", -1)) == int(dose_b)
                and float(r.get("cap16x", float("nan"))) == float(cap16)
                and r.get("parent_sha") == parent_sha
            )
            if compatible:
                return r
            raise RuntimeError(f"incompatible ARK-020 V4 arm result {rp}")
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
