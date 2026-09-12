"""ARK-020 V4 durability amendment A1.

This module is an engineering-only overlay around the frozen ARK-020 V4 scientific
runner. It does not change tasks, thresholds, arms, replay policy, or verdict rules.

It adds four fail-closed guarantees:
1. campaign executable identity is bound to an immutable receipt;
2. resume scans refuse partial/missing identity evidence;
3. exact phase-boundary checkpoints are advanced deterministically before resume;
4. the production exact-resume gate is supplemented with a controller/registry
   transition round-trip using the same checkpoint API.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch

import ark020_v4_core as C

AMENDMENT_ID = "ARK-020-V4-DURABILITY-A1"
EXECUTABLE_RECEIPT = "EXECUTABLE_IDENTITY_A1.json"
BOUNDARY_RECEIPT = "BOUNDARY_RESUME_A1.json"

_ORIGINALS: dict[str, Any] = {}


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def executable_identity(R) -> dict[str, Any]:
    files = {
        "core": R.HERE / "ark020_v4_core.py",
        "scientific_runner": R.HERE / "run_ark020_v4.py",
        "durability_overlay": R.HERE / "ark020_v4_durability.py",
        "operator_runner": R.HERE / "run_ark020_v4_hardened.py",
        "preregistration": R.HERE / "PREREGISTRATION.json",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"durability identity files missing: {missing}")
    return {
        "schema": "arkenstone-ark020-v4-executable-identity-a1/v1",
        "experiment": "ARK-020-V4",
        "engineering_amendment": AMENDMENT_ID,
        "git_commit": _git_head(R.REPO),
        "files": {name: _sha256(path) for name, path in files.items()},
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    x = json.loads(path.read_text())
    if not isinstance(x, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return x


def _identity_diff(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    errors = []
    for k in ("schema", "experiment", "engineering_amendment", "git_commit"):
        if actual.get(k) != expected.get(k):
            errors.append(f"{k}: {actual.get(k)!r} != {expected.get(k)!r}")
    ef = expected.get("files", {})
    af = actual.get("files", {})
    if af != ef:
        errors.append("file hashes differ")
    return errors


def ensure_executable_receipt(R, *, allow_create: bool) -> dict[str, Any]:
    """Require the exact engineering executable for every campaign continuation."""
    current = executable_identity(R)
    if current["git_commit"] == "unknown":
        raise RuntimeError("cannot establish immutable git executable identity")
    p = R.OUT / EXECUTABLE_RECEIPT
    if p.exists():
        actual = _read_json(p)
        diff = _identity_diff(current, actual)
        if diff:
            raise RuntimeError("executable identity mismatch: " + "; ".join(diff))
        return dict(actual)
    if not allow_create:
        raise RuntimeError("required executable identity receipt is absent")
    R.OUT.mkdir(parents=True, exist_ok=True)
    body = dict(current)
    body["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Exclusive creation is intentional: two fresh writers must not race.
    try:
        with p.open("x", encoding="utf-8") as f:
            json.dump(body, f, sort_keys=True, indent=2)
            f.write("\n")
    except FileExistsError:
        actual = _read_json(p)
        diff = _identity_diff(current, actual)
        if diff:
            raise RuntimeError("raced executable identity mismatch: " + "; ".join(diff))
        return dict(actual)
    return body


def _expected_arm_identity(R, ps, bs, arm, dose_b, parent_state, cap16, tasks) -> dict[str, Any]:
    oi = C.PHASE_ORDER_SEEDS["B"].index(bs)
    c_seed = C.PHASE_ORDER_SEEDS["C"][oi]
    d_seed = C.PHASE_ORDER_SEEDS["D"][oi]
    return {
        "schema": "arkenstone-ark020-v4-arm-ckpt/v1",
        "parent_seed": ps,
        "b_order_seed": bs,
        "c_order_seed": c_seed,
        "d_order_seed": d_seed,
        "arm": arm,
        "dose_b": int(dose_b),
        "parent_sha": R.V3.state_hash(parent_state["model"]),
        "task_hash": R.hjson({k: R.hjson(tasks[k]) for k in ("A", "B", "C", "D")}),
        "cap16x": float(cap16),
    }


def advance_boundary_payload(R, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Advance a checkpoint saved exactly at a phase horizon.

    The frozen runner checkpoints the final update before its post-phase controller
    transition. On resume that phase's inner loop is empty, so its local ``gstep``
    never exists. This function applies exactly that missing transition once and
    moves the checkpoint to the next phase (or finalization).
    """
    x = copy.deepcopy(dict(payload))
    phase_idx = int(x.get("phase_idx", -1))
    if not (0 <= phase_idx < len(C.PHASES)):
        return x, False
    phase = C.PHASES[phase_idx]
    if x.get("phase") != phase:
        raise RuntimeError(
            f"checkpoint phase identity mismatch: idx={phase_idx} says {phase}, "
            f"payload says {x.get('phase')!r}"
        )
    step = int(x.get("phase_step", -1))
    horizon = int(C.PHASE_UPDATES[phase])
    if step != horizon:
        return x, False

    end_gstep = R.global_step_of(phase_idx, horizon)
    if x.get("arm") in C.GUARDIAN_ARMS:
        reg = x.get("registry")
        ctrl = x.get("controller")
        if not isinstance(reg, dict) or not isinstance(ctrl, dict):
            raise RuntimeError("boundary checkpoint missing registry/controller")
        cap_ids = list(reg.get("capabilities", {}).keys())
        C.set_post_phase_floor(ctrl, cap_ids, end_gstep)
        x["controller"] = ctrl

    next_idx = phase_idx + 1
    x["phase_idx"] = next_idx
    if next_idx < len(C.PHASES):
        x["phase"] = C.PHASES[next_idx]
        x["phase_step"] = 0
    else:
        x["phase"] = "FINALIZE"
        x["phase_step"] = 0
    x["boundary_migration_a1"] = {
        "from_phase": phase,
        "from_phase_step": horizon,
        "phase_end_global_step": end_gstep,
        "to_phase_idx": next_idx,
        "to_phase": x["phase"],
        "amendment": AMENDMENT_ID,
    }
    return x, True


def migrate_boundary_checkpoint(R, cp: Path, expected: Mapping[str, Any]) -> bool:
    if not cp.exists():
        return False
    x = torch.load(cp, map_location="cpu", weights_only=False)
    for k, v in expected.items():
        if x.get(k) != v:
            raise RuntimeError(
                f"boundary checkpoint identity mismatch {k}: {x.get(k)!r} != {v!r}"
            )
    y, changed = advance_boundary_payload(R, x)
    if not changed:
        return False
    tmp = cp.with_suffix(cp.suffix + ".a1.tmp")
    torch.save(y, tmp)
    tmp.replace(cp)
    receipt = cp.parent / BOUNDARY_RECEIPT
    R.savej(
        receipt,
        {
            "schema": "arkenstone-ark020-v4-boundary-resume-a1/v1",
            "status": "APPLIED",
            "checkpoint": cp.name,
            **y["boundary_migration_a1"],
        },
    )
    return True


def hardened_run_set_arm(R, ps, bs, arm, dose_b, parent_state, cap16, bufs, tt, tasks, d,
                         deadline, session_log=print):
    expected = _expected_arm_identity(R, ps, bs, arm, dose_b, parent_state, cap16, tasks)
    _ad, _rp, cp, _pp = R.arm_paths(ps, bs, arm)
    migrate_boundary_checkpoint(R, cp, expected)
    return _ORIGINALS["run_set_arm"](
        ps, bs, arm, dose_b, parent_state, cap16, bufs, tt, tasks, d, deadline,
        session_log=session_log,
    )


def _validate_checkpoint_receipts(R, cp: Path) -> list[str]:
    errors: list[str] = []
    try:
        x = torch.load(cp, map_location="cpu", weights_only=False)
    except Exception as exc:
        return [f"checkpoint unreadable: {type(exc).__name__}: {exc}"]

    required = (
        "schema", "parent_seed", "b_order_seed", "c_order_seed", "d_order_seed", "arm",
        "dose_b", "parent_sha", "task_hash", "cap16x", "phase_idx", "phase", "phase_step",
        "registry", "controller", "counters", "phase_confirm", "global_confirm", "b_streaks",
    )
    for key in required:
        if key not in x:
            errors.append(f"checkpoint missing required field {key}")
    if x.get("schema") != "arkenstone-ark020-v4-arm-ckpt/v1":
        errors.append("checkpoint schema/version mismatch")
    if errors:
        return errors

    entry_path = R.OUT / "ENTRY_RECEIPT.json"
    parent_path = R.OUT / "PARENT_IDENTITIES.json"
    dose_path = R.OUT / "DOSE_SELECTION_IMPORTED.json"
    for p in (entry_path, parent_path, dose_path):
        if not p.exists():
            errors.append(f"required receipt absent: {p.name}")
    if errors:
        return errors

    try:
        entry = _read_json(entry_path)
        th = entry["task_hashes"]
        receipt_task_hash = R.hjson({k: th[k] for k in ("A", "B", "C", "D")})
        if x["task_hash"] != receipt_task_hash:
            errors.append("checkpoint task hash != entry receipt")
    except Exception as exc:
        errors.append(f"entry receipt invalid: {type(exc).__name__}: {exc}")

    try:
        parents = _read_json(parent_path)["parents"]
        acquired = parents[str(x["parent_seed"])]["acquired_parent_model_sha256"]
        if x["parent_sha"] != acquired:
            errors.append("checkpoint parent hash != acquired-parent receipt")
    except Exception as exc:
        errors.append(f"parent identity receipt invalid: {type(exc).__name__}: {exc}")

    try:
        dose = _read_json(dose_path)
        selected = int(dose.get("v4_local_selected_slots", dose.get("selected_b_slots", -1)))
        if dose.get("verification_status") != "verified":
            errors.append("dose receipt is not verified")
        if int(x["dose_b"]) != selected:
            errors.append("checkpoint dose != verified dose receipt")
    except Exception as exc:
        errors.append(f"dose receipt invalid: {type(exc).__name__}: {exc}")

    try:
        ensure_executable_receipt(R, allow_create=False)
    except Exception as exc:
        errors.append(f"executable receipt invalid: {exc}")

    return errors


def _orphan_partial_dirs(root: Path) -> list[str]:
    out = []
    for p in root.glob("matched_sets/p*_b*/*/PARTIAL.json"):
        if not (p.parent / "RESUME.pt").exists() and not (p.parent / "RESULT.json").exists():
            out.append(str(p.parent.relative_to(root)))
    return sorted(out)


def hardened_resume_scan(R, drive_ok: bool = True) -> dict[str, Any]:
    """Fail closed whenever exact identity cannot be proved."""
    info = dict(_ORIGINALS["resume_scan"](drive_ok=drive_ok))
    if info.get("SAFE_ACTION") not in {"START NEW CAMPAIGN", "RESUME"}:
        info["HARDENED_GATE"] = "NOT_APPLICABLE"
        return info

    root = R.OUT
    errors: list[str] = []
    resumes = sorted(root.glob("matched_sets/p*_b*/*/RESUME.pt"))
    orphan = _orphan_partial_dirs(root)
    if orphan:
        errors.append("orphan PARTIAL.json without exact checkpoint: " + ", ".join(orphan))
    if len(resumes) > 1:
        errors.append(f"multiple active checkpoints found ({len(resumes)}); expected at most one")

    base_identity = str(info.get("CHECKPOINT_IDENTITY", ""))
    if base_identity.startswith("PARTIAL"):
        errors.append("base scan reported only PARTIAL identity")

    if resumes:
        errors.extend(_validate_checkpoint_receipts(R, resumes[0]))
    elif info.get("SAFE_ACTION") == "RESUME":
        # Completed work without an active checkpoint is resumable only under the exact
        # executable that created the campaign.
        try:
            ensure_executable_receipt(R, allow_create=False)
        except Exception as exc:
            errors.append(f"completed campaign executable identity invalid: {exc}")

    # A stale/malformed lock from another experiment must never be silently inherited.
    lock = root / "CAMPAIGN_LOCK.json"
    if lock.exists() and info.get("WRITER_LOCK") != "ACTIVE":
        try:
            lock_data = _read_json(lock)
            if lock_data.get("experiment") not in {"ARK-020-V4", AMENDMENT_ID}:
                errors.append(f"stale lock belongs to {lock_data.get('experiment')!r}")
        except Exception as exc:
            errors.append(f"stale lock unreadable: {exc}")

    if errors:
        info["HARDENED_GATE"] = "FAIL"
        info["HARDENED_ERRORS"] = errors
        info["CHECKPOINT_IDENTITY"] = "FAIL"
        info["SAFE_ACTION"] = "STOP — CHECKPOINT IDENTITY FAILURE"
    else:
        info["HARDENED_GATE"] = "PASS"
        if resumes:
            info["CHECKPOINT_IDENTITY"] = "PASS"
            info["SAFE_ACTION"] = "RESUME"
    return info


def _transition_event(reg: dict, ctrl: dict, counters: dict, step: int, metrics: Mapping[str, Any]):
    for cap_id, m in metrics.items():
        if cap_id not in reg["capabilities"]:
            C.register_capability(reg, cap_id, step)
        C.observe_capability(reg, cap_id, step, m)
    caps = sorted(reg["capabilities"])
    C.update_controller("GUARDIAN_HYBRID", ctrl, step, reg, caps)
    replay, cap = C.treatment("GUARDIAN_HYBRID", ctrl, step, reg, caps, 1.0)
    counters["events"] = counters.get("events", 0) + 1
    counters["replay_slots"] = counters.get("replay_slots", 0) + len(replay)
    counters["replay_by_cap"] = dict(counters.get("replay_by_cap", {}))
    for c in replay:
        counters["replay_by_cap"][c] = counters["replay_by_cap"].get(c, 0) + 1
    return {"step": step, "replay": list(replay), "cap_active": cap is not None}


def controller_registry_resume_smoke(R, parent_state, d: torch.device) -> dict[str, Any]:
    """Durability fixture that crosses CONTROL -> controller -> replay -> registry change."""
    bad = {"canonical": 0.80, "order_only": 0.80, "query_order": 0.80}
    good = {"canonical": 0.99, "order_only": 0.99, "query_order": 0.99}

    def initial():
        reg = C.init_registry()
        C.register_capability(reg, "A", 0)
        return reg, C.initial_controller("GUARDIAN_HYBRID"), {
            "events": 0, "replay_slots": 0, "replay_by_cap": {}
        }

    reg, ctrl, cnt = initial()
    pre = _transition_event(reg, ctrl, cnt, 25, {"A": bad})
    if ctrl["states"].get("A") != "REPLAY32" or "A" not in pre["replay"]:
        raise RuntimeError("controller durability fixture failed to activate replay")
    C.register_capability(reg, "B", 50)
    mid = _transition_event(reg, ctrl, cnt, 50, {"A": good, "B": good})

    # Uninterrupted continuation reference.
    reg_ref, ctrl_ref, cnt_ref = copy.deepcopy(reg), copy.deepcopy(ctrl), copy.deepcopy(cnt)
    post_ref = _transition_event(reg_ref, ctrl_ref, cnt_ref, 75, {"A": good, "B": good})

    m, o, sc = R.restore(parent_state, d)
    expected = {
        "schema": "arkenstone-ark020-v4-durability-smoke/v1",
        "amendment": AMENDMENT_ID,
        "cut_step": 50,
    }
    payload = {
        **expected,
        "registry": reg,
        "controller": ctrl,
        "counters": cnt,
        "phase_idx": 0,
        "phase": "B",
        "phase_step": 50,
        "phase_confirm": {"B": None, "C": None, "D": None},
        "global_confirm": {"B": None, "C": None, "D": None},
        "b_streaks": {"B": 0, "C": 0, "D": 0},
        "control_trace": [pre, mid],
        "measure_trace": [],
    }
    q = R.OUT / "_durability_a1_ckpt.pt"
    R.save_checkpoint(q, payload, m, o, sc)
    before_model = R.model_state_hash(m)
    before_opt = R.optimizer_hash(o)
    before_cpu_rng = R.hjson(torch.get_rng_state().tolist())
    x, m2, o2, sc2 = R.load_checkpoint(q, expected, d)
    reg2, ctrl2, cnt2 = x["registry"], x["controller"], x["counters"]
    post_resumed = _transition_event(reg2, ctrl2, cnt2, 75, {"A": good, "B": good})

    checks = {
        "registry": R.hjson(reg_ref) == R.hjson(reg2),
        "controller": R.hjson(ctrl_ref) == R.hjson(ctrl2),
        "counters": R.hjson(cnt_ref) == R.hjson(cnt2),
        "post_event": R.hjson(post_ref) == R.hjson(post_resumed),
        "model": before_model == R.model_state_hash(m2),
        "optimizer": before_opt == R.optimizer_hash(o2),
        "cpu_rng": before_cpu_rng == R.hjson(torch.get_rng_state().tolist()),
        "registry_changed": "B" in reg2["capabilities"],
        "controller_transitioned": bool(ctrl2.get("transitions")),
        "replay_activated": cnt2.get("replay_slots", 0) > 0,
    }
    q.unlink(missing_ok=True)
    del m, o, m2, o2, sc, sc2
    if d.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "schema": "arkenstone-ark020-v4-controller-resume-a1/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cut_step": 50,
        "resume_step": 75,
    }


def hardened_exact_resume_smoke(R, parent_state, dose_b, bufs, tt, tasks, d):
    p = R.OUT / "EXACT_RESUME_SMOKE_V4_A1.json"
    if p.exists():
        x = _read_json(p)
        if x.get("status") != "PASS":
            raise RuntimeError("cached A1 durability smoke did not pass")
        return dict(x)

    production = _ORIGINALS["exact_resume_smoke"](parent_state, dose_b, bufs, tt, tasks, d)
    transition = controller_registry_resume_smoke(R, parent_state, d)
    ok = production.get("status") == "PASS" and transition.get("status") == "PASS"
    r = {
        "schema": "arkenstone-ark020-v4-exact-resume-a1/v1",
        "status": "PASS" if ok else "FAIL",
        "engineering_amendment": AMENDMENT_ID,
        "production_exact_resume": production,
        "controller_registry_transition": transition,
        "evidence_scope": (
            "production model/optimizer/scaler/RNG equivalence plus controller/registry/"
            "replay transition checkpoint round-trip"
        ),
    }
    R.savej(p, r)
    if not ok:
        raise RuntimeError("ARK-020 V4 A1 durability smoke failed")
    return r


def hardened_run_all(R):
    ensure_executable_receipt(R, allow_create=True)
    return _ORIGINALS["run_all"]()


def hardened_write_lock(R):
    body = _ORIGINALS["write_lock"]()
    body["experiment"] = "ARK-020-V4"
    body["engineering_amendment"] = AMENDMENT_ID
    (R.OUT / "CAMPAIGN_LOCK.json").write_text(json.dumps(body, sort_keys=True, indent=2) + "\n")
    return body


def install(R) -> None:
    """Install the A1 overlay exactly once into the imported frozen runner module."""
    if getattr(R, "_ARK020_V4_A1_INSTALLED", False):
        return
    for name in ("run_set_arm", "resume_scan", "exact_resume_smoke", "run_all", "write_lock"):
        _ORIGINALS[name] = getattr(R, name)

    R.run_set_arm = lambda *a, **kw: hardened_run_set_arm(R, *a, **kw)
    R.resume_scan = lambda drive_ok=True: hardened_resume_scan(R, drive_ok=drive_ok)
    R.exact_resume_smoke = lambda *a, **kw: hardened_exact_resume_smoke(R, *a, **kw)
    R.run_all = lambda: hardened_run_all(R)
    R.write_lock = lambda: hardened_write_lock(R)
    R._ARK020_V4_A1_INSTALLED = True
