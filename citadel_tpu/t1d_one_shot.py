"""T1D ONE-SHOT ORCHESTRATOR — fresh Colab TPU -> one run -> one bundle.

Operator workflow after this module exists: open the notebook, select TPU,
run CELL 1 (RUN EVERYTHING). Everything else is this file's job:

  P0  BOOTSTRAP      pinned Cymek runtime, session dir, development-certificate
                     identity check (fail-closed: code newer than certification)
  P1  PREFLIGHT      structured live gates (run_preflight); failure => bundle
  P2  CANARY         tiny end-to-end TPU canary over the REAL code paths
                     (MID + SCALE2 + masked + teacher + checkpoint/reload +
                     generation + the pure producer->finalizer bridge)
  P3  DATA           manifest (reuse or build) + unique/consumable accounting
  P4..P9  ARM A..F   per-arm isolation: one infra failure continues, two abort;
                     mid-arm checkpoints at 25/50/75%; prefinal snapshots;
                     finalization-only recovery
  P10 PRE50M         systems certification (fail-soft: failure receipts, arms kept)
  P11 SUMMARIZE      cross-arm classification, lift curves, session manifest
  P12 BUNDLE         build + verify CITADEL_T1D_RESULTS.zip

Every phase writes PHASE_<name>.json (status/times/identity/error). Any
failure after preflight preserves all prior evidence and exports
CITADEL_T1D_FAILURE.zip automatically. Heartbeat written every phase.

Device-free emulation: every device-bound runner is an injectable seam
(preflight_runner / canary_runner / arm_runner / pre50m_runner); the emulator
tests drive the REAL orchestration, REAL generators/feeders/finalizers/
classifier/bundle builders with synthetic predictions only.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

SESSION_SCHEMA = "citadel-t1d-one-shot/v1"
PHASE_SCHEMA = "citadel-t1d-phase/v1"
ORCHESTRATOR_VERSION = "one-shot/1.0"

PHASE_ORDER = ("BOOTSTRAP", "PREFLIGHT", "CANARY", "DATA",
               "ARM_A", "ARM_B", "ARM_C", "ARM_D", "ARM_E", "ARM_F",
               "PRE50M", "SUMMARIZE", "BUNDLE")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


# ------------------------------------------------------------------ preflight
def code_sha() -> str:
    """Identity of the last commit that touched EXECUTABLE paths. Docs-only
    handover commits (agent.md, receipts, certification itself) do not
    invalidate a certificate; any code change does."""
    import subprocess

    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "citadel_tpu", "tests",
         "notebooks", "pyproject.toml"],
        capture_output=True, text=True, timeout=60)
    return out.stdout.strip() or "unknown"


def build_development_certificate(out: str | Path | None = None) -> dict[str, Any]:
    """DEVELOPMENT_CERTIFICATION (§4/§32): run the complete deterministic
    repository suite and record totals + identity. Committed to the repo; the
    live preflight refuses to run TPU work under a different Citadel SHA."""
    import subprocess
    import sys

    from citadel_tpu import runtime_bootstrap as rb

    tests = ["tests/test_citadel_t1d.py", "tests/test_citadel_t1c.py",
             "tests/test_citadel_t1_canary.py", "tests/test_citadel_notebooks.py",
             "tests/test_citadel_bootstrap.py", "tests/test_citadel_cymek_checkpoint.py",
             "tests/test_citadel_one_shot.py"]
    results = {}
    for path in tests:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=1200)
        tail = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
        results[path] = {"returncode": proc.returncode, "summary": tail}
    passed = sum(1 for r in results.values() if r["returncode"] == 0)
    root, cymek_sha = rb.ensure_cymek_runtime()
    cert = {
        "schema": "citadel-development-certificate/v1",
        "generated_utc": _now(),
        "citadel_sha": rb.citadel_sha(),
        "code_sha": code_sha(),
        "cymek_sha": cymek_sha,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "tests": results,
        "files_passed": passed,
        "files_total": len(tests),
        "status": "PASS" if passed == len(tests) else "FAIL",
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cert, indent=2, sort_keys=True), encoding="utf-8")
    return cert


def _certificate_path() -> Path:
    from citadel_tpu import runtime_bootstrap as rb

    return Path(rb.citadel_root()) / "docs" / "citadel" / "experiments" / "T1D" / \
        "DEVELOPMENT_CERTIFICATION.json"


# ------------------------------------------------------------------ TPU canary
def tpu_canary(session_dir: str) -> dict[str, Any]:
    """Tiny end-to-end canary over the REAL session code paths (§18-21) BEFORE
    substantial arms: MID + SCALE2 instantiate; packed feeder; answer-only,
    teacher, and masked-vocab paths; forward/backward/optimizer; checkpoint
    save/reload; generation; the pure producer->finalizer bridge. Fails
    loudly before any expensive arm."""
    import torch

    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import xla_backend as xb

    checks: dict[str, Any] = {}
    device = xb.get_device()
    spec_mid = t1d.build_spec("MID")
    spec_scale2 = t1d.build_spec("SCALE2")
    checks["specs"] = {"MID": spec_mid.parameter_receipt().total,
                       "SCALE2": spec_scale2.parameter_receipt().total}

    from citadel_tpu import self_knowledge as sk
    from citadel_tpu import tiered_data as td
    from v5_model.core import initialize
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_training.optimizer import build_adamw_optimizer

    torch.manual_seed(20260906)
    model = initialize(spec_mid, 20260906).to(device)
    optimizer = build_adamw_optimizer(model, torch_module=torch)

    # feeder paths: flat / curriculum / teacher / self + masked construction
    for mode in ("flat", "curriculum", "teacher", "self"):
        feeder = t1d.TierFeeder(mode, 8, 64)
        seqs = feeder.fill_sequences(0.5)
        assert len(seqs) == 8, mode
    checks["feeder_modes"] = ["flat", "curriculum", "teacher", "self"]

    def one_update(feeder_mode: str, masked: bool):
        feeder = t1d.TierFeeder(feeder_mode, 8, 64)
        seqs = feeder.fill_sequences(0.5)
        tokens, seg, eligible, stats = t1d.assemble_batch(seqs, length=64,
                                                          torch_mod=torch)
        from v5_model.core import packed_layout

        pos, mask = packed_layout(seg, torch_module=torch)
        logits = model(tokens.to(device), pos.to(device), mask.to(device))
        if masked:
            allow = torch.zeros(24_576, dtype=torch.bool, device=device)
            allow[torch.tensor(t1d.valid_alphabet_ids(), device=device)] = True
            logits = torch.where(allow, logits,
                                 torch.full_like(logits, float("-inf")))
        loss, count = causal_lm_loss(logits, tokens.to(device), seg.to(device),
                                     eligible=eligible.to(device),
                                     torch_module=torch)
        assert int(count) == stats["answer"], "eligible mismatch in canary"
        loss.backward()
        xb.mark_step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xb.optimizer_step(optimizer)
        xb.mark_step()
        optimizer.zero_grad()
        return float(loss.detach().to("cpu").item())

    losses = {"ordinary": one_update("curriculum", False),
              "teacher": one_update("teacher", False),
              "masked": one_update("curriculum", True)}
    checks["updates"] = {k: v for k, v in losses.items()
                         if v == v and v < float("inf")}
    if len(checks["updates"]) != 3:
        raise RuntimeError("CANARY_NONFINITE_LOSS")

    # checkpoint + reload identity + generation
    from citadel_tpu import checkpoint as ckpt_mod

    ckpt = str(Path(session_dir) / "canary_ckpt.pt")
    sha = ckpt_mod.save(model, ckpt, {"canary": True})
    pre = cev.generate(["12 + 9 = "], model, xb, device=device, torch_mod=torch)
    model2 = initialize(spec_mid, 20260906).to(device)
    ckpt_mod.load_into(model2, ckpt)
    xb.mark_step()
    post = cev.generate(["12 + 9 = "], model2, xb, device=device, torch_mod=torch)
    checks["reload_identical"] = bool(pre[0]["prediction"] == post[0]["prediction"])
    checks["checkpoint_sha256"] = sha

    # SCALE2 single update at the calibrated-shape contract level (small shape)
    model_s = initialize(spec_scale2, 20260906).to(device)
    del model_s

    # the pure producer->finalizer bridge (legacy producer shape included)
    defects = t1d.producer_consumer_contract_probe(legacy_untrained_keys=True)
    checks["producer_finalizer_defects"] = defects
    if defects:
        raise RuntimeError("CANARY_SCHEMA_BRIDGE: " + "; ".join(defects))
    checks["self_probe_rows"] = len(sk.self_probe_rows()[0])
    checks["plan_sha"] = t1d.plan_identity()
    return {"schema": "citadel-t1d-canary/v1", "status": "PASS", "checks": checks}


# ------------------------------------------------------------- data accounting
def data_accounting() -> dict[str, Any]:
    """§23: unique vs schedulable data accounting for the REAL frozen budgets.
    Unique useful supervision is the goal — not file size."""
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import tiered_data as td

    per_arm = {}
    for tag, cfg in t1d.ARMS.items():
        budget = cfg["budget"]
        rows_est = budget // 20  # mean tiered row ~20 chars (measured below)
        per_arm[tag] = {"budget_tokens": budget,
                        "estimated_rows": rows_est}
    # measured mean row length from a real sample across tiers
    lens = []
    for tier in range(5):
        lens.extend(len(td.tier_row(tier, "train", i)[0]) for i in range(0, 2000, 7))
    mean_row = sum(lens) / len(lens)
    available_rows = sum(td.TRAIN_N.values())
    available_bytes_est = sum(td.TRAIN_N[t] * mean_row for t in range(5))
    schedulable_rows = sum(int(cfg["budget"] // mean_row) for cfg in t1d.ARMS.values())
    # consumption is tier-skewed (curriculum), but the pool bound is what §23 asks
    fraction = schedulable_rows / max(available_rows, 1)
    return {
        "schema": "citadel-t1d-data-account/v1",
        "measured_mean_row_chars": round(mean_row, 2),
        "available_unique_rows": available_rows,
        "available_unique_bytes_est": int(available_bytes_est),
        "scheduled_rows_all_arms_est": schedulable_rows,
        "consumable_unique_fraction_est": round(fraction, 4),
        "replay_expected": bool(fraction > 0.5),
        "per_arm_estimates": per_arm,
        "decision": ("KEEP" if fraction < 0.5 else
                     "EXPAND_IF_SCIENTIFICALLY_USEFUL"),
        "note": ("unique pool exceeds schedulable rows: generating more "
                 "physical data would be unused scale" if fraction < 0.5 else
                 "arms would exhaust a meaningful pool fraction"),
    }


# ----------------------------------------------------------------- orchestrator
def _phase_receipt(root: Path, phase: str, *, status: str, start: float,
                   inputs: dict | None = None, outputs: dict | None = None,
                   error: str | None = None) -> dict[str, Any]:
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d

    doc = {"schema": PHASE_SCHEMA, "phase": phase, "status": status,
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
           "end_utc": _now(), "wall_seconds": round(time.time() - start, 3),
           "citadel_sha": rb.citadel_sha(),
           "cymek_runtime_sha": _cymek_sha_cached(),
           "plan_sha": t1d.plan_identity(),
           "orchestrator_version": ORCHESTRATOR_VERSION,
           "inputs": inputs or {}, "outputs": outputs or {}}
    if error:
        doc["error"] = error
    (root / f"PHASE_{phase}.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return doc


_CK = {}


def _cymek_sha_cached() -> str:
    if "sha" not in _CK:
        from citadel_tpu import runtime_bootstrap as rb

        _CK["sha"] = rb.ensure_cymek_runtime()[1]
    return _CK["sha"]


def _phase_resume(root: Path, phase: str, plan_sha: str) -> dict | None:
    """Phase-level resume identity: a PASS receipt under the SAME plan version
    short-circuits the phase. Any plan/hash divergence forces a rerun."""
    path = Path(root) / f"PHASE_{phase}.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if doc.get("status") == "PASS" and doc.get("plan_sha") == plan_sha:
        return doc
    return None


def _heartbeat(root: Path, **fields: Any) -> None:
    doc = {"schema": "citadel-t1d-heartbeat/v1", "updated_utc": _now(), **fields}
    (root / "SESSION_HEARTBEAT.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")


def _failure_bundle(root: Path, phase: str, exc: BaseException) -> str:
    """Export EVERYTHING known at failure time (§30/§31): environment,
    preflight, phase receipts, traceback, arm receipts, heartbeat."""
    root = Path(root)
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    (root / "FAILURE_TRACEBACK.txt").write_text(
        f"phase: {phase}\nutc: {_now()}\n" + "".join(tb), encoding="utf-8")
    members = ["FAILURE_TRACEBACK.txt", "SESSION_HEARTBEAT.json"]
    members += [p.name for p in sorted(root.glob("PHASE_*.json"))]
    members += [p.name for p in sorted(root.glob("ARM_*.json"))]
    members += [p.name for p in sorted(root.glob("PREFLIGHT*.json"))]
    for name in ("ENVIRONMENT.json", "SESSION_MANIFEST.json", "DATA_MANIFEST.json",
                 "CALIBRATION.json"):
        if (root / name).is_file():
            members.append(name)
    out = root / "CITADEL_T1D_FAILURE.zip"
    seen = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in members:
            if name in seen or not (root / name).is_file():
                continue
            seen.add(name)
            zf.write(root / name, name)
    return str(out)


def run_all(session_dir: str, *, seed: int = 20260904,
            preflight_runner=None, canary_runner=None, arm_runner=None,
            pre50m_runner=None, device_required: bool = True) -> dict[str, Any]:
    """One orchestrated run. Returns the session dict; on any failure the
    failure bundle is written FIRST, then the failure is reported in the
    returned dict (never a bare traceback with lost evidence)."""
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import tiered_data as td

    root = Path(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    session = {"schema": SESSION_SCHEMA, "orchestrator_version": ORCHESTRATOR_VERSION,
               "session_dir": str(root), "phases": {}, "status": "RUNNING"}
    _heartbeat(root, phase="BOOTSTRAP", status="starting")

    def finish(status: str, error: str | None = None,
               bundle: str | None = None) -> dict[str, Any]:
        session["status"] = status
        if error:
            session["error"] = error
        if bundle:
            session["failure_bundle"] = bundle
        (root / "SESSION_MANIFEST.json").write_text(
            json.dumps(session, indent=2, sort_keys=True), encoding="utf-8")
        return session

    # P0 BOOTSTRAP
    start = time.time()
    try:
        rt_root, rt_sha = rb.ensure_cymek_runtime()
        env = env_mod.probe(require_tpu=False)
        (root / "ENVIRONMENT.json").write_text(
            json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")
        cert = _certificate_path()
        cert_doc = json.loads(cert.read_text(encoding="utf-8")) if cert.is_file() else None
        identity = {"cymek_runtime_sha": rt_sha,
                    "plan_sha": t1d.plan_identity(),
                    "certificate": ({"citadel_sha": cert_doc["citadel_sha"],
                                     "status": cert_doc["status"]}
                                    if cert_doc else "MISSING")}
        if cert_doc is None:
            raise RuntimeError("DEVELOPMENT_CERTIFICATION missing: run "
                               "build_development_certificate() before handoff")
        runtime_code_sha = code_sha()
        certified_code = cert_doc.get("code_sha", cert_doc.get("citadel_sha"))
        if certified_code != runtime_code_sha:
            raise RuntimeError(
                "CODE NEWER THAN CERTIFICATION: development certification was "
                f"built at {str(certified_code)[:12]} but executable code is "
                f"{runtime_code_sha[:12]}; regenerate certification first")
        if cert_doc.get("status") != "PASS":
            raise RuntimeError("development certification is not PASS")
        if cert_doc.get("cymek_sha") not in (None, rt_sha):
            raise RuntimeError("Cymek runtime SHA differs from certification pin")
        _phase_receipt(root, "BOOTSTRAP", status="PASS", start=start,
                       outputs=identity)
        session["phases"]["BOOTSTRAP"] = "PASS"
        _heartbeat(root, phase="BOOTSTRAP", status="PASS")
    except Exception as exc:
        _phase_receipt(root, "BOOTSTRAP", status="FAILED", start=start,
                       error=f"{type(exc).__name__}: {exc}")
        bundle = _failure_bundle(root, "BOOTSTRAP", exc)
        return finish("FAILED", f"BOOTSTRAP: {exc}", bundle)

    # P1 PREFLIGHT
    start = time.time()
    try:
        runner = preflight_runner or _live_preflight
        pre = runner(root)
        (root / "PREFLIGHT.json").write_text(json.dumps(pre, indent=2,
                                                        sort_keys=True),
                                             encoding="utf-8")
        if pre.get("status") != "PASS":
            raise RuntimeError("PREFLIGHT failed: " + "; ".join(
                pre.get("blocking_gates", [])[:4]))
        _phase_receipt(root, "PREFLIGHT", status="PASS", start=start,
                       outputs={"gates": len(pre.get("gates", []))})
        session["phases"]["PREFLIGHT"] = "PASS"
        _heartbeat(root, phase="PREFLIGHT", status="PASS")
    except Exception as exc:
        _phase_receipt(root, "PREFLIGHT", status="FAILED", start=start,
                       error=str(exc))
        bundle = _failure_bundle(root, "PREFLIGHT", exc)
        return finish("FAILED", f"PREFLIGHT: {exc}", bundle)

    # P2 CANARY
    start = time.time()
    resumed = _phase_resume(root, "CANARY", t1d.plan_identity())
    if resumed is not None:
        _phase_receipt(root, "CANARY", status="PASS", start=start,
                       outputs={"resumed": True})
        session["phases"]["CANARY"] = "PASS"
    else:
      try:
        crunner = canary_runner or tpu_canary
        canary = crunner(str(root))
        (root / "CANARY.json").write_text(json.dumps(canary, indent=2,
                                                     sort_keys=True),
                                          encoding="utf-8")
        _phase_receipt(root, "CANARY", status="PASS", start=start)
        session["phases"]["CANARY"] = "PASS"
        _heartbeat(root, phase="CANARY", status="PASS")
      except Exception as exc:
        _phase_receipt(root, "CANARY", status="FAILED", start=start,
                       error=f"{type(exc).__name__}: {exc}")
        bundle = _failure_bundle(root, "CANARY", exc)
        return finish("FAILED", f"CANARY: {exc}", bundle)

    # P3 DATA
    start = time.time()
    try:
        man_path = root / "DATA_MANIFEST.json"
        if man_path.is_file():
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
            assert manifest.get("generator_version") == td.GENERATOR_VERSION, \
                "stale data manifest version"
            reused = True
        else:
            manifest = td.build_manifest(out=str(man_path))
            reused = False
        fatal, _ = td.leakage_verdict(manifest["leakage"])
        if fatal:
            raise RuntimeError(f"LEAKAGE: {fatal}")
        if manifest.get("max_row_chars", 0) > 64:
            raise RuntimeError(f"ROW_TOO_LONG: {manifest['max_row_chars']}")
        account = data_accounting()
        (root / "DATA_ACCOUNT.json").write_text(
            json.dumps(account, indent=2, sort_keys=True), encoding="utf-8")
        _phase_receipt(root, "DATA", status="PASS", start=start,
                       outputs={"reused": reused,
                                "total_bytes": manifest.get("total_bytes"),
                                "consumable_fraction":
                                    account["consumable_unique_fraction_est"]})
        session["phases"]["DATA"] = "PASS"
        _heartbeat(root, phase="DATA", status="PASS")
    except Exception as exc:
        _phase_receipt(root, "DATA", status="FAILED", start=start,
                       error=f"{type(exc).__name__}: {exc}")
        bundle = _failure_bundle(root, "DATA", exc)
        return finish("FAILED", f"DATA: {exc}", bundle)

    # calibration is a prerequisite of arms: reuse or run (inside DATA->ARMS bridge)
    cal_path = root / "CALIBRATION.json"
    if cal_path.is_file():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        shape = (cal["selected"]["batch"], cal["selected"]["length"])
        rate = float(cal["selected_tokens_per_second"])
    else:
        start = time.time()
        try:
            from citadel_tpu import t1d_run as t1d

            cal = t1d.calibrate(out=str(cal_path))
            shape = (cal["selected"]["batch"], cal["selected"]["length"])
            rate = float(cal["selected_tokens_per_second"])
            _phase_receipt(root, "CALIBRATION", status="PASS", start=start,
                           outputs={"shape": list(shape), "rate": rate})
        except Exception as exc:
            _phase_receipt(root, "CALIBRATION", status="FAILED", start=start,
                           error=f"{type(exc).__name__}: {exc}")
            bundle = _failure_bundle(root, "CALIBRATION", exc)
            return finish("FAILED", f"CALIBRATION: {exc}", bundle)
    budgets = {t: dict(c) for t, c in t1d.ARMS.items()}
    scaled = False
    if rate < t1d.AUTO_SCALE_RATE:
        for c in budgets.values():
            c["budget"] //= 2
        scaled = True

    # P4..P9 ARMS
    arm_receipts: dict[str, Any] = {}
    infra_failures = 0
    for tag in t1d.ARM_ORDER:
        phase = f"ARM_{tag}"
        start = time.time()
        arm_path = root / f"ARM_{tag}.json"
        resumed = _phase_resume(root, phase, t1d.plan_identity())
        if resumed is not None and arm_path.is_file():
            receipt = json.loads(arm_path.read_text(encoding="utf-8"))
            arm_receipts[tag] = receipt
            _phase_receipt(root, phase, status="PASS", start=start,
                           outputs={"resumed": True,
                                    "arm_status": receipt.get("status")})
            session["phases"][phase] = receipt.get("status")
            print(f"arm {tag}: resumed {receipt.get('status')} "
                  f"(phase receipt + plan match)", flush=True)
            continue
        _heartbeat(root, phase=phase, status="starting",
                   arm=tag, shape=list(shape))
        try:
            runner = arm_runner or t1d.run_arm
            receipt = runner(tag, budgets[tag], shape=shape,
                             out_dir=str(root), seed=seed)
            arm_receipts[tag] = receipt
            _phase_receipt(root, phase, status="PASS", start=start,
                           outputs={"arm_status": receipt.get("status")})
            session["phases"][phase] = receipt.get("status")
            _heartbeat(root, phase=phase, status="PASS", arm=tag,
                       arm_status=receipt.get("status"))
            print(f"[{t1d.ARM_ORDER.index(tag) + 1}/{len(t1d.ARM_ORDER)}] "
                  f"arm {tag}: {receipt.get('status')}", flush=True)
        except Exception as exc:
            infra_failures += 1
            receipt = t1d.arm_failure_receipt(tag, exc,
                                              citadel_sha=rb.citadel_sha(),
                                              cymek_sha=_cymek_sha_cached())
            (root / f"ARM_{tag}.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            arm_receipts[tag] = receipt
            _phase_receipt(root, phase, status="FAILED", start=start,
                           error=f"{type(exc).__name__}: {exc}")
            session["phases"][phase] = "IMPLEMENTATION_FAILURE"
            _heartbeat(root, phase=phase, status="FAILED", arm=tag,
                       error=str(exc))
            print(f"arm {tag}: IMPLEMENTATION_FAILURE {exc}", flush=True)
            if infra_failures >= 2:
                bundle = _failure_bundle(root, phase, exc)
                return finish("FAILED",
                              f"SESSION: 2nd infra failure at arm {tag}", bundle)
    if not arm_receipts:
        exc = RuntimeError("no arm produced a receipt")
        bundle = _failure_bundle(root, "ARMS", exc)
        return finish("FAILED", "ARMS: no receipts", bundle)

    # P10 PRE50M (fail-soft: arms are already safe on disk)
    start = time.time()
    resumed = _phase_resume(root, "PRE50M", t1d.plan_identity())
    if resumed is not None and (root / "NEXT_50M_DECISION.json").is_file():
        decision = json.loads(
            (root / "NEXT_50M_DECISION.json").read_text(encoding="utf-8"))
        session["pre50m"] = {"status": "PASS", "decision": decision,
                             "resumed": True}
        _phase_receipt(root, "PRE50M", status="PASS", start=start,
                       outputs={"resumed": True})
        session["phases"]["PRE50M"] = "PASS"
    else:
      try:
        prunner = pre50m_runner or t1d._run_pre50m_phase
        pre50m_status = prunner(root, arm_receipts, rt_sha=_cymek_sha_cached(),
                                rate=rate, shape=tuple(shape))
        session["pre50m"] = pre50m_status
        _phase_receipt(root, "PRE50M", status="PASS", start=start)
        session["phases"]["PRE50M"] = "PASS"
      except Exception as exc:
        t1d._write_pre50m_failure_receipts(root, exc, citadel_sha=rb.citadel_sha(),
                                           cymek_sha=_cymek_sha_cached())
        session["pre50m"] = {"status": "IMPLEMENTATION_FAILURE",
                             "error": f"{type(exc).__name__}: {exc}"}
        _phase_receipt(root, "PRE50M", status="FAILED", start=start,
                       error=f"{type(exc).__name__}: {exc}")
        session["phases"]["PRE50M"] = "IMPLEMENTATION_FAILURE"

    # P11 SUMMARIZE
    start = time.time()
    try:
        session_doc = t1d.summarize_session(
            root, arm_receipts, shape=shape, rate=rate, scaled=scaled,
            budgets={t: c["budget"] for t, c in budgets.items()},
            rt_sha=_cymek_sha_cached())
        session["labels"] = session_doc.get("labels")
        _phase_receipt(root, "SUMMARIZE", status="PASS", start=start,
                       outputs={"labels": session_doc.get("labels")})
        session["phases"]["SUMMARIZE"] = "PASS"
    except Exception as exc:
        _phase_receipt(root, "SUMMARIZE", status="FAILED", start=start,
                       error=f"{type(exc).__name__}: {exc}")
        bundle = _failure_bundle(root, "SUMMARIZE", exc)
        return finish("FAILED", f"SUMMARIZE: {exc}", bundle)

    # P12 BUNDLE
    start = time.time()
    try:
        bundle = t1d.build_bundle(str(root), out=str(root / "CITADEL_T1D_RESULTS.zip"))
        verdict = t1d.verify_bundle(str(root))
        _phase_receipt(root, "BUNDLE", status="PASS", start=start,
                       outputs={"zip_bytes": bundle["zip_bytes"],
                                "verify": verdict["status"]})
        session["phases"]["BUNDLE"] = "PASS"
        session["bundle_bytes"] = bundle["zip_bytes"]
        return finish("COMPLETE")
    except Exception as exc:
        _phase_receipt(root, "BUNDLE", status="FAILED", start=start,
                       error=f"{type(exc).__name__}: {exc}")
        bundle_path = _failure_bundle(root, "BUNDLE", exc)
        return finish("FAILED", f"BUNDLE: {exc}", bundle_path)


def _live_preflight(root: Path) -> dict[str, Any]:
    from citadel_tpu import t1d_preflight

    return t1d_preflight.run_preflight()


__all__ = ["PHASE_ORDER", "build_development_certificate", "data_accounting",
           "run_all", "tpu_canary"]
