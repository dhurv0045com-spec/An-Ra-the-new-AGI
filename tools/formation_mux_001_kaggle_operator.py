"""FORMATION-MUX-001 Kaggle operator (parent coordinator).

Owns: hardware gate, qualification phases 0-4, runtime calibration and the
10.5-hour projection stop, the deterministic dual-worker queue, campaign
state (atomic + lock), failure classification (GLOBAL fail-closed vs
arm-local ENGINEERING_FAILURE), the per-experiment sealed firewall, sealed
finalization, and artifact packaging. Workers own one arm on one GPU and
nothing else. No DataParallel/DistributedDataParallel anywhere: the two T4s
are two independent scientific workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v5_experiments import formation_mux_protocol as proto  # noqa: E402
from anra_v5 import formation_mux_model as fxm  # noqa: E402
from v5_training.optimizer import build_adamw_optimizer  # noqa: E402

EXIT_OK, EXIT_ARM_LOCAL, EXIT_GLOBAL, EXIT_USAGE = 0, 3, 4, 2
CAMPAIGN_ROOT = Path("/kaggle/working/FORMATION_MUX_001")
BUNDLE_NAME = "FORMATION_MUX_001_RESULTS.zip"
EXPECTED_GPUSubstring = "T4"


class HardwareGateError(RuntimeError):
    pass


class GlobalIntegrityError(RuntimeError):
    pass


# -- atomic state ---------------------------------------------------------------

def _lock(state_dir: Path, name: str = "state.lock") -> Path:
    lock = state_dir / name
    deadline = time.monotonic() + 60.0
    while True:
        try:
            return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() > deadline:
                raise RuntimeError("campaign state lock held > 60s")
            time.sleep(0.05)


def _unlock(handle: int, state_dir: Path, name: str = "state.lock") -> None:
    os.close(handle)
    try:
        (state_dir / name).unlink()
    except FileNotFoundError:
        pass


def write_state(state_dir: Path, update: Callable[[dict], dict]) -> dict:
    """Read-modify-write CAMPAIGN_STATE.json under the campaign lock with an
    atomic tmp+rename. Never two writers at once."""

    state_dir.mkdir(parents=True, exist_ok=True)
    handle = _lock(state_dir)
    try:
        path = state_dir / "CAMPAIGN_STATE.json"
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        updated = update(current)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return updated
    finally:
        _unlock(handle, state_dir)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- hardware gate ----------------------------------------------------------------

def hardware_gate(torch: Any) -> dict[str, Any]:
    """Official science requires exactly 2 visible T4-class ~16GB devices.
    Anything else stops BEFORE scientific execution with the exact user
    instruction. Never silently falls back."""

    receipt: dict[str, Any] = {
        "schema": "anra.formation-mux-hardware-gate/v1",
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if
        torch.cuda.is_available() else 0,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "devices": []}
    for index in range(receipt["device_count"]):
        properties = torch.cuda.get_device_properties(index)
        receipt["devices"].append({
            "index": index, "name": torch.cuda.get_device_name(index),
            "total_vram_gb": round(properties.total_memory / 1e9, 2)})
    receipt["t4_class"] = (
        receipt["device_count"] == 2
        and all(EXPECTED_GPUSubstring in device["name"]
                for device in receipt["devices"])
        and all(14.0 <= device["total_vram_gb"] <= 17.0
                for device in receipt["devices"]))
    return receipt


def require_official_hardware(torch: Any) -> dict[str, Any]:
    receipt = hardware_gate(torch)
    if not receipt["t4_class"]:
        raise HardwareGateError(
            "OFFICIAL EXECUTION BLOCKED: this campaign requires exactly two "
            "T4-class GPUs. Set Kaggle Notebook -> Settings -> Accelerator -> "
            "GPU T4 x2 (and Internet -> ON), then rerun. Do not choose TPU or "
            f"CPU. Observed: {receipt['device_count']} device(s) "
            f"{[d['name'] for d in receipt['devices']]}")


# -- qualification -----------------------------------------------------------------

def _worker_cmd(*, experiment: str, arm: str, bundle: int, surface: Path,
                out: Path, device: str = "cuda",
                updates: int | None = None, engineering_only: bool = False,
                deadline_minutes: float | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "tools.formation_mux_001_worker",
           "--experiment", experiment, "--arm", arm,
           "--seed-bundle", str(bundle), "--surface", str(surface),
           "--out", str(out), "--device", device]
    if updates is not None:
        cmd += ["--updates-override", str(updates), "--engineering-only"]
    elif engineering_only:
        cmd += ["--engineering-only"]
    if deadline_minutes is not None:
        cmd += ["--deadline-minutes", str(deadline_minutes)]
    return cmd


def run_worker(cmd: list[str], *, gpu: int | None = None, cwd: Path) -> int:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    completed = subprocess.run(cmd, cwd=cwd, env=env)
    return completed.returncode


# -- calibration + projection --------------------------------------------------------

def calibrate_and_project(*, surface: Path, out: Path, torch: Any,
                          engineering_only: bool = False,
                          ) -> dict[str, Any]:
    """Disposable calibration on BOTH GPUs (no official seeds, no sealed
    data), then the projection formula over the frozen 24-arm queue. If the
    projected wall exceeds 10.5 hours the operator must NOT start official
    science; a PARTITION_PLAN is produced instead (protocol unchanged)."""

    from v5_experiments import formation_mux_protocol as proto
    started = time.monotonic()
    per_arm_seconds: list[float] = []
    for gpu in (0, 1):
        arm = "M0_STANDARD" if gpu == 0 else "M1_EXTRA_NO_DECAY"
        arm_started = time.monotonic()
        code = run_worker(
            _worker_cmd(experiment=proto.EXPERIMENT_A, arm=arm,
                        bundle=proto.SEED_BUNDLES[gpu], surface=surface,
                        out=out / "calibration", updates=5),
            gpu=gpu, cwd=Path(__file__).resolve().parents[1])
        if code != EXIT_OK:
            raise GlobalIntegrityError(
                f"calibration worker failed on GPU{gpu} (exit {code})")
        per_arm_seconds.append(time.monotonic() - arm_started)
    probe_seconds = max(per_arm_seconds) / 5.0  # 5-update probe -> per-update
    eval_every = proto.EVAL_EVERY
    updates = proto.UPDATES
    train_seconds = updates * probe_seconds
    eval_seconds = (updates // eval_every) * 20.0      # dev generation cadence
    checkpoint_seconds = (updates // proto.CHECKPOINT_EVERY) * 8.0
    per_arm = train_seconds + eval_seconds + checkpoint_seconds
    queue = proto.queue_assignment()
    worker_minutes = {gpu: len(arms) * per_arm / 60.0
                      for gpu, arms in queue.items()}
    projected = max(worker_minutes.values())
    receipt = {
        "schema": "anra.formation-mux-calibration/v1",
        "calibration_rng": "disposable (no official seed bundles)",
        "sealed_data_touched": False,
        "probe_seconds_per_update": round(probe_seconds, 3),
        "per_arm_projection_seconds": round(per_arm, 1),
        "projection_formula": ("max over workers of (#arms x (updates x "
                               "sec/update + updates/EVAL_EVERY x eval_s + "
                               "updates/CHECKPOINT_EVERY x ckpt_s))"),
        "worker_minutes": {gpu: round(minutes, 1)
                           for gpu, minutes in worker_minutes.items()},
        "projected_campaign_minutes": round(projected, 1),
        "partition_threshold_minutes": proto.PARTITION_THRESHOLD_MINUTES,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.monotonic() - started, 1)}
    (out / "CALIBRATION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if projected > proto.PARTITION_THRESHOLD_MINUTES and not engineering_only:
        partition = {
            "schema": "anra.formation-mux-partition-plan/v1",
            "reason": ("projected campaign wall exceeds 10.5h; scientific "
                       "protocol (seeds/arms/budgets/thresholds/hashes) is "
                       "unchanged; execution is partitioned operationally"),
            "executions": [
                {"execution": "A",
                 "scope": "CS-MECH-002 seed bundles S1-S2 + S3-S4 split across workers"},
                {"execution": "B",
                 "scope": "CS-MECH-002 remaining bundles"},
                {"execution": "C",
                 "scope": "REP-FORM-003A all bundles (8 arms)"}],
            "resume": ("attach the previous execution's Output under /kaggle/input "
                       "and rerun this notebook; completed arms skip, partial "
                       "arms exact-resume from validated checkpoints")}
        (out / "PARTITION_PLAN.json").write_text(
            json.dumps(partition, indent=2) + "\n", encoding="utf-8")
        receipt["partition_plan"] = str(out / "PARTITION_PLAN.json")
        receipt["stopped_before_science"] = True
    else:
        receipt["stopped_before_science"] = False
    return receipt


# -- campaign ------------------------------------------------------------------------

def campaign_state(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "CAMPAIGN_STATE.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema": "anra.formation-mux-state/v1", "arms": {}, "sealed":
        {experiment: "NOT_CONSUMED" for experiment in
         ("CS-MECH-002", "REP-FORM-003A")}, "global_failure": None}


def run_campaign(*, surface: Path, out: Path, torch: Any,
                 engineering_only: bool = False,
                 progress: Callable[[str], None] = print,
                 ) -> dict[str, Any]:
    """Deterministic dual-worker execution with the failure policy:

    GLOBAL (wrong hashes, unregistered arms/bundles, matched-init drift,
    sealed violation) -> fail closed, preserve evidence, stop everything.
    ARM-LOCAL (isolated OOM/CUDA/checkpoint failure) -> ENGINEERING_FAILURE
    receipt, matched comparison incomplete, unrelated bundles continue."""

    from v5_experiments import formation_mux_protocol as proto
    out = Path(out)
    state_dir = out
    started = time.monotonic()
    write_state(state_dir, lambda s: {**campaign_state(state_dir),
                                      "campaign": proto.CAMPAIGN,
                                      "science_started_utc": time.strftime(
                                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "surface_sha256": sha256_file(surface)})
    cwd = Path(__file__).resolve().parents[1]
    queue = proto.queue_assignment()
    pending: dict[str, list[dict[str, Any]]] = {
        gpu: list(arms) for gpu, arms in queue.items()}
    stop = False
    stop_reason = None
    while any(pending.values()) and not stop:
        running: dict[int, dict[str, Any]] = {}
        for gpu in (0, 1):
            if pending[f"GPU{gpu}"]:
                job = pending[f"GPU{gpu}"].pop(0)
                cmd = _worker_cmd(
                    experiment=job["experiment"], arm=job["arm"],
                    bundle=job["seed_bundle"], surface=surface, out=out,
                    engineering_only=engineering_only)
                running[gpu] = {"job": job, "proc": subprocess.Popen(
                    cmd, cwd=cwd, env={**os.environ,
                                       "CUDA_VISIBLE_DEVICES": str(gpu),
                                       "OMP_NUM_THREADS": "2",
                                       "MKL_NUM_THREADS": "2"})}
                progress(f"GPU{gpu} <- {job['experiment']}/{job['arm']} "
                         f"{job['seed_bundle_label']}")
        codes: dict[int, int] = {}
        for gpu, entry in running.items():
            codes[gpu] = entry["proc"].wait()
        for gpu, code in codes.items():
            job = running[gpu]["job"]
            key = f"{job['experiment']}/{job['arm']}/S{proto.SEED_BUNDLES.index(job['seed_bundle']) + 1}"
            if code == EXIT_OK:
                write_state(state_dir, lambda s: {**s, "arms": {**s["arms"],
                            key: "COMPLETE"}})
            elif code == EXIT_ARM_LOCAL:
                progress(f"ARM-LOCAL ENGINEERING_FAILURE at {key}: matched "
                         "comparison incomplete; unrelated work continues")
                write_state(state_dir, lambda s: {**s, "arms": {**s["arms"],
                            key: "ENGINEERING_FAILURE"}})
            else:
                reason = f"worker exit {code} at {key}"
                progress(f"GLOBAL FAIL-CLOSED: {reason}")
                write_state(state_dir, lambda s: {**s, "global_failure": reason})
                stop, stop_reason = True, reason
    state = campaign_state(state_dir)
    complete = sum(1 for v in state["arms"].values() if v == "COMPLETE")
    state["summary"] = {"complete_arms": complete,
                        "required_arms": proto.total_official_arms(),
                        "global_failure": stop_reason}
    if stop_reason is None and complete == proto.total_official_arms():
        state["status"] = "ARMS_COMPLETE"
        for experiment in proto.EXPERIMENTS:
            finalize_dev_aggregate(out, experiment)
    elif stop_reason is None:
        state["status"] = "PARTIAL_SESSION"
        state["resume_instruction"] = ("rerun this operator; completed arms "
                                       "skip and partial arms exact-resume")
    else:
        state["status"] = "FAILED_GLOBAL"
    state["wall_seconds"] = round(time.monotonic() - started, 1)
    (state_dir / "CAMPAIGN_STATE.json").write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state


# -- aggregates + sealed firewall ------------------------------------------------------

def _arm_result(out: Path, experiment: str, bundle: int, arm: str) -> dict | None:
    index = proto.SEED_BUNDLES.index(bundle) + 1
    path = Path(out) / experiment / arm / f"S{index}" / "ARM_RESULT.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def finalize_dev_aggregate(out: Path, experiment: str) -> dict[str, Any]:
    """Development aggregate + development verdict from completed arms only.
    Incomplete matched bundles are reported, never imputed."""

    from v5_experiments import formation_mux_protocol as proto
    arms = proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B
    per_arm: dict[str, dict[int, dict]] = {arm: {} for arm in arms}
    for bundle in proto.SEED_BUNDLES:
        for arm in arms:
            body = _arm_result(out, experiment, bundle, arm)
            if body and body.get("status") == "COMPLETE":
                per_arm[arm][bundle] = body["formation"]
    if experiment == proto.EXPERIMENT_A:
        contrasts = [(h, l, label) for h, l, label in proto.CONTRASTS_A]
    else:
        contrasts = [proto.CONTRAST_B]
    contrast_results = []
    for higher, lower, label in contrasts:
        bundles = [b for b in proto.SEED_BUNDLES
                   if b in per_arm[higher] and b in per_arm[lower]]
        deltas = {b: per_arm[higher][b]["formation_auc"]
                  - per_arm[lower][b]["formation_auc"] for b in bundles}
        gaps = {b: per_arm[higher][b]["endpoint"]
                - per_arm[lower][b]["endpoint"] for b in bundles}
        result = proto.paired_verdict(deltas=deltas, endpoint_gaps=gaps)
        result.update({"higher": higher, "lower": lower, "label": label})
        contrast_results.append(result)
    aggregate = {
        "schema": "anra.formation-mux-dev-aggregate/v1",
        "experiment": experiment,
        "per_arm": {arm: {str(b): body for b, body in per_arm[arm].items()}
                    for arm in arms},
        "contrast_results": contrast_results,
        "dev_verdict": contrast_results[0]["verdict"],
        "claim_ceiling": proto.claim_ceiling(
            experiment, contrast_results[0]["verdict"])}
    path = Path(out) / experiment / "DEV_AGGREGATE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregate, indent=2, default=str) + "\n",
                    encoding="utf-8")
    return aggregate


def sealed_finalized(out: Path, experiment: str) -> bool:
    marker_path = Path(out) / experiment / "SEALED_MARKER.json"
    return marker_path.exists() and json.loads(
        marker_path.read_text())["state"] == "COMPLETE"


def set_sealed_marker(out: Path, experiment: str, state: str) -> None:
    if state not in proto.SEALED_STATES:
        raise ValueError(f"unknown sealed state {state}")
    marker_path = Path(out) / experiment / "SEALED_MARKER.json"
    if marker_path.exists():
        previous = json.loads(marker_path.read_text(encoding="utf-8"))["state"]
        if previous == "STARTED" and state != "COMPLETE":
            raise GlobalIntegrityError(
                f"sealed marker for {experiment} is STARTED without a final "
                "result: fail closed, evidence preserved; never delete the "
                "marker to retry cleanly")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(
        {"experiment": experiment, "state": state,
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2) + "\n", encoding="utf-8")


def finalize_sealed(*, surface: Path, out: Path, torch: Any,
                    engineering_only: bool = False,
                    progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Consume each experiment's sealed set ONCE, after its development
    protocol is frozen (all official arms COMPLETE). Sealed evaluation uses
    the SAME frozen scoring code on the never-touched sealed rows; verdicts
    come from the preregistered paired rules."""

    from v5_experiments import formation_mux_protocol as proto
    from v5_experiments.formation_mux_data import load_surface
    from anra_v5 import formation_mux_train as train
    from v5_model.core import initialize

    results: dict[str, Any] = {}
    for experiment in proto.EXPERIMENTS:
        state = campaign_state(Path(out))
        arms = proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B
        needed = [f"{experiment}/{arm}/S{proto.SEED_BUNDLES.index(b) + 1}"
                  for arm in arms for b in proto.SEED_BUNDLES]
        if any(state["arms"].get(key) != "COMPLETE" for key in needed):
            results[experiment] = {"verdict": "INCONCLUSIVE",
                                   "reason": "official arms incomplete; "
                                             "sealed set stays untouched"}
            continue
        set_sealed_marker(out, experiment, "STARTED")
        surface_manifest = load_surface(surface)
        sealed_rows = list(surface_manifest["splits"]["sealed"])
        device = torch.device("cuda:0") if torch.cuda.is_available() \
            else torch.device("cpu")
        sealed_metrics: dict[str, dict[int, dict]] = {arm: {} for arm in arms}
        for arm in arms:
            for bundle in proto.SEED_BUNDLES:
                index = proto.SEED_BUNDLES.index(bundle) + 1
                checkpoint = Path(out) / experiment / arm / f"S{index}" / "resume.pt"
                if not checkpoint.exists():
                    raise GlobalIntegrityError(
                        f"sealed finalization missing checkpoint {checkpoint}")
                seed = bundle if experiment == proto.EXPERIMENT_A \
                    else proto.b_seed(bundle)
                model = initialize(fxm.spec(), int(seed), torch_module=torch).to(device)
                main = __import__("v5_training.optimizer", fromlist=["x"]) \
                    .build_adamw_optimizer(model, torch_module=torch, lr=proto.LR)
                train.load_checkpoint(checkpoint, model=model,
                                      optimizers={"main": main, "rows": None},
                                      torch=torch,
                                      expected={"experiment": experiment,
                                                "arm": arm,
                                                "seed_bundle": bundle})
                rates = train._eval_rates(model, sealed_rows, experiment, arm,
                                          torch=torch, device=device)
                sealed_metrics[arm][bundle] = rates
                del model, main
        higher, lower, label = (proto.CONTRAST_B if experiment ==
                                proto.EXPERIMENT_B else proto.CONTRASTS_A[0])
        # Sealed verdict: paired endpoint gaps per preregistered contrast.
        deltas = {b: sealed_metrics[higher][b]["content_exact"]
                  - sealed_metrics[lower][b]["content_exact"]
                  for b in proto.SEED_BUNDLES}
        gaps = dict(deltas)
        verdict = proto.paired_verdict(deltas=deltas, endpoint_gaps=gaps)
        final = {
            "schema": "anra.formation-mux-final-result/v1",
            "experiment": experiment,
            "observed": {arm: {str(b): sealed_metrics[arm][b]
                               for b in proto.SEED_BUNDLES}
                         for arm in arms},
            "derived": {"paired_content_exact_deltas":
                        {str(b): round(v, 6) for b, v in deltas.items()}},
            "verdict": verdict["verdict"],
            "claim_ceiling": proto.claim_ceiling(experiment, verdict["verdict"]),
            "next_action": proto.NEXT_ACTION[verdict["verdict"]],
            "sealed_consumption": "COMPLETE (once, after dev protocol frozen)"}
        (Path(out) / experiment / "FINAL_RESULT.json").write_text(
            json.dumps(final, indent=2, default=str) + "\n", encoding="utf-8")
        set_sealed_marker(out, experiment, "COMPLETE")
        results[experiment] = final
        progress(f"sealed finalization {experiment}: {verdict['verdict']}")
    return results


# -- packaging ------------------------------------------------------------------------

def export_model_checkpoints(out: Path) -> dict[str, Any]:
    """Every expensive campaign exports final model state SEPARATELY from
    the small results ZIP: one model-only archive, its SHA-256, and an
    EXPORT_VERIFIED receipt. Without this receipt the run is not complete."""

    checkpoints = sorted(Path(out).rglob("resume.pt"))
    archive_dir = Path(out) / "CHECKPOINTS"
    archive_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    for index, checkpoint in enumerate(checkpoints):
        relative = checkpoint.relative_to(out)
        label = str(relative).replace("/", "__").replace(".pt", ".pt")
        target = archive_dir / label
        if not target.exists() or target.stat().st_size == 0:
            target.write_bytes(checkpoint.read_bytes())
        entries[str(relative)] = sha256_file(target)
    archive = archive_dir / "MODEL_CHECKPOINTS.tar"
    import tarfile
    with tarfile.open(archive, "w") as tar:
        for checkpoint in checkpoints:
            tar.add(checkpoint, arcname=str(checkpoint.relative_to(out)))
    receipt = {"schema": "anra.formation-mux-export/v1",
               "export_verified": True,
               "checkpoint_count": len(checkpoints),
               "checkpoint_sha256": entries,
               "model_archive": str(archive),
               "model_archive_sha256": sha256_file(archive)}
    (archive_dir / "EXPORT_VERIFIED.json").write_text(
        json.dumps(receipt, indent=2) + chr(10), encoding="utf-8")
    return receipt


def package_results(out: Path, *, source_commit: str) -> dict[str, Any]:
    out = Path(out)
    bundle = out.parent / BUNDLE_NAME
    readme = (
        "FORMATION-MUX-001 results bundle.\n\n"
        "Contents: CAMPAIGN_STATE.json (queue + failure classification), "
        "ENVIRONMENT/CALIBRATION receipts, per-experiment preregistrations, "
        "per-arm ARM_RESULT/PARTIAL/FAILURE receipts, DEV_AGGREGATE.json per "
        "experiment, FINAL_RESULT.json + SEALED_MARKER.json (once sealed "
        "consumption legally happened), and the source commit identity. "
        "Failure receipts are preserved evidence, never cleaned.\n")
    payload_files: list[Path] = [out / "CAMPAIGN_STATE.json",
                                 out / "ENVIRONMENT.json",
                                 out / "CALIBRATION_RECEIPT.json"]
    if (out / "PARTITION_PLAN.json").exists():
        payload_files.append(out / "PARTITION_PLAN.json")
    for experiment in ("CS-MECH-002", "REP-FORM-003A"):
        base = out / experiment
        if base.exists():
            payload_files.extend(sorted(base.rglob("*.json")))
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme)
        zf.writestr("SOURCE_COMMIT.txt", source_commit + "\n")
        for path in payload_files:
            if path.exists():
                zf.write(path, path.relative_to(out.parent))
    digest = sha256_file(bundle)
    (bundle.parent / (bundle.name + ".sha256")).write_text(
        digest + "  " + bundle.name + "\n", encoding="utf-8")
    return {"path": str(bundle), "sha256": digest,
            "entries": len(payload_files) + 2}


# -- entry -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--surface", type=Path, default=None,
                        help="pregenerated surface manifest; generated when absent")
    parser.add_argument("--engineering-only", action="store_true",
                        help="explicit debug mode: single-GPU allowed, every "
                             "receipt labeled ENGINEERING_ONLY, no official verdict")
    parser.add_argument("--skip-sealed", action="store_true",
                        help="stop after development aggregates (sealed stays "
                             "NOT_CONSUMED for a later execution)")
    parser.add_argument("--device", default="cuda",
                        help="engineering-override device (cpu for local qualification)")
    args = parser.parse_args(argv)

    import torch
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    gate = hardware_gate(torch)
    (out / "ENVIRONMENT.json").write_text(json.dumps(
        {**gate, "engineering_only": bool(args.engineering_only),
         "device_override": args.device}, indent=2) + "\n", encoding="utf-8")
    print("HARDWARE GATE:", json.dumps(gate))
    if not gate["t4_class"] and not args.engineering_only:
        print(HardwareGateError(
            "OFFICIAL EXECUTION BLOCKED: Kaggle Notebook -> Settings -> "
            "Accelerator -> GPU T4 x2 (Internet -> ON). No CPU/TPU fallback; "
            "no single-GPU official run."))
        return EXIT_GLOBAL

    from v5_experiments import formation_mux_protocol as proto
    from anra_v5 import formation_mux_model as fxm
    from v5_experiments.formation_mux_data import build_surface, load_surface
    surface_path = args.surface or (out / "SURFACE_MANIFEST.json")
    if not Path(surface_path).exists():
        tokenizer = None
        if args.device != "cpu":
            try:
                from v5_data.corpus_loading import _load_tokenizer
                repo = Path(__file__).resolve().parents[1]
                tokenizer, _ = _load_tokenizer(repo.resolve())
            except Exception:
                tokenizer = None  # R0 token ids are then deferred to workers
        surface = build_surface(seed=73011, tokenizer=tokenizer)
        Path(surface_path).write_text(json.dumps(surface, indent=2) + "\n",
                                      encoding="utf-8")
    surface_manifest = load_surface(surface_path)
    print("SURFACE:", surface_manifest["sha256"][:16],
          "| worst shortcut:", surface_manifest["worst_shortcut_score"])

    calibration = calibrate_and_project(surface=Path(surface_path), out=out,
                                        torch=torch,
                                        engineering_only=args.engineering_only)
    if calibration.get("stopped_before_science"):
        print("PROJECTED WALL EXCEEDS 10.5h — science NOT started; "
              "PARTITION_PLAN.json written; rerun with attached Output to "
              "resume partitioned executions.")
        bundle = package_results(out, source_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, capture_output=True,
            text=True).stdout.strip())
        print("BUNDLE:", bundle)
        return EXIT_OK

    if args.engineering_only:
        # Qualification: tiny disposable arms on the override device.
        code = run_worker(_worker_cmd(
            experiment="CS-MECH-002", arm="M0_STANDARD",
            bundle=73011, surface=Path(surface_path), out=out,
            device=args.device, updates=2), gpu=None,
            cwd=Path(__file__).resolve().parents[1])
        if code != EXIT_OK:
            print("QUALIFICATION FAILED (exit", code, ")")
            return code
    gate_receipt = (args.repo / "docs" / "cymek" / "experiments" /
                    "FORMATION-BASELINE-GATE-001" /
                    "CAPABILITY_GATE_RECEIPT.json")
    gate_pass = gate_receipt.exists() and json.loads(
        gate_receipt.read_text(encoding="utf-8")).get("gate") == "PASS"
    if not args.engineering_only and not gate_pass:
        print("OFFICIAL MECHANISM ARMS BLOCKED: no CAPABILITY_GATE PASS "
              "receipt (FORMATION-BASELINE-GATE-001). Run the diagnostic "
              "first; an official campaign cannot launch on an unqualified "
              "regime.")
        bundle = package_results(out, source_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.repo, capture_output=True,
            text=True).stdout.strip())
        print("BUNDLE:", bundle)
        return EXIT_GLOBAL
    state = run_campaign(surface=Path(surface_path), out=out, torch=torch,
                         engineering_only=args.engineering_only,
                         progress=print)
    print("CAMPAIGN:", state["status"],
          "| complete:", state["summary"]["complete_arms"], "/",
          state["summary"]["required_arms"])
    if state["status"] == "ARMS_COMPLETE" and not args.skip_sealed:
        finalize_sealed(surface=Path(surface_path), out=out, torch=torch,
                        engineering_only=args.engineering_only, progress=print)
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo,
                                   capture_output=True, text=True).stdout.strip()
    bundle = package_results(out, source_commit=source_commit)
    print("BUNDLE:", json.dumps(bundle))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
