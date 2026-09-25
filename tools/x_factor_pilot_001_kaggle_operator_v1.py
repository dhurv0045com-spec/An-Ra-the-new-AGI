from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_v5 import x_factor_pilot_model_v1 as model_module
from anra_v5 import x_factor_pilot_train_v1 as train
from v5_experiments import x_factor_pilot_protocol_v1 as protocol

WORKER_MODULE = "tools.x_factor_pilot_001_worker_v1"
BUNDLE_NAME = "X_FACTOR_PILOT_001_RESULTS.zip"
FROZEN_PREFIXES = (
    "anra_v5/formation_mux_",
    "v5_experiments/formation_mux_",
    "v5_model/",
    "v5_training/",
    "v5_objectives/",
    "tools/formation_mux_001_",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def verify_frozen_files(repo: Path) -> dict[str, Any]:
    changed = [line for line in _git(repo, "diff", "--name-only", "HEAD").splitlines() if line]
    violations = [path for path in changed if any(path.startswith(prefix) for prefix in FROZEN_PREFIXES)]
    if violations:
        raise RuntimeError(f"FROZEN_FILE_MUTATION: {violations}")
    return {"changed_paths": changed, "frozen_prefixes": list(FROZEN_PREFIXES), "status": "PASS"}


def hardware_receipt(torch: Any) -> dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "vram_bytes": int(properties.total_memory),
            })
    passed = len(devices) == 2 and all("T4" in device["name"].upper() and 14 * 1024 ** 3 <= device["vram_bytes"] <= 17 * 1024 ** 3 for device in devices)
    return {
        "schema": "anra.x-factor-pilot-hardware/v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": devices,
        "required": "exactly two visible NVIDIA T4 devices",
        "passed": passed,
    }


def require_hardware(torch: Any) -> dict[str, Any]:
    receipt = hardware_receipt(torch)
    if not receipt["passed"]:
        raise RuntimeError(f"T4x2_REQUIRED: observed {receipt['devices']}")
    return receipt


def _environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def worker_command(
    *,
    mode: str,
    arm: str,
    seed: int,
    surface: Path,
    out: Path,
    target_updates: int | None = None,
    stop_after: int | None = None,
    deadline_epoch: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--mode",
        mode,
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--surface",
        str(surface),
        "--out",
        str(out),
        "--device",
        "cuda",
    ]
    if target_updates is not None:
        command.extend(["--target-updates", str(target_updates)])
    if stop_after is not None:
        command.extend(["--stop-after", str(stop_after)])
    if deadline_epoch is not None:
        command.extend(["--deadline-epoch", str(deadline_epoch)])
    return command


def _result_path(out: Path, mode: str, arm: str, seed: int) -> Path:
    return out / mode / arm / protocol.seed_label(seed) / "ARM_RESULT.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"expected result missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_equal(left: Any, right: Any, torch: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            return False
        return all(_tree_equal(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, type(right)) and not isinstance(right, type(left)):
            return False
        if len(left) != len(right):
            return False
        return all(_tree_equal(a, b, torch) for a, b in zip(left, right))
    return left == right


def _complete_result(
    path: Path,
    *,
    mode: str,
    arm: str,
    seed: int,
    surface_sha: str,
    target_updates: int,
) -> bool:
    if not path.exists() or path.is_symlink():
        return False
    checkpoint = path.parent / "resume.pt"
    receipt_path = path.parent / "CHECKPOINT_RECEIPT.json"
    if not checkpoint.exists() or checkpoint.is_symlink() or not receipt_path.exists():
        return False
    try:
        result = _load_json(path)
        receipt = _load_json(receipt_path)
    except Exception:
        return False
    required = {
        "status": "COMPLETE",
        "mode": mode,
        "arm": arm,
        "seed": int(seed),
        "data_manifest_sha256": surface_sha,
        "protocol_sha256": protocol.protocol_sha256(),
        "architecture_id": model_module.architecture_id(arm),
        "target_updates": int(target_updates),
        "updates": int(target_updates),
    }
    if any(result.get(key) != value for key, value in required.items()):
        return False
    identity_keys = ("mode", "arm", "seed", "data_manifest_sha256", "protocol_sha256", "architecture_id", "target_updates")
    if any(receipt.get(key) != required[key] for key in identity_keys):
        return False
    if result.get("resume_checkpoint_sha256") != receipt.get("sha256"):
        return False
    if receipt.get("sha256") != file_sha256(checkpoint) or receipt.get("current_model_sha256") != result.get("final_model_sha256"):
        return False
    if len(str(result.get("data_order_sha256", ""))) != 64 or not isinstance(result.get("formation"), dict) or not isinstance(result.get("trace"), list):
        return False
    eligible_from = protocol.OFFICIAL_ELIGIBLE_FROM if mode == "official" else protocol.CONTROL_ELIGIBLE_FROM
    if result["formation"] != train.formation_summary(result["trace"], eligible_from):
        return False
    return True


def _spawn(command: list[str], *, gpu: int, repo: Path, log_path: Path) -> tuple[subprocess.Popen[Any], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=_environment(gpu),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, handle


def _state_write(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(path, state)


def run_jobs(
    *,
    repo: Path,
    out: Path,
    surface: Path,
    surface_sha: str,
    mode: str,
    jobs: list[dict[str, Any]],
    state_path: Path,
    started: float,
    target_updates: int | None = None,
) -> dict[str, Any]:
    state = _load_json(state_path) if state_path.exists() else {
        "schema": "anra.x-factor-pilot-campaign-state/v1",
        "campaign": protocol.CAMPAIGN,
        "protocol_sha256": protocol.protocol_sha256(),
        "completed": {},
        "failures": {},
        "status": "RUNNING",
    }
    state.setdefault("completed", {})
    state.setdefault("failures", {})
    expected_updates = int(target_updates or (protocol.CONTROL_UPDATES if mode == "control" else protocol.OFFICIAL_UPDATES))
    pending: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for job in jobs:
        result_path = _result_path(out, mode, job["arm"], int(job["seed"]))
        if _complete_result(result_path, mode=mode, arm=job["arm"], seed=int(job["seed"]), surface_sha=surface_sha, target_updates=expected_updates):
            state["completed"][f"{mode}/{job['arm']}/{protocol.seed_label(int(job['seed']))}"] = "COMPLETE"
            continue
        pending[int(job["gpu"])].append(job)
    state["pending"] = {str(gpu): [job["arm"] for job in queue] for gpu, queue in pending.items()}
    if pending[0] or pending[1]:
        state["status"] = "RUNNING"
    _state_write(state_path, state)
    stopped = False
    running: dict[int, tuple[dict[str, Any], subprocess.Popen[Any], Any, Path]] = {}
    while (pending[0] or pending[1] or running) and not stopped:
        wall_guard = time.monotonic() - started >= protocol.MAX_SESSION_SECONDS - protocol.SESSION_GUARD_RESERVE_SECONDS
        if wall_guard and not running:
            state["status"] = "PARTIAL_SESSION"
            state["wall_guard_triggered"] = True
            _state_write(state_path, state)
            break
        for gpu in (0, 1):
            if stopped or gpu in running or not pending[gpu] or wall_guard:
                continue
            job = pending[gpu].pop(0)
            key = f"{mode}/{job['arm']}/{protocol.seed_label(int(job['seed']))}"
            state["failures"].pop(key, None)
            log_path = out / "logs" / mode / job["arm"] / f"{protocol.seed_label(int(job['seed']))}.log"
            deadline_epoch = None
            if mode == "official":
                remaining = protocol.MAX_SESSION_SECONDS - (time.monotonic() - started)
                deadline_epoch = time.time() + max(60.0, remaining - 60.0)
            command = worker_command(
                mode=mode,
                arm=job["arm"],
                seed=int(job["seed"]),
                surface=surface,
                out=out,
                target_updates=target_updates,
                stop_after=job.get("stop_after"),
                deadline_epoch=deadline_epoch,
            )
            process, handle = _spawn(command, gpu=gpu, repo=repo, log_path=log_path)
            running[gpu] = (job, process, handle, log_path)
            print(f"GPU{gpu} <- {key}", flush=True)
        completed_any = False
        for gpu in list(running):
            job, process, handle, _ = running[gpu]
            code = process.poll()
            if code is None:
                continue
            handle.close()
            key = f"{mode}/{job['arm']}/{protocol.seed_label(int(job['seed']))}"
            if code == 0:
                result = _load_json(_result_path(out, mode, job["arm"], int(job["seed"])))
                if result.get("status") == "COMPLETE":
                    state["completed"][key] = "COMPLETE"
                else:
                    state["completed"][key] = "PARTIAL"
                    stopped = True
            else:
                state["failures"][key] = code
                stopped = True
            del running[gpu]
            completed_any = True
            state["pending"] = {str(index): [item["arm"] for item in queue] for index, queue in pending.items()}
            _state_write(state_path, state)
        if running and not completed_any:
            time.sleep(0.5)
    for gpu, (job, process, handle, log_path) in list(running.items()):
        code = process.wait()
        handle.close()
        if code != 0:
            state["failures"][f"{mode}/{job['arm']}/{protocol.seed_label(int(job['seed']))}"] = code
            stopped = True
    if stopped:
        state["status"] = "FAILED" if state["failures"] else "PARTIAL_SESSION"
    elif all(_complete_result(_result_path(out, mode, job["arm"], int(job["seed"])), mode=mode, arm=job["arm"], seed=int(job["seed"]), surface_sha=surface_sha, target_updates=expected_updates) for job in jobs):
        state["status"] = "ARMS_COMPLETE"
    else:
        state["status"] = "PARTIAL_SESSION"
    _state_write(state_path, state)
    return state


def build_surfaces(repo: Path, out: Path) -> tuple[Path, Path, dict[str, Any]]:
    from v5_data.corpus_loading import _load_tokenizer
    from v5_experiments.formation_mux_surface_v5 import build_public_surface, load_public_surface

    tokenizer, tokenizer_evaluation = _load_tokenizer(repo)
    public_path = out / "PUBLIC_SURFACE_MANIFEST.json"
    if public_path.exists():
        public = load_public_surface(public_path)
        if public.get("seed") != protocol.SURFACE_SEED or public.get("tokenizer_artifact_sha256") != tokenizer.identity.artifact_sha256:
            raise RuntimeError("PUBLIC_SURFACE_IDENTITY_MISMATCH")
    else:
        public = build_public_surface(seed=protocol.SURFACE_SEED, tokenizer=tokenizer)
        atomic_json(public_path, public)
    for row in public["splits"]["training"] + public["splits"]["development"]:
        if "r0_prompt_ids" not in row or "r0_answer_ids" not in row:
            raise RuntimeError("PUBLIC_R0_RENDERING_MISSING")
        r0 = [int(value) for value in row["r0_prompt_ids"] + row["r0_answer_ids"]]
        if not r0 or min(r0) < 0 or max(r0) >= protocol.PHYSICAL_VOCAB or r0[0] != 2 or r0[-1] != 3:
            raise RuntimeError("PUBLIC_R0_RENDERING_INVALID")
    control_path = out / "POSITIVE_CONTROL_MANIFEST.json"
    expected_control = protocol.build_positive_control_surface()
    if control_path.exists():
        control = protocol.load_positive_control_surface(control_path)
        if control["sha256"] != expected_control["sha256"]:
            raise RuntimeError("POSITIVE_CONTROL_SURFACE_IDENTITY_MISMATCH")
    else:
        atomic_json(control_path, expected_control)
    receipt = {
        "schema": "anra.x-factor-pilot-surface-receipt/v1",
        "tokenizer_vocabulary": int(tokenizer.identity.vocabulary_size),
        "tokenizer_artifact_sha256": tokenizer.identity.artifact_sha256,
        "tokenizer_trainer_config_sha256": tokenizer.identity.trainer_config_sha256,
        "tokenizer_evaluation": tokenizer_evaluation,
        "public_surface_sha256": public["sha256"],
        "control_surface_sha256": protocol.load_positive_control_surface(control_path)["sha256"],
        "sealed_rows_persisted": False,
    }
    atomic_json(out / "SURFACE_RECEIPT.json", receipt)
    return public_path, control_path, receipt


def qualify(repo: Path, out: Path, control_surface: Path) -> dict[str, Any]:
    existing = out / "QUALIFICATION.json"
    if existing.exists():
        previous = _load_json(existing)
        if (
            previous.get("status") == "PASS"
            and previous.get("protocol_sha256") == protocol.protocol_sha256()
            and previous.get("source_commit") == _git(repo, "rev-parse", "HEAD")
            and previous.get("operator_blob") == _git(repo, "hash-object", "tools/x_factor_pilot_001_kaggle_operator_v1.py")
        ):
            return previous
    files = [
        "v5_experiments/x_factor_pilot_protocol_v1.py",
        "anra_v5/x_factor_pilot_model_v1.py",
        "anra_v5/x_factor_pilot_train_v1.py",
        "tools/x_factor_pilot_001_worker_v1.py",
        "tools/x_factor_pilot_001_kaggle_operator_v1.py",
        "tests/test_x_factor_pilot_v1.py",
    ]
    compile_run = subprocess.run([sys.executable, "-m", "py_compile", *files], cwd=repo, capture_output=True, text=True)
    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_x_factor_pilot_v1.py"], cwd=repo, capture_output=True, text=True)
    receipt: dict[str, Any] = {
        "schema": "anra.x-factor-pilot-qualification/v1",
        "protocol_sha256": protocol.protocol_sha256(),
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "operator_blob": _git(repo, "hash-object", "tools/x_factor_pilot_001_kaggle_operator_v1.py"),
        "compile_returncode": compile_run.returncode,
        "compile_stderr": compile_run.stderr[-8000:],
        "pytest_returncode": test_run.returncode,
        "pytest_stdout": test_run.stdout[-12000:],
        "pytest_stderr": test_run.stderr[-8000:],
    }
    if compile_run.returncode != 0 or test_run.returncode != 0:
        atomic_json(out / "QUALIFICATION.json", receipt)
        raise RuntimeError("CPU_STATIC_QUALIFICATION_FAILED")
    canary_out = out / "resume_canary"
    reference_out = out / "resume_reference"
    for path in (canary_out, reference_out):
        if path.exists():
            shutil.rmtree(path)
    canary_seed_label = protocol.seed_label(protocol.CALIBRATION_SEED)
    first_command = worker_command(
        mode="canary",
        arm=protocol.PRIMARY_ARM,
        seed=protocol.CALIBRATION_SEED,
        surface=control_surface,
        out=canary_out,
        target_updates=4,
        stop_after=2,
    )
    first = subprocess.run(first_command, cwd=repo, env=_environment(0))
    first_receipt = _load_json(canary_out / "canary" / protocol.PRIMARY_ARM / canary_seed_label / "CHECKPOINT_RECEIPT.json")
    atomic_json(out / "RESUME_CANARY_STEP2_RECEIPT.json", first_receipt)
    second_command = worker_command(
        mode="canary",
        arm=protocol.PRIMARY_ARM,
        seed=protocol.CALIBRATION_SEED,
        surface=control_surface,
        out=canary_out,
        target_updates=4,
        stop_after=4,
    )
    second = subprocess.run(second_command, cwd=repo, env=_environment(0))
    reference_command = worker_command(
        mode="canary",
        arm=protocol.PRIMARY_ARM,
        seed=protocol.CALIBRATION_SEED,
        surface=control_surface,
        out=reference_out,
        target_updates=4,
        stop_after=4,
    )
    reference = subprocess.run(reference_command, cwd=repo, env=_environment(0))
    import torch
    checkpoint = canary_out / "canary" / protocol.PRIMARY_ARM / canary_seed_label / "resume.pt"
    reference_checkpoint = reference_out / "canary" / protocol.PRIMARY_ARM / canary_seed_label / "resume.pt"
    body = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference_body = torch.load(reference_checkpoint, map_location="cpu", weights_only=True)
    expected_sha = first_receipt["sha256"]
    state_fields = ("updates", "processed_tokens", "supervised_tokens", "stream_cursor", "trace", "clip_events", "last_eval_axis", "last_checkpoint_axis")
    exact = {
        "model": _tree_equal(body.get("model"), reference_body.get("model"), torch),
        "optimizer": _tree_equal(body.get("optimizer"), reference_body.get("optimizer"), torch),
        "cpu_rng": _tree_equal(body.get("cpu_rng_state"), reference_body.get("cpu_rng_state"), torch),
        "cuda_rng": _tree_equal(body.get("cuda_rng_state"), reference_body.get("cuda_rng_state"), torch),
        "state": all(body.get(key) == reference_body.get(key) for key in state_fields),
        "lineage": body.get("resume_checkpoint_sha256") == expected_sha,
    }
    canary = {
        "schema": "anra.x-factor-pilot-resume-canary/v1",
        "first_returncode": first.returncode,
        "second_returncode": second.returncode,
        "reference_returncode": reference.returncode,
        "checkpoint_sha256": file_sha256(checkpoint),
        "reference_checkpoint_sha256": file_sha256(reference_checkpoint),
        "resumed_from_expected_sha256": body.get("resume_checkpoint_sha256"),
        "expected_sha256": expected_sha,
        "exact_components": exact,
        "status": "PASS" if first.returncode == 0 and second.returncode == 0 and reference.returncode == 0 and all(exact.values()) else "FAIL",
    }
    atomic_json(out / "RESUME_CANARY_RESULT.json", canary)
    receipt["resume_canary"] = canary
    receipt["status"] = "PASS" if canary["status"] == "PASS" else "FAIL"
    atomic_json(out / "QUALIFICATION.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError("RESUME_CANARY_FAILED")
    return receipt


def storage_preflight(out: Path, repo: Path) -> dict[str, Any]:
    canary_checkpoints = list((out / "resume_canary").rglob("resume.pt"))
    if not canary_checkpoints:
        raise RuntimeError("STORAGE_PREFLIGHT_MISSING_CANARY_CHECKPOINT")
    canary_bytes = max(path.stat().st_size for path in canary_checkpoints)
    maximum = int(canary_bytes * (protocol.MAX_TRAINABLE_PARAMETERS / protocol.LEV_TRAINABLE_PARAMETERS) * 1.25)
    target = out.parent if out.parent.exists() else out
    usage = shutil.disk_usage(target)
    slots = len(protocol.ARMS) * (len(protocol.MODEL_SEEDS) + 1)
    required = maximum * slots + protocol.STORAGE_RESERVE_BYTES
    receipt = {
        "schema": "anra.x-factor-pilot-storage-preflight/v1",
        "canary_checkpoint_bytes": canary_bytes,
        "estimated_max_checkpoint_bytes": maximum,
        "parameter_ratio_basis": [protocol.LEV_TRAINABLE_PARAMETERS, protocol.MAX_TRAINABLE_PARAMETERS],
        "remaining_slots": slots,
        "reserve_bytes": protocol.STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "free_bytes": usage.free,
        "pass": usage.free >= required,
    }
    atomic_json(out / "STORAGE_PREFLIGHT.json", receipt)
    if not receipt["pass"]:
        raise RuntimeError("STORAGE_PREFLIGHT_FAILED")
    return receipt


def _control_jobs() -> list[dict[str, Any]]:
    return [
        {"gpu": 0, "arm": arm, "seed": protocol.CONTROL_SEED}
        for arm in protocol.ARMS
    ]


def _official_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(protocol.MODEL_SEEDS):
        gpu = seed_index % 2
        for arm in protocol.ARMS:
            jobs.append({"gpu": gpu, "arm": arm, "seed": seed})
    return jobs


def _collect_results(out: Path, mode: str, arm: str, seed: int) -> dict[str, Any]:
    return _load_json(_result_path(out, mode, arm, seed))


def positive_control(out: Path, repo: Path, control_surface: Path, state_path: Path, started: float) -> dict[str, Any]:
    state = run_jobs(
        repo=repo,
        out=out,
        surface=control_surface,
        surface_sha=protocol.load_positive_control_surface(control_surface)["sha256"],
        mode="control",
        jobs=_control_jobs(),
        state_path=state_path,
        started=started,
    )
    results = {}
    for arm in protocol.ARMS:
        path = _result_path(out, "control", arm, protocol.CONTROL_SEED)
        if path.exists():
            results[arm] = _load_json(path)
    gate = protocol.positive_control_gate(results)
    if state.get("status") != "ARMS_COMPLETE":
        gate = {"status": "PARTIAL", "checks": gate.get("checks", {}), "campaign_state": state.get("status")}
    gate["schema"] = "anra.x-factor-pilot-positive-control-result/v1"
    gate["protocol_sha256"] = protocol.protocol_sha256()
    gate["campaign_state"] = state.get("status")
    atomic_json(out / "POSITIVE_CONTROL_RESULT.json", gate)
    return gate


def development_aggregate(out: Path) -> dict[str, Any]:
    results = {
        arm: {seed: _collect_results(out, "official", arm, seed) for seed in protocol.MODEL_SEEDS}
        for arm in protocol.ARMS
    }
    for seed in protocol.MODEL_SEEDS:
        if len({int(results[arm][seed]["processed_tokens"]) for arm in protocol.ARMS}) != 1 or len({int(results[arm][seed]["supervised_tokens"]) for arm in protocol.ARMS}) != 1 or len({str(results[arm][seed]["data_order_sha256"]) for arm in protocol.ARMS}) != 1:
            raise RuntimeError(f"OFFICIAL_EXPOSURE_OR_ORDER_MISMATCH:{seed}")
    contrasts = []
    for higher, lower in (("UNTIED_DENSE", "TIED_DENSE"), ("LEV_UNTIED", "TIED_DENSE"), ("LEV_UNTIED", "UNTIED_DENSE")):
        deltas = {
            seed: float(results[higher][seed]["formation"]["formation_auc"]) - float(results[lower][seed]["formation"]["formation_auc"])
            for seed in protocol.MODEL_SEEDS
        }
        endpoints = {
            seed: float(results[higher][seed]["formation"]["endpoint"]) - float(results[lower][seed]["formation"]["endpoint"])
            for seed in protocol.MODEL_SEEDS
        }
        contrasts.append({"higher": higher, "lower": lower, "formation_auc_deltas": deltas, "endpoint_deltas": endpoints})
    body = {
        "schema": "anra.x-factor-pilot-development/v1",
        "protocol_sha256": protocol.protocol_sha256(),
        "status": "DEVELOPMENT_COMPLETE",
        "contrasts": contrasts,
        "sealed_consumed": False,
    }
    atomic_json(out / "DEVELOPMENT_RESULT.json", body)
    return body


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _sealed_marker(out: Path) -> dict[str, Any]:
    path = out / "SEALED_MARKER.json"
    return _load_json(path) if path.exists() else {"state": "NOT_CONSUMED"}


def _set_sealed(out: Path, state: str) -> None:
    atomic_json(out / "SEALED_MARKER.json", {
        "schema": "anra.x-factor-pilot-sealed-marker/v1",
        "state": state,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def finalize_sealed(out: Path, public_path: Path, repo: Path, torch: Any) -> dict[str, Any]:
    final_path = out / "FINAL_RESULT.json"
    plan_path = out / "SEALED_PLAN.json"
    score_root = out / "sealed_scores"
    from v5_data.corpus_loading import _load_tokenizer
    from v5_experiments.formation_mux_surface_v5 import load_public_surface, regenerate_sealed_rows
    public = load_public_surface(public_path)
    if public.get("seed") != protocol.SURFACE_SEED:
        raise RuntimeError("SEALED_PUBLIC_SURFACE_IDENTITY_MISMATCH")
    development_path = out / "DEVELOPMENT_RESULT.json"
    development = _load_json(development_path)
    checkpoint_inventory = []
    for arm in protocol.ARMS:
        for seed in protocol.MODEL_SEEDS:
            result_path = _result_path(out, "official", arm, seed)
            checkpoint = result_path.parent / "resume.pt"
            checkpoint_inventory.append({"arm": arm, "seed": seed, "result": str(result_path), "checkpoint_sha256": file_sha256(checkpoint)})
    plan_body = {
        "schema": "anra.x-factor-pilot-sealed-plan/v1",
        "protocol_sha256": protocol.protocol_sha256(),
        "public_surface_sha256": public["sha256"],
        "sealed_commitment": public["sealed_commitments"]["CS-MECH-002"],
        "development_result_sha256": file_sha256(development_path),
        "checkpoint_inventory": checkpoint_inventory,
    }
    plan_body["sha256"] = _json_sha(plan_body)
    marker = _sealed_marker(out)
    if marker.get("state") == "COMPLETE":
        if not final_path.exists():
            raise RuntimeError("SEALED_COMPLETE_WITHOUT_FINAL")
        final = _load_json(final_path)
        if final.get("protocol_sha256") != protocol.protocol_sha256() or final.get("sealed_plan_sha256") != plan_body["sha256"]:
            raise RuntimeError("SEALED_FINAL_IDENTITY_MISMATCH")
        return final
    if marker.get("state") == "STARTED":
        if not plan_path.exists():
            raise RuntimeError("SEALED_STARTED_WITHOUT_PLAN")
        saved_plan = _load_json(plan_path)
        saved_body = {key: value for key, value in saved_plan.items() if key != "sha256"}
        if saved_plan.get("sha256") != _json_sha(saved_body) or saved_plan.get("protocol_sha256") != protocol.protocol_sha256() or saved_plan.get("public_surface_sha256") != public["sha256"] or saved_plan.get("checkpoint_inventory") != checkpoint_inventory:
            raise RuntimeError("SEALED_PLAN_IDENTITY_MISMATCH")
        plan_body = saved_plan
        if final_path.exists():
            final = _load_json(final_path)
            if final.get("sealed_plan_sha256") != plan_body["sha256"]:
                raise RuntimeError("SEALED_FINAL_PLAN_MISMATCH")
            _set_sealed(out, "COMPLETE")
            return final
    else:
        atomic_json(plan_path, plan_body)
        _set_sealed(out, "STARTED")
    tokenizer, _ = _load_tokenizer(repo)
    observed: dict[str, dict[str, float]] = {arm: {} for arm in protocol.ARMS}
    sealed_rows = None
    for arm in protocol.ARMS:
        for seed in protocol.MODEL_SEEDS:
            score_path = score_root / f"{arm}_{protocol.seed_label(seed)}.json"
            if score_path.exists():
                score = _load_json(score_path)
                if score.get("protocol_sha256") != protocol.protocol_sha256() or score.get("sealed_plan_sha256") != plan_body["sha256"] or score.get("arm") != arm or int(score.get("seed", -1)) != seed:
                    raise RuntimeError("SEALED_SCORE_IDENTITY_MISMATCH")
                observed[arm][str(seed)] = float(score["identity_exact_valid_eos"])
                continue
            if sealed_rows is None:
                sealed_rows = regenerate_sealed_rows(public_manifest=public, tokenizer=tokenizer, experiment="CS-MECH-002")
            checkpoint = _result_path(out, "official", arm, seed).parent / "resume.pt"
            model = train.load_model_for_evaluation(
                checkpoint,
                mode="official",
                arm=arm,
                seed=seed,
                data_manifest_sha256=public["sha256"],
                torch=torch,
                device=torch.device("cuda:0"),
            )
            scored = train.evaluate_development(model, sealed_rows, mode="official", torch=torch, device=torch.device("cuda:0"))
            value = float(scored["identity_exact_valid_eos"])
            observed[arm][str(seed)] = value
            atomic_json(score_path, {
                "schema": "anra.x-factor-pilot-sealed-score/v1",
                "protocol_sha256": protocol.protocol_sha256(),
                "sealed_plan_sha256": plan_body["sha256"],
                "public_surface_sha256": public["sha256"],
                "sealed_commitment": public["sealed_commitments"]["CS-MECH-002"],
                "arm": arm,
                "seed": seed,
                "identity_exact_valid_eos": value,
                "raw_sealed_rows_persisted": False,
            })
            del model
            torch.cuda.empty_cache()
    verdicts: dict[str, str] = {}
    contrast_results = []
    for row in development["contrasts"]:
        higher, lower = row["higher"], row["lower"]
        auc = {int(seed): float(value) for seed, value in row["formation_auc_deltas"].items()}
        endpoint = {int(seed): float(observed[higher][str(seed)] - observed[lower][str(seed)]) for seed in protocol.MODEL_SEEDS}
        verdict = protocol.paired_verdict(formation_auc_deltas=auc, sealed_endpoint_gaps=endpoint)
        key = f"{higher}_vs_{lower}"
        verdicts[key] = verdict["verdict"]
        contrast_results.append({"higher": higher, "lower": lower, **verdict})
    primary_key = f"{protocol.PRIMARY_ARM}_vs_TIED_DENSE"
    if primary_key not in verdicts:
        raise RuntimeError("PRIMARY_CONTRAST_MISSING")
    final = {
        "schema": "anra.x-factor-pilot-final/v1",
        "protocol_sha256": protocol.protocol_sha256(),
        "sealed_plan_sha256": plan_body["sha256"],
        "public_surface_sha256": public["sha256"],
        "sealed_commitment": public["sealed_commitments"]["CS-MECH-002"],
        "development_result_sha256": file_sha256(development_path),
        "status": "COMPLETE",
        "primary_arm": protocol.PRIMARY_ARM,
        "observed_sealed_identity_exact_valid_eos": observed,
        "contrast_results": contrast_results,
        "verdict": verdicts[primary_key],
        "diagnostic_verdicts": verdicts,
        "sealed_consumed": True,
        "raw_sealed_rows_persisted": False,
        "claim_ceiling": "Formation-Mux V24576 pilot evidence only; no production, scale, cognition, or AGI claim.",
    }
    atomic_json(final_path, final)
    _set_sealed(out, "COMPLETE")
    del sealed_rows
    return final


def package(out: Path, repo: Path) -> dict[str, Any]:
    bundle = out.parent / BUNDLE_NAME
    if bundle.exists():
        bundle.unlink()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("PROTOCOL.json", json.dumps(protocol.protocol_payload(), indent=2, sort_keys=True) + "\n")
        archive.writestr("HEAD.txt", _git(repo, "rev-parse", "HEAD") + "\n")
        archive.writestr("CHECKPOINT_POLICY.txt", "resume.pt files remain in the complete Kaggle working tree and are required for exact resume\n")
        for path in sorted(out.rglob("*.json")):
            archive.write(path, path.relative_to(out.parent))
        for path in sorted(out.rglob("*.log")):
            archive.write(path, Path("RUNTIME_LOGS") / path.relative_to(out))
    digest = file_sha256(bundle)
    Path(str(bundle) + ".sha256").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
    return {"path": str(bundle), "sha256": digest, "checkpoints_included": False, "raw_sealed_rows_included": False}


def import_prior_output(out: Path) -> str | None:
    if out.exists() and any(out.iterdir()):
        return None
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    valid: list[Path] = []
    campaign_like: list[Path] = []
    for state_path in input_root.rglob("CAMPAIGN_STATE.json"):
        root = state_path.parent
        if root.is_symlink() or root.name != "X_FACTOR_PILOT_001":
            continue
        campaign_like.append(root)
        try:
            state = _load_json(state_path)
        except Exception:
            continue
        if state.get("schema") == "anra.x-factor-pilot-campaign-state/v1" and state.get("campaign") == protocol.CAMPAIGN and state.get("protocol_sha256") == protocol.protocol_sha256() and not (root / "GLOBAL_FAILURE.json").exists():
            valid.append(root)
    if not valid:
        if campaign_like:
            raise RuntimeError("PRIOR_OUTPUT_PRESENT_BUT_INVALID")
        return None
    if len(valid) != 1:
        raise RuntimeError("PRIOR_OUTPUT_AMBIGUOUS")
    source = valid[0]
    shutil.copytree(source, out, dirs_exist_ok=True)
    return str(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/X_FACTOR_PILOT_001"))
    parser.add_argument("--skip-sealed", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    out = args.out.resolve()
    started = time.monotonic()
    try:
        import torch
        if not repo.exists():
            raise RuntimeError("repository checkout missing")
        imported = import_prior_output(out)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "GLOBAL_FAILURE.json").exists():
            raise RuntimeError("UNRESOLVED_GLOBAL_FAILURE")
        frozen = verify_frozen_files(repo)
        hardware = require_hardware(torch)
        public_path, control_path, surface_receipt = build_surfaces(repo, out)
        qualification = qualify(repo, out, control_path)
        storage_preflight(out, repo)
        state_path = out / "CAMPAIGN_STATE.json"
        gate = positive_control(out, repo, control_path, state_path, started)
        if gate.get("status") != "PASS":
            package(out, repo)
            print(f"X-FACTOR-PILOT-001 STATUS: {gate.get('status')}", flush=True)
            return 0
        state = run_jobs(
            repo=repo,
            out=out,
            surface=public_path,
            surface_sha=json.loads(public_path.read_text(encoding="utf-8"))["sha256"],
            mode="official",
            jobs=_official_jobs(),
            state_path=state_path,
            started=started,
        )
        if state.get("status") != "ARMS_COMPLETE":
            package(out, repo)
            print(f"X-FACTOR-PILOT-001 STATUS: {state.get('status')}", flush=True)
            return 0
        development_aggregate(out)
        if not args.skip_sealed:
            finalize_sealed(out, public_path, repo, torch)
        final_state = {
            "schema": "anra.x-factor-pilot-campaign-state/v1",
            "campaign": protocol.CAMPAIGN,
            "protocol_sha256": protocol.protocol_sha256(),
            "status": "COMPLETE",
            "positive_control": gate["status"],
            "official_arms": len(protocol.ARMS) * len(protocol.MODEL_SEEDS),
            "sealed_consumed": not args.skip_sealed,
            "head": _git(repo, "rev-parse", "HEAD"),
            "frozen_files": frozen,
            "hardware": hardware,
            "surface_receipt": surface_receipt,
            "qualification": qualification["status"],
            "imported_prior_output": imported,
        }
        atomic_json(state_path, final_state)
        bundle = package(out, repo)
        print("X-FACTOR-PILOT-001 STATUS: COMPLETE", flush=True)
        print("RESULT BUNDLE:", json.dumps(bundle), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema": "anra.x-factor-pilot-global-failure/v1",
            "protocol_sha256": protocol.protocol_sha256(),
            "exception": type(exc).__name__,
            "message": str(exc),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            atomic_json(out / "GLOBAL_FAILURE.json", failure)
            package(out, repo)
        except Exception:
            pass
        print("X-FACTOR-PILOT-001 GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr, flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
