"""FORMATION-MUX-001 Amendment-1 Kaggle T4 x2 operator.

This engineering layer pins immutable science commit S2, verifies that the
working scientific files are byte-identical to S2, qualifies the executable,
calibrates both T4s concurrently with non-scientific seeds, executes/resumes
matched seed bundles, keeps sealed sets firewalled, and packages results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v5_experiments import formation_mux_protocol_v2 as proto

SCIENCE_COMMIT = "534dcccfb8f96a30e80d71a30160e6a16eaa1ede"
SCIENCE_FILES = (
    "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1_PREEXECUTION.md",
    "docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION.json",
    "docs/cymek/experiments/CS-MECH-002/PREREGISTRATION_V2.json",
    "docs/cymek/experiments/REP-FORM-003A/PREREGISTRATION_V2.json",
    "anra_v5/formation_mux_model_v2.py",
    "anra_v5/formation_mux_train_v2.py",
    "v5_experiments/formation_mux_protocol_v2.py",
    "v5_experiments/formation_mux_surface_v2.py",
    "tools/formation_mux_001_worker_v2.py",
    "tests/test_formation_mux_001_v2.py",
    "tests/test_formation_mux_surface_v2.py",
)
CAMPAIGN_ROOT = Path("/kaggle/working/FORMATION_MUX_001")
BUNDLE_PATH = Path("/kaggle/working/FORMATION_MUX_001_RESULTS.zip")
EXIT_OK, EXIT_ARM_LOCAL, EXIT_GLOBAL = 0, 3, 4


class GlobalIntegrityError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify_science(repo: Path) -> dict[str, str]:
    """Working science must exactly equal immutable S2 even though operator is later."""
    subprocess.run(["git", "cat-file", "-e", f"{SCIENCE_COMMIT}^{{commit}}"], cwd=repo, check=True)
    hashes: dict[str, str] = {}
    for rel in SCIENCE_FILES:
        expected = subprocess.run(
            ["git", "show", f"{SCIENCE_COMMIT}:{rel}"],
            cwd=repo,
            capture_output=True,
            check=True,
        ).stdout
        actual = (repo / rel).read_bytes()
        if actual != expected:
            raise GlobalIntegrityError(f"science identity mismatch: {rel}")
        hashes[rel] = _sha(actual)
    return hashes


def hardware_receipt(torch: Any) -> dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": p.total_memory / (1024 ** 3),
                }
            )
    ok = (
        len(devices) == 2
        and all("T4" in d["name"] for d in devices)
        and all(14.0 <= d["vram_gb"] <= 17.0 for d in devices)
    )
    return {
        "schema": "anra.formation-mux-hardware/v2",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "devices": devices,
        "official_t4x2": ok,
    }


def require_hardware(torch: Any) -> dict[str, Any]:
    receipt = hardware_receipt(torch)
    if not receipt["official_t4x2"]:
        raise GlobalIntegrityError(
            "OFFICIAL EXECUTION BLOCKED: select Kaggle Notebook -> Settings -> "
            "Accelerator -> GPU T4 x2 and Internet -> ON. Exact official run "
            f"requires two visible T4s; observed {[d['name'] for d in receipt['devices']]}"
        )
    return receipt


def worker_cmd(
    *,
    experiment: str,
    arm: str,
    seed: int,
    surface: Path,
    out: Path,
    engineering_only: bool = False,
    a_updates: int | None = None,
    b_tokens: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "tools.formation_mux_001_worker_v2",
        "--experiment",
        experiment,
        "--arm",
        arm,
        "--seed-bundle",
        str(seed),
        "--surface",
        str(surface),
        "--out",
        str(out),
        "--device",
        "cuda",
    ]
    if engineering_only:
        cmd.append("--engineering-only")
    if a_updates is not None:
        cmd += ["--a-updates-override", str(a_updates)]
    if b_tokens is not None:
        cmd += ["--b-token-budget-override", str(b_tokens)]
    return cmd


def _env_for_gpu(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _spawn(cmd: list[str], *, gpu: int, repo: Path) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=repo, env=_env_for_gpu(gpu))


def _result_path(out: Path, experiment: str, arm: str, seed: int, official: bool) -> Path:
    label = f"S{proto.SEED_BUNDLES.index(seed) + 1}" if official else f"CAL{seed}"
    return out / experiment / arm / label / "ARM_RESULT.json"


def qualify(repo: Path, surface: Path, out: Path) -> dict[str, Any]:
    """CPU/static tests plus simultaneous tiny disposable T4 workers."""
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_formation_mux_001_v2.py",
            "tests/test_formation_mux_surface_v2.py",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    receipt: dict[str, Any] = {
        "pytest_returncode": tests.returncode,
        "pytest_stdout_tail": tests.stdout[-8000:],
        "pytest_stderr_tail": tests.stderr[-4000:],
    }
    if tests.returncode != 0:
        _atomic_json(out / "QUALIFICATION.json", receipt)
        raise GlobalIntegrityError("pre-science pytest qualification failed")

    cmds = [
        worker_cmd(
            experiment=proto.EXPERIMENT_A,
            arm="M0_STANDARD",
            seed=proto.CALIBRATION_SEEDS[0],
            surface=surface,
            out=out / "qualification",
            engineering_only=True,
            a_updates=2,
        ),
        worker_cmd(
            experiment=proto.EXPERIMENT_A,
            arm="M2_EXTRA_FROZEN",
            seed=proto.CALIBRATION_SEEDS[1],
            surface=surface,
            out=out / "qualification",
            engineering_only=True,
            a_updates=2,
        ),
    ]
    procs = [_spawn(cmds[i], gpu=i, repo=repo) for i in (0, 1)]
    codes = [p.wait() for p in procs]
    receipt["dual_worker_codes"] = codes
    if codes != [0, 0]:
        _atomic_json(out / "QUALIFICATION.json", receipt)
        raise GlobalIntegrityError(f"dual-T4 tiny qualification failed: {codes}")
    receipt["status"] = "GPU_E2E_PASS_ENGINEERING_ONLY"
    _atomic_json(out / "QUALIFICATION.json", receipt)
    return receipt


def _read_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GlobalIntegrityError(f"expected worker result missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def calibrate(repo: Path, surface: Path, out: Path) -> dict[str, Any]:
    """Two concurrent rounds: mechanism timing then rendering timing."""
    cal_root = out / "calibration"
    rounds = [
        (
            [
                worker_cmd(experiment=proto.EXPERIMENT_A, arm="M0_STANDARD", seed=proto.CALIBRATION_SEEDS[0], surface=surface, out=cal_root, engineering_only=True, a_updates=120),
                worker_cmd(experiment=proto.EXPERIMENT_A, arm="M1_EXTRA_NO_DECAY", seed=proto.CALIBRATION_SEEDS[1], surface=surface, out=cal_root, engineering_only=True, a_updates=120),
            ],
            [(proto.EXPERIMENT_A, "M0_STANDARD", proto.CALIBRATION_SEEDS[0]), (proto.EXPERIMENT_A, "M1_EXTRA_NO_DECAY", proto.CALIBRATION_SEEDS[1])],
        ),
        (
            [
                worker_cmd(experiment=proto.EXPERIMENT_B, arm="R0_PRODUCTION_BPE", seed=proto.CALIBRATION_SEEDS[2], surface=surface, out=cal_root, engineering_only=True, b_tokens=30_000),
                worker_cmd(experiment=proto.EXPERIMENT_B, arm="R1_ISOMORPHIC_RENDERING", seed=proto.CALIBRATION_SEEDS[3], surface=surface, out=cal_root, engineering_only=True, b_tokens=30_000),
            ],
            [(proto.EXPERIMENT_B, "R0_PRODUCTION_BPE", proto.CALIBRATION_SEEDS[2]), (proto.EXPERIMENT_B, "R1_ISOMORPHIC_RENDERING", proto.CALIBRATION_SEEDS[3])],
        ),
    ]
    results: list[dict[str, Any]] = []
    for cmds, identities in rounds:
        procs = [_spawn(cmds[i], gpu=i, repo=repo) for i in (0, 1)]
        codes = [p.wait() for p in procs]
        if codes != [0, 0]:
            raise GlobalIntegrityError(f"calibration worker failure: {codes}")
        for exp, arm, seed in identities:
            results.append(_read_result(_result_path(cal_root, exp, arm, seed, False)))

    a_results = [r for r in results if r["experiment"] == proto.EXPERIMENT_A]
    b_results = [r for r in results if r["experiment"] == proto.EXPERIMENT_B]

    def event_cost(rows: list[dict[str, Any]], key: str, events: int) -> float:
        vals = []
        for r in rows:
            observed_events = len(r["trace"]) if key == "eval_seconds" else 1
            vals.append(float(r["timing"][key]) / max(observed_events, 1))
        return max(vals) * events

    a_train_per_update = max(float(r["timing"]["train_seconds"]) / max(int(r["updates"]), 1) for r in a_results)
    a_eval_events = math.ceil(proto.A_UPDATES / proto.A_EVAL_EVERY_UPDATES)
    a_ckpt_events = math.ceil(proto.A_UPDATES / proto.A_CHECKPOINT_EVERY_UPDATES)
    a_seconds = (
        a_train_per_update * proto.A_UPDATES
        + event_cost(a_results, "eval_seconds", a_eval_events)
        + event_cost(a_results, "checkpoint_seconds", a_ckpt_events)
    )

    b_train_per_token = max(float(r["timing"]["train_seconds"]) / max(int(r["processed_tokens"]), 1) for r in b_results)
    b_eval_events = math.ceil(proto.B_PROCESSED_TOKEN_BUDGET / proto.B_EVAL_EVERY_TOKENS)
    b_ckpt_events = math.ceil(proto.B_PROCESSED_TOKEN_BUDGET / proto.B_CHECKPOINT_EVERY_TOKENS)
    b_seconds = (
        b_train_per_token * proto.B_PROCESSED_TOKEN_BUDGET
        + event_cost(b_results, "eval_seconds", b_eval_events)
        + event_cost(b_results, "checkpoint_seconds", b_ckpt_events)
    )

    # Each physical T4 owns two matched bundles: 8 mechanism arms + 4 rendering arms.
    projected_worker_seconds = 8 * a_seconds + 4 * b_seconds
    projected_minutes = projected_worker_seconds / 60.0
    partition_required = projected_minutes > proto.PARTITION_THRESHOLD_MINUTES
    half_minutes = projected_minutes / 2.0
    if partition_required and half_minutes > proto.PARTITION_THRESHOLD_MINUTES:
        raise GlobalIntegrityError(
            f"even a two-session seed partition projects {half_minutes:.1f} min > 10.5h; "
            "do not shorten science automatically"
        )
    receipt = {
        "schema": "anra.formation-mux-calibration/v2",
        "calibration_seeds": list(proto.CALIBRATION_SEEDS),
        "official_science_seeds_used": False,
        "sealed_rows_used": False,
        "rounds_concurrent_across_gpu0_gpu1": True,
        "mechanism_arm_projection_seconds": a_seconds,
        "rendering_arm_projection_seconds": b_seconds,
        "projected_full_campaign_minutes": projected_minutes,
        "partition_threshold_minutes": proto.PARTITION_THRESHOLD_MINUTES,
        "partition_required": partition_required,
        "projected_partition_minutes": half_minutes if partition_required else projected_minutes,
        "raw_calibration_results": results,
    }
    _atomic_json(out / "CALIBRATION_RECEIPT.json", receipt)
    if partition_required:
        _atomic_json(
            out / "PARTITION_PLAN.json",
            {
                "schema": "anra.formation-mux-partition-plan/v2",
                "science_protocol_unchanged": True,
                "sessions": [
                    {"session": 1, "seed_bundles": [73011, 73012]},
                    {"session": 2, "seed_bundles": [73013, 73014]},
                ],
                "rule": "each session keeps whole matched bundles on one T4; completed arms are immutable and skipped on resume",
            },
        )
    return receipt


def import_prior_state(out: Path) -> str | None:
    """Copy a prior Kaggle Output tree only before local state exists."""
    if (out / "CAMPAIGN_STATE.json").exists():
        return None
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    candidates = sorted(root.rglob("FORMATION_MUX_001/CAMPAIGN_STATE.json"))
    if not candidates:
        return None
    source = candidates[-1].parent
    shutil.copytree(source, out, dirs_exist_ok=True)
    return str(source)


def _scan_completed(out: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    for experiment, arms in ((proto.EXPERIMENT_A, proto.ARMS_A), (proto.EXPERIMENT_B, proto.ARMS_B)):
        for bundle in proto.SEED_BUNDLES:
            label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
            for arm in arms:
                key = f"{experiment}/{arm}/{label}"
                path = out / experiment / arm / label / "ARM_RESULT.json"
                if path.exists():
                    body = json.loads(path.read_text(encoding="utf-8"))
                    if body.get("status") == "COMPLETE" and body.get("protocol_sha256") == proto.protocol_sha(experiment):
                        states[key] = "COMPLETE"
    return states


def run_campaign(repo: Path, surface: Path, out: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    previous = _scan_completed(out)
    if calibration["partition_required"]:
        first_complete = all(
            any(key.endswith(f"/S{i}") for key in previous) for i in (1, 2)
        )
        allowed = set(proto.SEED_BUNDLES[2:] if first_complete else proto.SEED_BUNDLES[:2])
    else:
        allowed = set(proto.SEED_BUNDLES)

    queues = proto.queue_assignment()
    pending = {
        gpu: [j for j in jobs if j["seed_bundle"] in allowed and f"{j['experiment']}/{j['arm']}/{j['seed_bundle_label']}" not in previous]
        for gpu, jobs in queues.items()
    }
    state = {
        "schema": "anra.formation-mux-state/v2",
        "science_commit": SCIENCE_COMMIT,
        "allowed_seed_bundles_this_session": sorted(allowed),
        "arms": dict(previous),
        "global_failure": None,
        "sealed": {exp: "NOT_CONSUMED" for exp in proto.EXPERIMENTS},
    }
    old_state_path = out / "CAMPAIGN_STATE.json"
    if old_state_path.exists():
        old = json.loads(old_state_path.read_text(encoding="utf-8"))
        state["sealed"].update(old.get("sealed", {}))
    _atomic_json(old_state_path, state)

    stop = False
    while any(pending.values()) and not stop:
        running: dict[int, tuple[dict[str, Any], subprocess.Popen]] = {}
        for gpu in (0, 1):
            q = pending[f"GPU{gpu}"]
            if q:
                job = q.pop(0)
                cmd = worker_cmd(
                    experiment=job["experiment"],
                    arm=job["arm"],
                    seed=job["seed_bundle"],
                    surface=surface,
                    out=out,
                )
                running[gpu] = (job, _spawn(cmd, gpu=gpu, repo=repo))
                print(f"GPU{gpu} <- {job['experiment']}/{job['arm']}/{job['seed_bundle_label']}", flush=True)
        for gpu, (job, proc) in running.items():
            code = proc.wait()
            key = f"{job['experiment']}/{job['arm']}/{job['seed_bundle_label']}"
            if code == EXIT_OK:
                state["arms"][key] = "COMPLETE"
            elif code == EXIT_ARM_LOCAL:
                state["arms"][key] = "ENGINEERING_FAILURE"
            else:
                state["global_failure"] = f"worker exit {code} at {key}"
                stop = True
            _atomic_json(old_state_path, state)

    complete = _scan_completed(out)
    state["arms"].update(complete)
    state["complete_arms"] = len(complete)
    state["required_arms"] = proto.total_official_arms()
    if state["global_failure"]:
        state["status"] = "FAILED_GLOBAL"
    elif len(complete) == proto.total_official_arms():
        state["status"] = "ARMS_COMPLETE"
    else:
        state["status"] = "PARTIAL_SESSION"
        state["resume_instruction"] = "save this Kaggle version/output, attach it to the next run, rerun the same pinned notebook"
    _atomic_json(old_state_path, state)
    return state


def _arm_result(out: Path, experiment: str, arm: str, bundle: int) -> dict[str, Any] | None:
    label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
    path = out / experiment / arm / label / "ARM_RESULT.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def finalize_dev(out: Path, experiment: str) -> dict[str, Any]:
    arms = proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B
    results = {arm: {b: _arm_result(out, experiment, arm, b) for b in proto.SEED_BUNDLES} for arm in arms}
    if any(results[a][b] is None for a in arms for b in proto.SEED_BUNDLES):
        body = {"experiment": experiment, "status": "INCONCLUSIVE", "reason": "matched arms incomplete"}
        _atomic_json(out / experiment / "DEV_AGGREGATE.json", body)
        return body

    if experiment == proto.EXPERIMENT_B:
        mismatch = {
            b: proto.exposure_mismatch(results[arms[0]][b]["processed_tokens"], results[arms[1]][b]["processed_tokens"])
            for b in proto.SEED_BUNDLES
        }
        if any(v > proto.B_EXPOSURE_MISMATCH_TOLERANCE for v in mismatch.values()):
            body = {
                "experiment": experiment,
                "status": "INCONCLUSIVE_EXPOSURE_MISMATCH",
                "exposure_mismatch_fraction": {str(k): v for k, v in mismatch.items()},
                "tolerance": proto.B_EXPOSURE_MISMATCH_TOLERANCE,
            }
            _atomic_json(out / experiment / "DEV_AGGREGATE.json", body)
            return body
        contrasts = (proto.CONTRAST_B,)
    else:
        contrasts = proto.CONTRASTS_A

    contrast_rows = []
    for higher, lower, label in contrasts:
        deltas = {
            b: float(results[higher][b]["formation"]["formation_auc"]) - float(results[lower][b]["formation"]["formation_auc"])
            for b in proto.SEED_BUNDLES
        }
        dev_endpoint = {
            b: float(results[higher][b]["formation"]["endpoint"]) - float(results[lower][b]["formation"]["endpoint"])
            for b in proto.SEED_BUNDLES
        }
        contrast_rows.append(
            {
                "higher": higher,
                "lower": lower,
                "label": label,
                "development_formation_auc_deltas": {str(k): v for k, v in deltas.items()},
                "development_endpoint_gaps_secondary": {str(k): v for k, v in dev_endpoint.items()},
            }
        )
    body = {
        "schema": "anra.formation-mux-dev-aggregate/v2",
        "experiment": experiment,
        "status": "DEVELOPMENT_COMPLETE",
        "primary_family": "identity",
        "contrasts": contrast_rows,
    }
    _atomic_json(out / experiment / "DEV_AGGREGATE.json", body)
    return body


def _sealed_marker(out: Path, experiment: str) -> dict[str, Any]:
    path = out / experiment / "SEALED_MARKER.json"
    if not path.exists():
        return {"state": "NOT_CONSUMED"}
    return json.loads(path.read_text(encoding="utf-8"))


def _set_sealed(out: Path, experiment: str, state: str) -> None:
    path = out / experiment / "SEALED_MARKER.json"
    prior = _sealed_marker(out, experiment).get("state")
    if prior == "STARTED" and state != "COMPLETE":
        raise GlobalIntegrityError(f"sealed marker STARTED without FINAL_RESULT for {experiment}; preserve and fail closed")
    if prior == "COMPLETE" and state != "COMPLETE":
        raise GlobalIntegrityError(f"sealed result already complete for {experiment}")
    _atomic_json(path, {"experiment": experiment, "state": state, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def finalize_sealed(surface_path: Path, out: Path, torch: Any) -> dict[str, Any]:
    from v5_experiments.formation_mux_data import load_surface
    from anra_v5 import formation_mux_train_v2 as train

    surface = load_surface(surface_path)
    sealed_rows = list(surface["splits"]["sealed"])
    device = torch.device("cuda:0")
    finals: dict[str, Any] = {}
    for experiment in proto.EXPERIMENTS:
        final_path = out / experiment / "FINAL_RESULT.json"
        marker = _sealed_marker(out, experiment)
        if marker.get("state") == "COMPLETE" and final_path.exists():
            finals[experiment] = json.loads(final_path.read_text(encoding="utf-8"))
            continue
        if marker.get("state") == "STARTED" and not final_path.exists():
            raise GlobalIntegrityError(f"sealed marker STARTED without FINAL_RESULT for {experiment}")
        dev = finalize_dev(out, experiment)
        if dev.get("status") != "DEVELOPMENT_COMPLETE":
            finals[experiment] = {"verdict": dev.get("status", "INCONCLUSIVE"), "sealed_consumed": False}
            continue
        _set_sealed(out, experiment, "STARTED")
        arms = proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B
        observed: dict[str, dict[int, float]] = {a: {} for a in arms}
        per_family: dict[str, dict[int, Any]] = {a: {} for a in arms}
        for arm in arms:
            for bundle in proto.SEED_BUNDLES:
                label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
                checkpoint = out / experiment / arm / label / "resume.pt"
                model = train.load_model_for_evaluation(
                    checkpoint,
                    experiment=experiment,
                    arm=arm,
                    seed_bundle=bundle,
                    data_manifest_sha256=str(surface["sha256"]),
                    torch=torch,
                    device=device,
                )
                scored = train.evaluate_development(model, sealed_rows, experiment, arm, torch=torch, device=device)
                observed[arm][bundle] = float(scored["identity_exact_valid_eos"])
                per_family[arm][bundle] = scored["per_family"]
                del model
                torch.cuda.empty_cache()

        contrast_results = []
        for contrast in dev["contrasts"]:
            higher, lower = contrast["higher"], contrast["lower"]
            auc = {int(k): float(v) for k, v in contrast["development_formation_auc_deltas"].items()}
            endpoint = {b: observed[higher][b] - observed[lower][b] for b in proto.SEED_BUNDLES}
            verdict = proto.paired_verdict(formation_auc_deltas=auc, sealed_endpoint_gaps=endpoint)
            contrast_results.append({**contrast, "sealed_endpoint_gaps": {str(k): v for k, v in endpoint.items()}, **verdict})
        overall = proto.aggregate_experiment_verdict([r["verdict"] for r in contrast_results])
        final = {
            "schema": "anra.formation-mux-final-result/v2",
            "experiment": experiment,
            "observed_sealed_identity_exact_valid_eos": {a: {str(k): v for k, v in values.items()} for a, values in observed.items()},
            "observed_sealed_per_family": {a: {str(k): v for k, v in values.items()} for a, values in per_family.items()},
            "contrast_results": contrast_results,
            "verdict": overall,
            "claim_ceiling": proto.claim_ceiling(experiment, overall),
            "next_action": proto.NEXT_ACTION[overall],
            "sealed_consumed": True,
        }
        _atomic_json(final_path, final)
        _set_sealed(out, experiment, "COMPLETE")
        finals[experiment] = final
    return finals


def package(out: Path, repo: Path) -> dict[str, Any]:
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()
    with zipfile.ZipFile(BUNDLE_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SCIENCE_COMMIT.txt", SCIENCE_COMMIT + "\n")
        zf.writestr("OPERATOR_HEAD.txt", subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout)
        for path in sorted(out.rglob("*.json")):
            zf.write(path, path.relative_to(out.parent))
    digest = _sha(BUNDLE_PATH.read_bytes())
    Path(str(BUNDLE_PATH) + ".sha256").write_text(f"{digest}  {BUNDLE_PATH.name}\n", encoding="utf-8")
    return {"path": str(BUNDLE_PATH), "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--skip-sealed", action="store_true")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        science_hashes = verify_science(repo)
        environment = require_hardware(torch)
        environment["science_commit"] = SCIENCE_COMMIT
        environment["science_files_sha256"] = science_hashes
        imported = import_prior_state(out)
        environment["imported_prior_state"] = imported
        _atomic_json(out / "ENVIRONMENT.json", environment)

        from v5_data.corpus_loading import _load_tokenizer
        from v5_experiments.formation_mux_surface_v2 import build_official_surface, validate_official_surface
        tokenizer, tokenizer_eval = _load_tokenizer(repo)
        surface_path = out / "SURFACE_MANIFEST.json"
        if surface_path.exists():
            from v5_experiments.formation_mux_data import load_surface
            surface = load_surface(surface_path)
            validate_official_surface(surface)
        else:
            surface = build_official_surface(seed=73011, tokenizer=tokenizer)
            _atomic_json(surface_path, surface)
        _atomic_json(out / "TOKENIZER_RECEIPT.json", {"identity": tokenizer.identity.__dict__ if hasattr(tokenizer.identity, "__dict__") else str(tokenizer.identity), "evaluation": tokenizer_eval})

        qualify(repo, surface_path, out)
        calibration = calibrate(repo, surface_path, out)
        state = run_campaign(repo, surface_path, out, calibration)
        if state["status"] == "ARMS_COMPLETE":
            for experiment in proto.EXPERIMENTS:
                finalize_dev(out, experiment)
            if not args.skip_sealed:
                finalize_sealed(surface_path, out, torch)
        bundle = package(out, repo)
        print("FORMATION-MUX-001 STATUS:", state["status"])
        print("RESULT BUNDLE:", json.dumps(bundle))
        return EXIT_OK
    except Exception as exc:
        failure = {
            "schema": "anra.formation-mux-global-failure/v2",
            "exception": type(exc).__name__,
            "message": str(exc),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(out / "GLOBAL_FAILURE.json", failure)
        try:
            bundle = package(out, repo)
            print("FAILURE BUNDLE:", json.dumps(bundle))
        except Exception:
            pass
        print("GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr)
        return EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
