"""Canonical FORMATION-MUX-001 Kaggle operator — Science S4, ops v7.

Properties:
- immutable Science-S4 verification;
- exact Kaggle T4 x2 hardware gate;
- public training/development manifest only (raw sealed rows never persisted);
- two independent matched-bundle GPU workers, no DDP;
- concurrent non-scientific calibration;
- measured 9.5h campaign projection gate plus an actual elapsed-time launch guard;
- exact-resume operational partitioning;
- one-shot post-development sealed regeneration/verification in coordinator memory;
- append-only worker logs and safe result/failure packaging.
"""

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

from tools import formation_mux_001_kaggle_operator_v2 as base
from tools import formation_mux_001_kaggle_operator_v3 as partition
from v5_experiments import formation_mux_protocol_v4 as proto

SCIENCE_COMMIT = "ad25f6944cdd5ac5f47a6bf3a667322cb985314b"
OPERATOR_NAME = "tools/formation_mux_001_kaggle_operator_v7.py"
CAMPAIGN_ROOT = Path("/kaggle/working/FORMATION_MUX_001")
BUNDLE_NAME = "FORMATION_MUX_001_RESULTS.zip"
OPS_SAFE_CAMPAIGN_MINUTES = 570.0   # 9.5h projected science work
MAX_OPERATOR_WALL_MINUTES = 660.0   # 11h from operator entry
FINALIZATION_PACKAGE_RESERVE_MINUTES = 45.0
ARM_RUNTIME_SAFETY_FACTOR = 1.25

SCIENCE_FILES = (
    "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1_PREEXECUTION.md",
    "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1B_PREEXECUTION_CLIP_ISOLATION.md",
    "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1C_PREEXECUTION_SEALED_FIREWALL.md",
    "docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION_V4.json",
    "docs/cymek/experiments/CS-MECH-002/PREREGISTRATION_V4.json",
    "docs/cymek/experiments/REP-FORM-003A/PREREGISTRATION_V4.json",
    "anra_v5/formation_mux_model_v2.py",
    "anra_v5/formation_mux_model_v3.py",
    "anra_v5/formation_mux_train_v2.py",
    "anra_v5/formation_mux_train_v4.py",
    "v5_experiments/formation_mux_data.py",
    "v5_experiments/formation_mux_surface_v2.py",
    "v5_experiments/formation_mux_surface_v4.py",
    "v5_experiments/formation_mux_protocol_v2.py",
    "v5_experiments/formation_mux_protocol_v3.py",
    "v5_experiments/formation_mux_protocol_v4.py",
    "tools/formation_mux_001_worker_v4.py",
    "tests/test_formation_mux_001_v2.py",
    "tests/test_formation_mux_001_v3.py",
    "tests/test_formation_mux_001_v4.py",
    "tests/test_formation_mux_surface_v2.py",
    "v5_model/core.py",
    "v5_training/production_backend.py",
    "v5_training/optimizer.py",
    "v5_training/state.py",
    "v5_objectives/causal_lm.py",
    "v5_data/corpus_loading.py",
    "artifacts/e1/local_tournament/result.json",
    "artifacts/e1/local_tournament/tokenizer-24576.json.gz",
)

# Rebind the audited engineering helpers to Science S4.
base.SCIENCE_COMMIT = SCIENCE_COMMIT
base.SCIENCE_FILES = SCIENCE_FILES
base.proto = proto
partition.SCIENCE_COMMIT = SCIENCE_COMMIT
partition.proto = proto


def worker_cmd(
    *, experiment: str, arm: str, seed: int, surface: Path, out: Path,
    engineering_only: bool = False, a_updates: int | None = None,
    b_tokens: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "tools.formation_mux_001_worker_v4",
        "--experiment", experiment,
        "--arm", arm,
        "--seed-bundle", str(seed),
        "--surface", str(surface),
        "--out", str(out),
        "--device", "cuda",
    ]
    if engineering_only:
        cmd.append("--engineering-only")
    if a_updates is not None:
        cmd += ["--a-updates-override", str(a_updates)]
    if b_tokens is not None:
        cmd += ["--b-token-budget-override", str(b_tokens)]
    return cmd


base.worker_cmd = worker_cmd


def _arg(cmd: list[str], name: str, default: str = "unknown") -> str:
    try:
        return str(cmd[cmd.index(name) + 1])
    except (ValueError, IndexError):
        return default


def spawn_logged(cmd: list[str], *, gpu: int, repo: Path) -> subprocess.Popen:
    out = Path(_arg(cmd, "--out", str(CAMPAIGN_ROOT)))
    log_dir = out / "_worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{_arg(cmd, '--experiment')}__{_arg(cmd, '--arm')}__"
        f"seed{_arg(cmd, '--seed-bundle')}__gpu{gpu}.log"
    )
    path = log_dir / name
    handle = path.open("a", encoding="utf-8", buffering=1)
    handle.write("\n=== worker invocation ===\n")
    handle.write("utc: " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    handle.write("command: " + " ".join(cmd) + "\n")
    handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=repo,
        env=base._env_for_gpu(gpu),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._formation_mux_log_handle = handle
    proc._formation_mux_log_path = str(path)
    return proc


base._spawn = spawn_logged


def import_prior_state_s4(out: Path) -> str | None:
    """Import only prior output created under the exact S4 science identity."""
    if (out / "CAMPAIGN_STATE.json").exists():
        existing = json.loads((out / "CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
        if existing.get("science_commit") != SCIENCE_COMMIT:
            raise base.GlobalIntegrityError(
                "local FORMATION-MUX state belongs to another science commit; "
                "preserve it separately and do not mix campaigns"
            )
        return None
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    candidates = sorted(root.rglob("FORMATION_MUX_001/CAMPAIGN_STATE.json"))
    for state_path in reversed(candidates):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("science_commit") != SCIENCE_COMMIT:
            continue
        source = state_path.parent
        shutil.copytree(source, out, dirs_exist_ok=True)
        return str(source)
    return None


def qualify(repo: Path, public_surface: Path, out: Path) -> dict[str, Any]:
    """CPU/static qualification plus simultaneous disposable T4 workers."""
    compile_files = [
        "anra_v5/formation_mux_model_v3.py",
        "anra_v5/formation_mux_train_v4.py",
        "v5_experiments/formation_mux_protocol_v4.py",
        "v5_experiments/formation_mux_surface_v4.py",
        "tools/formation_mux_001_worker_v4.py",
        OPERATOR_NAME,
    ]
    compile_run = subprocess.run(
        [sys.executable, "-m", "py_compile", *compile_files],
        cwd=repo, capture_output=True, text=True,
    )
    tests = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_formation_mux_001_v2.py",
            "tests/test_formation_mux_001_v3.py",
            "tests/test_formation_mux_001_v4.py",
            "tests/test_formation_mux_surface_v2.py",
        ],
        cwd=repo, capture_output=True, text=True,
    )
    receipt: dict[str, Any] = {
        "schema": "anra.formation-mux-qualification/v4",
        "science_commit": SCIENCE_COMMIT,
        "py_compile_returncode": compile_run.returncode,
        "py_compile_stderr_tail": compile_run.stderr[-4000:],
        "pytest_returncode": tests.returncode,
        "pytest_stdout_tail": tests.stdout[-12000:],
        "pytest_stderr_tail": tests.stderr[-6000:],
    }
    if compile_run.returncode != 0 or tests.returncode != 0:
        base._atomic_json(out / "QUALIFICATION.json", receipt)
        raise base.GlobalIntegrityError("Science S4 CPU/static qualification failed")

    cmds = [
        worker_cmd(
            experiment=proto.EXPERIMENT_A,
            arm="M1_EXTRA_NO_DECAY",
            seed=proto.CALIBRATION_SEEDS[0],
            surface=public_surface,
            out=out / "qualification_s4",
            engineering_only=True,
            a_updates=2,
        ),
        worker_cmd(
            experiment=proto.EXPERIMENT_A,
            arm="M2_EXTRA_FROZEN",
            seed=proto.CALIBRATION_SEEDS[1],
            surface=public_surface,
            out=out / "qualification_s4",
            engineering_only=True,
            a_updates=2,
        ),
    ]
    procs = [spawn_logged(cmds[i], gpu=i, repo=repo) for i in (0, 1)]
    codes = [p.wait() for p in procs]
    for p in procs:
        handle = getattr(p, "_formation_mux_log_handle", None)
        if handle is not None:
            handle.close()
    receipt["dual_t4_worker_codes"] = codes
    if codes != [0, 0]:
        base._atomic_json(out / "QUALIFICATION.json", receipt)
        raise base.GlobalIntegrityError(f"Science S4 dual-T4 tiny qualification failed: {codes}")
    receipt["status"] = "CPU_STATIC_PASS / GPU_E2E_PASS_ENGINEERING_ONLY"
    receipt["sealed_rows_visible_to_workers"] = False
    base._atomic_json(out / "QUALIFICATION.json", receipt)
    return receipt


def calibrate_safe(repo: Path, public_surface: Path, out: Path) -> dict[str, Any]:
    receipt = base.calibrate(repo, public_surface, out)
    projected = float(receipt["projected_full_campaign_minutes"])
    safe_partition = projected > OPS_SAFE_CAMPAIGN_MINUTES
    partition_required = bool(receipt.get("partition_required") or safe_partition)
    partition_minutes = projected / 2.0 if partition_required else projected
    if partition_required and partition_minutes > OPS_SAFE_CAMPAIGN_MINUTES:
        raise base.GlobalIntegrityError(
            f"full campaign projects {projected:.1f} min and a two-session matched-bundle "
            f"partition still projects {partition_minutes:.1f} min > "
            f"the ops-safe {OPS_SAFE_CAMPAIGN_MINUTES:.1f} min campaign ceiling; "
            "do not shorten the frozen science automatically"
        )
    receipt.update(
        {
            "science_commit": SCIENCE_COMMIT,
            "ops_safe_campaign_minutes": OPS_SAFE_CAMPAIGN_MINUTES,
            "partition_required": partition_required,
            "projected_partition_minutes": partition_minutes,
            "operator_wall_minutes": MAX_OPERATOR_WALL_MINUTES,
            "finalization_package_reserve_minutes": FINALIZATION_PACKAGE_RESERVE_MINUTES,
            "arm_runtime_safety_factor": ARM_RUNTIME_SAFETY_FACTOR,
            "raw_sealed_rows_used": False,
        }
    )
    base._atomic_json(out / "CALIBRATION_RECEIPT.json", receipt)
    if partition_required:
        base._atomic_json(
            out / "PARTITION_PLAN.json",
            {
                "schema": "anra.formation-mux-partition-plan/v4",
                "science_commit": SCIENCE_COMMIT,
                "science_protocol_unchanged": True,
                "reason": "measured campaign does not fit the ops-safe single-session wall budget",
                "execution_policy": (
                    "select at most one incomplete matched seed bundle per T4; "
                    "completed arms remain immutable; incomplete bundles remain pending"
                ),
                "projected_full_campaign_minutes": projected,
                "projected_partition_minutes": partition_minutes,
            },
        )
    return receipt


def _bundle_complete(bundle: int, complete: dict[str, str]) -> bool:
    label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
    for experiment, arms in (
        (proto.EXPERIMENT_A, proto.ARMS_A),
        (proto.EXPERIMENT_B, proto.ARMS_B),
    ):
        for arm in arms:
            if complete.get(f"{experiment}/{arm}/{label}") != "COMPLETE":
                return False
    return True


def _choose_partition_bundles(complete: dict[str, str]) -> set[int]:
    pending = [b for b in proto.SEED_BUNDLES if not _bundle_complete(b, complete)]
    selected: list[int] = []
    for parity in (0, 1):
        candidates = [b for b in pending if proto.SEED_BUNDLES.index(b) % 2 == parity]
        if candidates:
            selected.append(candidates[0])
    return set(selected)


def _estimated_arm_seconds(job: dict[str, Any], calibration: dict[str, Any]) -> float:
    key = (
        "mechanism_arm_projection_seconds"
        if job["experiment"] == proto.EXPERIMENT_A
        else "rendering_arm_projection_seconds"
    )
    return float(calibration[key]) * ARM_RUNTIME_SAFETY_FACTOR


def run_campaign(
    repo: Path,
    public_surface: Path,
    out: Path,
    calibration: dict[str, Any],
    *,
    operator_started: float,
) -> dict[str, Any]:
    complete = base._scan_completed(out)
    allowed = (
        _choose_partition_bundles(complete)
        if calibration["partition_required"]
        else {b for b in proto.SEED_BUNDLES if not _bundle_complete(b, complete)}
    )
    queues = proto.queue_assignment()
    pending = {
        gpu: [
            j for j in jobs
            if j["seed_bundle"] in allowed
            and f"{j['experiment']}/{j['arm']}/{j['seed_bundle_label']}" not in complete
        ]
        for gpu, jobs in queues.items()
    }

    state_path = out / "CAMPAIGN_STATE.json"
    old = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if old and old.get("science_commit") not in (None, SCIENCE_COMMIT):
        raise base.GlobalIntegrityError("campaign-state science identity mismatch")
    state = {
        "schema": "anra.formation-mux-state/v4",
        "science_commit": SCIENCE_COMMIT,
        "allowed_seed_bundles_this_session": sorted(allowed),
        "arms": {**old.get("arms", {}), **complete},
        "global_failure": old.get("global_failure"),
        "sealed": old.get(
            "sealed", {exp: "NOT_CONSUMED" for exp in proto.EXPERIMENTS}
        ),
        "wall_guard_triggered": False,
    }
    base._atomic_json(state_path, state)

    stop = False
    while any(pending.values()) and not stop:
        candidates = {
            gpu: pending[f"GPU{gpu}"][0]
            for gpu in (0, 1)
            if pending[f"GPU{gpu}"]
        }
        if candidates:
            longest = max(
                _estimated_arm_seconds(job, calibration)
                for job in candidates.values()
            )
            elapsed = time.monotonic() - operator_started
            latest_safe_finish = (
                MAX_OPERATOR_WALL_MINUTES - FINALIZATION_PACKAGE_RESERVE_MINUTES
            ) * 60.0
            if elapsed + longest > latest_safe_finish:
                state["wall_guard_triggered"] = True
                state["wall_guard_reason"] = (
                    f"elapsed={elapsed:.1f}s + conservative_next_arm={longest:.1f}s "
                    f"would exceed launch deadline={latest_safe_finish:.1f}s"
                )
                base._atomic_json(state_path, state)
                break

        running: dict[int, tuple[dict[str, Any], subprocess.Popen]] = {}
        for gpu in (0, 1):
            q = pending[f"GPU{gpu}"]
            if not q:
                continue
            job = q.pop(0)
            cmd = worker_cmd(
                experiment=job["experiment"],
                arm=job["arm"],
                seed=job["seed_bundle"],
                surface=public_surface,
                out=out,
            )
            proc = spawn_logged(cmd, gpu=gpu, repo=repo)
            running[gpu] = (job, proc)
            print(
                f"GPU{gpu} <- {job['experiment']}/{job['arm']}/{job['seed_bundle_label']}",
                flush=True,
            )

        for gpu, (job, proc) in running.items():
            code = proc.wait()
            handle = getattr(proc, "_formation_mux_log_handle", None)
            if handle is not None:
                handle.close()
            key = f"{job['experiment']}/{job['arm']}/{job['seed_bundle_label']}"
            if code == base.EXIT_OK:
                state["arms"][key] = "COMPLETE"
            elif code == base.EXIT_ARM_LOCAL:
                state["arms"][key] = "ENGINEERING_FAILURE"
            else:
                state["global_failure"] = f"worker exit {code} at {key}"
                stop = True
            base._atomic_json(state_path, state)

    complete = base._scan_completed(out)
    state["arms"].update(complete)
    state["complete_arms"] = len(complete)
    state["required_arms"] = proto.total_official_arms()
    state["completed_seed_bundles"] = [
        b for b in proto.SEED_BUNDLES if _bundle_complete(b, complete)
    ]
    state["pending_seed_bundles"] = [
        b for b in proto.SEED_BUNDLES if not _bundle_complete(b, complete)
    ]
    if state.get("global_failure"):
        state["status"] = "FAILED_GLOBAL"
    elif len(complete) == proto.total_official_arms():
        state["status"] = "ARMS_COMPLETE"
    else:
        state["status"] = "PARTIAL_SESSION"
        state["resume_instruction"] = (
            "Save this Kaggle version/output, attach that output to the next run, "
            "and rerun the exact same pinned notebook. Completed arms are immutable; "
            "incomplete matched bundles remain pending."
        )
    state["operator_elapsed_seconds"] = time.monotonic() - operator_started
    base._atomic_json(state_path, state)
    return state


def _sync_sealed_state(out: Path) -> None:
    path = out / "CAMPAIGN_STATE.json"
    if not path.exists():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    for experiment in proto.EXPERIMENTS:
        state.setdefault("sealed", {})[experiment] = base._sealed_marker(
            out, experiment
        ).get("state", "NOT_CONSUMED")
    base._atomic_json(path, state)


def finalize_sealed(
    public_surface_path: Path,
    out: Path,
    torch: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Consume raw sealed rows only after each experiment's dev result is frozen."""
    from anra_v5 import formation_mux_train_v4 as train
    from v5_experiments.formation_mux_surface_v4 import (
        load_public_surface,
        regenerate_sealed_rows,
    )

    public = load_public_surface(public_surface_path)
    device = torch.device("cuda:0")
    finals: dict[str, Any] = {}
    for experiment in proto.EXPERIMENTS:
        final_path = out / experiment / "FINAL_RESULT.json"
        marker = base._sealed_marker(out, experiment)
        if marker.get("state") == "COMPLETE" and final_path.exists():
            finals[experiment] = json.loads(final_path.read_text(encoding="utf-8"))
            continue
        if marker.get("state") == "STARTED" and not final_path.exists():
            raise base.GlobalIntegrityError(
                f"sealed marker STARTED without FINAL_RESULT for {experiment}; "
                "preserve evidence and fail closed"
            )
        dev = base.finalize_dev(out, experiment)
        if dev.get("status") != "DEVELOPMENT_COMPLETE":
            finals[experiment] = {
                "verdict": dev.get("status", "INCONCLUSIVE"),
                "sealed_consumed": False,
            }
            continue

        base._set_sealed(out, experiment, "STARTED")
        # Only now does raw sealed content enter coordinator memory.
        sealed_rows = regenerate_sealed_rows(
            public_manifest=public,
            tokenizer=tokenizer,
            experiment=experiment,
        )
        arms = proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B
        observed: dict[str, dict[int, float]] = {arm: {} for arm in arms}
        secondary: dict[str, dict[int, Any]] = {arm: {} for arm in arms}
        for arm in arms:
            for bundle in proto.SEED_BUNDLES:
                label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
                checkpoint = out / experiment / arm / label / "resume.pt"
                model = train.load_model_for_evaluation(
                    checkpoint,
                    experiment=experiment,
                    arm=arm,
                    seed_bundle=bundle,
                    data_manifest_sha256=str(public["sha256"]),
                    torch=torch,
                    device=device,
                )
                scored = train.evaluate_development(
                    model,
                    sealed_rows,
                    experiment,
                    arm,
                    torch=torch,
                    device=device,
                )
                observed[arm][bundle] = float(scored["identity_exact_valid_eos"])
                secondary[arm][bundle] = scored["per_family"]
                del model
                torch.cuda.empty_cache()

        contrasts = []
        for row in dev["contrasts"]:
            higher, lower = row["higher"], row["lower"]
            auc = {
                int(k): float(v)
                for k, v in row["development_formation_auc_deltas"].items()
            }
            endpoint = {
                b: observed[higher][b] - observed[lower][b]
                for b in proto.SEED_BUNDLES
            }
            verdict = proto.paired_verdict(
                formation_auc_deltas=auc,
                sealed_endpoint_gaps=endpoint,
            )
            contrasts.append(
                {
                    **row,
                    "sealed_endpoint_gaps": {
                        str(k): v for k, v in endpoint.items()
                    },
                    **verdict,
                }
            )
        overall = proto.aggregate_experiment_verdict(
            [x["verdict"] for x in contrasts]
        )
        final = {
            "schema": "anra.formation-mux-final-result/v4",
            "science_commit": SCIENCE_COMMIT,
            "experiment": experiment,
            "sealed_commitment_sha256": public["sealed_commitments"][experiment],
            "raw_sealed_rows_persisted": False,
            "observed_sealed_identity_exact_valid_eos": {
                arm: {str(k): v for k, v in values.items()}
                for arm, values in observed.items()
            },
            "observed_sealed_per_family": {
                arm: {str(k): v for k, v in values.items()}
                for arm, values in secondary.items()
            },
            "contrast_results": contrasts,
            "verdict": overall,
            "claim_ceiling": proto.claim_ceiling(experiment, overall),
            "next_action": proto.NEXT_ACTION[overall],
            "sealed_consumed": True,
        }
        base._atomic_json(final_path, final)
        base._set_sealed(out, experiment, "COMPLETE")
        finals[experiment] = final
        # Do not serialize raw sealed rows anywhere.
        del sealed_rows
    return finals


def package_safe(out: Path, repo: Path) -> dict[str, Any]:
    bundle = out.parent / BUNDLE_NAME
    if bundle.exists():
        bundle.unlink()
    public_path = out / "PUBLIC_SURFACE_MANIFEST.json"
    if public_path.exists():
        from v5_experiments.formation_mux_surface_v4 import load_public_surface
        load_public_surface(public_path)  # proves raw sealed rows are absent
    forbidden = out / "SURFACE_MANIFEST.json"
    if forbidden.exists():
        raise base.GlobalIntegrityError(
            "refusing to package legacy raw SURFACE_MANIFEST.json under Science S4"
        )

    frozen_docs = [
        "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1_PREEXECUTION.md",
        "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1B_PREEXECUTION_CLIP_ISOLATION.md",
        "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1C_PREEXECUTION_SEALED_FIREWALL.md",
        "docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION_V4.json",
        "docs/cymek/experiments/CS-MECH-002/PREREGISTRATION_V4.json",
        "docs/cymek/experiments/REP-FORM-003A/PREREGISTRATION_V4.json",
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SCIENCE_COMMIT.txt", SCIENCE_COMMIT + "\n")
        zf.writestr(
            "OPERATOR_HEAD.txt",
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                text=True,
                capture_output=True,
                check=True,
            ).stdout,
        )
        zf.writestr(
            "README_S4.txt",
            "FORMATION-MUX-001 Science S4 results/failure bundle.\n"
            "Raw sealed examples are never persisted or included.\n"
            "Checkpoints needed for exact resume live in the Kaggle Output tree, not this ZIP.\n",
        )
        for rel in frozen_docs:
            zf.write(repo / rel, "FROZEN_PROTOCOL/" + rel)
        for path in sorted(out.rglob("*.json")):
            if path.name == "SURFACE_MANIFEST.json":
                continue
            zf.write(path, path.relative_to(out.parent))
        for path in sorted(out.rglob("*.log")):
            zf.write(path, "RUNTIME_LOGS/" + path.relative_to(out).as_posix())
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    Path(str(bundle) + ".sha256").write_text(
        f"{digest}  {bundle.name}\n", encoding="utf-8"
    )
    return {
        "path": str(bundle),
        "sha256": digest,
        "science_commit": SCIENCE_COMMIT,
        "raw_sealed_rows_included": False,
        "resume_state_location": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--skip-sealed", action="store_true")
    args = parser.parse_args(argv)
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    operator_started = time.monotonic()

    try:
        import torch

        science_hashes = base.verify_science(repo)
        environment = base.require_hardware(torch)
        imported = import_prior_state_s4(out)
        environment.update(
            {
                "science_commit": SCIENCE_COMMIT,
                "science_files_sha256": science_hashes,
                "imported_prior_state": imported,
                "canonical_operator": OPERATOR_NAME,
                "sealed_firewall": proto.SEALED_FIREWALL,
                "operator_wall_minutes": MAX_OPERATOR_WALL_MINUTES,
            }
        )
        base._atomic_json(out / "ENVIRONMENT.json", environment)

        if (out / "SURFACE_MANIFEST.json").exists():
            raise base.GlobalIntegrityError(
                "legacy raw SURFACE_MANIFEST.json detected; Science S4 refuses to mix it with public-only state"
            )

        from v5_data.corpus_loading import _load_tokenizer
        from v5_experiments.formation_mux_surface_v4 import (
            build_public_surface,
            load_public_surface,
        )

        tokenizer, tokenizer_eval = _load_tokenizer(repo)
        public_path = out / "PUBLIC_SURFACE_MANIFEST.json"
        if public_path.exists():
            public = load_public_surface(public_path)
            if public["tokenizer_artifact_sha256"] != tokenizer.identity.artifact_sha256:
                raise base.GlobalIntegrityError("public surface tokenizer identity drift")
        else:
            public = build_public_surface(seed=73011, tokenizer=tokenizer)
            base._atomic_json(public_path, public)
        base._atomic_json(
            out / "TOKENIZER_RECEIPT.json",
            {
                "vocabulary_size": int(tokenizer.identity.vocabulary_size),
                "special_token_ids": dict(tokenizer.identity.special_token_ids),
                "artifact_sha256": tokenizer.identity.artifact_sha256,
                "evaluation": tokenizer_eval,
                "raw_sealed_rows_persisted": False,
                "full_surface_sha256_commitment": public["full_surface_sha256"],
                "sealed_commitments": public["sealed_commitments"],
            },
        )

        qualify(repo, public_path, out)
        calibration = calibrate_safe(repo, public_path, out)
        state = run_campaign(
            repo,
            public_path,
            out,
            calibration,
            operator_started=operator_started,
        )
        if state["status"] == "ARMS_COMPLETE":
            for experiment in proto.EXPERIMENTS:
                base.finalize_dev(out, experiment)
            if not args.skip_sealed:
                finalize_sealed(public_path, out, torch, tokenizer)
                _sync_sealed_state(out)
        bundle = package_safe(out, repo)
        print("FORMATION-MUX-001 STATUS:", state["status"])
        print("SCIENCE COMMIT S4:", SCIENCE_COMMIT)
        print("RESULT BUNDLE:", json.dumps(bundle))
        return base.EXIT_OK
    except Exception as exc:
        base._atomic_json(
            out / "GLOBAL_FAILURE.json",
            {
                "schema": "anra.formation-mux-global-failure/v7",
                "science_commit": SCIENCE_COMMIT,
                "operator": OPERATOR_NAME,
                "exception": type(exc).__name__,
                "message": str(exc),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        try:
            print("FAILURE BUNDLE:", json.dumps(package_safe(out, repo)))
        except Exception as package_exc:
            print("FAILURE PACKAGE ALSO FAILED:", repr(package_exc), file=sys.stderr)
        print("GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr)
        return base.EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
