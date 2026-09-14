"""Canonical FORMATION-MUX operator: frozen S5 + preregistered frontier science.

Stage 1 executes frozen Science S5 through operator v9 with sealed evaluation
suppressed. Stage 2 executes the independently preregistered TIE-ROLE frontier
through the same two-T4 matched-seed discipline and exact-resume checkpoints.
Only after all development results and development-only diagnostics are frozen
does the coordinator regenerate sealed rows in memory and score both programs.
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v9 as v9
from v5_experiments import tie_role_protocol_v1 as frontier

OPERATOR_NAME = "tools/formation_mux_001_kaggle_operator_v10.py"
FRONTIER_SPEC = "docs/cymek/experiments/FORMATION-MUX-001/TIE_ROLE_FRONTIER_PREREGISTRATION_V1.json"
FRONTIER_WORKER = "tools.formation_mux_001_tie_role_worker_v1"
FRONTIER_TEST = "tests/test_formation_mux_tie_role_v1.py"
FRONTIER_STATE = "TIE_ROLE_FRONTIER_STATE.json"
FRONTIER_QUALIFICATION = "TIE_ROLE_FRONTIER_QUALIFICATION.json"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify_frontier_preregistration(repo: Path) -> dict[str, Any]:
    path = repo / FRONTIER_SPEC
    if not path.exists():
        raise v9.v8.v7.base.GlobalIntegrityError(f"frontier preregistration missing: {FRONTIER_SPEC}")
    body = json.loads(path.read_text(encoding="utf-8"))
    relation = body.get("relationship_to_science_s5", {})
    if body.get("written_before_any_official_formation_mux_outcomes") is not True:
        raise v9.v8.v7.base.GlobalIntegrityError("frontier preregistration is not prospective")
    if (
        relation.get("changes_s5_training") is not False
        or relation.get("changes_s5_arms") is not False
        or relation.get("changes_s5_primary_or_sealed_verdict") is not False
        or relation.get("may_change_frozen_s5_verdict") is not False
    ):
        raise v9.v8.v7.base.GlobalIntegrityError("frontier preregistration violates frozen S5")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": "anra.tie-role-preregistration-receipt/v1",
        "path": FRONTIER_SPEC,
        "sha256": digest,
        "prospective": True,
        "science_s5_unchanged": True,
        "extension": frontier.EXTENSION,
    }


def _worker_cmd(*, experiment: str, arm: str, seed: int, surface: Path, out: Path,
                engineering_only: bool = False, a_updates: int | None = None,
                b_tokens: int | None = None) -> list[str]:
    cmd = [
        sys.executable, "-m", FRONTIER_WORKER,
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


def qualify_frontier(repo: Path, public_path: Path, out: Path) -> dict[str, Any]:
    prereg = verify_frontier_preregistration(repo)
    compile_files = [
        "anra_v5/tie_role_model_v1.py",
        "anra_v5/tie_role_train_v1.py",
        "v5_experiments/tie_role_protocol_v1.py",
        "tools/formation_mux_001_tie_role_worker_v1.py",
        "tools/formation_mux_001_tie_role_gradient_diag_v1.py",
        OPERATOR_NAME,
    ]
    compile_run = subprocess.run(
        [sys.executable, "-m", "py_compile", *compile_files],
        cwd=repo, capture_output=True, text=True,
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", FRONTIER_TEST],
        cwd=repo, capture_output=True, text=True,
    )
    receipt: dict[str, Any] = {
        "schema": "anra.tie-role-qualification/v1",
        "preregistration": prereg,
        "compile_files": compile_files,
        "py_compile_returncode": compile_run.returncode,
        "py_compile_stderr_tail": compile_run.stderr[-6000:],
        "pytest_returncode": tests.returncode,
        "pytest_stdout_tail": tests.stdout[-12000:],
        "pytest_stderr_tail": tests.stderr[-6000:],
        "sealed_rows_visible_to_workers": False,
    }
    if compile_run.returncode != 0 or tests.returncode != 0:
        receipt["status"] = "CPU_STATIC_FAIL"
        _atomic_json(out / FRONTIER_QUALIFICATION, receipt)
        raise v9.v8.v7.base.GlobalIntegrityError("TIE-ROLE CPU/static qualification failed")

    qout = out / "qualification_tie_role"
    cmds = [
        _worker_cmd(
            experiment=frontier.EXPERIMENT_A,
            arm="T3_BALANCED_X4_X025",
            seed=frontier.CALIBRATION_SEEDS[0],
            surface=public_path,
            out=qout,
            engineering_only=True,
            a_updates=2,
        ),
        _worker_cmd(
            experiment=frontier.EXPERIMENT_B,
            arm="X1_BALANCED_R0",
            seed=frontier.CALIBRATION_SEEDS[1],
            surface=public_path,
            out=qout,
            engineering_only=True,
            b_tokens=2_000,
        ),
    ]
    procs = [v9.v8.v7.spawn_logged(cmds[i], gpu=i, repo=repo) for i in (0, 1)]
    codes = [p.wait() for p in procs]
    for p in procs:
        handle = getattr(p, "_formation_mux_log_handle", None)
        if handle is not None:
            handle.close()
    receipt["dual_t4_worker_codes"] = codes
    if codes != [0, 0]:
        receipt["status"] = "GPU_E2E_FAIL"
        _atomic_json(out / FRONTIER_QUALIFICATION, receipt)
        raise v9.v8.v7.base.GlobalIntegrityError(f"TIE-ROLE dual-T4 qualification failed: {codes}")
    receipt["status"] = "CPU_STATIC_PASS / GPU_E2E_PASS_ENGINEERING_ONLY"
    _atomic_json(out / FRONTIER_QUALIFICATION, receipt)
    return receipt


def _result_path(out: Path, experiment: str, arm: str, index: int) -> Path:
    return out / experiment / arm / f"S{index}" / "ARM_RESULT.json"


def _scan_frontier_completed(out: Path) -> dict[str, str]:
    complete: dict[str, str] = {}
    for experiment, arms in ((frontier.EXPERIMENT_A, frontier.ARMS_A), (frontier.EXPERIMENT_B, frontier.ARMS_B)):
        for arm in arms:
            for index, bundle in enumerate(frontier.SEED_BUNDLES, start=1):
                path = _result_path(out, experiment, arm, index)
                if not path.exists():
                    continue
                body = json.loads(path.read_text(encoding="utf-8"))
                if (
                    body.get("status") == "COMPLETE"
                    and body.get("experiment") == experiment
                    and body.get("arm") == arm
                    and int(body.get("seed_bundle")) == bundle
                    and body.get("protocol_sha256") == frontier.protocol_sha(experiment)
                ):
                    complete[f"{experiment}/{arm}/S{index}"] = "COMPLETE"
    return complete


def _bundle_complete(bundle: int, complete: dict[str, str]) -> bool:
    index = frontier.SEED_BUNDLES.index(bundle) + 1
    for experiment, arms in ((frontier.EXPERIMENT_A, frontier.ARMS_A), (frontier.EXPERIMENT_B, frontier.ARMS_B)):
        for arm in arms:
            if complete.get(f"{experiment}/{arm}/S{index}") != "COMPLETE":
                return False
    return True


def _estimated_arm_seconds(job: dict[str, Any], calibration: dict[str, Any]) -> float:
    key = "mechanism_arm_projection_seconds" if job["experiment"] == frontier.EXPERIMENT_A else "rendering_arm_projection_seconds"
    return float(calibration[key]) * float(v9.v8.v7.ARM_RUNTIME_SAFETY_FACTOR)


def run_frontier(repo: Path, public_path: Path, out: Path, calibration: dict[str, Any], *, operator_started: float) -> dict[str, Any]:
    complete = _scan_frontier_completed(out)
    queues = frontier.queue_assignment()
    pending = {
        gpu: [
            j for j in jobs
            if f"{j['experiment']}/{j['arm']}/{j['seed_bundle_label']}" not in complete
        ]
        for gpu, jobs in queues.items()
    }
    state_path = out / FRONTIER_STATE
    old = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state: dict[str, Any] = {
        "schema": "anra.tie-role-frontier-state/v1",
        "extension": frontier.EXTENSION,
        "arms": {**old.get("arms", {}), **complete},
        "global_failure": old.get("global_failure"),
        "wall_guard_triggered": False,
        "sealed": old.get("sealed", {e: "NOT_CONSUMED" for e in frontier.EXPERIMENTS}),
    }
    _atomic_json(state_path, state)

    stop = False
    while any(pending.values()) and not stop:
        candidates = {gpu: pending[f"GPU{gpu}"][0] for gpu in (0, 1) if pending[f"GPU{gpu}"]}
        if candidates:
            longest = max(_estimated_arm_seconds(job, calibration) for job in candidates.values())
            elapsed = time.monotonic() - operator_started
            latest_safe_finish = (
                float(v9.v8.v7.MAX_OPERATOR_WALL_MINUTES)
                - float(v9.v8.v7.FINALIZATION_PACKAGE_RESERVE_MINUTES)
            ) * 60.0
            if elapsed + longest > latest_safe_finish:
                state["wall_guard_triggered"] = True
                state["wall_guard_reason"] = (
                    f"elapsed={elapsed:.1f}s + conservative_next_frontier_arm={longest:.1f}s "
                    f"would exceed launch deadline={latest_safe_finish:.1f}s"
                )
                _atomic_json(state_path, state)
                break

        running: dict[int, tuple[dict[str, Any], subprocess.Popen]] = {}
        for gpu in (0, 1):
            q = pending[f"GPU{gpu}"]
            if not q:
                continue
            job = q.pop(0)
            cmd = _worker_cmd(
                experiment=job["experiment"], arm=job["arm"],
                seed=job["seed_bundle"], surface=public_path, out=out,
            )
            proc = v9.v8.v7.spawn_logged(cmd, gpu=gpu, repo=repo)
            running[gpu] = (job, proc)
            print(f"GPU{gpu} <- FRONTIER {job['experiment']}/{job['arm']}/{job['seed_bundle_label']}", flush=True)

        for gpu, (job, proc) in running.items():
            code = proc.wait()
            handle = getattr(proc, "_formation_mux_log_handle", None)
            if handle is not None:
                handle.close()
            key = f"{job['experiment']}/{job['arm']}/{job['seed_bundle_label']}"
            if code == 0:
                state["arms"][key] = "COMPLETE"
            elif code == 3:
                state["arms"][key] = "ENGINEERING_FAILURE"
            else:
                state["global_failure"] = f"frontier worker exit {code} at {key}"
                stop = True
            _atomic_json(state_path, state)

    complete = _scan_frontier_completed(out)
    state["arms"].update(complete)
    state["complete_arms"] = len(complete)
    state["required_arms"] = frontier.total_official_arms()
    state["completed_seed_bundles"] = [b for b in frontier.SEED_BUNDLES if _bundle_complete(b, complete)]
    state["pending_seed_bundles"] = [b for b in frontier.SEED_BUNDLES if not _bundle_complete(b, complete)]
    if state.get("global_failure"):
        state["status"] = "FAILED_GLOBAL"
    elif len(complete) == frontier.total_official_arms():
        state["status"] = "ARMS_COMPLETE"
    else:
        state["status"] = "PARTIAL_SESSION"
        state["resume_instruction"] = (
            "Save Kaggle output, attach it to the next run, and rerun the exact pinned notebook. "
            "Completed frontier arms are immutable and incomplete arms exact-resume."
        )
    state["operator_elapsed_seconds"] = time.monotonic() - operator_started
    _atomic_json(state_path, state)
    return state


def _arm_result(out: Path, experiment: str, arm: str, bundle: int) -> dict[str, Any]:
    index = frontier.SEED_BUNDLES.index(bundle) + 1
    path = _result_path(out, experiment, arm, index)
    if not path.exists():
        raise RuntimeError(f"frontier arm result missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_frontier_dev(out: Path, experiment: str) -> dict[str, Any]:
    path = out / experiment / "DEVELOPMENT_RESULT.json"
    arms = frontier.ARMS_A if experiment == frontier.EXPERIMENT_A else frontier.ARMS_B
    contrasts = frontier.CONTRASTS_A if experiment == frontier.EXPERIMENT_A else (frontier.CONTRAST_B,)
    arm_results: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in arms}
    for arm in arms:
        for bundle in frontier.SEED_BUNDLES:
            arm_results[arm][bundle] = _arm_result(out, experiment, arm, bundle)

    if experiment == frontier.EXPERIMENT_B:
        mismatches = {}
        for bundle in frontier.SEED_BUNDLES:
            a = int(arm_results[frontier.ARMS_B[0]][bundle]["processed_tokens"])
            b = int(arm_results[frontier.ARMS_B[1]][bundle]["processed_tokens"])
            mismatches[str(bundle)] = frontier.exposure_mismatch(a, b)
        if any(v > frontier.B_EXPOSURE_MISMATCH_TOLERANCE for v in mismatches.values()):
            result = {
                "schema": "anra.tie-role-development-result/v1",
                "experiment": experiment,
                "status": "INCONCLUSIVE_EXPOSURE_MISMATCH",
                "exposure_mismatch": mismatches,
            }
            _atomic_json(path, result)
            return result

    contrast_rows = []
    for higher, lower, label in contrasts:
        deltas = {
            bundle: float(arm_results[higher][bundle]["formation"]["formation_auc"])
            - float(arm_results[lower][bundle]["formation"]["formation_auc"])
            for bundle in frontier.SEED_BUNDLES
        }
        dev_endpoint = {
            bundle: float(arm_results[higher][bundle]["formation"]["endpoint"])
            - float(arm_results[lower][bundle]["formation"]["endpoint"])
            for bundle in frontier.SEED_BUNDLES
        }
        contrast_rows.append({
            "higher": higher,
            "lower": lower,
            "label": label,
            "development_formation_auc_deltas": {str(k): v for k, v in deltas.items()},
            "development_endpoint_deltas": {str(k): v for k, v in dev_endpoint.items()},
            "mean_development_formation_auc_delta": sum(deltas.values()) / len(deltas),
        })

    result: dict[str, Any] = {
        "schema": "anra.tie-role-development-result/v1",
        "extension": frontier.EXTENSION,
        "experiment": experiment,
        "status": "DEVELOPMENT_COMPLETE",
        "contrasts": contrast_rows,
        "sealed_consumed": False,
    }
    if experiment == frontier.EXPERIMENT_A:
        interaction = {}
        for bundle in frontier.SEED_BUNDLES:
            f = {arm: float(arm_results[arm][bundle]["formation"]["formation_auc"]) for arm in frontier.ARMS_A}
            interaction[str(bundle)] = (f["T3_BALANCED_X4_X025"] - f["T2_OUTPUT_X025"]) - (f["T1_INPUT_X4"] - f["T0_CANONICAL"])
        result["factorial_interaction_auc"] = interaction
        result["mean_factorial_interaction_auc"] = sum(interaction.values()) / len(interaction)
    _atomic_json(path, result)
    return result


def _marker_path(out: Path, experiment: str) -> Path:
    return out / experiment / "TIE_ROLE_SEALED_MARKER.json"


def _marker(out: Path, experiment: str) -> dict[str, Any]:
    path = _marker_path(out, experiment)
    if not path.exists():
        return {"state": "NOT_CONSUMED"}
    return json.loads(path.read_text(encoding="utf-8"))


def _set_marker(out: Path, experiment: str, state: str) -> None:
    _atomic_json(_marker_path(out, experiment), {
        "schema": "anra.tie-role-sealed-marker/v1",
        "experiment": experiment,
        "state": state,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def finalize_frontier_sealed(public_path: Path, out: Path, torch: Any, tokenizer: Any) -> dict[str, Any]:
    from anra_v5 import tie_role_train_v1 as train
    from v5_experiments.formation_mux_surface_v5 import load_public_surface, regenerate_sealed_rows

    public = load_public_surface(public_path)
    device = torch.device("cuda:0")
    finals: dict[str, Any] = {}
    for experiment in frontier.EXPERIMENTS:
        final_path = out / experiment / "FINAL_RESULT.json"
        marker = _marker(out, experiment)
        if marker.get("state") == "COMPLETE" and final_path.exists():
            finals[experiment] = json.loads(final_path.read_text(encoding="utf-8"))
            continue
        if marker.get("state") == "STARTED" and not final_path.exists():
            raise v9.v8.v7.base.GlobalIntegrityError(
                f"frontier sealed marker STARTED without FINAL_RESULT for {experiment}; fail closed"
            )
        dev = finalize_frontier_dev(out, experiment)
        if dev.get("status") != "DEVELOPMENT_COMPLETE":
            finals[experiment] = {"verdict": dev.get("status"), "sealed_consumed": False}
            continue
        _set_marker(out, experiment, "STARTED")
        # Reuse the already precommitted deterministic S5 sealed surface. The
        # raw rows enter coordinator memory only here and are never persisted.
        sealed_rows = regenerate_sealed_rows(
            public_manifest=public, tokenizer=tokenizer, experiment="CS-MECH-002"
        )
        arms = frontier.ARMS_A if experiment == frontier.EXPERIMENT_A else frontier.ARMS_B
        observed = {arm: {} for arm in arms}
        secondary = {arm: {} for arm in arms}
        for arm in arms:
            for index, bundle in enumerate(frontier.SEED_BUNDLES, start=1):
                model = train.load_model_for_evaluation(
                    out / experiment / arm / f"S{index}" / "resume.pt",
                    experiment=experiment,
                    arm=arm,
                    seed_bundle=bundle,
                    data_manifest_sha256=str(public["sha256"]),
                    torch=torch,
                    device=device,
                )
                scored = train.evaluate_development(
                    model, sealed_rows, experiment, arm, torch=torch, device=device
                )
                observed[arm][bundle] = float(scored["identity_exact_valid_eos"])
                secondary[arm][bundle] = scored["per_family"]
                del model
                torch.cuda.empty_cache()
        contrast_results = []
        for row in dev["contrasts"]:
            higher, lower = row["higher"], row["lower"]
            auc = {int(k): float(v) for k, v in row["development_formation_auc_deltas"].items()}
            endpoint = {b: observed[higher][b] - observed[lower][b] for b in frontier.SEED_BUNDLES}
            verdict = frontier.paired_verdict(
                formation_auc_deltas=auc,
                sealed_endpoint_gaps=endpoint,
            )
            contrast_results.append({
                **row,
                "sealed_endpoint_gaps": {str(k): v for k, v in endpoint.items()},
                **verdict,
            })
        final = {
            "schema": "anra.tie-role-final-result/v1",
            "extension": frontier.EXTENSION,
            "experiment": experiment,
            "sealed_source_commitment_experiment": "CS-MECH-002",
            "sealed_source_commitment_sha256": public["sealed_commitments"]["CS-MECH-002"],
            "raw_sealed_rows_persisted": False,
            "observed_sealed_identity_exact_valid_eos": {
                arm: {str(k): v for k, v in values.items()} for arm, values in observed.items()
            },
            "observed_sealed_per_family": {
                arm: {str(k): v for k, v in values.items()} for arm, values in secondary.items()
            },
            "contrast_results": contrast_results,
            "sealed_consumed": True,
            "claim_ceiling": (
                "controlled mechanism/transfer evidence only; no automatic V6, 3x/6x, "
                "large-scale, cognition, or AGI authorization"
            ),
        }
        _atomic_json(final_path, final)
        _set_marker(out, experiment, "COMPLETE")
        finals[experiment] = final
        del sealed_rows
    return finals


def architecture_gate(out: Path, gradient_diag: dict[str, Any]) -> dict[str, Any]:
    a = json.loads((out / frontier.EXPERIMENT_A / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    b = json.loads((out / frontier.EXPERIMENT_B / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    a_primary = next(x for x in a["contrast_results"] if x["higher"] == frontier.PRIMARY_CONTRAST_A[0] and x["lower"] == frontier.PRIMARY_CONTRAST_A[1])
    b_primary = b["contrast_results"][0]
    canonical_keys = [
        f"{frontier.EXPERIMENT_A}/T0_CANONICAL",
        f"{frontier.EXPERIMENT_B}/X0_CANONICAL_R0",
    ]
    diagnostic_rows = [gradient_diag["per_arm"][k] for k in canonical_keys]
    mechanism_support = any(
        float(x["mean_output_to_input_gradient_norm_ratio"]) > 1.0
        and float(x["mean_input_output_gradient_cosine"]) < 0.0
        for x in diagnostic_rows
    )
    av, bv = a_primary["verdict"], b_primary["verdict"]
    if av == "SUCCESS" and bv == "SUCCESS" and mechanism_support:
        classification = "ROBUST_ROLE_INTERFERENCE"
        next_action = "nominate role-balanced tying and an untied-input/output successor for separate production-scale confirmation"
    elif av == "SUCCESS" and bv == "SUCCESS":
        classification = "ROBUST_EFFECT_MECHANISM_NOT_CONFIRMED"
        next_action = "effect transfers, but do not claim gradient interference until a smaller mechanism probe closes the diagnostic gap"
    elif av == "SUCCESS" and bv != "SUCCESS":
        classification = "LATENT_ONLY"
        next_action = "do not promote to Core; effect failed production-BPE transfer"
    elif av == "NULL" and bv == "NULL":
        classification = "NULL"
        next_action = "stop tied-role balancing and move to the next isolated bottleneck"
    elif "REVERSE_EFFECT" in (av, bv):
        classification = "REVERSE_OR_INTERACTION"
        next_action = "retain canonical tying; preserve the reversed result"
    else:
        classification = "TRANSFER_OR_REPRESENTATION_INTERACTION"
        next_action = "do not promote; isolate the smallest tying-by-representation interaction"
    gate = {
        "schema": "anra.tie-role-architecture-gate/v1",
        "extension": frontier.EXTENSION,
        "latent_primary_verdict": av,
        "production_bpe_primary_verdict": bv,
        "mechanism_support": mechanism_support,
        "classification": classification,
        "next_action": next_action,
        "automatic_version_bump": False,
        "automatic_3x_or_6x_claim": False,
    }
    _atomic_json(out / "TIE_ROLE_ARCHITECTURE_GATE.json", gate)
    return gate


def package_frontier(out: Path, repo: Path) -> dict[str, Any]:
    base = v9.v8.package(out, repo)
    bundle = Path(base["path"])
    with zipfile.ZipFile(bundle, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.write(repo / FRONTIER_SPEC, "FROZEN_PROTOCOL/" + FRONTIER_SPEC)
        for rel in (
            "v5_experiments/tie_role_protocol_v1.py",
            "anra_v5/tie_role_model_v1.py",
            "anra_v5/tie_role_train_v1.py",
            "tools/formation_mux_001_tie_role_worker_v1.py",
            "tools/formation_mux_001_tie_role_gradient_diag_v1.py",
        ):
            zf.write(repo / rel, "FROZEN_PROTOCOL/" + rel)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    Path(str(bundle) + ".sha256").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
    return {**base, "sha256": digest, "frontier_extension": frontier.EXTENSION}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=v9.v8.CAMPAIGN_ROOT)
    p.add_argument("--skip-sealed", action="store_true")
    args = p.parse_args(argv)
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    operator_started = time.monotonic()
    try:
        # S5 remains byte-for-byte frozen. v9 handles S5 qualification,
        # exact-resume training, storage checks and packaging, but cannot touch sealed.
        code = v9.main([
            "--repo", str(repo), "--out", str(out), "--skip-sealed"
        ])
        if code != 0:
            return code
        s5_state_path = out / "CAMPAIGN_STATE.json"
        s5_state = json.loads(s5_state_path.read_text(encoding="utf-8"))
        if s5_state.get("status") != "ARMS_COMPLETE":
            print("FORMATION-MUX S5 remains partial; frontier waits for frozen S5 arms.", flush=True)
            package_frontier(out, repo)
            return 0

        import torch
        from v5_data.corpus_loading import _load_tokenizer
        from v5_experiments.formation_mux_surface_v5 import load_public_surface

        public_path = out / "PUBLIC_SURFACE_MANIFEST.json"
        load_public_surface(public_path)
        tokenizer, _ = _load_tokenizer(repo)
        qualify_frontier(repo, public_path, out)
        calibration = json.loads((out / "CALIBRATION_RECEIPT.json").read_text(encoding="utf-8"))
        state = run_frontier(
            repo, public_path, out, calibration, operator_started=operator_started
        )
        if state["status"] != "ARMS_COMPLETE":
            print("TIE-ROLE FRONTIER STATUS:", state["status"], flush=True)
            package_frontier(out, repo)
            return 0

        for experiment in frontier.EXPERIMENTS:
            dev = finalize_frontier_dev(out, experiment)
            if dev.get("status") != "DEVELOPMENT_COMPLETE":
                package_frontier(out, repo)
                return 0

        # All development is now frozen. Run both development-only diagnostics
        # before raw sealed rows can enter coordinator memory.
        from tools.formation_mux_001_tie_role_gradient_diag_v1 import run_diagnostic
        gradient_diag = run_diagnostic(
            public_path=public_path,
            out=out,
            torch=torch,
            device=torch.device("cuda:0"),
        )
        v9.run_xfactor(public_path, out, torch)

        if not args.skip_sealed:
            finalize_frontier_sealed(public_path, out, torch, tokenizer)
            # Frozen S5 sealed evaluation happens only after frontier dev+diagnostics.
            v9._ORIGINAL_FINALIZE_SEALED(public_path, out, torch, tokenizer)
            v9.v8.v7._sync_sealed_state(out)
            architecture_gate(out, gradient_diag)

        environment_path = out / "ENVIRONMENT.json"
        if environment_path.exists():
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment["canonical_operator"] = OPERATOR_NAME
            environment["frontier_extension"] = frontier.EXTENSION
            _atomic_json(environment_path, environment)
        bundle = package_frontier(out, repo)
        print("FORMATION-MUX + TIE-ROLE STATUS:", state["status"], flush=True)
        print("FRONTIER ARMS:", state["complete_arms"], "/", state["required_arms"], flush=True)
        if (out / "TIE_ROLE_ARCHITECTURE_GATE.json").exists():
            gate = json.loads((out / "TIE_ROLE_ARCHITECTURE_GATE.json").read_text(encoding="utf-8"))
            print("TIE-ROLE ARCHITECTURE GATE:", gate["classification"], flush=True)
        print("RESULT BUNDLE:", json.dumps(bundle), flush=True)
        return 0
    except Exception as exc:
        _atomic_json(out / "GLOBAL_FAILURE_V10.json", {
            "schema": "anra.formation-mux-global-failure/v10",
            "operator": OPERATOR_NAME,
            "frontier_extension": frontier.EXTENSION,
            "exception": type(exc).__name__,
            "message": str(exc),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        try:
            print("FAILURE BUNDLE:", json.dumps(package_frontier(out, repo)), flush=True)
        except Exception:
            pass
        print("GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
