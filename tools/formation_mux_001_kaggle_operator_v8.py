"""Canonical FORMATION-MUX-001 Kaggle operator — Science S5, ops v8.

Runs the frozen campaign for as many Kaggle sessions as required. A session
uses the available wall safely, checkpoints are exact-resume, and no scientific
exposure is shortened to fit a session.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time, zipfile
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v7 as v7
from v5_experiments import formation_mux_protocol_v5 as proto

SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
OPERATOR_NAME = "tools/formation_mux_001_kaggle_operator_v8.py"
CAMPAIGN_ROOT = Path("/kaggle/working/FORMATION_MUX_001")
BUNDLE_NAME = "FORMATION_MUX_001_RESULTS.zip"

SCIENCE_FILES = v7.SCIENCE_FILES + (
    "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1D_PREEXECUTION_LONG_RUN_DATA_CUSTODY.md",
    "docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION_V5.json",
    "docs/cymek/experiments/CS-MECH-002/PREREGISTRATION_V5.json",
    "docs/cymek/experiments/REP-FORM-003A/PREREGISTRATION_V5.json",
    "v5_experiments/formation_mux_surface_v5.py",
    "v5_experiments/formation_mux_protocol_v5.py",
    "anra_v5/formation_mux_train_v5.py",
    "tools/formation_mux_001_worker_v5.py",
    "tests/test_formation_mux_001_v5.py",
)

# Rebind audited S4 engineering helpers to frozen S5.
v7.SCIENCE_COMMIT = SCIENCE_COMMIT
v7.SCIENCE_FILES = SCIENCE_FILES
v7.OPERATOR_NAME = OPERATOR_NAME
v7.proto = proto
v7.base.SCIENCE_COMMIT = SCIENCE_COMMIT
v7.base.SCIENCE_FILES = SCIENCE_FILES
v7.base.proto = proto
v7.partition.SCIENCE_COMMIT = SCIENCE_COMMIT
v7.partition.proto = proto


def worker_cmd(*, experiment: str, arm: str, seed: int, surface: Path, out: Path,
               engineering_only: bool = False, a_updates: int | None = None,
               b_tokens: int | None = None) -> list[str]:
    cmd = [
        sys.executable, "-m", "tools.formation_mux_001_worker_v5",
        "--experiment", experiment, "--arm", arm,
        "--seed-bundle", str(seed), "--surface", str(surface),
        "--out", str(out), "--device", "cuda",
    ]
    if engineering_only:
        cmd.append("--engineering-only")
    if a_updates is not None:
        cmd += ["--a-updates-override", str(a_updates)]
    if b_tokens is not None:
        cmd += ["--b-token-budget-override", str(b_tokens)]
    return cmd


v7.worker_cmd = worker_cmd
v7.base.worker_cmd = worker_cmd


def calibrate_long_run(repo: Path, public_surface: Path, out: Path) -> dict[str, Any]:
    """Measure ETA but never truncate frozen science because of projected duration."""
    receipt = v7.base.calibrate(repo, public_surface, out)
    receipt.update({
        "science_commit": SCIENCE_COMMIT,
        "partition_required": False,
        "science_wall_time_stop": False,
        "execution_policy": "run until actual per-session wall guard, package exact-resume state, continue next Kaggle session",
        "operator_wall_minutes": v7.MAX_OPERATOR_WALL_MINUTES,
        "finalization_package_reserve_minutes": v7.FINALIZATION_PACKAGE_RESERVE_MINUTES,
        "raw_sealed_rows_used": False,
    })
    v7.base._atomic_json(out / "CALIBRATION_RECEIPT.json", receipt)
    return receipt


def finalize_sealed(public_path: Path, out: Path, torch: Any, tokenizer: Any) -> dict[str, Any]:
    from anra_v5 import formation_mux_train_v5 as train
    from v5_experiments.formation_mux_surface_v5 import load_public_surface, regenerate_sealed_rows
    public = load_public_surface(public_path)
    device = torch.device("cuda:0")
    finals = {}
    for experiment in proto.EXPERIMENTS:
        final_path = out / experiment / "FINAL_RESULT.json"
        marker = v7.base._sealed_marker(out, experiment)
        if marker.get("state") == "COMPLETE" and final_path.exists():
            finals[experiment] = json.loads(final_path.read_text(encoding="utf-8"))
            continue
        if marker.get("state") == "STARTED" and not final_path.exists():
            raise v7.base.GlobalIntegrityError(
                f"sealed marker STARTED without FINAL_RESULT for {experiment}; preserve evidence and fail closed"
            )
        dev = v7.base.finalize_dev(out, experiment)
        if dev.get("status") != "DEVELOPMENT_COMPLETE":
            finals[experiment] = {"verdict": dev.get("status", "INCONCLUSIVE"), "sealed_consumed": False}
            continue
        v7.base._set_sealed(out, experiment, "STARTED")
        sealed_rows = regenerate_sealed_rows(
            public_manifest=public, tokenizer=tokenizer, experiment=experiment
        )
        arms = proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B
        observed = {arm: {} for arm in arms}
        secondary = {arm: {} for arm in arms}
        for arm in arms:
            for bundle in proto.SEED_BUNDLES:
                label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
                model = train.load_model_for_evaluation(
                    out / experiment / arm / label / "resume.pt",
                    experiment=experiment, arm=arm, seed_bundle=bundle,
                    data_manifest_sha256=str(public["sha256"]),
                    torch=torch, device=device,
                )
                scored = train.evaluate_development(
                    model, sealed_rows, experiment, arm, torch=torch, device=device
                )
                observed[arm][bundle] = float(scored["identity_exact_valid_eos"])
                secondary[arm][bundle] = scored["per_family"]
                del model
                torch.cuda.empty_cache()
        contrasts = []
        for row in dev["contrasts"]:
            higher, lower = row["higher"], row["lower"]
            auc = {int(k): float(v) for k, v in row["development_formation_auc_deltas"].items()}
            endpoint = {b: observed[higher][b] - observed[lower][b] for b in proto.SEED_BUNDLES}
            verdict = proto.paired_verdict(formation_auc_deltas=auc, sealed_endpoint_gaps=endpoint)
            contrasts.append({
                **row,
                "sealed_endpoint_gaps": {str(k): v for k, v in endpoint.items()},
                **verdict,
            })
        overall = proto.aggregate_experiment_verdict([x["verdict"] for x in contrasts])
        final = {
            "schema": "anra.formation-mux-final-result/v5",
            "science_commit": SCIENCE_COMMIT,
            "experiment": experiment,
            "sealed_commitment_sha256": public["sealed_commitments"][experiment],
            "raw_sealed_rows_persisted": False,
            "observed_sealed_identity_exact_valid_eos": {
                arm: {str(k): v for k, v in values.items()} for arm, values in observed.items()
            },
            "observed_sealed_per_family": {
                arm: {str(k): v for k, v in values.items()} for arm, values in secondary.items()
            },
            "contrast_results": contrasts,
            "verdict": overall,
            "claim_ceiling": proto.claim_ceiling(experiment, overall),
            "next_action": proto.NEXT_ACTION[overall],
            "sealed_consumed": True,
        }
        v7.base._atomic_json(final_path, final)
        v7.base._set_sealed(out, experiment, "COMPLETE")
        finals[experiment] = final
        del sealed_rows
    return finals


def package(out: Path, repo: Path) -> dict[str, Any]:
    from v5_experiments.formation_mux_surface_v5 import load_public_surface
    bundle = out.parent / BUNDLE_NAME
    if bundle.exists():
        bundle.unlink()
    public_path = out / "PUBLIC_SURFACE_MANIFEST.json"
    if public_path.exists():
        load_public_surface(public_path)
    if (out / "SURFACE_MANIFEST.json").exists():
        raise v7.base.GlobalIntegrityError("legacy raw sealed-containing manifest detected")
    docs = [
        "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1C_PREEXECUTION_SEALED_FIREWALL.md",
        "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1D_PREEXECUTION_LONG_RUN_DATA_CUSTODY.md",
        "docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION_V5.json",
        "docs/cymek/experiments/CS-MECH-002/PREREGISTRATION_V5.json",
        "docs/cymek/experiments/REP-FORM-003A/PREREGISTRATION_V5.json",
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SCIENCE_COMMIT.txt", SCIENCE_COMMIT + "\n")
        zf.writestr("OPERATOR_HEAD.txt", subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            capture_output=True, check=True).stdout)
        zf.writestr("README_S5.txt",
            "FORMATION-MUX-001 Science S5. Campaign may span multiple Kaggle sessions.\n"
            "Raw sealed rows are never persisted. Exact-resume checkpoints stay in Kaggle Output.\n"
            "LATEST_PROGRESS.json and progress/*.json are diagnostic snapshots.\n")
        for rel in docs:
            zf.write(repo / rel, "FROZEN_PROTOCOL/" + rel)
        for path in sorted(out.rglob("*.json")):
            if path.name != "SURFACE_MANIFEST.json":
                zf.write(path, path.relative_to(out.parent))
        for path in sorted(out.rglob("*.log")):
            zf.write(path, "RUNTIME_LOGS/" + path.relative_to(out).as_posix())
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    Path(str(bundle) + ".sha256").write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
    return {"path": str(bundle), "sha256": digest, "science_commit": SCIENCE_COMMIT,
            "resume_state_location": str(out), "raw_sealed_rows_included": False}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=CAMPAIGN_ROOT)
    p.add_argument("--skip-sealed", action="store_true")
    args = p.parse_args(argv)
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        import torch
        v7.base.verify_science(repo)
        environment = v7.base.require_hardware(torch)
        imported = v7.import_prior_state_s4(out)
        environment.update({
            "science_commit": SCIENCE_COMMIT,
            "imported_prior_state": imported,
            "canonical_operator": OPERATOR_NAME,
            "session_policy": "continue exact frozen campaign across as many sessions as needed",
            "operator_wall_minutes": v7.MAX_OPERATOR_WALL_MINUTES,
        })
        v7.base._atomic_json(out / "ENVIRONMENT.json", environment)
        if (out / "SURFACE_MANIFEST.json").exists():
            raise v7.base.GlobalIntegrityError("legacy raw SURFACE_MANIFEST.json detected")
        from v5_data.corpus_loading import _load_tokenizer
        from v5_experiments.formation_mux_surface_v5 import build_public_surface, load_public_surface
        tokenizer, tokenizer_eval = _load_tokenizer(repo)
        public_path = out / "PUBLIC_SURFACE_MANIFEST.json"
        if public_path.exists():
            public = load_public_surface(public_path)
            if public["tokenizer_artifact_sha256"] != tokenizer.identity.artifact_sha256:
                raise v7.base.GlobalIntegrityError("S5 public surface tokenizer identity drift")
        else:
            public = build_public_surface(seed=73011, tokenizer=tokenizer)
            v7.base._atomic_json(public_path, public)
        v7.base._atomic_json(out / "TOKENIZER_RECEIPT.json", {
            "vocabulary_size": int(tokenizer.identity.vocabulary_size),
            "artifact_sha256": tokenizer.identity.artifact_sha256,
            "evaluation": tokenizer_eval,
            "training_rows": 60000,
            "raw_sealed_rows_persisted": False,
            "sealed_commitments": public["sealed_commitments"],
        })
        v7.qualify(repo, public_path, out)
        calibration = calibrate_long_run(repo, public_path, out)
        state = v7.run_campaign(repo, public_path, out, calibration, operator_started=started)
        if state["status"] == "ARMS_COMPLETE":
            for experiment in proto.EXPERIMENTS:
                v7.base.finalize_dev(out, experiment)
            if not args.skip_sealed:
                finalize_sealed(public_path, out, torch, tokenizer)
                v7._sync_sealed_state(out)
        bundle = package(out, repo)
        print("FORMATION-MUX-001 STATUS:", state["status"])
        print("SCIENCE COMMIT S5:", SCIENCE_COMMIT)
        print("RESULT BUNDLE:", json.dumps(bundle))
        return v7.base.EXIT_OK
    except Exception as exc:
        v7.base._atomic_json(out / "GLOBAL_FAILURE.json", {
            "schema": "anra.formation-mux-global-failure/v8",
            "science_commit": SCIENCE_COMMIT, "operator": OPERATOR_NAME,
            "exception": type(exc).__name__, "message": str(exc),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        try:
            print("FAILURE BUNDLE:", json.dumps(package(out, repo)))
        except Exception:
            pass
        print("GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr)
        return v7.base.EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
