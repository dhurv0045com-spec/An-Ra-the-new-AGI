"""Canonical FORMATION-MUX-001 operator for prospective Science S3.

S3 adds Amendment-1B: M2 frozen extra-row gradients remain in the global
clip while the row optimizer excludes those rows from updates/state/decay.
This operator reuses the audited dual-T4 engineering primitives, rebinds them
to the S3 protocol/worker, verifies every imported science dependency against
immutable S3, and performs S3 model-only sealed evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v2 as base
from tools import formation_mux_001_kaggle_operator_v3 as partition
from v5_experiments import formation_mux_protocol_v3 as proto

SCIENCE_COMMIT = "e157835a52a41696ca56512477584fd267391ced"
SCIENCE_FILES = (
    "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1_PREEXECUTION.md",
    "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1B_PREEXECUTION_CLIP_ISOLATION.md",
    "docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION.json",
    "docs/cymek/experiments/CS-MECH-002/PREREGISTRATION_V3.json",
    "docs/cymek/experiments/REP-FORM-003A/PREREGISTRATION_V3.json",
    "anra_v5/formation_mux_model_v2.py",
    "anra_v5/formation_mux_model_v3.py",
    "anra_v5/formation_mux_train_v2.py",
    "anra_v5/formation_mux_train_v3.py",
    "v5_experiments/formation_mux_protocol_v2.py",
    "v5_experiments/formation_mux_protocol_v3.py",
    "v5_experiments/formation_mux_surface_v2.py",
    "tools/formation_mux_001_worker_v3.py",
    "tests/test_formation_mux_001_v2.py",
    "tests/test_formation_mux_001_v3.py",
    "tests/test_formation_mux_surface_v2.py",
)

# Rebind v2/v3 engineering helpers to S3 before any helper is called.
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
        sys.executable, "-m", "tools.formation_mux_001_worker_v3",
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


base.worker_cmd = worker_cmd


def qualify(repo: Path, surface: Path, out: Path) -> dict[str, Any]:
    tests = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_formation_mux_001_v2.py",
            "tests/test_formation_mux_001_v3.py",
            "tests/test_formation_mux_surface_v2.py",
        ],
        cwd=repo, capture_output=True, text=True,
    )
    receipt: dict[str, Any] = {
        "pytest_returncode": tests.returncode,
        "pytest_stdout_tail": tests.stdout[-10000:],
        "pytest_stderr_tail": tests.stderr[-5000:],
    }
    if tests.returncode != 0:
        base._atomic_json(out / "QUALIFICATION.json", receipt)
        raise base.GlobalIntegrityError("Science S3 pre-execution tests failed")
    cmds = [
        worker_cmd(
            experiment=proto.EXPERIMENT_A, arm="M1_EXTRA_NO_DECAY",
            seed=proto.CALIBRATION_SEEDS[0], surface=surface,
            out=out / "qualification_s3", engineering_only=True, a_updates=2,
        ),
        worker_cmd(
            experiment=proto.EXPERIMENT_A, arm="M2_EXTRA_FROZEN",
            seed=proto.CALIBRATION_SEEDS[1], surface=surface,
            out=out / "qualification_s3", engineering_only=True, a_updates=2,
        ),
    ]
    procs = [base._spawn(cmds[i], gpu=i, repo=repo) for i in (0, 1)]
    codes = [p.wait() for p in procs]
    receipt["dual_t4_s3_codes"] = codes
    if codes != [0, 0]:
        base._atomic_json(out / "QUALIFICATION.json", receipt)
        raise base.GlobalIntegrityError(f"Science S3 dual-T4 tiny qualification failed: {codes}")
    receipt["status"] = "S3_GPU_E2E_PASS_ENGINEERING_ONLY"
    base._atomic_json(out / "QUALIFICATION.json", receipt)
    return receipt


def finalize_sealed(surface_path: Path, out: Path, torch: Any) -> dict[str, Any]:
    """One-shot S3 sealed endpoint: identity exact+valid-EOS for every contrast."""
    from v5_experiments.formation_mux_data import load_surface
    from anra_v5 import formation_mux_train_v3 as train

    surface = load_surface(surface_path)
    sealed_rows = list(surface["splits"]["sealed"])
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
                f"sealed marker STARTED without FINAL_RESULT for {experiment}; preserve and fail closed"
            )
        dev = base.finalize_dev(out, experiment)
        if dev.get("status") != "DEVELOPMENT_COMPLETE":
            finals[experiment] = {
                "verdict": dev.get("status", "INCONCLUSIVE"),
                "sealed_consumed": False,
            }
            continue
        base._set_sealed(out, experiment, "STARTED")
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
                    data_manifest_sha256=str(surface["sha256"]),
                    torch=torch,
                    device=device,
                )
                scored = train.evaluate_development(
                    model, sealed_rows, experiment, arm,
                    torch=torch, device=device,
                )
                observed[arm][bundle] = float(scored["identity_exact_valid_eos"])
                secondary[arm][bundle] = scored["per_family"]
                del model
                torch.cuda.empty_cache()

        contrasts = []
        for row in dev["contrasts"]:
            higher, lower = row["higher"], row["lower"]
            auc = {int(k): float(v) for k, v in row["development_formation_auc_deltas"].items()}
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
                    "sealed_endpoint_gaps": {str(k): v for k, v in endpoint.items()},
                    **verdict,
                }
            )
        overall = proto.aggregate_experiment_verdict([x["verdict"] for x in contrasts])
        final = {
            "schema": "anra.formation-mux-final-result/v3",
            "science_commit": SCIENCE_COMMIT,
            "experiment": experiment,
            "observed_sealed_identity_exact_valid_eos": {
                arm: {str(k): v for k, v in rows.items()} for arm, rows in observed.items()
            },
            "observed_sealed_per_family": {
                arm: {str(k): v for k, v in rows.items()} for arm, rows in secondary.items()
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
    return finals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=base.CAMPAIGN_ROOT)
    parser.add_argument("--skip-sealed", action="store_true")
    args = parser.parse_args(argv)
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        science_hashes = base.verify_science(repo)
        environment = base.require_hardware(torch)
        imported = base.import_prior_state(out)
        environment.update(
            {
                "science_commit": SCIENCE_COMMIT,
                "science_files_sha256": science_hashes,
                "imported_prior_state": imported,
                "canonical_operator": "tools/formation_mux_001_kaggle_operator_v4.py",
                "clip_isolation": "M2 frozen extra gradients retained through global clip",
            }
        )
        base._atomic_json(out / "ENVIRONMENT.json", environment)

        from v5_data.corpus_loading import _load_tokenizer
        from v5_experiments.formation_mux_surface_v2 import build_official_surface, validate_official_surface
        from v5_experiments.formation_mux_data import load_surface

        tokenizer, tokenizer_eval = _load_tokenizer(repo)
        surface_path = out / "SURFACE_MANIFEST.json"
        if surface_path.exists():
            surface = load_surface(surface_path)
            validate_official_surface(surface)
        else:
            surface = build_official_surface(seed=73011, tokenizer=tokenizer)
            base._atomic_json(surface_path, surface)
        base._atomic_json(
            out / "TOKENIZER_RECEIPT.json",
            {
                "vocabulary_size": int(tokenizer.identity.vocabulary_size),
                "special_token_ids": dict(tokenizer.identity.special_token_ids),
                "artifact_sha256": tokenizer.identity.artifact_sha256,
                "evaluation": tokenizer_eval,
            },
        )

        qualify(repo, surface_path, out)
        calibration = base.calibrate(repo, surface_path, out)
        state = partition.run_campaign(repo, surface_path, out, calibration)
        if state["status"] == "ARMS_COMPLETE":
            for experiment in proto.EXPERIMENTS:
                base.finalize_dev(out, experiment)
            if not args.skip_sealed:
                finalize_sealed(surface_path, out, torch)
                partition._sync_sealed_state(out)
        bundle = base.package(out, repo)
        print("FORMATION-MUX-001 STATUS:", state["status"])
        print("SCIENCE COMMIT S3:", SCIENCE_COMMIT)
        print("RESULT BUNDLE:", json.dumps(bundle))
        return base.EXIT_OK
    except Exception as exc:
        base._atomic_json(
            out / "GLOBAL_FAILURE.json",
            {
                "schema": "anra.formation-mux-global-failure/v4",
                "science_commit": SCIENCE_COMMIT,
                "exception": type(exc).__name__,
                "message": str(exc),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        try:
            print("FAILURE BUNDLE:", json.dumps(base.package(out, repo)))
        except Exception:
            pass
        print("GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr)
        return base.EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
