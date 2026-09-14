"""Canonical FORMATION-MUX-001 Kaggle operator.

Wraps the Amendment-1 v2 engineering primitives but fixes partition/resume
selection so an incomplete matched bundle (including an arm-local failure)
is never abandoned merely because another arm in that bundle completed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v2 as base
from v5_experiments import formation_mux_protocol_v2 as proto

SCIENCE_COMMIT = base.SCIENCE_COMMIT


def _expected_keys(bundle: int) -> list[str]:
    label = f"S{proto.SEED_BUNDLES.index(bundle) + 1}"
    keys = []
    for experiment, arms in ((proto.EXPERIMENT_A, proto.ARMS_A), (proto.EXPERIMENT_B, proto.ARMS_B)):
        keys.extend(f"{experiment}/{arm}/{label}" for arm in arms)
    return keys


def _bundle_complete(bundle: int, complete: dict[str, str]) -> bool:
    return all(complete.get(key) == "COMPLETE" for key in _expected_keys(bundle))


def _choose_partition_bundles(complete: dict[str, str]) -> set[int]:
    pending = [b for b in proto.SEED_BUNDLES if not _bundle_complete(b, complete)]
    if not pending:
        return set()
    selected: list[int] = []
    for parity in (0, 1):
        candidates = [b for b in pending if proto.SEED_BUNDLES.index(b) % 2 == parity]
        if candidates:
            selected.append(candidates[0])
    return set(selected)


def run_campaign(repo: Path, surface: Path, out: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    complete = base._scan_completed(out)
    allowed = (
        _choose_partition_bundles(complete)
        if calibration["partition_required"]
        else {b for b in proto.SEED_BUNDLES if not _bundle_complete(b, complete)}
    )

    queues = proto.queue_assignment()
    pending = {
        gpu: [
            job for job in jobs
            if job["seed_bundle"] in allowed
            and f"{job['experiment']}/{job['arm']}/{job['seed_bundle_label']}" not in complete
        ]
        for gpu, jobs in queues.items()
    }

    state_path = out / "CAMPAIGN_STATE.json"
    old = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state = {
        "schema": "anra.formation-mux-state/v3",
        "science_commit": SCIENCE_COMMIT,
        "allowed_seed_bundles_this_session": sorted(allowed),
        "arms": {**old.get("arms", {}), **complete},
        "global_failure": old.get("global_failure"),
        "sealed": old.get("sealed", {exp: "NOT_CONSUMED" for exp in proto.EXPERIMENTS}),
    }
    base._atomic_json(state_path, state)

    stop = False
    while any(pending.values()) and not stop:
        running = {}
        for gpu in (0, 1):
            queue = pending[f"GPU{gpu}"]
            if not queue:
                continue
            job = queue.pop(0)
            cmd = base.worker_cmd(
                experiment=job["experiment"],
                arm=job["arm"],
                seed=job["seed_bundle"],
                surface=surface,
                out=out,
            )
            proc = base._spawn(cmd, gpu=gpu, repo=repo)
            running[gpu] = (job, proc)
            print(f"GPU{gpu} <- {job['experiment']}/{job['arm']}/{job['seed_bundle_label']}", flush=True)

        for gpu, (job, proc) in running.items():
            code = proc.wait()
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
    state["completed_seed_bundles"] = [b for b in proto.SEED_BUNDLES if _bundle_complete(b, complete)]
    state["pending_seed_bundles"] = [b for b in proto.SEED_BUNDLES if not _bundle_complete(b, complete)]
    if state.get("global_failure"):
        state["status"] = "FAILED_GLOBAL"
    elif len(complete) == proto.total_official_arms():
        state["status"] = "ARMS_COMPLETE"
    else:
        state["status"] = "PARTIAL_SESSION"
        state["resume_instruction"] = (
            "Save this Kaggle version/output, attach that output to the next run, "
            "and rerun the exact same pinned notebook. Completed arms are immutable; "
            "incomplete bundles remain selected until complete."
        )
    base._atomic_json(state_path, state)
    return state


def _sync_sealed_state(out: Path) -> None:
    path = out / "CAMPAIGN_STATE.json"
    if not path.exists():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    for experiment in proto.EXPERIMENTS:
        state.setdefault("sealed", {})[experiment] = base._sealed_marker(out, experiment).get("state", "NOT_CONSUMED")
    base._atomic_json(path, state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=base.CAMPAIGN_ROOT)
    parser.add_argument("--skip-sealed", action="store_true")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    out = args.out.resolve()
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
                "canonical_operator": "tools/formation_mux_001_kaggle_operator_v3.py",
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

        base.qualify(repo, surface_path, out)
        calibration = base.calibrate(repo, surface_path, out)
        state = run_campaign(repo, surface_path, out, calibration)
        if state["status"] == "ARMS_COMPLETE":
            for experiment in proto.EXPERIMENTS:
                base.finalize_dev(out, experiment)
            if not args.skip_sealed:
                base.finalize_sealed(surface_path, out, torch)
                _sync_sealed_state(out)
        bundle = base.package(out, repo)
        print("FORMATION-MUX-001 STATUS:", state["status"])
        print("RESULT BUNDLE:", json.dumps(bundle))
        return base.EXIT_OK
    except Exception as exc:
        failure = {
            "schema": "anra.formation-mux-global-failure/v3",
            "exception": type(exc).__name__,
            "message": str(exc),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        base._atomic_json(out / "GLOBAL_FAILURE.json", failure)
        try:
            print("FAILURE BUNDLE:", json.dumps(base.package(out, repo)))
        except Exception:
            pass
        print("GLOBAL FAIL-CLOSED:", repr(exc), file=sys.stderr)
        return base.EXIT_GLOBAL


if __name__ == "__main__":
    raise SystemExit(main())
