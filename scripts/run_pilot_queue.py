"""Validate, plan, and resumably execute signed pilot seed runs.

Dry-run planning is the default. Actual execution requires both the
owner-authorized signed manifests and an explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from anra.anra_paths import ROOT
from training.forecast_ledger import audit_pre_launch
from training.launch_manifest import load_and_validate_manifest

DEFAULT_ROOT = ROOT / "output" / "v2" / "campaigns" / "pilots"


def build_training_command(python: str, manifest_path: Path) -> list[str]:
    return [
        python,
        "-m",
        "training.train_unified",
        "--mode",
        "session",
        "--prepare-data",
        "never",
        "--launch-manifest",
        str(manifest_path.resolve()),
    ]


def artifact_paths(manifest: dict[str, object]) -> tuple[Path, Path]:
    artifact = Path(str(manifest["artifact_path"]))
    if not artifact.is_absolute():
        artifact = (ROOT / artifact).resolve()
    return artifact, artifact.with_suffix(".run.json")


def completed_run(manifest: dict[str, object]) -> bool:
    artifact, report_path = artifact_paths(manifest)
    if not artifact.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stages = report.get("stages", {})
    base = stages.get("base", {}) if isinstance(stages, dict) else {}
    return (
        isinstance(base, dict)
        and int(base.get("exit_code", -1)) == 0
        and int(report.get("seed", -1)) == int(manifest["seed"])
        and str((report.get("launch_manifest") or {}).get("run_id", ""))
        == str(manifest["run_id"])
    )


def _cuda_available(python: str) -> bool:
    result = subprocess.run(
        [python, "-c", "import torch;raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        check=False,
    )
    return result.returncode == 0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def plan_jobs(
    root: Path,
    *,
    include_moonshots: bool,
    cells: set[str],
    seeds: set[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ready: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for path in sorted((root / "cells").rglob("seed-*.json")):
        manifest = load_and_validate_manifest(path, allow_blocked=True)
        audit_pre_launch(manifest)
        cell_id = str(manifest.get("pilot_cell_id", ""))
        seed = int(manifest["seed"])
        reason = ""
        if cells and cell_id not in cells:
            reason = "cell-filter"
        elif seeds and seed not in seeds:
            reason = "seed-filter"
        elif bool(manifest.get("moonshot")) and not include_moonshots:
            reason = "moonshot-excluded"
        elif str(manifest.get("blocked_on", "")).strip():
            reason = f"blocked:{manifest['blocked_on']}"
        elif completed_run(manifest):
            reason = "already-complete"
        item = {
            "manifest_path": str(path.resolve()),
            "cell_id": cell_id,
            "seed": seed,
            "run_id": str(manifest["run_id"]),
            "artifact_path": str(artifact_paths(manifest)[0]),
            "moonshot": bool(manifest.get("moonshot")),
        }
        if reason:
            skipped.append({**item, "reason": reason})
        else:
            ready.append({**item, "manifest": manifest})
    return ready, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--python",
        default=str(ROOT / ".venv-cuda" / "Scripts" / "python.exe"),
    )
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--include-moonshots", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANRA_MANIFEST_SIGNING_KEY", ""):
        raise PermissionError("ANRA_MANIFEST_SIGNING_KEY is required to validate pilot jobs")
    ready, skipped = plan_jobs(
        args.root,
        include_moonshots=args.include_moonshots,
        cells=set(args.cell),
        seeds=set(args.seed),
    )
    public_ready = [{k: v for k, v in item.items() if k != "manifest"} for item in ready]
    plan = {
        "schema_version": 1,
        "generated_at": time.time(),
        "execute": bool(args.execute),
        "python": str(args.python),
        "ready": public_ready,
        "skipped": skipped,
    }
    _write_json(args.root / "queue_plan.json", plan)
    print(json.dumps({"ready": len(ready), "skipped": len(skipped)}, sort_keys=True))
    if not args.execute:
        return 0
    if not Path(args.python).is_file():
        raise FileNotFoundError(f"Pilot Python runtime is missing: {args.python}")
    if not args.allow_cpu and not _cuda_available(args.python):
        raise RuntimeError(
            "Pilot execution requires a CUDA runtime; use --allow-cpu only for a smoke"
        )

    failures = 0
    for index, item in enumerate(ready):
        manifest_path = Path(str(item["manifest_path"]))
        log_path = manifest_path.with_suffix(".stdout.log")
        status_path = manifest_path.with_suffix(".status.json")
        started = time.time()
        _write_json(
            status_path,
            {**public_ready[index], "status": "running", "started_at": started},
        )
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                build_training_command(args.python, manifest_path),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status = {
            **public_ready[index],
            "status": "complete" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "started_at": started,
            "ended_at": time.time(),
            "log_path": str(log_path.resolve()),
        }
        _write_json(status_path, status)
        if result.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
