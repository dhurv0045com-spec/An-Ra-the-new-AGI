"""Canonical FORMATION-MUX-001 operator wrapper — Science S3, ops v5.

Adds failure-evidence custody without changing science: every child worker has
an append-only runtime log under the campaign root, and result bundles include
those logs plus the frozen preregistration/amendment documents.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v2 as base
from tools import formation_mux_001_kaggle_operator_v4 as science_operator

SCIENCE_COMMIT = science_operator.SCIENCE_COMMIT

_orig_package = base.package


def _arg(cmd: list[str], name: str, default: str = "unknown") -> str:
    try:
        return str(cmd[cmd.index(name) + 1])
    except (ValueError, IndexError):
        return default


def logged_spawn(cmd: list[str], *, gpu: int, repo: Path) -> subprocess.Popen:
    out = Path(_arg(cmd, "--out", str(base.CAMPAIGN_ROOT)))
    exp = _arg(cmd, "--experiment")
    arm = _arg(cmd, "--arm")
    seed = _arg(cmd, "--seed-bundle")
    log_dir = out / "_worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp}__{arm}__seed{seed}__gpu{gpu}.log"
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    handle.write("\n=== worker invocation ===\n")
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
    proc._formation_mux_log_handle = handle  # keep parent descriptor alive
    proc._formation_mux_log_path = str(log_path)
    return proc


def package_with_custody(out: Path, repo: Path):
    receipt = _orig_package(out, repo)
    bundle = Path(receipt["path"])
    frozen_docs = [
        "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1_PREEXECUTION.md",
        "docs/cymek/experiments/FORMATION-MUX-001/AMENDMENT_1B_PREEXECUTION_CLIP_ISOLATION.md",
        "docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION.json",
        "docs/cymek/experiments/CS-MECH-002/PREREGISTRATION_V3.json",
        "docs/cymek/experiments/REP-FORM-003A/PREREGISTRATION_V3.json",
    ]
    readme = (
        "FORMATION-MUX-001 Science S3 results/failure bundle.\n"
        "Science is immutable at commit " + SCIENCE_COMMIT + ".\n"
        "Worker logs are preserved even for arm-local or global engineering failures.\n"
        "S1/S2 are pre-execution audit history and are not valid run authorities.\n"
    )
    with zipfile.ZipFile(bundle, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README_S3.txt", readme)
        for rel in frozen_docs:
            path = repo / rel
            if path.exists():
                zf.write(path, "FROZEN_PROTOCOL/" + rel)
        for path in sorted(Path(out).rglob("*.log")):
            zf.write(path, "RUNTIME_LOGS/" + path.relative_to(out).as_posix())
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    Path(str(bundle) + ".sha256").write_text(
        f"{digest}  {bundle.name}\n", encoding="utf-8"
    )
    return {**receipt, "sha256": digest, "science_commit": SCIENCE_COMMIT,
            "worker_logs_included": True, "frozen_protocol_included": True}


# Dynamic calls inside the S3 operator resolve through this shared base module.
base._spawn = logged_spawn
base.package = package_with_custody


def main(argv=None) -> int:
    return science_operator.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
