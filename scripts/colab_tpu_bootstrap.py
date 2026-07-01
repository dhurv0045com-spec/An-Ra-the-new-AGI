"""Google Colab TPU bootstrap and runtime report for iterate500."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from anra.anra_paths import DRIVE_DIR
from training.tpu_runtime import TPUUnavailableError, require_torch_xla


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path.cwd()))
    parser.add_argument("--drive-root", default=str(DRIVE_DIR))
    parser.add_argument("--install-repo", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    drive = Path(args.drive_root)
    drive.mkdir(parents=True, exist_ok=True)
    if args.install_repo:
        run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo)])

    try:
        xm, _pl = require_torch_xla()
    except TPUUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    supported = []
    try:
        supported = list(xm.get_xla_supported_devices())
    except Exception:
        supported = []
    if not supported:
        raise SystemExit(
            "No XLA devices were found. In Colab use Runtime -> Change runtime type -> TPU."
        )
    device = xm.xla_device()
    probe = torch.tensor([1.0, 2.0], device=device).sum()
    xm.mark_step()

    colab_root = Path(os.environ.get("ANRA_COLAB_ROOT", str(Path("/") / "content")))
    scratch = colab_root / "anra-tpu-scratch" if colab_root.exists() else repo / "output" / "tpu-scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "generated_at": time.time(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_xla_device": str(device),
        "xla_supported_devices": supported,
        "pjrt_device": os.environ.get("PJRT_DEVICE"),
        "colab_tpu_addr": os.environ.get("COLAB_TPU_ADDR"),
        "probe_sum": float(probe.cpu().item()),
        "repo": str(repo),
        "scratch": str(scratch),
        "drive": str(drive),
        "readme_sha256": sha256(repo / "README.md"),
    }
    report_path = scratch / "bootstrap_tpu_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
