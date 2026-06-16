"""Canonical Google Colab bootstrap and environment report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from anra.anra_paths import DRIVE_DIR


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def install_thirdeye(repo: Path) -> Path:
    target = repo.parent / "thirdeye"
    if not target.exists():
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/dhurv0045com-spec/thirdeye.git",
                str(target),
            ]
        )
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(target)])
    return target


def require_cuda_gpu() -> None:
    if torch.cuda.is_available():
        return
    raise SystemExit(
        "AN-RA iterate900 requires a CUDA GPU runtime. "
        "In Colab, use Runtime -> Change runtime type -> T4 GPU. "
        "TPU v5e/CPU runtimes are not supported by this PyTorch trainer."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path.cwd()))
    parser.add_argument("--drive-root", default=str(DRIVE_DIR))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-thirdeye", action="store_true")
    parser.add_argument("--model-size", default="frontier", choices=["frontier"])
    parser.add_argument("--allow-non-cuda", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    constraints = repo / "constraints-colab-t4.txt"
    if not args.allow_non_cuda:
        require_cuda_gpu()
    if args.install:
        run([sys.executable, "-m", "pip", "install", "-e", f"{repo}[evidence]", "-c", str(constraints)])
    thirdeye_path = None
    if args.install or args.install_thirdeye:
        thirdeye_path = install_thirdeye(repo)
    drive = Path(args.drive_root)
    colab_root = Path(os.environ.get("ANRA_COLAB_ROOT", str(Path("/") / "content")))
    scratch = colab_root / "anra-scratch" if colab_root.exists() else repo / "output" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    drive.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "generated_at": time.time(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "flash_sdp": bool(torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled()),
        "repo": str(repo),
        "thirdeye": str(thirdeye_path) if thirdeye_path else None,
        "scratch": str(scratch),
        "drive": str(drive),
        "constraints_sha256": sha256(constraints),
        "secrets_in_environment": {
            "owner_token": bool(os.environ.get("ANRA_OWNER_TOKEN")),
            "manifest_key": bool(os.environ.get("ANRA_MANIFEST_SIGNING_KEY")),
        },
    }
    report_path = scratch / "bootstrap_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    command = [
        sys.executable,
        "-m",
        "training.train_unified",
        "--mode",
        "preflight",
        "--model-size",
        args.model_size,
        "--prepare_data",
        "never",
    ]
    completed = subprocess.run(command, cwd=repo, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
