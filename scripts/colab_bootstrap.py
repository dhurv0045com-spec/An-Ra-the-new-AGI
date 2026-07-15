"""Canonical Google Colab bootstrap and environment report."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from anra.anra_paths import DRIVE_DIR  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


RUNTIME_REQUIREMENTS = {
    "aiosqlite": ("aiosqlite", "0.19.0"),
    "cryptography": ("cryptography", "42.0.0"),
    "datasets": ("datasets", "3.2.0"),
    "fastapi": ("fastapi", "0.110.0"),
    "git": ("GitPython", "3.1.40"),
    "httpx": ("httpx", "0.27.0"),
    "networkx": ("networkx", "3.2.0"),
    "pydantic": ("pydantic", "2.5.0"),
    "transformers": ("transformers", "4.40.0"),
    "tokenizers": ("tokenizers", "0.19.0"),
    "uvicorn": ("uvicorn", "0.27.0"),
    "yaml": ("PyYAML", "6.0.0"),
    "psutil": ("psutil", "5.9.0"),
    "scipy": ("scipy", "1.12.0"),
    "sympy": ("sympy", "1.12.0"),
    "tqdm": ("tqdm", "4.66.0"),
}


def _version_at_least(installed: str, minimum: str) -> bool:
    try:
        from packaging.version import Version

        return Version(installed) >= Version(minimum)
    except Exception:
        return installed >= minimum


def missing_runtime_packages(
    requirements: dict[str, tuple[str, str]] = RUNTIME_REQUIREMENTS,
) -> list[str]:
    """Return only the small runtime packages absent from the Colab image."""
    missing: list[str] = []
    for module_name, (package_name, minimum_version) in requirements.items():
        try:
            importlib.import_module(module_name)
            installed = metadata.version(package_name)
        except Exception:
            missing.append(f"{package_name}>={minimum_version}")
            continue
        if not _version_at_least(installed, minimum_version):
            missing.append(f"{package_name}>={minimum_version}")
    return missing


def install_runtime_packages(packages: Iterable[str]) -> list[str]:
    packages = list(packages)
    if not packages:
        print("[Colab] runtime dependencies already available; skipping pip download.")
        return []
    print(f"[Colab] installing only missing runtime packages: {', '.join(packages)}")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            "--prefer-binary",
            *packages,
        ]
    )
    return packages


def configure_local_pip_cache() -> Path:
    """Keep pip's content-addressed cache out of mounted Google Drive."""
    requested = Path(os.environ.get("ANRA_PIP_CACHE", "/content/.cache/pip"))
    cache = requested.expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("PIP_CACHE_DIR", "")
    os.environ["PIP_CACHE_DIR"] = str(cache)
    drive_mount_prefix = "/content" + "/drive/"
    if drive_mount_prefix in previous.replace("\\", "/"):
        print(f"[Colab] moved pip cache off Drive: {cache}")
    return cache


def install_project(repo: Path) -> None:
    """Expose this checkout without asking pip to replace Colab's CUDA torch."""
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])


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
    else:
        run(["git", "-C", str(target), "fetch", "origin", "main"])
        run(["git", "-C", str(target), "checkout", "main"])
        run(["git", "-C", str(target), "pull", "--ff-only", "origin", "main"])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(target)])
    return target


def require_cuda_gpu() -> None:
    if torch.cuda.is_available():
        return
    raise SystemExit(
        "AN-RA iterate500 requires a CUDA GPU runtime. "
        "In Colab, use Runtime -> Change runtime type -> T4 GPU. "
        "TPU v5e/CPU runtimes are not supported by this PyTorch trainer."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path.cwd()))
    parser.add_argument("--drive-root", default=str(DRIVE_DIR))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-thirdeye", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run the slower full unified-trainer preflight after bootstrap.",
    )
    parser.add_argument(
        "--model-size", default="anra-v4-180m", choices=["anra-v4-180m"]
    )
    parser.add_argument("--allow-non-cuda", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    constraints = repo / "constraints-colab-t4.txt"
    if not args.allow_non_cuda:
        require_cuda_gpu()
    pip_cache = configure_local_pip_cache()
    installed_packages: list[str] = []
    if args.install:
        installed_packages = install_runtime_packages(missing_runtime_packages())
        install_project(repo)
    thirdeye_path = None
    # ThirdEye is useful for training evidence, but chat/UI bootstrap must not
    # depend on a second Git checkout or an avoidable network operation.
    if args.install_thirdeye:
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
        "pip_cache": str(pip_cache),
        "drive": str(drive),
        "constraints_sha256": sha256(constraints),
        "installed_runtime_packages": installed_packages,
        "secrets_in_environment": {
            "owner_token": bool(os.environ.get("ANRA_OWNER_TOKEN")),
            "manifest_key": bool(os.environ.get("ANRA_MANIFEST_SIGNING_KEY")),
        },
    }
    report_path = scratch / "bootstrap_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.preflight:
        print("[Colab] fast bootstrap complete; full preflight skipped for faster session start.")
        return
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
