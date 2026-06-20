"""Prepare or restore a Drive-backed AN-RA training corpus for Colab."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anra.anra_paths import DRIVE_DATA_DIR


CACHE_SCHEMA_VERSION = 1
CACHE_FILES = (
    "anra_training.txt",
    "reasoning.jsonl",
    "frontier_dfc.jsonl",
    "teacher_reasoning_v2.jsonl",
)
MANIFEST_NAME = "colab_data_cache.json"


def cache_dir(drive_root: Path, profile: str) -> Path:
    return drive_root / "data" / "iterate500" / profile


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _read_manifest(root: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def cache_is_valid(root: Path, profile: str) -> bool:
    manifest = _read_manifest(root)
    if not manifest:
        return False
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION or manifest.get("profile") != profile:
        return False
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    for name in CACHE_FILES:
        path = root / name
        expected = files.get(name)
        if not path.is_file() or not isinstance(expected, dict):
            return False
        if int(expected.get("bytes", -1)) != path.stat().st_size:
            return False
    return True


def write_manifest(root: Path, profile: str) -> Path:
    files = {
        name: {"bytes": (root / name).stat().st_size}
        for name in CACHE_FILES
    }
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "profile": profile,
        "generated_at": time.time(),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path(root).with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path(root))
    return manifest_path(root)


def copy_cached_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES:
        source_path = source / name
        target_path = destination / name
        if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
            print(f"[Data Cache] local file already current: {name}", flush=True)
            continue
        print(f"[Data Cache] restoring {name}", flush=True)
        shutil.copy2(source_path, target_path)
    shutil.copy2(manifest_path(source), manifest_path(destination))


def cache_local_files(source: Path, destination: Path, profile: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES:
        source_path = source / name
        if not source_path.is_file():
            raise FileNotFoundError(f"Prepared training file is missing: {source_path}")
        target_path = destination / name
        if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
            print(f"[Data Cache] Drive file already current: {name}", flush=True)
            continue
        print(f"[Data Cache] caching {name} in MyDrive", flush=True)
        shutil.copy2(source_path, target_path)
    write_manifest(destination, profile)


def print_cache_summary(root: Path, label: str) -> None:
    manifest = _read_manifest(root)
    total = int(manifest.get("total_bytes", 0)) if manifest else 0
    print(f"[Data Cache] {label}: {total / 1024**3:.2f} GB at {root}", flush=True)


def build_fresh_data(repo: Path, profile: str) -> None:
    command = [
        sys.executable,
        "scripts/download_training_data.py",
        "--profile",
        profile,
        "--prepare-corpus",
    ]
    print(f"[Data Cache] no valid cache found; building profile={profile} once.", flush=True)
    subprocess.run(command, cwd=repo, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore or build a persistent Colab data cache")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--profile", default="t4-cached")
    parser.add_argument("--drive-root", default=str(DRIVE_DATA_DIR.parent))
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    local = repo / "training_data"
    drive = cache_dir(Path(args.drive_root), args.profile)

    if not args.force_rebuild and cache_is_valid(local, args.profile):
        print_cache_summary(local, "local prepared corpus ready")
        return 0

    if not args.force_rebuild and cache_is_valid(drive, args.profile):
        print_cache_summary(drive, "restoring prepared corpus from MyDrive")
        copy_cached_files(drive, local)
        if not cache_is_valid(local, args.profile):
            raise RuntimeError("Data cache restore finished but local validation failed.")
        print_cache_summary(local, "local prepared corpus ready")
        return 0

    build_fresh_data(repo, args.profile)
    cache_local_files(local, drive, args.profile)
    write_manifest(local, args.profile)
    if not cache_is_valid(local, args.profile) or not cache_is_valid(drive, args.profile):
        raise RuntimeError("Fresh data build completed but cache validation failed.")
    print_cache_summary(drive, "MyDrive cache ready for future sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
