"""Prepare or restore a Drive-backed AN-RA training corpus for Colab."""

from __future__ import annotations

import argparse
import hashlib
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


CACHE_SCHEMA_VERSION = 2
CACHE_FILES = (
    "anra_training.txt",
    "reasoning.jsonl",
    "frontier_dfc.jsonl",
    "teacher_reasoning_v2.jsonl",
)
MANIFEST_NAME = "colab_data_cache.json"
TOKEN_SHARD_RELATIVE = Path("output") / "v2" / "data_manifests" / "native_foundation_v3"


def _profile_files(profile: str) -> tuple[str, ...]:
    if profile in {"smoke", "15gb", "30gb"}:
        return (*CACHE_FILES, "foundation_records.jsonl")
    return CACHE_FILES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for name in _profile_files(profile):
        path = root / name
        expected = files.get(name)
        if not path.is_file() or not isinstance(expected, dict):
            return False
        if int(expected.get("bytes", -1)) != path.stat().st_size:
            return False
        if str(expected.get("sha256", "")) != _sha256(path):
            return False
    return True


def write_manifest(root: Path, profile: str) -> Path:
    files = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": _sha256(root / name),
        }
        for name in _profile_files(profile)
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


def copy_cached_files(
    source: Path,
    destination: Path,
    profile: str = "t4-cached",
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in _profile_files(profile):
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
    for name in _profile_files(profile):
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
        "--publish-token-shards",
    ]
    print(f"[Data Cache] no valid cache found; building profile={profile} once.", flush=True)
    subprocess.run(command, cwd=repo, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore or build a persistent Colab data cache")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--profile", choices=["smoke", "15gb", "30gb"], default="30gb")
    parser.add_argument("--drive-root", default=str(DRIVE_DATA_DIR.parent))
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    local = repo / "training_data"
    drive = cache_dir(Path(args.drive_root), args.profile)
    local_tokens = repo / TOKEN_SHARD_RELATIVE / args.profile
    drive_tokens = (
        Path(args.drive_root)
        / "data"
        / "iterate500"
        / "token_shards"
        / args.profile
    )

    def token_cache_valid(root: Path) -> bool:
        manifests = list(root.rglob("manifest.json")) if root.exists() else []
        if not manifests:
            return False
        for token_manifest in manifests:
            try:
                payload = json.loads(token_manifest.read_text(encoding="utf-8"))
            except Exception:
                return False
            for item in payload.get("shards", []):
                path = token_manifest.parent / str(item.get("path", ""))
                if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
                    return False
        train_manifest = root / "manifest.json"
        if not train_manifest.is_file():
            return False
        train_payload = json.loads(train_manifest.read_text(encoding="utf-8"))
        return bool(train_payload.get("shards"))

    def restore_token_cache(source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for source_path in source.rglob("*"):
            if not source_path.is_file():
                continue
            target_path = destination / source_path.relative_to(source)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists() and _sha256(target_path) == _sha256(source_path):
                continue
            shutil.copy2(source_path, target_path)

    if (
        not args.force_rebuild
        and cache_is_valid(local, args.profile)
        and token_cache_valid(local_tokens)
    ):
        print_cache_summary(local, "local prepared corpus ready")
        return 0

    if (
        not args.force_rebuild
        and cache_is_valid(drive, args.profile)
        and token_cache_valid(drive_tokens)
    ):
        print_cache_summary(drive, "restoring prepared corpus from MyDrive")
        copy_cached_files(drive, local, args.profile)
        restore_token_cache(drive_tokens, local_tokens)
        if not cache_is_valid(local, args.profile):
            raise RuntimeError("Data cache restore finished but local validation failed.")
        print_cache_summary(local, "local prepared corpus ready")
        return 0

    build_fresh_data(repo, args.profile)
    cache_local_files(local, drive, args.profile)
    restore_token_cache(local_tokens, drive_tokens)
    write_manifest(local, args.profile)
    if (
        not cache_is_valid(local, args.profile)
        or not cache_is_valid(drive, args.profile)
        or not token_cache_valid(local_tokens)
        or not token_cache_valid(drive_tokens)
    ):
        raise RuntimeError("Fresh data build completed but cache validation failed.")
    print_cache_summary(drive, "MyDrive cache ready for future sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
