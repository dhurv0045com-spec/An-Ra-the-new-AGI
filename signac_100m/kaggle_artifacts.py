"""Package Signac Kaggle run evidence for download and notebook-output saving."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile
from typing import Any

from .source_identity import SOURCE_DIRECTORIES, SOURCE_FILES


BUNDLE_SCHEMA = "signac-100m-kaggle-results-bundle/v1"
DEFAULT_CHECKPOINT_EVERY_UPDATES = 200
_TEXT_SUFFIXES = {".ipynb", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
_SOURCE_DIRS = SOURCE_DIRECTORIES
_EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _source_files(repo_root: Path):
    for dirname in _SOURCE_DIRS:
        directory = repo_root / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if (not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES
                    or any(part in _EXCLUDED_PARTS for part in path.parts)
                    or not _inside(path, repo_root)):
                continue
            yield path, f"source/{path.relative_to(repo_root).as_posix()}", "source"

    for relative in SOURCE_FILES:
        path = repo_root / relative
        if path.is_file() and _inside(path, repo_root):
            yield path, f"source/{relative}", "source"

    docs = repo_root / "docs" / "signac_100m"
    if docs.is_dir():
        for path in sorted(docs.rglob("*.md")):
            if path.is_file() and _inside(path, repo_root):
                yield path, f"source/{path.relative_to(repo_root).as_posix()}", "source"


def create_kaggle_results_bundle(
    *,
    output_root: str | Path,
    repo_root: str | Path,
    run_id: str,
    status: str,
    receipt_root: str | Path | None = None,
    source_identity: dict[str, Any] | None = None,
    error: str | None = None,
    checkpoint_every_updates: int = DEFAULT_CHECKPOINT_EVERY_UPDATES,
) -> dict[str, Any]:
    """Create and verify a results ZIP, including partial receipts on failure.

    Only Signac-prefixed top-level JSON results, the selected run's receipt
    tree, and the small reproducibility source bundle are included. Checkpoint
    payloads remain byte-for-byte intact in the ZIP. The archive is published
    atomically only after its central directory and CRCs verify successfully.
    """

    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", run_id):
        raise ValueError("run_id must be a short filename-safe identifier")
    if status not in {"complete", "failed", "interrupted"}:
        raise ValueError("status must be complete, failed, or interrupted")
    if type(checkpoint_every_updates) is not int or checkpoint_every_updates <= 0:
        raise ValueError("checkpoint_every_updates must be a positive integer")

    output = Path(output_root).resolve()
    repo = Path(repo_root).resolve()
    if not output.is_dir():
        raise ValueError(f"Kaggle output directory does not exist: {output}")
    if not repo.is_dir():
        raise ValueError(f"repository root does not exist: {repo}")

    candidates: dict[str, tuple[Path, str]] = {}

    def add(path: Path, archive_name: str, category: str, anchor: Path) -> None:
        if not path.is_file() or not _inside(path, anchor):
            return
        normalized = Path(archive_name).as_posix()
        if normalized.startswith("/") or ".." in Path(normalized).parts:
            raise ValueError(f"unsafe archive path: {archive_name}")
        if normalized in candidates:
            raise ValueError(f"duplicate archive path: {normalized}")
        candidates[normalized] = (path.resolve(), category)

    stale_on_failure = {
        "signac_100m_all_core_canaries.json",
        "signac_100m_post_canary_preflight.json",
    }
    for path in sorted(output.glob("signac_100m_*.json")):
        if status != "complete" and path.name in stale_on_failure:
            continue
        add(path, f"results/{path.name}", "result", output)

    if receipt_root is not None:
        receipts = Path(receipt_root).resolve()
        if _inside(receipts, output) and receipts.is_dir():
            for path in sorted(receipts.rglob("*")):
                if path.is_file() and not any(part in _EXCLUDED_PARTS for part in path.parts):
                    relative = path.relative_to(receipts).as_posix()
                    add(path, f"receipts/{receipts.name}/{relative}", "receipt", output)

    for path, archive_name, category in _source_files(repo):
        add(path, archive_name, category, repo)

    manifest_entries = []
    total_input_bytes = 0
    for archive_name, (path, category) in sorted(candidates.items()):
        size = path.stat().st_size
        total_input_bytes += size
        manifest_entries.append({
            "path": archive_name,
            "category": category,
            "size_bytes": size,
            "sha256": _sha256_file(path),
        })

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "run_id": run_id,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity or {},
        "checkpoint_policy": {
            "every_optimizer_updates": checkpoint_every_updates,
            "note": "Production campaign checkpoints at this cadence; shorter Kaggle qualification canaries do not reach update 200.",
        },
        "error": error[:2000] if error else None,
        "files": manifest_entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    archive_path = output / f"signac_100m_results_{run_id}.zip"
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".partial")
    existing_archive_bytes = archive_path.stat().st_size if archive_path.exists() else 0
    required_free = (total_input_bytes + existing_archive_bytes
                     + max(64 * 1024 * 1024, total_input_bytes // 100))
    free_bytes = shutil.disk_usage(output).free
    if free_bytes < required_free:
        raise OSError(
            f"not enough free space to create results bundle: need about "
            f"{required_free} bytes, have {free_bytes}; loose Kaggle outputs were left intact"
        )

    try:
        with zipfile.ZipFile(temporary_path, "w", allowZip64=True) as bundle:
            for archive_name, (path, _category) in sorted(candidates.items()):
                compression = zipfile.ZIP_STORED if path.stat().st_size >= 64 * 1024 * 1024 else zipfile.ZIP_DEFLATED
                bundle.write(path, arcname=archive_name, compress_type=compression)
            bundle.writestr(
                "BUNDLE_MANIFEST.json", manifest_bytes,
                compress_type=zipfile.ZIP_DEFLATED,
            )
        with zipfile.ZipFile(temporary_path, "r") as bundle:
            corrupt_member = bundle.testzip()
            if corrupt_member is not None:
                raise OSError(f"ZIP integrity check failed at {corrupt_member}")
        os.replace(temporary_path, archive_path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    archive_sha256 = _sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_temp = checksum_path.with_suffix(checksum_path.suffix + ".partial")
    checksum_temp.write_text(f"{archive_sha256}  {archive_path.name}\n", encoding="ascii")
    os.replace(checksum_temp, checksum_path)
    return {
        "path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "sha256": archive_sha256,
        "checksum_path": str(checksum_path),
        "file_count": len(manifest_entries),
        "status": status,
    }


__all__ = ["BUNDLE_SCHEMA", "DEFAULT_CHECKPOINT_EVERY_UPDATES", "create_kaggle_results_bundle"]
