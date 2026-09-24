"""Reproducible Signac source identity for repository and Kaggle snapshots."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SOURCE_IDENTITY_SCHEMA = "anra-signac-source-identity/v2"
SOURCE_DIRECTORIES = (
    "e0_cognition",
    "signac_100m",
    "v5_contracts",
    "v5_data",
    "v5_evaluation",
    "v5_model",
    "v5_objectives",
    "v5_registry",
    "v5_tokenizer",
    "v5_training",
)
SOURCE_FILES = (
    "notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb",
    "tools/signac_100m_model_smoke.py",
    "tools/signac_100m_preflight.py",
    "v5_checkpointing.py",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _git_commit(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else None


def build_source_identity(repo_root: Path) -> dict[str, Any]:
    """Hash the executable Signac dependency tree, with Git as optional metadata.

    Kaggle input datasets commonly omit ``.git``. A deterministic tree digest
    still identifies the code that actually ran, while a Git commit is
    included whenever it is available. File names and each file's content
    digest are both bound, so renames and edits change the tree identity.
    """

    root = repo_root.resolve()
    files: dict[str, str] = {}
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise ValueError(f"Signac source identity is missing required directory: {directory}")
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in {".py", ".md", ".json"}:
                continue
            relative = path.relative_to(root).as_posix()
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not any(name.startswith(directory + "/") for name in files):
            raise ValueError(f"Signac source identity found no source files in {directory}")
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Signac source identity is missing required file: {relative}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    inventory = [{"path": name, "sha256": files[name]} for name in sorted(files)]
    tree_sha256 = hashlib.sha256(_canonical_json(inventory)).hexdigest()
    return {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "source_commit": _git_commit(root),
        "source_tree_sha256": tree_sha256,
        "file_count": len(inventory),
        "files": inventory,
        "commit_note": "Git commit is optional for Kaggle dataset snapshots; source_tree_sha256 always binds the executable dependency inventory.",
    }


__all__ = ["SOURCE_IDENTITY_SCHEMA", "build_source_identity"]
