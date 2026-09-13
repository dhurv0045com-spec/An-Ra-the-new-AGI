"""Source provenance helpers (B2.2 R1/R4): the real code-snapshot identity.

A generic HEAD-dirty suffix is not an identity for dirty code bytes. The
source closure digest covers every Python file under ``bramastra_lab`` so
any edit changes identity; runs and checkpoints record it and restore
validates compatibility, with migration handled explicitly.
"""
from __future__ import annotations

import hashlib
import os
import subprocess


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _package_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def source_closure_sha256() -> str:
    """SHA-256 over (relative path, file digest) for every bramastra_lab .py file."""
    package_root = _package_root()
    digest = hashlib.sha256()
    files: list[tuple[str, str]] = []
    for base, _dirs, names in os.walk(package_root):
        if "__pycache__" in base:
            continue
        for name in names:
            if name.endswith(".py"):
                absolute = os.path.join(base, name)
                files.append((os.path.relpath(absolute, package_root), absolute))
    for relative, absolute in sorted(files):
        digest.update(relative.replace(os.sep, "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(open(absolute, "rb").read()).hexdigest().encode())
        digest.update(b"\0")
    return digest.hexdigest()


def source_identity() -> dict[str, str]:
    identity = {"git_head": "unavailable", "source_closure_sha256": "unavailable",
                "dirty": "unknown"}
    try:
        identity["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True, timeout=10).strip()
        identity["dirty"] = "true" if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_repo_root(), text=True,
            timeout=10).strip() else "false"
    except Exception:
        pass
    try:
        identity["source_closure_sha256"] = source_closure_sha256()
    except Exception:
        pass
    return identity
