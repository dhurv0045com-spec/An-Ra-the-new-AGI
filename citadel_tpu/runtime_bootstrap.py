"""Pinned Cymek runtime bootstrap. No torch, no XLA, stdlib only.

Citadel descends from ESOES and never merges Cymek, so a bare `citadel`
checkout has no `anra_v5` / `v5_*` packages. This module resolves a READ-ONLY
detached Cymek runtime at a pinned SHA and puts it on `sys.path` — without
changing Citadel ancestry or touching `origin/cymek`.

Resolution order:
  1. `CITADEL_CYMEK_RUNTIME` env / explicit arg: use that directory as-is.
  2. Otherwise `<parent-of-citadel-root>/An-Ra-cymek-runtime`, created via
     `git fetch origin cymek --depth=1` + detached `git worktree add`.
SHA pin: `CITADEL_CYMEK_SHA` env / explicit arg, else PINNED_CYMEK_SHA.
Every failure raises RuntimeError with a PRECHECK_ code before any import,
compile, or device use. stdlib only: safe to run anywhere, including preflight.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PINNED_CYMEK_SHA = "298c91ac04f756f0833a7edcf63e73af3d5af688"
PIN_REASON = (
    "current origin/cymek HEAD at time of pinning; T0-relevant surface "
    "(anra_v5.miniature_run MINI_SPEC, v5_model/*, optimizer, checkpoint, pack) "
    "verified unchanged vs the last audited SHA 105ad22"
)

# Cymek-controlled modules the T0/T1/T2 path actually needs (nothing more).
REQUIRED_RELATIVE_PATHS = (
    "anra_v5/miniature_run.py",  # MINI_SPEC
    "v5_model/config.py",  # from_spec
    "v5_model/core.py",  # initialize, packed_layout
    "v5_contracts/model_spec.py",  # QK_NORM_EPSILON
    "v5_objectives/causal_lm.py",  # causal_lm_loss
    "v5_training/optimizer.py",  # build_adamw_optimizer
    "v5_training/distributed.py",  # T2 ledger schema
)


def _run(cmd: list[str], *, cwd: str | Path) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(cwd))
    except Exception as exc:
        raise RuntimeError(f"PRECHECK_GIT_FAILED: {' '.join(cmd)}: {exc!r}")
    if out.returncode != 0:
        raise RuntimeError(f"PRECHECK_GIT_FAILED: {' '.join(cmd)}: {(out.stderr or '').strip()[:300]}")
    return (out.stdout or "").strip()


def citadel_root() -> Path:
    """Locate the Citadel checkout root without importing anything else."""
    override = os.environ.get("CITADEL_ROOT", "").strip()
    if override and Path(override).is_dir():
        return Path(override)
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("PRECHECK_NO_CITADEL_ROOT: cannot locate Citadel checkout (.git not found)")


def citadel_sha() -> str | None:
    try:
        return _run(["git", "rev-parse", "HEAD"], cwd=citadel_root()) or None
    except RuntimeError:
        return None


def desired_sha(explicit: str | None = None) -> str:
    for candidate in (explicit, os.environ.get("CITADEL_CYMEK_SHA", "").strip()):
        if candidate:
            return candidate
    return PINNED_CYMEK_SHA


def default_runtime_dir() -> Path:
    override = os.environ.get("CITADEL_CYMEK_RUNTIME_DIR", "").strip()
    if override:
        return Path(override)
    return citadel_root().parent / "An-Ra-cymek-runtime"


def verify_files(root: str | Path) -> list[tuple[str, bool]]:
    """Check required Cymek paths exist. Pure file I/O: no imports, no torch."""
    root = Path(root)
    return [(rel, (root / rel).is_file()) for rel in REQUIRED_RELATIVE_PATHS]


def _worktree_sha(path: Path) -> str | None:
    try:
        return _run(["git", "-C", str(path), "rev-parse", "HEAD"], cwd=path) or None
    except RuntimeError:
        try:
            text = (path / ".git").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if text.startswith("gitdir:"):
            try:
                return _run(["git", "rev-parse", "HEAD"], cwd=path) or None
            except RuntimeError:
                return None
        return None


def ensure_cymek_runtime(*, runtime_dir: str | Path | None = None,
                         cymek_sha: str | None = None) -> tuple[Path, str]:
    """Resolve the pinned read-only Cymek runtime; return (root, sha).

    Raises RuntimeError(PRECHECK_*) with one clear cause on any problem.
    Never modifies branches; the worktree is detached.
    """
    explicit = os.environ.get("CITADEL_CYMEK_RUNTIME", "").strip() or runtime_dir
    sha = desired_sha(cymek_sha)
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise RuntimeError(f"PRECHECK_RUNTIME_MISSING: Cymek runtime dir does not exist: {root}")
        missing = [rel for rel, ok in verify_files(root) if not ok]
        if missing:
            raise RuntimeError(f"PRECHECK_IMPORT_FAILURE: runtime at {root} lacks: {', '.join(missing)}")
        found = _worktree_sha(root) or "explicit-path:unknown"
        _prepend_sys_path(root)
        return root, found
    root = Path(default_runtime_dir())
    if root.is_dir():
        missing = [rel for rel, ok in verify_files(root) if not ok]
        found = _worktree_sha(root)
        if not missing and found == sha:
            _prepend_sys_path(root)
            return root, found
        if not missing and found is not None and found != sha:
            raise RuntimeError(
                f"PRECHECK_PIN_MISMATCH: runtime at {root} is {found}, want {sha}; "
                "remove it or set CITADEL_CYMEK_SHA explicitly."
            )
    croot = citadel_root()
    _run(["git", "fetch", "origin", "cymek", "--depth=1"], cwd=croot)
    fetched = _run(["git", "rev-parse", "FETCH_HEAD"], cwd=croot)
    if fetched != sha:
        raise RuntimeError(
            f"PRECHECK_PIN_MISMATCH: origin/cymek fetched as {fetched}, want pinned {sha}; "
            "set CITADEL_CYMEK_SHA explicitly to adopt the new HEAD after auditing it."
        )
    _run(["git", "worktree", "add", "--detach", str(root), sha], cwd=croot)
    missing = [rel for rel, ok in verify_files(root) if not ok]
    if missing:
        raise RuntimeError(f"PRECHECK_IMPORT_FAILURE: fetched runtime lacks: {', '.join(missing)}")
    _prepend_sys_path(root)
    return root, sha


def _prepend_sys_path(root: Path) -> None:
    text = str(root)
    while text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def codebase_identities() -> dict[str, str | None]:
    """Both codebase SHAs for receipts: Citadel checkout + Cymek runtime pin."""
    return {"citadel_sha": citadel_sha(), "cymek_runtime_sha": desired_sha()}


__all__ = [
    "PINNED_CYMEK_SHA",
    "PIN_REASON",
    "REQUIRED_RELATIVE_PATHS",
    "citadel_root",
    "citadel_sha",
    "codebase_identities",
    "desired_sha",
    "ensure_cymek_runtime",
    "verify_files",
]
