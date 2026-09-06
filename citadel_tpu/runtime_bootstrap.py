"""Pinned Cymek runtime bootstrap. No torch, no XLA, stdlib only.

Citadel descends from ESOES and never merges Cymek, so a bare `citadel`
checkout has no `anra_v5` / `v5_*` packages. This module resolves a READ-ONLY
detached Cymek runtime at a pinned SHA and puts it on `sys.path` — without
changing Citadel ancestry or touching `origin/cymek`.

Resolution order:
  1. `CITADEL_CYMEK_RUNTIME` env / explicit arg: use that directory as-is.
  2. Otherwise `<parent-of-citadel-root>/An-Ra-cymek-runtime`: reuse when it is
     exactly the pin, else recreate deterministically at the pin.
SHA pin: `CITADEL_CYMEK_SHA` env / explicit arg, else PINNED_CYMEK_SHA.
The pin is fetched by EXACT commit SHA, never by branch HEAD: a future
origin/cymek HEAD can neither break nor silently change a pinned experiment.
Every failure raises RuntimeError with a PRECHECK_ code before any import,
compile, or device use. stdlib only: safe to run anywhere, including preflight.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PINNED_CYMEK_SHA = "28bf57a0d299a2c13a99fe0046616c00a1b8530c"
PIN_REASON = (
    "current origin/cymek HEAD at time of pinning; T1D-relevant surface "
    "(anra_v5.miniature_run MINI_SPEC, v5_model/*, optimizer, checkpoint, "
    "state, pack, causal_lm) verified byte-identical vs the T0-certified SHA "
    "298c91a (delta is additive tokenizer/data-pipeline/registry/eval files, "
    "CLI and packaging entries only). See RUNTIME_AMENDMENT_001.md."
)

# Cymek-controlled modules the T1D/PRE50M path actually needs (nothing more).
REQUIRED_RELATIVE_PATHS = (
    "anra_v5/miniature_run.py",  # MINI_SPEC
    "v5_model/config.py",  # from_spec
    "v5_model/core.py",  # initialize, packed_layout
    "v5_contracts/model_spec.py",  # QK_NORM_EPSILON
    "v5_objectives/causal_lm.py",  # causal_lm_loss
    "v5_training/optimizer.py",  # build_adamw_optimizer
    "v5_training/distributed.py",  # T2 ledger schema
    "v5_training/checkpoint.py",  # CheckpointStore (production transactions)
    "v5_training/state.py",  # TrainingState/CursorState/IdentityBindings
    # production-lineage identity guards: these files exist ONLY on the
    # pinned production lineage. Cymek carries a second, disconnected
    # history (no common ancestor) whose tip LACKS mutation/provenance/
    # artifact - if the runtime ever resolves to that wrong lineage, the
    # presence check below fails loudly instead of silently training
    # against an unaudited variant (divergence recorded in
    # docs/citadel/CROSS_BRANCH_INGESTION.md).
    "v5_training/mutation.py",  # real-mutation certification
    "v5_training/provenance.py",  # step provenance
    "v5_tokenizer/artifact.py",  # tokenizer artifact identity
    "v5_data/pack.py",  # true multi-segment packing
    "v5_data/cursor.py",  # coordinate cursor
    "v5_data/mixture.py",  # frozen mixture allocation
    "v5_evaluation/checkpoint_adapter.py",  # checkpoint-backed evaluation
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


def _registered_worktrees(croot: str | Path) -> list[str]:
    """Worktree paths git knows about (read-only query)."""
    try:
        out = _run(["git", "worktree", "list", "--porcelain"], cwd=croot)
    except RuntimeError:
        return []
    roots = []
    for line in out.splitlines():
        if line.startswith("worktree "):
            roots.append(line[len("worktree "):].strip())
    return roots


def _fetch_exact_sha(croot: str | Path, sha: str) -> None:
    """Ensure commit object `sha` exists locally, independent of branch HEAD.

    Primary: fetch the exact SHA (works regardless of where origin/cymek
    points). Fallback: shallow branch fetch, then require the object.
    Raises RuntimeError(PRECHECK_*) when the commit is truly unavailable.
    """
    try:
        _run(["git", "fetch", "origin", sha, "--depth=1"], cwd=croot)
        _run(["git", "cat-file", "-e", sha], cwd=croot)
        return
    except RuntimeError:
        pass
    try:
        _run(["git", "fetch", "origin", "cymek", "--depth=50"], cwd=croot)
    except RuntimeError as exc:
        raise RuntimeError(
            f"PRECHECK_GIT_FAILED: cannot fetch cymek history for {sha}: {exc}")
    try:
        _run(["git", "cat-file", "-e", sha], cwd=croot)
    except RuntimeError:
        raise RuntimeError(
            f"PRECHECK_PIN_UNAVAILABLE: commit {sha} is not reachable from "
            "origin (neither direct-SHA fetch nor recent branch history "
            "contains it); audit a reachable SHA explicitly.")


def _remove_worktree(croot: str | Path, path: Path) -> None:
    """Remove a runtime worktree we own (registered) or a stale plain dir."""
    registered = [Path(p) for p in _registered_worktrees(croot)]
    if any(path == r or path in r.parents for r in registered):
        _run(["git", "worktree", "remove", "--force", str(path)], cwd=croot)
        return
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()
        return
    raise RuntimeError(
        f"PRECHECK_RUNTIME_CONFLICT: {path} exists but is not a registered "
        "Citadel runtime worktree; remove it manually or set "
        "CITADEL_CYMEK_RUNTIME_DIR explicitly.")


def ensure_cymek_runtime(*, runtime_dir: str | Path | None = None,
                          cymek_sha: str | None = None) -> tuple[Path, str]:
    """Resolve the pinned read-only Cymek runtime; return (root, sha).

    Exact-SHA semantics: a future origin/cymek HEAD never breaks (or silently
    changes) a pinned experiment. Reuses a valid existing runtime; recreates a
    stale/corrupt one; fetches the exact commit independent of branch HEAD.
    Raises RuntimeError(PRECHECK_*) with one clear cause on any problem.
    Never modifies branches; the worktree is always detached.
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
    croot = citadel_root()
    root = Path(default_runtime_dir())
    if root.is_dir():
        missing = [rel for rel, ok in verify_files(root) if not ok]
        found = _worktree_sha(root)
        if not missing and found == sha:
            _prepend_sys_path(root)
            return root, found
        # Stale SHA or corrupt/incomplete tree: recreate deterministically.
        _remove_worktree(croot, root)
    _fetch_exact_sha(croot, sha)
    _run(["git", "worktree", "add", "--detach", str(root), sha], cwd=croot)
    actual = _worktree_sha(root)
    if actual != sha:
        raise RuntimeError(
            f"PRECHECK_RUNTIME_MISMATCH: worktree resolved to {actual}, want {sha}")
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
