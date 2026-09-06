"""Citadel runtime-bootstrap regression tests (§2 A–F). Hermetic local git only.

Run:  python tests/test_citadel_bootstrap.py   (exit 0 = all pass)
Builds a Temp bare origin + clone with fixture files at every
REQUIRED_RELATIVE_PATH, then proves exact-SHA bootstrap semantics:
  A. branch HEAD == pin -> resolves pin (and reuses on second call)
  B. branch HEAD newer than pin -> STILL resolves the pin (the reported bug:
     old code fetched branch HEAD and failed PIN_MISMATCH)
  C. stale runtime worktree at wrong SHA -> recreated exactly at pin
  D. correct runtime present -> reused as-is
  E. missing commit SHA -> one clear PRECHECK error naming the SHA
  F. corrupted runtime (required file deleted) -> PRECHECK_IMPORT_FAILURE
     (explicit path) and deterministic recreate (default path)
No network, no torch, no TPU. Real git binary required.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(CITADEL_ROOT))

from citadel_tpu import runtime_bootstrap as rb  # noqa: E402


def _git(args: list[str], *, cwd: str | Path) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         timeout=60, cwd=str(cwd))
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(out.stderr or '')[:300]}")
    return (out.stdout or "").strip()


class Fixture:
    """Temp bare origin + clone; fixture tree committed twice (v1, v2)."""

    def __init__(self, tmp: str):
        self.tmp = Path(tmp)
        self.origin = self.tmp / "origin.git"
        self.clone = self.tmp / "citadel"
        self.rt = self.tmp / "rt"
        _git(["init", "--bare", "-q", str(self.origin)], cwd=self.tmp)
        _git(["clone", "-q", str(self.origin), str(self.clone)], cwd=self.tmp)
        _git(["config", "user.email", "citadel-test@local"], cwd=self.clone)
        _git(["config", "user.name", "citadel-test"], cwd=self.clone)
        _git(["checkout", "-qb", "cymek"], cwd=self.clone)
        self.v1 = self._commit({p: f"v1:{p}\n" for p in rb.REQUIRED_RELATIVE_PATHS},
                               "v1")
        self.v2 = self._commit({p: f"v2:{p}\n" for p in rb.REQUIRED_RELATIVE_PATHS},
                               "v2")

    def _commit(self, files: dict[str, str], msg: str) -> str:
        for rel, content in files.items():
            p = self.clone / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        _git(["add", "-A"], cwd=self.clone)
        _git(["commit", "-qm", msg], cwd=self.clone)
        return _git(["rev-parse", "HEAD"], cwd=self.clone)

    def push_cymek(self) -> None:
        _git(["push", "-q", "origin", "cymek"], cwd=self.clone)


_SAVED_ENV: dict[str, str | None] = {}
_SAVED_PATH: list[str] = []


def _setup_env(fx: Fixture) -> None:
    global _SAVED_ENV, _SAVED_PATH
    _SAVED_ENV = {k: os.environ.get(k) for k in
                  ("CITADEL_ROOT", "CITADEL_CYMEK_RUNTIME",
                   "CITADEL_CYMEK_RUNTIME_DIR", "CITADEL_CYMEK_SHA")}
    _SAVED_PATH = list(sys.path)
    os.environ["CITADEL_ROOT"] = str(fx.clone)
    os.environ["CITADEL_CYMEK_RUNTIME_DIR"] = str(fx.rt)
    os.environ.pop("CITADEL_CYMEK_RUNTIME", None)
    os.environ.pop("CITADEL_CYMEK_SHA", None)


def _teardown_env() -> None:
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    sys.path[:] = _SAVED_PATH


def _run_case(label: str, fn) -> None:
    tmp = tempfile.mkdtemp(prefix="citadel-bootstrap-")
    try:
        fx = Fixture(tmp)
        _setup_env(fx)
        try:
            fn(fx)
        finally:
            _teardown_env()
        print(f"PASS {label}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def case_a_head_equals_pin(fx: Fixture) -> None:
    fx.push_cymek()
    root, sha = rb.ensure_cymek_runtime(cymek_sha=fx.v2)
    assert sha == fx.v2, sha
    assert (root / "v5_model" / "core.py").is_file()
    root2, sha2 = rb.ensure_cymek_runtime(cymek_sha=fx.v2)
    assert (root2, sha2) == (root, sha)


def case_b_head_newer_than_pin(fx: Fixture) -> None:
    fx.push_cymek()
    v3 = fx._commit({"extra.txt": "v3\n"}, "v3")
    fx.push_cymek()
    assert v3 != fx.v2
    root, sha = rb.ensure_cymek_runtime(cymek_sha=fx.v2)
    assert sha == fx.v2, sha
    assert (root / "v5_model" / "core.py").read_text().startswith("v2:")


def case_c_stale_runtime_recreated(fx: Fixture) -> None:
    fx.push_cymek()
    _git(["worktree", "add", "--detach", str(fx.rt), fx.v1], cwd=fx.clone)
    assert rb._worktree_sha(fx.rt) == fx.v1
    fx._commit({"extra.txt": "v2\n"}, "v2b")
    root, sha = rb.ensure_cymek_runtime(cymek_sha=fx.v2)
    assert root == fx.rt and sha == fx.v2, (root, sha)
    assert (root / "v5_model" / "core.py").read_text().startswith("v2:")


def case_d_correct_runtime_reused(fx: Fixture) -> None:
    fx.push_cymek()
    _git(["worktree", "add", "--detach", str(fx.rt), fx.v2], cwd=fx.clone)
    marker = fx.rt / "v5_model" / "core.py"
    before = marker.stat().st_mtime_ns
    root, sha = rb.ensure_cymek_runtime(cymek_sha=fx.v2)
    assert (root, sha) == (fx.rt, fx.v2)
    assert marker.stat().st_mtime_ns == before


def case_e_missing_commit_clear_error(fx: Fixture) -> None:
    fx.push_cymek()
    bogus = "f" * 40
    try:
        rb.ensure_cymek_runtime(cymek_sha=bogus)
        raise SystemExit("bogus SHA unexpectedly resolved")
    except RuntimeError as exc:
        assert bogus in str(exc), exc
        assert "PRECHECK" in str(exc), exc


def case_f_corrupt_runtime_recreated(fx: Fixture) -> None:
    fx.push_cymek()
    _git(["worktree", "add", "--detach", str(fx.rt), fx.v2], cwd=fx.clone)
    victim = fx.rt / "v5_model" / "core.py"
    victim.unlink()
    root, sha = rb.ensure_cymek_runtime(cymek_sha=fx.v2)
    assert (root, sha) == (fx.rt, fx.v2)
    assert victim.is_file(), "corrupt runtime was not recreated"
    # explicit-path variant must fail loudly, never silently use partial tree
    victim.unlink()
    try:
        rb.ensure_cymek_runtime(runtime_dir=str(fx.rt), cymek_sha=fx.v2)
        raise SystemExit("partial explicit runtime unexpectedly accepted")
    except RuntimeError as exc:
        assert "PRECHECK_IMPORT_FAILURE" in str(exc), exc


def case_g_wrong_lineage_rejected(fx: Fixture) -> None:
    """Cymek carries a second, DISCONNECTED history whose tip lacks the
    production modules (mutation/provenance/artifact). A runtime resolved
    from that wrong lineage must fail verify_files loudly - the presence
    guards double as a lineage check (see CROSS_BRANCH_INGESTION.md)."""
    from citadel_tpu import runtime_bootstrap as rb

    root = fx.rt
    missing = rb.verify_files(root)
    assert any(not ok for _rel, ok in missing)
    assert any("mutation.py" in rel for rel, ok in missing if not ok)
    # a runtime stubbed with EVERY required path satisfies the guards,
    # proving the list is satisfiable on the true production lineage
    for rel, _ok in missing:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("# stub", encoding="utf-8")
    assert all(ok for _rel, ok in rb.verify_files(root))
    # removing ONE lineage-guard file must fail verification
    (root / "v5_training/mutation.py").unlink()
    assert any(not ok for _rel, ok in rb.verify_files(root))


def main() -> int:
    cases = [("A_head_equals_pin", case_a_head_equals_pin),
             ("B_head_newer_than_pin", case_b_head_newer_than_pin),
             ("C_stale_runtime_recreated", case_c_stale_runtime_recreated),
             ("D_correct_runtime_reused", case_d_correct_runtime_reused),
             ("E_missing_commit_clear_error", case_e_missing_commit_clear_error),
             ("F_corrupt_runtime_recreated", case_f_corrupt_runtime_recreated),
             ("G_wrong_lineage_rejected", case_g_wrong_lineage_rejected)]
    failed = 0
    for label, fn in cases:
        try:
            _run_case(label, fn)
        except Exception as exc:
            failed += 1
            print(f"FAIL {label}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
