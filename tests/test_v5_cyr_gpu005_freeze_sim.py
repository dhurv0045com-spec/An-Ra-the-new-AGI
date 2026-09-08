"""Freeze/checkout simulation (sections 43, 44): the two-commit protocol
and the notebook CELL 0 order, exercised against a real git repository."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from v5_experiments import cyr_gpu005 as core

EXECUTABLE_FILES = {
    "v5_experiments/cyr_gpu005.py": None,
    "anra_v5/cyr_gpu005_run.py": None,
    "notebooks/cymek_colab_gpu_research_v5.ipynb": None,
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture()
def frozen_repo(tmp_path: Path) -> tuple[Path, str, Path]:
    """Commit A = executable freeze; commit B = preregistration freeze."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "audit@example.com")
    _git(repo, "config", "user.name", "Freeze Simulation")
    file_hashes: dict[str, str] = {}
    for relative in EXECUTABLE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = relative.encode("utf-8")
        path.write_bytes(payload)
        file_hashes[relative] = hashlib.sha256(payload).hexdigest()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "COMMIT A: executable freeze",
         "--author=freeze <freeze@example.com>")
    executable_sha = _git(repo, "rev-parse", "HEAD")
    prereg = {"schema": "anra-cyr-gpu005-preregistration/v1",
              "experiment": core.CYR5_ID,
              "executable_sha256": executable_sha,
              "executable_files": file_hashes}
    (repo / "docs").mkdir()
    (repo / "docs" / "PREREGISTRATION.json").write_text(
        json.dumps(prereg, indent=2), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "COMMIT B: preregistration freeze",
         "--author=freeze <freeze@example.com>")
    external = tmp_path / "outside"
    external.mkdir()
    return repo, executable_sha, external


def _file_hashes(repo: Path) -> dict[str, str]:
    return {relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            for relative in EXECUTABLE_FILES}


def test_two_commit_freeze_and_checkout_sequence(frozen_repo):
    repo, executable_sha, external = frozen_repo
    assert _git(repo, "rev-parse", "HEAD") != executable_sha  # HEAD is commit B
    prereg = json.loads(
        (repo / "docs" / "PREREGISTRATION.json").read_text("utf-8"))
    outside_copy = external / "CYR-GPU-005-PREREGISTRATION.json"
    outside_copy.write_text(json.dumps(prereg), encoding="utf-8")
    _git(repo, "checkout", "-q", executable_sha)
    assert _git(repo, "rev-parse", "HEAD") == executable_sha
    receipt = core.assert_freeze_contract(
        preregistration=json.loads(outside_copy.read_text("utf-8")),
        executable_sha=executable_sha,
        head_sha=_git(repo, "rev-parse", "HEAD"),
        file_hashes=_file_hashes(repo))
    assert receipt["files_verified"] == sorted(EXECUTABLE_FILES)


def test_tampered_executable_fails_the_gate(frozen_repo):
    repo, executable_sha, external = frozen_repo
    prereg = json.loads(
        (repo / "docs" / "PREREGISTRATION.json").read_text("utf-8"))
    _git(repo, "checkout", "-q", executable_sha)
    target = repo / "v5_experiments" / "cyr_gpu005.py"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        core.assert_freeze_contract(
            preregistration=prereg, executable_sha=executable_sha,
            head_sha=executable_sha, file_hashes=_file_hashes(repo))


def test_executable_change_after_prereg_means_superseded(frozen_repo):
    """Section 45: a post-freeze executable change must be detected, never
    silently regenerated under the same experiment identity."""
    repo, executable_sha, external = frozen_repo
    prereg = json.loads(
        (repo / "docs" / "PREREGISTRATION.json").read_text("utf-8"))
    _git(repo, "checkout", "-q", executable_sha)
    target = repo / "anra_v5" / "cyr_gpu005_run.py"
    target.write_bytes(target.read_bytes() + b"# changed after freeze")
    hashes = _file_hashes(repo)
    with pytest.raises(ValueError, match="hash mismatch"):
        core.assert_freeze_contract(
            preregistration=prereg, executable_sha=executable_sha,
            head_sha=executable_sha, file_hashes=hashes)
