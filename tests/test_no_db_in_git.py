"""Fails if any .db or .sqlite files are tracked by git.
Run: pytest tests/test_no_db_in_git.py
"""
from __future__ import annotations
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def test_no_database_files_tracked_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "*.db"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    tracked = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    assert not tracked, (
        "These database files are tracked in git. Run: git rm --cached <file>\n"
        + "\n".join(tracked)
    )

def test_no_sqlite_files_tracked_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    sqlite_files = [f for f in result.stdout.splitlines() if f.endswith(".sqlite") or f.endswith(".db")]
    assert not sqlite_files, (
        "Binary database files are tracked in git:\n" + "\n".join(sqlite_files)
    )
