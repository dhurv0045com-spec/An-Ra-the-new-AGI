"""Git-frozen source sets: exact commit bytes, dirty-worktree immunity (M6-M8)."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from v5_data.sourceset import (
    freeze_sourceset,
    resolve_commit,
    verify_sourceset,
)

ROOT = Path(__file__).resolve().parents[1]

SUFFIXES = (".py", ".md")
EXCLUDE = (".git", ".venv", ".venv-cuda", "__pycache__", "artifacts", ".codex-worktrees", "tests")
CATEGORIES = {".py": "code", ".md": "prose"}


class SourcesetTests(unittest.TestCase):
    def test_freeze_binds_commit_blobs_and_bytes(self) -> None:
        manifest = freeze_sourceset(
            ROOT, revision="HEAD", suffixes=SUFFIXES, exclude_parts=EXCLUDE,
            categories=CATEGORIES, max_bytes=200_000,
        )
        self.assertEqual(manifest.source_commit, resolve_commit(ROOT, "HEAD"))
        self.assertTrue(len(manifest.entries) > 5)
        verify_sourceset(ROOT, manifest)
        self.assertEqual(len(manifest.sha256()), 64)

    def test_untracked_files_cannot_enter(self) -> None:
        marker = ROOT / "temp-sourceset-probe.py"
        try:
            marker.write_text("untracked scientific content\n", encoding="utf-8")
            manifest = freeze_sourceset(
                ROOT, revision="HEAD", suffixes=(".py",), exclude_parts=EXCLUDE,
                categories={".py": "code"}, max_bytes=5_000_000,
            )
            paths = [entry.path for entry in manifest.entries]
            self.assertNotIn("temp-sourceset-probe.py", paths)
        finally:
            if marker.exists():
                marker.unlink()

    def test_dirty_tracked_file_uses_frozen_bytes(self) -> None:
        target = ROOT / "v5_data" / "stream.py"
        committed = subprocess.run(
            ["git", "show", f"HEAD:{target.relative_to(ROOT).as_posix()}"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# DIRTY WORKTREE MARKER\n")
            manifest = freeze_sourceset(
                ROOT, revision="HEAD", suffixes=(".py",), exclude_parts=EXCLUDE,
                categories={".py": "code"}, max_bytes=5_000_000,
            )
            entry = next(e for e in manifest.entries if e.path == "v5_data/stream.py")
            import hashlib

            self.assertEqual(entry.raw_sha256, hashlib.sha256(committed).hexdigest())
            self.assertNotIn(b"DIRTY", committed)
            verify_sourceset(ROOT, manifest)
        finally:
            target.write_bytes(original)

    def test_tampered_manifest_fails_verification(self) -> None:
        import dataclasses

        manifest = freeze_sourceset(
            ROOT, revision="HEAD", suffixes=(".md",), exclude_parts=EXCLUDE,
            categories={".md": "prose"}, max_bytes=200_000,
        )
        tampered_entries = tuple(
            dataclasses.replace(entry, raw_sha256="0" * 64) if index == 0 else entry
            for index, entry in enumerate(manifest.entries)
        )
        tampered = dataclasses.replace(manifest, entries=tampered_entries)
        with self.assertRaises(ValueError):
            verify_sourceset(ROOT, tampered)


if __name__ == "__main__":
    unittest.main()
