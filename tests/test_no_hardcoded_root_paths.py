from __future__ import annotations

import re
from pathlib import Path

from anra_paths import DRIVE_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPES = ("training", "scripts", "phase2", "phase3", "ui")
ALLOWED_FILES = {
    REPO_ROOT / "anra_paths.py",
}
BLOCKED_PATTERNS = (
    re.compile(re.escape(str(DRIVE_DIR))),
    re.compile(r'Path\(\s*["\']/content'),
    re.compile(r'Path\(\s*r?["\'][A-Za-z]:\\'),
)


def test_no_hardcoded_root_or_drive_paths_outside_registry() -> None:
    offenders: list[str] = []
    for scope in SCOPES:
        for path in (REPO_ROOT / scope).rglob("*.py"):
            if path in ALLOWED_FILES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in BLOCKED_PATTERNS:
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0)}")
    assert not offenders, "Hardcoded root/Drive paths found:\n" + "\n".join(offenders)
