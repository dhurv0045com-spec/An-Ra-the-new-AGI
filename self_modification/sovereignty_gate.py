from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from anra_paths import ROOT, SELF_MOD_AUDIT_LOG


def _under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def sovereignty_audit_change(file_path: str | Path, new_content: str, *, reason: str = "") -> dict[str, Any]:
    path = Path(file_path)
    allowed = True
    findings: list[str] = []
    if _under(path, ROOT / "history"):
        allowed = False
        findings.append("history is immutable")
    if path.suffix == ".pt":
        allowed = False
        findings.append("checkpoint mutation is blocked")
    if path.suffix == ".py":
        try:
            ast.parse(new_content, filename=str(path))
        except SyntaxError as exc:
            allowed = False
            findings.append(f"syntax error: {exc}")

    event = {
        "ts": time.time(),
        "event_type": "SELF_MOD_AUDIT",
        "component": "self_modification",
        "action": "approve" if allowed else "reject",
        "file": str(path),
        "reason": reason,
        "findings": findings,
    }
    try:
        SELF_MOD_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SELF_MOD_AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        pass
    try:
        from sovereignty.logger import audit_event

        audit_event("self_modification", "SELF_MOD_AUDIT", event["action"], details=event)
    except Exception:
        pass
    return {"allowed": allowed, "event": event}


class SovereigntyRollback:
    """
    Context manager that backs up affected files before
    modification and rolls back if tests fail after.
    """

    def __init__(self, files_to_modify: list[Path], test_cmd: list[str] = None):
        self.files = files_to_modify
        self.test_cmd = test_cmd or [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-x",
            "-q",
            "--timeout=60",
        ]
        self.backup_dir: Path | None = None

    def __enter__(self):
        self.backup_dir = Path(tempfile.mkdtemp(prefix="anra_rollback_"))
        for f in self.files:
            if f.exists():
                dest = self.backup_dir / f.name
                shutil.copy2(f, dest)
        return self

    def commit(self) -> bool:
        """Run tests. Return True if passed, auto-rollback if failed."""
        result = subprocess.run(
            self.test_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            self._rollback()
            return False
        self._cleanup()
        return True

    def _rollback(self):
        if self.backup_dir is None:
            return
        for f in self.files:
            backup = self.backup_dir / f.name
            if backup.exists():
                shutil.copy2(backup, f)
        self._cleanup()

    def _cleanup(self):
        if self.backup_dir and self.backup_dir.exists():
            shutil.rmtree(self.backup_dir, ignore_errors=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._rollback()
        return False
