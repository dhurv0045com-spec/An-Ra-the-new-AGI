from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import tempfile
import time


@dataclass
class FSAction:
    ts: float
    action: str
    path: str


class FSAgent:
    def __init__(self, root: str | Path = ".", audit_log: str | Path = "state/fs_audit.log") -> None:
        self.root = Path(root).resolve()
        self.audit_log = Path(audit_log)
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_git_repo()

    def _ensure_git_repo(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / ".git").exists():
            initialized = True
        else:
            proc = subprocess.run(["git", "init"], capture_output=True, text=True, cwd=str(self.root))
            if proc.returncode != 0:
                raise RuntimeError(f"[fs_agent] git init failed: {(proc.stdout + proc.stderr).strip()}")
            initialized = False
        email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, cwd=str(self.root))
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, cwd=str(self.root))
        if not email.stdout.strip():
            subprocess.run(["git", "config", "user.email", "anra-fs-agent@example.local"], capture_output=True, text=True, cwd=str(self.root))
        if not name.stdout.strip():
            subprocess.run(["git", "config", "user.name", "AnRa FS Agent"], capture_output=True, text=True, cwd=str(self.root))

    def _resolve(self, path: str | Path) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("Path escapes workspace root")
        return p

    def _log(self, action: str, path: Path) -> None:
        rec = FSAction(ts=time.time(), action=action, path=str(path))
        with self.audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__) + "\n")

    def read(self, path: str | Path) -> str:
        p = self._resolve(path)
        out = p.read_text(encoding="utf-8")
        self._log("read", p)
        return out

    def write(self, path: str | Path, content: str) -> Path:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, p)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        self._log("write", p)
        return p

    def append(self, path: str | Path, content: str) -> Path:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        self._log("append", p)
        return p

    def delete(self, path: str | Path) -> bool:
        p = self._resolve(path)
        if not p.exists():
            return False
        p.unlink()
        self._log("delete", p)
        return True

    def git_commit(self, message: str) -> tuple[int, str]:
        add = subprocess.run(["git", "add", "-A"], capture_output=True, text=True, cwd=str(self.root))
        if add.returncode != 0:
            self._log("git_commit_failed", self.root)
            return int(add.returncode), (add.stdout + add.stderr).strip()
        proc = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, cwd=str(self.root))
        self._log("git_commit", self.root)
        return int(proc.returncode), (proc.stdout + proc.stderr).strip()
