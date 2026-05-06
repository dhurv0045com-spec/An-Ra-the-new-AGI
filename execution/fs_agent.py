from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
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
        tmp = p.with_name(f".{p.name}.{int(time.time() * 1000)}.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, p)
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
        if not (self.root / ".git").exists():
            init = subprocess.run(["git", "init"], capture_output=True, text=True, cwd=str(self.root))
            if init.returncode != 0:
                self._log("git_commit", self.root)
                return 1, (init.stdout + init.stderr).strip()
            subprocess.run(["git", "config", "user.email", "anra-local@example.invalid"], cwd=str(self.root), capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "An-Ra Local"], cwd=str(self.root), capture_output=True, text=True)
        subprocess.run(["git", "add", "-A"], capture_output=True, text=True, cwd=str(self.root))
        proc = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, cwd=str(self.root))
        self._log("git_commit", self.root)
        code = int(proc.returncode)
        if code not in {0, 1}:
            code = 1
        return code, (proc.stdout + proc.stderr).strip()
