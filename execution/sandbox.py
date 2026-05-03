from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from anra_paths import WORKSPACE_DIR


@dataclass
class SandboxResult:
    success: bool
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool


class CodeSandbox:
    def __init__(self, workspace: str | Path | None = None, timeout: int = 30) -> None:
        self.workspace = Path(workspace) if workspace is not None else WORKSPACE_DIR / "sandbox"
        self.timeout = int(timeout)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str) -> SandboxResult:
        fp = None
        try:
            fd, p = tempfile.mkstemp(suffix=".py", dir=str(self.workspace))
            Path(p).write_text(code, encoding="utf-8")
            fp = Path(p)
            proc = subprocess.run(
                ["python3", str(fp)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.workspace),
            )
            return SandboxResult(proc.returncode == 0, int(proc.returncode), proc.stdout[:4096], proc.stderr[:4096], False)
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(False, 124, (exc.stdout or "")[:4096] if isinstance(exc.stdout, str) else "", (exc.stderr or "")[:4096] if isinstance(exc.stderr, str) else "", True)
        except Exception as exc:
            return SandboxResult(False, 1, "", str(exc)[:4096], False)
        finally:
            if fp and fp.exists():
                fp.unlink(missing_ok=True)
