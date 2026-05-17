from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys

try:
    import resource
except ImportError:  # Windows and other platforms without POSIX rlimits
    resource = None  # type: ignore[assignment]

from anra_paths import WORKSPACE_DIR


SANDBOX_PREAMBLE = '''
import os
if os.environ.get("SANDBOX_NO_NETWORK"):
    import socket as _s
    _real_connect = _s.socket.connect
    def _blocked_connect(self, *a, **kw):
        raise ConnectionRefusedError("Network blocked in sandbox")
    _s.socket.connect = _blocked_connect
'''


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
        wrapped_code = SANDBOX_PREAMBLE + "\n" + code

        def _set_limits():
            if resource is None or sys.platform == "win32":
                return
            if not hasattr(sys, "getandroidapilevel"):
                # AN: Termux's Android linker can fail before user code under a 512 MB RLIMIT_AS.
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (512 * 1024 * 1024, 512 * 1024 * 1024),
                )
            resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

        try:
            env = self._clean_env()
            env["SANDBOX_NO_NETWORK"] = "1"
            proc = subprocess.run(
                ["python", "-c", wrapped_code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.workspace),
                env=env,
                preexec_fn=_set_limits if sys.platform != "win32" else None,
            )
            return SandboxResult(proc.returncode == 0, int(proc.returncode), proc.stdout[:4096], proc.stderr[:4096], False)
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(False, 124, (exc.stdout or "")[:4096] if isinstance(exc.stdout, str) else "", (exc.stderr or "")[:4096] if isinstance(exc.stderr, str) else "", True)
        except Exception as exc:
            return SandboxResult(False, 1, "", str(exc)[:4096], False)

    def _clean_env(self) -> dict[str, str]:
        allowed = {"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "PYTHONPATH"}
        blocked_fragments = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        clean = {}
        for key, value in os.environ.items():
            upper = key.upper()
            if upper in allowed and not any(fragment in upper for fragment in blocked_fragments):
                clean[key] = value
        return clean


class Sandbox(CodeSandbox):
    def run(self, code: str) -> SandboxResult:
        return self.execute(code)
