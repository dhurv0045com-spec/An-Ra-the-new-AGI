from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
except ImportError:  # Hard OS limits still apply without the optional monitor.
    psutil = None  # type: ignore[assignment]

try:
    import resource
except ImportError:  # Windows and other platforms without POSIX rlimits
    resource = None  # type: ignore[assignment]

from anra.anra_paths import WORKSPACE_DIR

SANDBOX_PREAMBLE = r'''
import os as _os
import sys as _sys

_ROOT = _os.path.realpath(_os.environ["ANRA_SANDBOX_ROOT"])
_WRITE_FLAGS = 1 | 2 | 64 | 512 | 1024

class SandboxPolicyError(PermissionError):
    pass

def _deny(reason):
    _os.write(2, ("ANRA_SANDBOX_POLICY:" + reason + "\n").encode("utf-8"))
    raise SandboxPolicyError(reason)

def _inside(path):
    if isinstance(path, int):
        return True
    try:
        resolved = _os.path.realpath(_os.fspath(path))
        return _os.path.commonpath((_ROOT, resolved)) == _ROOT
    except (TypeError, ValueError, OSError):
        return False

def _audit(event, args):
    if event == "open":
        path = args[0] if args else ""
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        writing = (
            (isinstance(mode, str) and any(mark in mode for mark in "wax+"))
            or (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
        )
        if writing and not _inside(path):
            _deny("filesystem write outside sandbox denied")
    elif event in {
        "os.remove", "os.rmdir", "os.rename", "os.replace", "os.mkdir",
        "os.chmod", "os.chown", "os.truncate", "os.symlink", "os.link",
    }:
        paths = (
            args[:2]
            if event in {"os.rename", "os.replace", "os.symlink", "os.link"}
            else args[:1]
        )
        if any(not _inside(path) for path in paths):
            _deny("filesystem mutation outside sandbox denied")
    elif event == "os.chdir" and args and not _inside(args[0]):
        _deny("chdir outside sandbox denied")
    elif event.startswith(("subprocess.", "os.spawn", "os.exec", "os.fork", "pty.spawn")):
        _deny("child processes denied")
    elif event.startswith(("socket.connect", "socket.bind", "socket.getaddrinfo")):
        _deny("network denied")
    elif event in {"ctypes.dlopen", "ctypes.dlsym", "ctypes.call_function"}:
        _deny("native library loading denied")

_sys.addaudithook(_audit)
'''


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_seconds: float = 5.0
    cpu_seconds: float = 3.0
    memory_bytes: int = 256 * 1024 * 1024
    file_bytes: int = 4 * 1024 * 1024
    output_bytes: int = 4096
    open_files: int = 32


@dataclass
class SandboxResult:
    success: bool
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    limit_reason: str = ""
    output_truncated: bool = False
    duration_ms: float = 0.0


class CodeSandbox:
    def __init__(
        self,
        workspace: str | Path | None = None,
        timeout: int | float = 5,
        *,
        policy: SandboxPolicy | None = None,
    ) -> None:
        self.workspace = (
            Path(workspace) if workspace is not None else WORKSPACE_DIR / "sandbox"
        ).resolve()
        self.policy = policy or SandboxPolicy(timeout_seconds=float(timeout))
        self.timeout = float(self.policy.timeout_seconds)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str) -> SandboxResult:
        wrapped_code = SANDBOX_PREAMBLE + "\n" + code
        started = time.perf_counter()
        baseline_bytes = self._workspace_bytes()
        env = self._clean_env()
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-B", "-c", wrapped_code],
                cwd=str(self.workspace),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                start_new_session=sys.platform != "win32",
                preexec_fn=self._set_posix_limits if sys.platform != "win32" else None,
            )
        except Exception as exc:
            return SandboxResult(
                False,
                1,
                "",
                str(exc)[: self.policy.output_bytes],
                False,
                "launch_error",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        job_handle = self._apply_windows_job(proc) if sys.platform == "win32" else None

        stdout = bytearray()
        stderr = bytearray()
        truncated = [False]
        threads = [
            threading.Thread(
                target=self._drain_pipe,
                args=(proc.stdout, stdout, truncated),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain_pipe,
                args=(proc.stderr, stderr, truncated),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        limit_reason = ""
        timed_out = False
        process = psutil.Process(proc.pid) if psutil is not None else None
        while proc.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed > self.policy.timeout_seconds:
                timed_out = True
                limit_reason = "wall_time"
            elif process is not None:
                try:
                    cpu = process.cpu_times().user + process.cpu_times().system
                    rss = process.memory_info().rss + sum(
                        child.memory_info().rss for child in process.children(recursive=True)
                    )
                    if cpu > self.policy.cpu_seconds:
                        limit_reason = "cpu"
                    elif rss > self.policy.memory_bytes:
                        limit_reason = "memory"
                    elif self._workspace_bytes() - baseline_bytes > self.policy.file_bytes:
                        limit_reason = "file_size"
                except Exception:
                    pass
            if limit_reason:
                self._kill_tree(proc)
                break
            time.sleep(0.01)

        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            proc.wait(timeout=1.0)
        for thread in threads:
            thread.join(timeout=1.0)

        peak_memory = self._windows_peak_memory(proc) if sys.platform == "win32" else 0
        peak_cpu = self._windows_cpu_seconds(proc) if sys.platform == "win32" else 0.0
        if job_handle is not None:
            self._close_windows_handle(job_handle)
        stderr_text = stderr.decode("utf-8", errors="replace")
        if not limit_reason and (
            "SandboxPolicyError" in stderr_text or "ANRA_SANDBOX_POLICY:" in stderr_text
        ):
            limit_reason = "policy"
        if not limit_reason and (
            "MemoryError" in stderr_text or peak_memory > self.policy.memory_bytes
        ):
            limit_reason = "memory"
        if not limit_reason and peak_cpu >= self.policy.cpu_seconds * 0.9:
            limit_reason = "cpu"
        if not limit_reason and (
            "File too large" in stderr_text
            or "Errno 27" in stderr_text
            or self._workspace_bytes() - baseline_bytes > self.policy.file_bytes
        ):
            limit_reason = "file_size"
        if (
            not limit_reason
            and sys.platform != "win32"
            and proc.returncode == -getattr(signal, "SIGXCPU", 24)
        ):
            limit_reason = "cpu"
        return_code = (
            124
            if timed_out
            else 137
            if limit_reason and limit_reason != "policy"
            else int(proc.returncode or 0)
        )
        return SandboxResult(
            success=return_code == 0 and not limit_reason,
            return_code=return_code,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr_text,
            timed_out=timed_out,
            limit_reason=limit_reason,
            output_truncated=truncated[0],
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def _set_posix_limits(self) -> None:
        if resource is None:
            return
        memory = int(self.policy.memory_bytes)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = max(1, int(self.policy.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        file_bytes = int(self.policy.file_bytes)
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (self.policy.open_files, self.policy.open_files))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def _apply_windows_job(self, proc: subprocess.Popen[bytes]) -> int | None:
        """Apply kernel-enforced memory/CPU limits when Windows Jobs are available."""
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return None
        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x2 | 0x100 | 0x2000
        limits.BasicLimitInformation.PerProcessUserTimeLimit = int(
            self.policy.cpu_seconds * 10_000_000
        )
        limits.ProcessMemoryLimit = int(self.policy.memory_bytes)
        configured = kernel32.SetInformationJobObject(
            handle,
            9,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        assigned = configured and kernel32.AssignProcessToJobObject(
            handle, int(proc._handle)
        )
        if not assigned:
            kernel32.CloseHandle(handle)
            return None
        return int(handle)

    @staticmethod
    def _windows_peak_memory(proc: subprocess.Popen[bytes]) -> int:
        if sys.platform != "win32":
            return 0
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            int(proc._handle), ctypes.byref(counters), ctypes.sizeof(counters)
        ):
            return 0
        return int(max(counters.PeakWorkingSetSize, counters.PeakPagefileUsage))

    @staticmethod
    def _windows_cpu_seconds(proc: subprocess.Popen[bytes]) -> float:
        if sys.platform != "win32":
            return 0.0
        import ctypes
        from ctypes import wintypes

        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        if not kernel32.GetProcessTimes(
            int(proc._handle),
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return 0.0

        def ticks(value: wintypes.FILETIME) -> int:
            return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

        return (ticks(kernel) + ticks(user)) / 10_000_000.0

    @staticmethod
    def _close_windows_handle(handle: int) -> None:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle(handle)

    def _drain_pipe(
        self,
        pipe: object,
        retained: bytearray,
        truncated: list[bool],
    ) -> None:
        if pipe is None or not hasattr(pipe, "read"):
            return
        while True:
            chunk = pipe.read(8192)
            if not chunk:
                break
            remaining = self.policy.output_bytes - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True

    def _kill_tree(self, proc: subprocess.Popen[bytes]) -> None:
        if psutil is not None:
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.kill()
                parent.kill()
                psutil.wait_procs(children, timeout=1.0)
                return
            except Exception:
                pass
        if proc.poll() is None:
            if sys.platform != "win32":
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()

    def _workspace_bytes(self) -> int:
        total = 0
        for path in self.workspace.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def _clean_env(self) -> dict[str, str]:
        clean: dict[str, str] = {}
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL"):
            value = os.environ.get(key)
            if value:
                clean[key] = value
        clean.update(
            {
                "ANRA_SANDBOX_ROOT": str(self.workspace),
                "HOME": str(self.workspace),
                "USERPROFILE": str(self.workspace),
                "TMP": str(self.workspace),
                "TEMP": str(self.workspace),
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return clean


class Sandbox(CodeSandbox):
    def run(self, code: str) -> SandboxResult:
        return self.execute(code)
