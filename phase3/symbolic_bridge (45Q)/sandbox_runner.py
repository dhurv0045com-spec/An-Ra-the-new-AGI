"""
sandbox_runner.py — Sandboxed Test Executor for 45Q.

Executes generated test code in an isolated subprocess with:
  - Hard timeout (SANDBOX_TIMEOUT_SEC)
  - Network access blocked (socket patched)
  - Filesystem write access blocked (open for write patched)
  - Memory limit via resource module where available
  - Captured stdout, stderr, return code

The sandbox works by prepending a safety preamble to the test code that
patches dangerous builtins and modules before the test runs.
"""

from __future__ import annotations
import subprocess
import sys
import os
import textwrap
from dataclasses import dataclass, field
from typing import Optional
from . import config


@dataclass
class SandboxResult:
    """
    Result of a sandboxed test execution.

    Attributes
    ----------
    returncode : int
        Process return code (0 = all tests passed).
    stdout : str
        Captured standard output.
    stderr : str
        Captured standard error.
    tests_passed : int
        Number of tests that passed.
    tests_failed : int
        Number of tests that failed.
    tests_error : int
        Number of tests that errored.
    tests_skipped : int
        Number of tests that were skipped.
    timed_out : bool
        True if the sandbox killed the process due to timeout.
    failure_details : list[str]
        Descriptions of each failure/error.
    """
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    tests_error: int = 0
    tests_skipped: int = 0
    timed_out: bool = False
    failure_details: list[str] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return self.tests_passed + self.tests_failed + self.tests_error + self.tests_skipped

    @property
    def all_passed(self) -> bool:
        return self.tests_failed == 0 and self.tests_error == 0 and not self.timed_out


# ── Safety preamble injected before all test code ──────────────────────────────

_SAFETY_PREAMBLE = '''
import socket as _socket
import builtins as _builtins
import sys as _sys
import os as _os

# Block network access
_real_socket = _socket.socket
class _BlockedSocket:
    def __init__(self, *a, **kw):
        raise PermissionError("45Q sandbox: network access is blocked")
_socket.socket = _BlockedSocket

# Block filesystem writes
_real_open = _builtins.open
def _safe_open(file, mode='r', *args, **kwargs):
    if isinstance(mode, str) and any(m in mode for m in ('w', 'a', 'x', '+')):
        raise PermissionError(f"45Q sandbox: filesystem writes blocked (mode={mode!r})")
    return _real_open(file, mode, *args, **kwargs)
_builtins.open = _safe_open

# Block os.system and dangerous calls
_os.system = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("45Q sandbox: os.system blocked"))

# Block subprocess creation from within sandbox
import subprocess as _subprocess
_subprocess.Popen = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("45Q sandbox: subprocess blocked"))
_subprocess.call = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("45Q sandbox: subprocess blocked"))
_subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(PermissionError("45Q sandbox: subprocess blocked"))

# Apply memory limit where available
try:
    import resource as _resource
    _mb = {memory_mb}
    _resource.setrlimit(_resource.RLIMIT_AS, (_mb * 1024 * 1024, _mb * 1024 * 1024))
except Exception:
    pass  # resource module not available (Windows)

'''


def run_tests(test_code: str) -> SandboxResult:
    """
    Execute test code in a sandboxed subprocess.

    The test code is prepended with a safety preamble that disables
    network access and filesystem writes, then executed via a fresh
    Python interpreter subprocess with a hard timeout.

    Parameters
    ----------
    test_code : str
        Complete Python test module code (unittest.TestCase subclass).

    Returns
    -------
    SandboxResult
        Results including pass/fail counts and any failure details.
    """
    preamble = _SAFETY_PREAMBLE.replace("{memory_mb}", str(config.SANDBOX_MEMORY_MB))
    full_code = preamble + "\n" + test_code

    # Write to a temporary string for subprocess via stdin
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", "__main__"],
            input=full_code,
            capture_output=True,
            text=True,
            timeout=config.SANDBOX_TIMEOUT_SEC,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            # Run via -c to execute code from stdin via -m unittest __main__
            # Actually, we need to run via -c:
        )
        # The above doesn't work for -m unittest; use -c approach instead
        proc = subprocess.run(
            [sys.executable, "-c", full_code + "\nimport unittest\nunittest.main(module='__main__', argv=['test'], verbosity=2, exit=False)"],
            capture_output=True,
            text=True,
            timeout=config.SANDBOX_TIMEOUT_SEC,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return _parse_result(proc.stdout, proc.stderr, proc.returncode, timed_out=False)

    except subprocess.TimeoutExpired:
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=f"TIMEOUT: process killed after {config.SANDBOX_TIMEOUT_SEC}s",
            timed_out=True,
            failure_details=[f"Test execution timed out after {config.SANDBOX_TIMEOUT_SEC} seconds"],
        )
    except Exception as e:
        return SandboxResult(
            returncode=-2,
            stderr=str(e),
            failure_details=[f"Sandbox execution error: {e}"],
        )


def _parse_result(
    stdout: str,
    stderr: str,
    returncode: int,
    timed_out: bool,
) -> SandboxResult:
    """
    Parse unittest output to extract pass/fail counts.

    Parameters
    ----------
    stdout : str
        Captured stdout from the test run.
    stderr : str
        Captured stderr (unittest writes to stderr by default).
    returncode : int
        Process exit code.
    timed_out : bool
        Whether the process timed out.

    Returns
    -------
    SandboxResult
        Parsed result.
    """
    result = SandboxResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )

    # Combined output (unittest writes test results to stderr)
    combined = (stderr + "\n" + stdout).lower()

    # Parse summary line: "Ran N tests in X.XXXs"
    import re
    ran_match = re.search(r'ran (\d+) test', combined)
    if ran_match:
        total = int(ran_match.group(1))
    else:
        total = 0

    # Parse failure counts: "FAILED (failures=N, errors=M)"
    fail_match = re.search(r'failed \(([^)]+)\)', combined)
    if fail_match:
        detail = fail_match.group(1)
        f_match = re.search(r'failures?=(\d+)', detail)
        e_match = re.search(r'errors?=(\d+)', detail)
        s_match = re.search(r'skipped=(\d+)', detail)
        result.tests_failed = int(f_match.group(1)) if f_match else 0
        result.tests_error = int(e_match.group(1)) if e_match else 0
        result.tests_skipped = int(s_match.group(1)) if s_match else 0
        result.tests_passed = total - result.tests_failed - result.tests_error - result.tests_skipped
    elif 'ok' in combined and ran_match:
        result.tests_passed = total

    # Extract failure details
    # Look for "FAIL: test_name" or "ERROR: test_name" sections
    fail_sections = re.findall(
        r'(FAIL|ERROR): (\w+).*?\n(.*?)(?=\n-{10,}|\Z)',
        stderr + stdout,
        re.DOTALL,
    )
    for (ftype, test_name, detail) in fail_sections:
        result.failure_details.append(
            f"{ftype}: {test_name}\n{detail.strip()}"
        )

    return result
