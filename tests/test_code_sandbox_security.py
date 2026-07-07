from __future__ import annotations

import os
from pathlib import Path

from execution.sandbox import CodeSandbox, SandboxPolicy
from training.verifier import VerifierHierarchy


def test_normal_code_executes_in_isolated_workspace(tmp_path: Path) -> None:
    sandbox = CodeSandbox(tmp_path / "sandbox", timeout=2)
    result = sandbox.execute("from pathlib import Path\nPath('ok.txt').write_text('ok')\nprint(6 * 7)")
    assert result.success is True
    assert result.stdout.strip() == "42"
    assert (tmp_path / "sandbox" / "ok.txt").read_text(encoding="utf-8") == "ok"


def test_secret_environment_and_pythonpath_are_not_inherited(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANRA_TEST_SECRET_TOKEN", "do-not-leak")
    monkeypatch.setenv("PYTHONPATH", "sensitive-import-path")
    result = CodeSandbox(tmp_path / "sandbox").execute(
        "import os\nprint(os.getenv('ANRA_TEST_SECRET_TOKEN'))\nprint(os.getenv('PYTHONPATH'))"
    )
    assert result.success is True
    assert "do-not-leak" not in result.stdout
    assert "sensitive-import-path" not in result.stdout


def test_filesystem_escape_is_denied_even_when_exception_is_caught(tmp_path: Path) -> None:
    escaped = tmp_path / "escaped.txt"
    code = (
        "try:\n"
        f"    open({str(escaped)!r}, 'w').write('owned')\n"
        "except PermissionError:\n"
        "    print('caught')\n"
    )
    result = CodeSandbox(tmp_path / "sandbox").execute(code)
    assert result.success is False
    assert result.limit_reason == "policy"
    assert not escaped.exists()


def test_child_process_and_network_are_denied(tmp_path: Path) -> None:
    child = CodeSandbox(tmp_path / "child").execute(
        "import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'print(1)'])"
    )
    network = CodeSandbox(tmp_path / "network").execute(
        "import socket\nsocket.create_connection(('127.0.0.1', 9), timeout=0.1)"
    )
    assert child.limit_reason == "policy"
    assert network.limit_reason == "policy"


def test_wall_timeout_kills_run(tmp_path: Path) -> None:
    policy = SandboxPolicy(timeout_seconds=0.2, cpu_seconds=10.0)
    result = CodeSandbox(tmp_path / "sandbox", policy=policy).execute("while True: pass")
    assert result.success is False
    assert result.timed_out is True
    assert result.return_code == 124
    assert result.limit_reason == "wall_time"


def test_cpu_ceiling_is_enforced_before_wall_timeout(tmp_path: Path) -> None:
    policy = SandboxPolicy(timeout_seconds=5.0, cpu_seconds=0.2)
    result = CodeSandbox(tmp_path / "sandbox", policy=policy).execute("while True: pass")
    assert result.success is False
    assert result.timed_out is False
    assert result.limit_reason == "cpu"
    assert result.return_code == 137


def test_output_is_drained_but_retained_at_a_hard_cap(tmp_path: Path) -> None:
    policy = SandboxPolicy(output_bytes=1024)
    result = CodeSandbox(tmp_path / "sandbox", policy=policy).execute("print('x' * 1000000)")
    assert result.success is True
    assert result.output_truncated is True
    assert len(result.stdout.encode("utf-8")) <= 1024


def test_memory_ceiling_is_enforced(tmp_path: Path) -> None:
    policy = SandboxPolicy(
        timeout_seconds=3.0,
        cpu_seconds=3.0,
        memory_bytes=64 * 1024 * 1024,
    )
    result = CodeSandbox(tmp_path / "sandbox", policy=policy).execute(
        "payload = bytearray(256 * 1024 * 1024)\nprint(len(payload))"
    )
    assert result.success is False
    assert result.limit_reason == "memory"
    assert result.return_code == 137


def test_file_size_ceiling_is_enforced(tmp_path: Path) -> None:
    policy = SandboxPolicy(file_bytes=64 * 1024)
    result = CodeSandbox(tmp_path / "sandbox", policy=policy).execute(
        "open('large.bin', 'wb').write(b'x' * (2 * 1024 * 1024))"
    )
    assert result.success is False
    assert result.limit_reason == "file_size"
    assert result.return_code == 137


def test_registered_code_verifier_uses_sandbox_policy(tmp_path: Path) -> None:
    verifier = VerifierHierarchy(tmp_path / "verifier")
    escaped = tmp_path / "verifier-escape.txt"
    result = verifier.score("code", code=f"open({str(escaped)!r}, 'w').write('owned')")
    assert result.score == 0.0
    assert result.reason == "sandbox_policy"
    assert not escaped.exists()


def test_sandbox_does_not_change_parent_working_directory(tmp_path: Path) -> None:
    before = Path.cwd()
    result = CodeSandbox(tmp_path / "sandbox").execute("import os\nos.chdir('.')\nprint(os.getcwd())")
    assert result.success is True
    assert Path.cwd() == before
    assert str(tmp_path / "sandbox") in result.stdout
    assert os.getcwd() == str(before)
