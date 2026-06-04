from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT / "phase2" / "agent_loop_45k"


@pytest.fixture(autouse=True)
def _agent_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("AGENT_FILE_ROOT", str(ws))
    if str(AGENT_DIR) not in sys.path:
        sys.path.append(str(AGENT_DIR))
    yield ws


def test_file_manager_write_read():
    from builtin import file_manager

    w = file_manager("write demo.txt hello operator")
    assert w.success
    r = file_manager("read demo.txt")
    assert r.success
    assert "hello operator" in r.output


def test_cad_generate_raptor_engine():
    from builtin import cad_generate

    result = cad_generate("raptor_engine")
    assert result.success
    scad = Path(os.environ["AGENT_FILE_ROOT"]) / "engineering" / "raptor_engine" / "raptor_engine.scad"
    report = scad.parent / "REPORT.md"
    assert scad.exists()
    assert report.exists()


def test_os_action_rejects_escape():
    from builtin import os_action

    result = os_action("open ../../../etc/passwd")
    assert not result.success


def test_slash_write_command():
    from runtime.operator_commands import handle_slash_command

    calls = []

    def _fake_goal(text: str):
        calls.append(text)
        return {"success": True, "output": "ok"}

    handled, msg = handle_slash_command("/write notes.txt line one", run_goal=_fake_goal)
    assert handled
    assert "line one" in msg or "Written" in msg

    root = Path(os.environ["AGENT_FILE_ROOT"])
    assert (root / "notes.txt").read_text(encoding="utf-8") == "line one"
