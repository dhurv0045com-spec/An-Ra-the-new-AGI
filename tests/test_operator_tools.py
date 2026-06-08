from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _agent_workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("AGENT_FILE_ROOT", str(ws))
    return ws


def test_file_manager_write_read():
    from phase2.agent_loop_45k.builtin import file_manager

    w = file_manager("write demo.txt hello operator")
    assert w.success
    r = file_manager("read demo.txt")
    assert r.success
    assert "hello operator" in r.output


def test_cad_generate_raptor_engine():
    from phase2.agent_loop_45k.builtin import cad_generate

    result = cad_generate("raptor_engine")
    assert result.success
    scad = Path(os.environ["AGENT_FILE_ROOT"]) / "engineering" / "raptor_engine" / "raptor_engine.scad"
    report = scad.parent / "REPORT.md"
    assert scad.exists()
    assert report.exists()


def test_os_action_rejects_escape():
    from phase2.agent_loop_45k.builtin import os_action

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


def test_natural_write_command_routes_to_file_tool():
    from runtime.operator_commands import handle_natural_operator_request

    calls = []

    def _fake_goal(text: str):
        calls.append(text)
        return {"success": True, "output": "ok"}

    handled, msg = handle_natural_operator_request(
        "create file notes.txt with content line one",
        run_goal=_fake_goal,
    )
    assert handled
    assert "Written" in msg
    assert calls == []

    root = Path(os.environ["AGENT_FILE_ROOT"])
    assert (root / "notes.txt").read_text(encoding="utf-8") == "line one"


def test_natural_write_without_path_stays_chat():
    from runtime.operator_commands import handle_natural_operator_request

    handled, msg = handle_natural_operator_request(
        "write a poem about operators",
        run_goal=lambda text: {"success": True, "output": text},
    )
    assert not handled
    assert msg == ""
