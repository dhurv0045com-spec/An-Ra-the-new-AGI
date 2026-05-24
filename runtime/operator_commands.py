"""Slash-command operator surface for interactive An-Ra sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from anra_paths import OPERATOR_AUDIT_LOG, get_agent_workspace


HELP_TEXT = """
An-Ra operator commands (chat mode):
  /help                         Show this help
  /workspace                    Print agent sandbox path
  /goal <text>                  Run full agent loop on a goal
  /write <path> <content>       Create/overwrite file in workspace
  /read <path>                  Read file from workspace
  /open <path>                  Open file/folder in OS default app
  /list [path]                  List workspace directory
  /cad <template>               Generate engineering CAD stub (e.g. raptor_engine)

Plain text without / runs chat. Prefix with goal: also works (legacy).
""".strip()


def _audit(action: str, detail: str, success: bool) -> None:
    try:
        OPERATOR_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "action": action,
            "detail": detail[:500],
            "success": success,
        }
        with OPERATOR_AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _file_tool_call(instruction: str) -> dict[str, Any]:
    import sys
    from pathlib import Path as P

    agent_dir = P(__file__).resolve().parent.parent / "phase2" / "agent_loop (45k)"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))
    from builtin import file_manager
    from registry import ToolResult

    result: ToolResult = file_manager(instruction)
    return result.to_dict()


def _os_tool_call(instruction: str) -> dict[str, Any]:
    import sys
    from pathlib import Path as P

    agent_dir = P(__file__).resolve().parent.parent / "phase2" / "agent_loop (45k)"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))
    from builtin import os_action
    from registry import ToolResult

    result: ToolResult = os_action(instruction)
    return result.to_dict()


def _cad_tool_call(instruction: str) -> dict[str, Any]:
    import sys
    from pathlib import Path as P

    agent_dir = P(__file__).resolve().parent.parent / "phase2" / "agent_loop (45k)"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))
    from builtin import cad_generate
    from registry import ToolResult

    result: ToolResult = cad_generate(instruction)
    return result.to_dict()


def handle_slash_command(
    line: str,
    *,
    run_goal: Callable[[str], dict[str, Any]],
) -> tuple[bool, str]:
    """
    Returns (handled, message).
    If handled is False, caller should continue normal chat.
    """
    raw = line.strip()
    if not raw.startswith("/"):
        return False, ""

    parts = raw.split(maxsplit=2)
    cmd = parts[0].lower()
    arg1 = parts[1] if len(parts) > 1 else ""
    arg2 = parts[2] if len(parts) > 2 else ""

    if cmd in ("/help", "/?"):
        return True, HELP_TEXT

    if cmd == "/workspace":
        ws = get_agent_workspace()
        _audit("workspace", str(ws), True)
        return True, f"Agent workspace: {ws}"

    if cmd == "/goal":
        if not arg1:
            return True, "Usage: /goal <what to do>"
        _audit("goal", arg1, True)
        result = run_goal(raw[len("/goal") :].strip())
        ok = bool(result.get("success"))
        out = str(result.get("output", ""))[:2000]
        return True, f"Success: {ok}\n{out}"

    if cmd == "/write":
        if not arg1 or not arg2:
            return True, "Usage: /write <relative-path> <content>"
        r = _file_tool_call(f"write {arg1} {arg2}")
        _audit("write", arg1, r.get("success", False))
        return True, r.get("output") or r.get("error", "write failed")

    if cmd == "/read":
        if not arg1:
            return True, "Usage: /read <relative-path>"
        r = _file_tool_call(f"read {arg1}")
        _audit("read", arg1, r.get("success", False))
        return True, r.get("output") or r.get("error", "read failed")

    if cmd == "/list":
        path = arg1 or "."
        r = _file_tool_call(f"list {path}")
        _audit("list", path, r.get("success", False))
        return True, r.get("output") or r.get("error", "list failed")

    if cmd == "/open":
        if not arg1:
            return True, "Usage: /open <relative-path-or-url>"
        r = _os_tool_call(f"open {arg1}")
        _audit("open", arg1, r.get("success", False))
        return True, r.get("output") or r.get("error", "open failed")

    if cmd == "/cad":
        template = arg1 or "raptor_engine"
        r = _cad_tool_call(template)
        _audit("cad", template, r.get("success", False))
        return True, r.get("output") or r.get("error", "cad failed")

    return True, f"Unknown command {cmd}. Type /help"
