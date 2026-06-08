"""Slash-command operator surface for interactive An-Ra sessions."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable


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


def _agent_workspace() -> Path:
    try:
        from anra.anra_paths import get_agent_workspace

        return get_agent_workspace()
    except Exception:
        import os

        root = Path(os.environ.get("AGENT_FILE_ROOT", "./agent_workspace")).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()


def _operator_audit_log() -> Path:
    try:
        from anra.anra_paths import OPERATOR_AUDIT_LOG

        return OPERATOR_AUDIT_LOG
    except Exception:
        return _agent_workspace() / "operator_actions.jsonl"


def _audit(action: str, detail: str, success: bool) -> None:
    try:
        audit_log = _operator_audit_log()
        audit_log.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "action": action,
            "detail": detail[:500],
            "success": success,
        }
        with audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _file_tool_call(instruction: str) -> dict[str, Any]:
    from phase2.agent_loop_45k.builtin import file_manager
    from phase2.agent_loop_45k.registry import ToolResult

    result: ToolResult = file_manager(instruction)
    return result.to_dict()


def _os_tool_call(instruction: str) -> dict[str, Any]:
    from phase2.agent_loop_45k.builtin import os_action
    from phase2.agent_loop_45k.registry import ToolResult

    result: ToolResult = os_action(instruction)
    return result.to_dict()


def _cad_tool_call(instruction: str) -> dict[str, Any]:
    from phase2.agent_loop_45k.builtin import cad_generate
    from phase2.agent_loop_45k.registry import ToolResult

    result: ToolResult = cad_generate(instruction)
    return result.to_dict()


def _clean_natural_path(raw: str) -> str:
    return raw.strip().strip(":,;.").strip('"').strip("'")


def _looks_like_workspace_path(path: str) -> bool:
    if not path or path.startswith(("/", "\\")):
        return False
    if ".." in Path(path).parts:
        return False
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", Path(path).name))


def _write_from_natural(raw: str) -> tuple[bool, str]:
    patterns = [
        r"""(?ix)
        \b(?:please\s+)?(?:create|make|write|save)\s+
        (?:a\s+)?(?:new\s+)?(?:file\s+)?(?:called\s+|named\s+)?
        ["']?(?P<path>[\w./\\-]+\.[A-Za-z0-9]{1,8})["']?
        \s*(?::|-|\bwith\s+content\b|\bwith\b|\bcontaining\b|\bthat\s+says\b|\bcontent\b)
        \s*(?P<content>.+)
        """,
        r"""(?ix)
        \b(?:please\s+)?(?:put|save)\s+
        (?P<content>.+?)
        \s+\b(?:in|to)\s+
        (?:a\s+)?(?:file\s+)?(?:called\s+|named\s+)?
        ["']?(?P<path>[\w./\\-]+\.[A-Za-z0-9]{1,8})["']?\s*$
        """,
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if not match:
            continue
        path = _clean_natural_path(match.group("path"))
        content = match.group("content").strip().strip('"').strip("'")
        if _looks_like_workspace_path(path) and content:
            r = _file_tool_call(f"write {path} {content}")
            _audit("natural_write", path, r.get("success", False))
            return True, r.get("output") or r.get("error", "write failed")
    return False, ""


def handle_natural_operator_request(
    line: str,
    *,
    run_goal: Callable[[str], dict[str, Any]],
) -> tuple[bool, str]:
    """
    Route plain-English operator requests to the same safe tools used by slash
    commands. Conservative by design: file writes need a concrete path and
    explicit content, so normal chat like "write a poem" still goes to the LLM.
    """
    raw = line.strip()
    low = raw.lower()
    if not raw or raw.startswith("/"):
        return False, ""

    if re.fullmatch(r"(show|print|where is|open)?\s*(the\s+)?workspace\s*(path)?", low):
        ws = _agent_workspace()
        _audit("natural_workspace", str(ws), True)
        return True, f"Agent workspace: {ws}"

    handled, msg = _write_from_natural(raw)
    if handled:
        return True, msg

    read_match = re.search(
        r"\b(?:read|show|display)\s+(?:file\s+)?(?:called\s+|named\s+)?"
        r"['\"]?(?P<path>[\w./\\-]+\.[A-Za-z0-9]{1,8})['\"]?\s*$",
        raw,
        re.IGNORECASE,
    )
    if read_match:
        path = _clean_natural_path(read_match.group("path"))
        if _looks_like_workspace_path(path):
            r = _file_tool_call(f"read {path}")
            _audit("natural_read", path, r.get("success", False))
            return True, r.get("output") or r.get("error", "read failed")

    list_match = re.search(
        r"\b(?:list|show)\s+(?:files|contents)(?:\s+(?:in|under)\s+(?P<path>[\w./\\-]+))?\s*$",
        raw,
        re.IGNORECASE,
    )
    if list_match:
        path = _clean_natural_path(list_match.group("path") or ".")
        r = _file_tool_call(f"list {path}")
        _audit("natural_list", path, r.get("success", False))
        return True, r.get("output") or r.get("error", "list failed")

    open_match = re.search(
        r"\bopen\s+(?:file\s+|folder\s+)?(?:called\s+|named\s+)?"
        r"['\"]?(?P<target>https?://\S+|[\w./\\-]+\.[A-Za-z0-9]{1,8})['\"]?\s*$",
        raw,
        re.IGNORECASE,
    )
    if open_match:
        target = _clean_natural_path(open_match.group("target"))
        r = _os_tool_call(f"open {target}")
        _audit("natural_open", target, r.get("success", False))
        return True, r.get("output") or r.get("error", "open failed")

    cad_match = re.search(r"\b(?:generate|create|make)\s+(?:cad|openscad)\s+(?P<template>[\w-]+)?", raw, re.IGNORECASE)
    if cad_match:
        template = cad_match.group("template") or "raptor_engine"
        r = _cad_tool_call(template)
        _audit("natural_cad", template, r.get("success", False))
        return True, r.get("output") or r.get("error", "cad failed")

    goal_match = re.search(r"\b(?:run|execute|start)\s+(?:a\s+)?goal\s+(?P<goal>.+)", raw, re.IGNORECASE)
    if goal_match:
        goal = goal_match.group("goal").strip()
        _audit("natural_goal", goal, True)
        result = run_goal(goal)
        ok = bool(result.get("success"))
        out = str(result.get("output", ""))[:2000]
        return True, f"Success: {ok}\n{out}"

    return False, ""


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
        ws = _agent_workspace()
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
