"""
================================================================================
FILE: tools/registry.py
PROJECT: Agent Loop — 45K
PURPOSE: Tool registration, dispatch, rate limiting, safety checks, call logging
================================================================================
"""

import time
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    SAFE       = "safe"       # No restrictions
    RESTRICTED = "restricted" # External side-effects (web, files)
    DANGEROUS  = "dangerous"  # Irreversible (delete, write external)


@dataclass
class ToolResult:
    success:     bool
    output:      str
    data:        Any           = None
    error:       Optional[str] = None
    tool_name:   str           = ""
    duration_ms: float         = 0.0
    metadata:    Dict          = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"success": self.success, "output": self.output,
                "data": self.data, "error": self.error,
                "tool_name": self.tool_name, "duration_ms": round(self.duration_ms, 2)}

    def __str__(self) -> str:
        return self.output if self.success else f"[ERROR:{self.tool_name}] {self.error}"


@dataclass
class ToolDefinition:
    name:         str
    description:  str
    fn:           Callable
    parameters:   Dict[str, str]
    safety_level: SafetyLevel  = SafetyLevel.SAFE
    rate_limit:   int          = 60   # calls per minute; 0 = unlimited
    examples:     List[str]    = field(default_factory=list)
    _call_times:  deque        = field(default_factory=lambda: deque(maxlen=500))

    def is_rate_limited(self) -> bool:
        if self.rate_limit == 0:
            return False
        now = time.time()
        while self._call_times and self._call_times[0] < now - 60:
            self._call_times.popleft()
        return len(self._call_times) >= self.rate_limit

    def record_call(self):
        self._call_times.append(time.time())


@dataclass
class ToolCall:
    tool_name:  str
    input_text: str
    result:     ToolResult
    timestamp:  float = field(default_factory=time.time)
    step_id:    str   = ""
    approved:   bool  = True

    def to_dict(self) -> Dict:
        return {"tool_name": self.tool_name, "input": self.input_text,
                "result": self.result.to_dict(), "timestamp": self.timestamp,
                "step_id": self.step_id}


class ToolRegistry:
    """
    Central hub for all agent tools.
    Handles: registration, dispatch, rate-limiting, safety, logging.
    Never raises — all errors surfaced as ToolResult(success=False).
    """

    def __init__(self, require_approval_for_dangerous: bool = True):
        self._tools: Dict[str, ToolDefinition] = {}
        self._call_log: List[ToolCall] = []
        self._require_approval = require_approval_for_dangerous
        self._approval_callback: Optional[Callable] = None

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool
        logger.info(f"Tool registered: {tool.name} [{tool.safety_level.value}]")

    def deregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        return [{"name": t.name, "description": t.description,
                 "parameters": t.parameters, "safety": t.safety_level.value,
                 "examples": t.examples} for t in self._tools.values()]

    def set_approval_callback(self, fn: Callable) -> None:
        self._approval_callback = fn

    def call(self, tool_name: str, input_text: str, step_id: str = "", **kwargs) -> ToolResult:
        """Dispatch one tool call. Always returns ToolResult."""
        tool = self._tools.get(tool_name)
        if tool is None:
            avail = ", ".join(self._tools) or "none"
            r = ToolResult(False, "", error=f"Tool '{tool_name}' not found. Available: {avail}", tool_name=tool_name)
            self._log(tool_name, input_text, r, step_id)
            return r

        if tool.is_rate_limited():
            r = ToolResult(False, "", error=f"Rate limit: {tool.rate_limit}/min for '{tool_name}'", tool_name=tool_name)
            self._log(tool_name, input_text, r, step_id)
            return r

        if tool.safety_level == SafetyLevel.DANGEROUS and self._require_approval:
            approved = self._approval_callback(tool_name, input_text) if self._approval_callback else False
            if not approved:
                r = ToolResult(False, "", error=f"Dangerous tool '{tool_name}' denied — no approval.", tool_name=tool_name)
                self._log(tool_name, input_text, r, step_id, approved=False)
                return r

        t0 = time.time()
        try:
            result = tool.fn(input_text, **kwargs)
            result.tool_name   = tool_name
            result.duration_ms = (time.time() - t0) * 1000
            tool.record_call()
            logger.debug(f"Tool '{tool_name}' OK {result.duration_ms:.0f}ms")
        except Exception as e:
            result = ToolResult(False, "", error=f"{e}\n{traceback.format_exc()}",
                                tool_name=tool_name, duration_ms=(time.time()-t0)*1000)
            logger.error(f"Tool '{tool_name}' crashed: {e}")

        self._log(tool_name, input_text, result, step_id)
        return result

    def _log(self, tool_name, input_text, result, step_id, approved=True):
        self._call_log.append(ToolCall(tool_name, input_text, result,
                                        step_id=step_id, approved=approved))

    def get_call_log(self, step_id: str = "") -> List[ToolCall]:
        if step_id:
            return [c for c in self._call_log if c.step_id == step_id]
        return list(self._call_log)

    def summary(self) -> Dict:
        total = len(self._call_log)
        ok    = sum(1 for c in self._call_log if c.result.success)
        by_tool = {}
        for c in self._call_log:
            s = by_tool.setdefault(c.tool_name, {"calls": 0, "errors": 0})
            s["calls"] += 1
            if not c.result.success:
                s["errors"] += 1
        return {"total": total, "success": ok, "failed": total - ok,
                "rate": round(ok/total, 3) if total else 0, "by_tool": by_tool}

    def __contains__(self, n): return n in self._tools
    def __len__(self):         return len(self._tools)


_default_registry: Optional[ToolRegistry] = None

def get_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry
