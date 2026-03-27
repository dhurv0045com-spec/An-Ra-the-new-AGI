"""
================================================================================
FILE: agent/core/dispatcher.py
PROJECT: Agent Loop — 45K
PURPOSE: Natural language → best tool selection and call formatting
================================================================================
"""

import re
import logging
from typing import List, Optional, Tuple
from tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Keyword → tool priority mapping for heuristic selection
_TOOL_KEYWORDS = {
    "web_search":    ["search", "find", "look up", "research", "what is", "who is",
                      "current", "latest", "best", "reviews", "compare online"],
    "code_executor": ["run", "execute", "compute", "python", "calculate with code",
                      "script", "program", "implement", "test code"],
    "file_manager":  ["read file", "write file", "save", "load", "open", "list files",
                      "create file", "delete file", "directory"],
    "calculator":    ["calculate", "math", "compute", "sum", "average", "sqrt",
                      "multiply", "divide", "equation", "formula"],
    "memory_tool":   ["remember", "store", "recall", "retrieve", "memorize", "forget",
                      "memory", "stored", "previously"],
    "summarizer":    ["summarize", "summary", "condense", "key points", "tldr",
                      "brief", "shorten", "extract"],
    "task_manager":  ["task", "todo", "track", "create task", "complete task",
                      "mark done", "subtask"],
}


class Dispatcher:
    """
    Resolves natural language requests to the best matching tool.

    When the agent says "I need to search for X", the dispatcher:
      1. Scores each registered tool against the request
      2. Picks the highest scorer
      3. Formats the input correctly for that tool
      4. Calls it via the registry
      5. Returns the ToolResult

    Also supports explicit tool override: "web_search: query here"
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def dispatch(self, request: str, step_id: str = "") -> ToolResult:
        """
        Route a natural language request to the best tool.

        Supports explicit tool prefix: "tool_name: input"
        Falls back to scoring if no explicit prefix.
        """
        request = request.strip()

        # Explicit prefix override: "tool_name: actual input"
        if ":" in request:
            prefix, _, body = request.partition(":")
            prefix = prefix.strip().lower().replace(" ", "_")
            if prefix in self.registry:
                logger.debug(f"Dispatcher: explicit tool '{prefix}'")
                return self.registry.call(prefix, body.strip(), step_id=step_id)

        # Score each tool
        tool_name = self._score_and_select(request)
        logger.debug(f"Dispatcher: selected '{tool_name}' for: {request[:80]!r}")
        return self.registry.call(tool_name, request, step_id=step_id)

    def _score_and_select(self, request: str) -> str:
        """Pick the highest-scoring tool for a request."""
        request_lower = request.lower()
        scores = {}
        tools  = self.registry.list_tools()

        for tool_info in tools:
            name     = tool_info["name"]
            keywords = _TOOL_KEYWORDS.get(name, [])
            score    = sum(1 for kw in keywords if kw in request_lower)
            # Bonus: tool name appears in request
            if name.replace("_", " ") in request_lower or name in request_lower:
                score += 3
            scores[name] = score

        if not scores:
            return "memory_tool"  # Fallback

        # Tie-break: prefer safer tools
        safety_order = {"safe": 0, "restricted": 1, "dangerous": 2}
        def sort_key(item):
            name, score = item
            tool = self.registry.get(name)
            safety = safety_order.get(tool.safety_level.value if tool else "safe", 0)
            return (-score, safety)

        best_name, best_score = sorted(scores.items(), key=sort_key)[0]
        if best_score == 0:
            # No good match — default to memory_tool for recording
            return "memory_tool"
        return best_name

    def describe_tools(self) -> str:
        """Return a formatted string describing all available tools."""
        lines = ["Available tools:"]
        for t in self.registry.list_tools():
            lines.append(f"  {t['name']}: {t['description']}")
            if t.get("examples"):
                lines.append(f"    Example: {t['examples'][0]}")
        return "\n".join(lines)
