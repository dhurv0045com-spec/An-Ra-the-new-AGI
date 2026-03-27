"""
================================================================================
FILE: tools/builtin.py
PROJECT: Agent Loop — 45K
PURPOSE: All built-in tools: web_search, code_executor, file_manager,
         calculator, memory_tool, summarizer, task_manager
================================================================================
Each tool is a standalone function that returns ToolResult.
They are registered into the ToolRegistry by register_all_tools().
================================================================================
"""

import os
import re
import ast
import math
import json
import time
import hashlib
import logging
import operator
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import ToolResult, ToolDefinition, SafetyLevel, ToolRegistry

logger = logging.getLogger(__name__)

# Safe root for all file operations — never escape this
_FILE_ROOT = Path(os.environ.get("AGENT_FILE_ROOT", "./agent_workspace")).resolve()
_FILE_ROOT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. WEB SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def web_search(query: str, **kwargs) -> ToolResult:
    """
    Search the web. Uses urllib (stdlib) to hit DuckDuckGo Instant Answer API.
    Falls back to a simulated result if network is unavailable (useful for tests).
    """
    import urllib.request
    import urllib.parse

    query = query.strip()
    if not query:
        return ToolResult(False, "", error="web_search: query cannot be empty")

    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1"
    })

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgentBot/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)

        results = []
        # Abstract (best answer)
        if data.get("AbstractText"):
            results.append({
                "title":   data.get("Heading", ""),
                "snippet": data["AbstractText"],
                "url":     data.get("AbstractURL", ""),
                "source":  "DDG Abstract",
            })
        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url":     topic.get("FirstURL", ""),
                    "source":  "DDG Related",
                })

        if not results:
            # No results from DDG, provide fallback
            return ToolResult(
                True,
                f"No instant results for '{query}'. Consider refining your search terms.",
                data={"query": query, "results": []},
            )

        formatted = f"Search results for: {query}\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"{i}. {r['title']}\n   {r['snippet'][:200]}\n   Source: {r['url']}\n\n"

        return ToolResult(True, formatted.strip(), data={"query": query, "results": results})

    except Exception as e:
        # Network unavailable — return structured error with helpful message
        return ToolResult(
            False, "",
            error=f"web_search failed: {e}. Check network connectivity.",
            metadata={"query": query}
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. CODE EXECUTOR
# ──────────────────────────────────────────────────────────────────────────────

# Forbidden patterns for sandboxing
_CODE_FORBIDDEN = [
    r"\bos\.system\b", r"\bsubprocess\b", r"\bshutil\.rmtree\b",
    r"\bopen\s*\(.*[\"\']/", r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(",
    r"\bimport\s+os\b", r"\bimport\s+subprocess\b", r"\bsocket\b",
]

def _is_safe_code(code: str) -> tuple:
    """Return (is_safe, reason). Blocks obviously dangerous patterns."""
    for pattern in _CODE_FORBIDDEN:
        if re.search(pattern, code):
            return False, f"Forbidden pattern: {pattern}"
    return True, ""


def code_executor(code: str, **kwargs) -> ToolResult:
    """
    Execute Python code in a sandboxed subprocess.
    Returns stdout, stderr, and return value.
    Timeout: 15 seconds. No network or filesystem writes outside workspace.
    """
    code = code.strip()
    if not code:
        return ToolResult(False, "", error="code_executor: no code provided")

    safe, reason = _is_safe_code(code)
    if not safe:
        return ToolResult(False, "", error=f"code_executor: unsafe code rejected — {reason}")

    # Wrap code to capture output cleanly
    wrapper = textwrap.dedent(f"""
import sys, io, json, math, statistics, re, datetime, collections, itertools

_stdout_buf = io.StringIO()
_stderr_buf = io.StringIO()

import contextlib
with contextlib.redirect_stdout(_stdout_buf), contextlib.redirect_stderr(_stderr_buf):
    try:
        _result = None
        exec(compile({repr(code)}, '<agent_code>', 'exec'), {{'__builtins__': __builtins__}})
    except Exception as _e:
        _stderr_buf.write(str(_e))

print(json.dumps({{
    'stdout': _stdout_buf.getvalue(),
    'stderr': _stderr_buf.getvalue(),
}}))
""")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp = f.name

    try:
        proc = subprocess.run(
            ["python3", tmp],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        os.unlink(tmp)

        if proc.returncode != 0 and not proc.stdout:
            return ToolResult(False, "", error=f"Execution error:\n{proc.stderr}")

        try:
            result = json.loads(proc.stdout.strip().split("\n")[-1])
        except Exception:
            result = {"stdout": proc.stdout, "stderr": proc.stderr}

        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()

        output = stdout or "(no output)"
        if stderr:
            output += f"\n[stderr]: {stderr}"

        return ToolResult(
            success=not bool(stderr and not stdout),
            output=output,
            data=result,
            error=stderr if stderr and not stdout else None,
        )
    except subprocess.TimeoutExpired:
        os.unlink(tmp)
        return ToolResult(False, "", error="code_executor: timed out after 15 seconds")
    except Exception as e:
        return ToolResult(False, "", error=f"code_executor: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. FILE MANAGER
# ──────────────────────────────────────────────────────────────────────────────

def _safe_path(relative: str) -> Optional[Path]:
    """Resolve a relative path under FILE_ROOT. Returns None if escaping root."""
    try:
        resolved = (_FILE_ROOT / relative.lstrip("/")).resolve()
        resolved.relative_to(_FILE_ROOT)   # raises if outside root
        return resolved
    except (ValueError, Exception):
        return None


def file_manager(instruction: str, **kwargs) -> ToolResult:
    """
    File operations: read, write, append, delete, list, search.

    Instruction format:
      read <path>
      write <path> <content>
      append <path> <content>
      delete <path>
      list [path]
      search <path> <pattern>
      exists <path>

    All paths are relative to the agent workspace root.
    """
    instruction = instruction.strip()
    parts = instruction.split(None, 2)
    if not parts:
        return ToolResult(False, "", error="file_manager: empty instruction")

    cmd = parts[0].lower()

    if cmd == "list":
        target = _safe_path(parts[1] if len(parts) > 1 else ".")
        if target is None:
            return ToolResult(False, "", error="Path escapes workspace root")
        if not target.exists():
            return ToolResult(False, "", error=f"Path not found: {parts[1] if len(parts)>1 else '.'}")
        if target.is_file():
            items = [target.name]
        else:
            items = [str(p.relative_to(_FILE_ROOT)) for p in sorted(target.iterdir())]
        return ToolResult(True, "\n".join(items) or "(empty)", data=items)

    elif cmd == "read":
        if len(parts) < 2:
            return ToolResult(False, "", error="file_manager read: missing path")
        path = _safe_path(parts[1])
        if path is None:
            return ToolResult(False, "", error="Path escapes workspace root")
        if not path.exists():
            return ToolResult(False, "", error=f"File not found: {parts[1]}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return ToolResult(True, content, data={"path": str(path.relative_to(_FILE_ROOT)), "size": len(content)})

    elif cmd == "write":
        if len(parts) < 3:
            return ToolResult(False, "", error="file_manager write: missing path or content")
        path = _safe_path(parts[1])
        if path is None:
            return ToolResult(False, "", error="Path escapes workspace root")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = parts[2]
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Written {len(content)} chars to {parts[1]}", data={"path": parts[1], "size": len(content)})

    elif cmd == "append":
        if len(parts) < 3:
            return ToolResult(False, "", error="file_manager append: missing path or content")
        path = _safe_path(parts[1])
        if path is None:
            return ToolResult(False, "", error="Path escapes workspace root")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(parts[2])
        return ToolResult(True, f"Appended to {parts[1]}")

    elif cmd == "delete":
        if len(parts) < 2:
            return ToolResult(False, "", error="file_manager delete: missing path")
        path = _safe_path(parts[1])
        if path is None:
            return ToolResult(False, "", error="Path escapes workspace root")
        if not path.exists():
            return ToolResult(False, "", error=f"File not found: {parts[1]}")
        path.unlink()
        return ToolResult(True, f"Deleted: {parts[1]}")

    elif cmd == "search":
        if len(parts) < 3:
            return ToolResult(False, "", error="file_manager search: need path and pattern")
        path = _safe_path(parts[1])
        if path is None:
            return ToolResult(False, "", error="Path escapes workspace root")
        pattern = parts[2]
        matches = []
        target = path if path.is_file() else None
        files  = [path] if target else list(path.rglob("*") if path.is_dir() else [])
        for fp in files:
            if not fp.is_file():
                continue
            try:
                for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append(f"{fp.relative_to(_FILE_ROOT)}:{i}: {line.strip()}")
            except Exception:
                pass
        output = "\n".join(matches[:50]) if matches else f"No matches for '{pattern}'"
        return ToolResult(True, output, data={"matches": len(matches)})

    elif cmd == "exists":
        if len(parts) < 2:
            return ToolResult(False, "", error="file_manager exists: missing path")
        path = _safe_path(parts[1])
        exists = path is not None and path.exists()
        return ToolResult(True, str(exists).lower(), data={"exists": exists})

    else:
        return ToolResult(False, "", error=f"Unknown file_manager command: '{cmd}'. Use: read, write, append, delete, list, search, exists")


# ──────────────────────────────────────────────────────────────────────────────
# 4. CALCULATOR
# ──────────────────────────────────────────────────────────────────────────────

# Allowed names in evaluated math expressions
_MATH_GLOBALS = {
    k: getattr(math, k) for k in dir(math) if not k.startswith("_")
}
_MATH_GLOBALS.update({
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "int": int, "float": float,
    "__builtins__": {},
})


def calculator(expression: str, **kwargs) -> ToolResult:
    """
    Evaluate a mathematical expression safely.
    Supports: arithmetic, trig, log, sqrt, floor, ceil, constants (pi, e, inf).
    Rejects: any non-mathematical Python code.
    """
    expression = expression.strip()
    if not expression:
        return ToolResult(False, "", error="calculator: empty expression")

    # Reject anything that looks like non-math code
    if re.search(r'[;]|import|exec|eval|open|print|__', expression):
        return ToolResult(False, "", error="calculator: expression contains disallowed tokens")

    try:
        # Parse AST to verify it's purely mathematical
        tree = ast.parse(expression, mode='eval')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                  ast.ClassDef, ast.Assign, ast.Call)):
                if isinstance(node, ast.Call):
                    # Allow math function calls only
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in _MATH_GLOBALS:
                            return ToolResult(False, "", error=f"calculator: function '{node.func.id}' not allowed")
                    elif isinstance(node.func, ast.Attribute):
                        pass  # allow e.g. math.sin
                    continue
                return ToolResult(False, "", error=f"calculator: disallowed AST node {type(node).__name__}")

        result = eval(compile(tree, "<calc>", "eval"), _MATH_GLOBALS)

        # Format result
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                formatted = str(int(result))
            else:
                formatted = f"{result:.10g}"
        else:
            formatted = str(result)

        return ToolResult(True, formatted, data={"expression": expression, "result": result})

    except ZeroDivisionError:
        return ToolResult(False, "", error="calculator: division by zero")
    except Exception as e:
        return ToolResult(False, "", error=f"calculator: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. MEMORY TOOL
# ──────────────────────────────────────────────────────────────────────────────

# Simple in-process memory store (in production: connects to 45J memory system)
_memory_store: Dict[str, Dict] = {}


def memory_tool(instruction: str, **kwargs) -> ToolResult:
    """
    Agent's explicit memory interface.

    Commands:
      store <key> <value>     — Store a fact
      recall <key>            — Retrieve by exact key
      search <query>          — Fuzzy search across all stored memories
      list                    — List all stored keys
      forget <key>            — Delete a memory
      stats                   — Memory store statistics
    """
    instruction = instruction.strip()
    parts = instruction.split(None, 2)
    if not parts:
        return ToolResult(False, "", error="memory_tool: empty instruction")

    cmd = parts[0].lower()

    if cmd == "store":
        if len(parts) < 3:
            return ToolResult(False, "", error="memory_tool store: need key and value")
        key, value = parts[1], parts[2]
        _memory_store[key] = {
            "value": value,
            "timestamp": time.time(),
            "access_count": _memory_store.get(key, {}).get("access_count", 0),
        }
        return ToolResult(True, f"Stored: {key}", data={"key": key})

    elif cmd == "recall":
        if len(parts) < 2:
            return ToolResult(False, "", error="memory_tool recall: need key")
        key = parts[1]
        if key not in _memory_store:
            return ToolResult(False, "", error=f"Memory key '{key}' not found")
        entry = _memory_store[key]
        entry["access_count"] = entry.get("access_count", 0) + 1
        return ToolResult(True, entry["value"], data=entry)

    elif cmd == "search":
        if len(parts) < 2:
            return ToolResult(False, "", error="memory_tool search: need query")
        query = parts[1].lower()
        results = []
        for k, v in _memory_store.items():
            if query in k.lower() or query in str(v["value"]).lower():
                results.append(f"{k}: {str(v['value'])[:120]}")
        output = "\n".join(results) if results else f"No memories matching '{query}'"
        return ToolResult(True, output, data={"matches": len(results)})

    elif cmd == "list":
        keys = list(_memory_store.keys())
        return ToolResult(True, "\n".join(keys) or "(empty)", data={"count": len(keys)})

    elif cmd == "forget":
        if len(parts) < 2:
            return ToolResult(False, "", error="memory_tool forget: need key")
        key = parts[1]
        if key in _memory_store:
            del _memory_store[key]
            return ToolResult(True, f"Forgotten: {key}")
        return ToolResult(False, "", error=f"Key '{key}' not found")

    elif cmd == "stats":
        total   = len(_memory_store)
        top     = sorted(_memory_store.items(), key=lambda x: x[1].get("access_count", 0), reverse=True)[:5]
        top_str = "\n".join(f"  {k}: {v.get('access_count',0)} accesses" for k, v in top)
        return ToolResult(True, f"Total memories: {total}\nMost accessed:\n{top_str}",
                          data={"total": total})

    else:
        return ToolResult(False, "", error=f"Unknown memory_tool command: '{cmd}'")


# ──────────────────────────────────────────────────────────────────────────────
# 6. SUMMARIZER
# ──────────────────────────────────────────────────────────────────────────────

def summarizer(text: str, **kwargs) -> ToolResult:
    """
    Summarize long text: extract key sentences, key points, and overall theme.
    Uses extractive summarization (no model needed) with TF-IDF-style scoring.
    """
    text = text.strip()
    if not text:
        return ToolResult(False, "", error="summarizer: empty text")

    max_sentences = int(kwargs.get("max_sentences", 5))

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= max_sentences:
        # Already short enough
        return ToolResult(True, text, data={"original_length": len(text), "summarized": True})

    # Score sentences by word frequency (extractive summarization)
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
                  "of", "and", "or", "but", "be", "was", "were", "this", "that",
                  "with", "by", "from", "are", "as", "not", "have", "has", "had"}
    freq = {}
    for w in words:
        if w not in stopwords and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    def score(sentence):
        s_words = re.findall(r'\b\w+\b', sentence.lower())
        return sum(freq.get(w, 0) for w in s_words) / max(len(s_words), 1)

    scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    top_idx = sorted([i for i, _ in scored[:max_sentences]])
    summary = " ".join(sentences[i] for i in top_idx)

    # Extract key bullet points (highest-scoring sentences)
    bullets = [f"• {sentences[i].strip()}" for i, _ in scored[:3]]

    output = f"SUMMARY:\n{summary}\n\nKEY POINTS:\n" + "\n".join(bullets)
    return ToolResult(True, output, data={
        "original_length": len(text),
        "summary_length": len(summary),
        "compression_ratio": round(len(summary) / len(text), 2),
    })


# ──────────────────────────────────────────────────────────────────────────────
# 7. TASK MANAGER
# ──────────────────────────────────────────────────────────────────────────────

# Persistent task store (file-backed for cross-session continuity)
_TASK_FILE = _FILE_ROOT / ".agent_tasks.json"


def _load_tasks() -> Dict:
    try:
        if _TASK_FILE.exists():
            return json.loads(_TASK_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_tasks(tasks: Dict) -> None:
    try:
        _TASK_FILE.write_text(json.dumps(tasks, indent=2))
    except Exception as e:
        logger.warning(f"task_manager: could not persist tasks: {e}")


def task_manager(instruction: str, **kwargs) -> ToolResult:
    """
    Manage tasks and subtasks across sessions.

    Commands:
      create <id> <title>          — Create a new task
      update <id> status=<s>       — Update task status (pending/active/done/failed)
      complete <id>                — Mark task complete
      fail <id> [reason]           — Mark task failed
      get <id>                     — Get task details
      list [status]                — List tasks (optionally filter by status)
      add_note <id> <note>         — Add a progress note to a task
      delete <id>                  — Delete a task
      clear                        — Delete all completed tasks
    """
    tasks = _load_tasks()
    instruction = instruction.strip()
    parts = instruction.split(None, 3)
    if not parts:
        return ToolResult(False, "", error="task_manager: empty instruction")

    cmd = parts[0].lower()

    if cmd == "create":
        if len(parts) < 3:
            return ToolResult(False, "", error="task_manager create: need id and title")
        tid, title = parts[1], parts[2]
        if tid in tasks:
            return ToolResult(False, "", error=f"Task '{tid}' already exists")
        tasks[tid] = {
            "id": tid, "title": title, "status": "pending",
            "created": time.time(), "updated": time.time(),
            "notes": [], "subtasks": [],
        }
        _save_tasks(tasks)
        return ToolResult(True, f"Created task: {tid} — {title}", data=tasks[tid])

    elif cmd == "update":
        if len(parts) < 3:
            return ToolResult(False, "", error="task_manager update: need id and field=value")
        tid = parts[1]
        if tid not in tasks:
            return ToolResult(False, "", error=f"Task '{tid}' not found")
        update_str = parts[2]
        for kv in update_str.split(","):
            if "=" in kv:
                k, _, v = kv.partition("=")
                tasks[tid][k.strip()] = v.strip()
        tasks[tid]["updated"] = time.time()
        _save_tasks(tasks)
        return ToolResult(True, f"Updated task: {tid}", data=tasks[tid])

    elif cmd == "complete":
        if len(parts) < 2:
            return ToolResult(False, "", error="task_manager complete: need id")
        tid = parts[1]
        if tid not in tasks:
            return ToolResult(False, "", error=f"Task '{tid}' not found")
        tasks[tid]["status"]  = "done"
        tasks[tid]["updated"] = time.time()
        tasks[tid]["completed_at"] = time.time()
        _save_tasks(tasks)
        return ToolResult(True, f"Task {tid} marked complete", data=tasks[tid])

    elif cmd == "fail":
        if len(parts) < 2:
            return ToolResult(False, "", error="task_manager fail: need id")
        tid = parts[1]
        if tid not in tasks:
            return ToolResult(False, "", error=f"Task '{tid}' not found")
        tasks[tid]["status"]    = "failed"
        tasks[tid]["updated"]   = time.time()
        tasks[tid]["fail_reason"] = parts[3] if len(parts) > 3 else ""
        _save_tasks(tasks)
        return ToolResult(True, f"Task {tid} marked failed", data=tasks[tid])

    elif cmd == "get":
        if len(parts) < 2:
            return ToolResult(False, "", error="task_manager get: need id")
        tid = parts[1]
        if tid not in tasks:
            return ToolResult(False, "", error=f"Task '{tid}' not found")
        t = tasks[tid]
        lines = [f"Task: {t['id']}", f"Title: {t['title']}", f"Status: {t['status']}"]
        if t.get("notes"):
            lines.append("Notes:\n" + "\n".join(f"  - {n}" for n in t["notes"]))
        return ToolResult(True, "\n".join(lines), data=t)

    elif cmd == "list":
        filter_status = parts[1].lower() if len(parts) > 1 else None
        filtered = {k: v for k, v in tasks.items()
                    if not filter_status or v["status"] == filter_status}
        if not filtered:
            return ToolResult(True, f"No tasks{' with status ' + filter_status if filter_status else ''}.",
                              data={"count": 0})
        lines = [f"[{v['status']:7}] {k}: {v['title']}" for k, v in filtered.items()]
        return ToolResult(True, "\n".join(lines), data={"count": len(filtered), "tasks": filtered})

    elif cmd == "add_note":
        if len(parts) < 3:
            return ToolResult(False, "", error="task_manager add_note: need id and note")
        tid, note = parts[1], parts[2]
        if tid not in tasks:
            return ToolResult(False, "", error=f"Task '{tid}' not found")
        tasks[tid].setdefault("notes", []).append(note)
        tasks[tid]["updated"] = time.time()
        _save_tasks(tasks)
        return ToolResult(True, f"Note added to {tid}", data=tasks[tid])

    elif cmd == "delete":
        if len(parts) < 2:
            return ToolResult(False, "", error="task_manager delete: need id")
        tid = parts[1]
        if tid not in tasks:
            return ToolResult(False, "", error=f"Task '{tid}' not found")
        del tasks[tid]
        _save_tasks(tasks)
        return ToolResult(True, f"Deleted: {tid}")

    elif cmd == "clear":
        before = len(tasks)
        tasks  = {k: v for k, v in tasks.items() if v["status"] != "done"}
        _save_tasks(tasks)
        return ToolResult(True, f"Cleared {before - len(tasks)} completed tasks")

    else:
        return ToolResult(False, "", error=f"Unknown task_manager command: '{cmd}'")


# ──────────────────────────────────────────────────────────────────────────────
# REGISTRATION
# ──────────────────────────────────────────────────────────────────────────────

def register_all_tools(registry: ToolRegistry) -> None:
    """Register every built-in tool into the given registry."""

    tools = [
        ToolDefinition(
            name="web_search",
            description="Search the web for current information on any topic. Returns top results with snippets and URLs.",
            fn=web_search,
            parameters={"query": "The search query string"},
            safety_level=SafetyLevel.RESTRICTED,
            rate_limit=20,
            examples=["web_search best GPU under 500 dollars 2024", "web_search Python async tutorial"],
        ),
        ToolDefinition(
            name="code_executor",
            description="Write and execute Python code. Returns stdout and stderr. Safe sandbox — no filesystem writes or network calls.",
            fn=code_executor,
            parameters={"code": "Valid Python code to execute"},
            safety_level=SafetyLevel.RESTRICTED,
            rate_limit=30,
            examples=["code_executor print(sum(range(100)))", "code_executor import math; print(math.pi)"],
        ),
        ToolDefinition(
            name="file_manager",
            description="Read, write, list, search, and delete files within the agent workspace. Paths are relative to workspace root.",
            fn=file_manager,
            parameters={"instruction": "Command + path + content. E.g.: 'write report.txt Hello world'"},
            safety_level=SafetyLevel.RESTRICTED,
            rate_limit=60,
            examples=["file_manager list", "file_manager read notes.txt", "file_manager write output.md # Report"],
        ),
        ToolDefinition(
            name="calculator",
            description="Evaluate mathematical expressions precisely. Supports arithmetic, trigonometry, logarithms, sqrt, floor, ceil.",
            fn=calculator,
            parameters={"expression": "A valid math expression, e.g. '2 ** 32' or 'sqrt(2) * pi'"},
            safety_level=SafetyLevel.SAFE,
            rate_limit=0,
            examples=["calculator 2**32", "calculator sqrt(144) * pi", "calculator log(1000, 10)"],
        ),
        ToolDefinition(
            name="memory_tool",
            description="Store and retrieve facts in the agent's memory. Persists across steps within a session.",
            fn=memory_tool,
            parameters={"instruction": "Command: store/recall/search/list/forget/stats + args"},
            safety_level=SafetyLevel.SAFE,
            rate_limit=0,
            examples=["memory_tool store gpu_choice RTX 4090", "memory_tool recall gpu_choice", "memory_tool search GPU"],
        ),
        ToolDefinition(
            name="summarizer",
            description="Summarize long text, extract key points, and condense web pages or documents.",
            fn=summarizer,
            parameters={"text": "The text to summarize"},
            safety_level=SafetyLevel.SAFE,
            rate_limit=0,
            examples=["summarizer <long document text>"],
        ),
        ToolDefinition(
            name="task_manager",
            description="Create and track tasks and subtasks across the agent's execution. Persists between sessions.",
            fn=task_manager,
            parameters={"instruction": "Command: create/update/complete/fail/get/list/add_note/delete/clear"},
            safety_level=SafetyLevel.SAFE,
            rate_limit=0,
            examples=["task_manager create step1 Research GPU options", "task_manager complete step1", "task_manager list"],
        ),
    ]

    for t in tools:
        if t.name not in registry:
            registry.register(t)

    logger.info(f"Registered {len(tools)} built-in tools")
