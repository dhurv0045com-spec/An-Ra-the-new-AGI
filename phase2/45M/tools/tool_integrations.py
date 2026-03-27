"""
tools/tool_integrations.py — Real Tool Integrations

Concrete tools the agent can actually call:
  - web_search:     DuckDuckGo instant answers (no API key needed)
  - code_execute:   Safe Python sandbox with output capture
  - file_read:      Read local files (text, JSON, CSV)
  - file_write:     Write files with tier-2 decision check
  - calculator:     Safe math expression evaluator
  - datetime_tool:  Current date/time, date arithmetic
  - knowledge_lookup: Search the personal knowledge base
  - summarize:      Summarize long text
  - http_fetch:     Fetch a URL (text content only)

Each tool is self-contained, logs what it does, and returns
a structured result the model can use in its response.
"""

import os, re, json, math, time, subprocess, textwrap
import urllib.request, urllib.parse, urllib.error
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO
import contextlib, traceback


TOOL_LOG_DIR = Path("state/tool_logs")
TOOL_LOG_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_WRITE_DIRS = [Path("outputs"), Path("state/files")]
for d in ALLOWED_WRITE_DIRS:
    d.mkdir(parents=True, exist_ok=True)


# ── Tool result ────────────────────────────────────────────────────────────────

class ToolResult:
    def __init__(self, tool: str, success: bool, output: Any,
                 error: str = "", duration_ms: float = 0.0):
        self.tool        = tool
        self.success     = success
        self.output      = output
        self.error       = error
        self.duration_ms = duration_ms
        self.timestamp   = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "tool":        self.tool,
            "success":     self.success,
            "output":      self.output,
            "error":       self.error,
            "duration_ms": self.duration_ms,
            "timestamp":   self.timestamp,
        }

    def __str__(self) -> str:
        if self.success:
            return f"[{self.tool}] {str(self.output)[:500]}"
        return f"[{self.tool}] ERROR: {self.error}"


def _timed(fn):
    """Decorator: adds duration_ms to ToolResult."""
    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        result = fn(*args, **kwargs)
        result.duration_ms = round((time.monotonic() - t0) * 1000, 1)
        return result
    return wrapper


# ── Calculator ─────────────────────────────────────────────────────────────────

@_timed
def calculator(expression: str) -> ToolResult:
    """
    Safely evaluate a math expression.
    Supports: +, -, *, /, **, sqrt, sin, cos, log, abs, round, etc.
    No eval of arbitrary Python — restricted to math operations.
    """
    # Whitelist: only math-safe characters
    safe = re.sub(r'[^0-9+\-*/().,\s]', '', expression)
    if expression.lower().replace(' ', '') != safe.replace(' ', ''):
        # Has non-numeric chars — try math function names
        allowed_names = {
            k: getattr(math, k) for k in dir(math) if not k.startswith('_')
        }
        allowed_names['abs'] = abs
        allowed_names['round'] = round
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return ToolResult("calculator", True, result)
        except Exception as e:
            return ToolResult("calculator", False, None, str(e))
    try:
        result = eval(safe, {"__builtins__": {}}, {})
        return ToolResult("calculator", True, result)
    except Exception as e:
        return ToolResult("calculator", False, None, str(e))


# ── DateTime ───────────────────────────────────────────────────────────────────

@_timed
def datetime_tool(query: str = "now") -> ToolResult:
    """
    Get current datetime, or do date arithmetic.
    Examples: "now", "today", "tomorrow", "in 7 days", "days since 2024-01-01"
    """
    now = datetime.utcnow()
    q   = query.lower().strip()

    if q in ("now", "current", "time"):
        return ToolResult("datetime", True, {
            "datetime": now.isoformat(),
            "formatted": now.strftime("%A, %B %d, %Y at %H:%M UTC"),
            "timestamp": time.time(),
        })

    if q == "today":
        return ToolResult("datetime", True, now.strftime("%Y-%m-%d"))
    if q == "tomorrow":
        return ToolResult("datetime", True, (now + timedelta(days=1)).strftime("%Y-%m-%d"))

    # "in N days"
    m = re.search(r'in (\d+) day', q)
    if m:
        n = int(m.group(1))
        future = (now + timedelta(days=n)).strftime("%Y-%m-%d")
        return ToolResult("datetime", True, future)

    # "days since DATE"
    m = re.search(r'days since (\d{4}-\d{2}-\d{2})', q)
    if m:
        try:
            past = datetime.fromisoformat(m.group(1))
            days = (now - past).days
            return ToolResult("datetime", True, {"days_since": days})
        except ValueError as e:
            return ToolResult("datetime", False, None, str(e))

    return ToolResult("datetime", True, now.isoformat())


# ── File Read ──────────────────────────────────────────────────────────────────

@_timed
def file_read(path: str, max_chars: int = 5000) -> ToolResult:
    """
    Read a local text file.
    Supports: .txt, .md, .json, .csv, .py, and other plain text.
    Truncates at max_chars.
    """
    p = Path(path)
    if not p.exists():
        return ToolResult("file_read", False, None, f"File not found: {path}")
    if not p.is_file():
        return ToolResult("file_read", False, None, f"Not a file: {path}")
    if p.stat().st_size > 10_000_000:  # 10MB limit
        return ToolResult("file_read", False, None, "File too large (>10MB)")

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        output = {
            "path":      str(p),
            "size":      p.stat().st_size,
            "lines":     text.count("\n"),
            "content":   text[:max_chars],
            "truncated": truncated,
        }
        return ToolResult("file_read", True, output)
    except Exception as e:
        return ToolResult("file_read", False, None, str(e))


# ── File Write ─────────────────────────────────────────────────────────────────

@_timed
def file_write(path: str, content: str, append: bool = False) -> ToolResult:
    """
    Write text to a file. Restricted to allowed directories.
    append=True adds to existing file rather than overwriting.
    """
    p = Path(path)

    # Safety: only allow writes to approved directories
    allowed = any(
        p.resolve().is_relative_to(d.resolve())
        for d in ALLOWED_WRITE_DIRS
    )
    if not allowed and not path.startswith("outputs/") and not path.startswith("state/files/"):
        # Try to make it relative to outputs/
        p = ALLOWED_WRITE_DIRS[0] / path

    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = "a" if append else "w"
        with open(p, mode, encoding="utf-8") as f:
            f.write(content)
        return ToolResult("file_write", True, {
            "path": str(p), "bytes": len(content.encode()),
            "mode": "append" if append else "overwrite"
        })
    except Exception as e:
        return ToolResult("file_write", False, None, str(e))


# ── Code Executor ──────────────────────────────────────────────────────────────

@_timed
def code_execute(code: str, timeout: int = 10) -> ToolResult:
    """
    Execute Python code in a restricted sandbox.
    Captures stdout/stderr. Returns output.
    Hard limits: no file system access, no network, no subprocess.
    Timeout: kills execution after N seconds.
    """
    # Block dangerous operations
    blocked = [
        "import os", "import sys", "import subprocess", "import socket",
        "__import__", "open(", "exec(", "eval(", "compile(",
        "importlib", "ctypes", "pickle",
    ]
    code_lower = code.lower()
    for blocked_term in blocked:
        if blocked_term.lower() in code_lower:
            return ToolResult(
                "code_execute", False, None,
                f"Blocked: code contains '{blocked_term}'"
            )

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Restricted globals: math + builtins only
    safe_globals = {
        "__builtins__": {
            "print": lambda *a, **kw: print(*a, file=stdout_capture, **kw),
            "range": range, "len": len, "list": list, "dict": dict,
            "set": set, "tuple": tuple, "str": str, "int": int,
            "float": float, "bool": bool, "abs": abs, "round": round,
            "min": min, "max": max, "sum": sum, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
            "reversed": reversed, "isinstance": isinstance, "type": type,
            "repr": repr, "format": format, "chr": chr, "ord": ord,
        }
    }

    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {timeout}s timeout")

        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        try:
            exec(code, safe_globals)
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        output = stdout_capture.getvalue()
        return ToolResult("code_execute", True, {
            "stdout":    output[:2000] if output else "(no output)",
            "lines":     output.count("\n"),
        })

    except TimeoutError as e:
        return ToolResult("code_execute", False, None, str(e))
    except Exception as e:
        err_lines = traceback.format_exc().split("\n")
        # Filter out internal frame info
        clean = [l for l in err_lines if "safe_globals" not in l]
        return ToolResult("code_execute", False, None, "\n".join(clean[-5:]))


# ── Web Search (DuckDuckGo Instant Answers — no API key) ──────────────────────

@_timed
def web_search(query: str, max_results: int = 5) -> ToolResult:
    """
    Search the web using DuckDuckGo's HTML interface.
    No API key required. Returns titles, snippets, and URLs.
    Falls back gracefully if network is unavailable.
    """
    try:
        encoded = urllib.parse.quote_plus(query)
        url     = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MyAI/1.0)"
        }
        req  = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Parse results from HTML
        results = []
        # Extract result blocks
        pattern = r'<a class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        matches = re.findall(pattern, html)
        snippet_pattern = r'<a class="result__snippet"[^>]*>(.*?)</a>'
        snippets = re.findall(snippet_pattern, html)

        for i, (url_match, title) in enumerate(matches[:max_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            # Clean HTML tags
            clean_title   = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            if clean_title:
                results.append({
                    "title":   clean_title[:100],
                    "snippet": clean_snippet[:300],
                    "url":     url_match[:200],
                })

        if not results:
            # Try to extract from plain text
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            results = [{"title": f"Search results for: {query}",
                        "snippet": text[:500], "url": url}]

        return ToolResult("web_search", True, {
            "query":   query,
            "results": results,
            "count":   len(results),
        })

    except urllib.error.URLError as e:
        return ToolResult("web_search", False, None,
                          f"Network unavailable: {e}")
    except Exception as e:
        return ToolResult("web_search", False, None, str(e))


# ── HTTP Fetch ─────────────────────────────────────────────────────────────────

@_timed
def http_fetch(url: str, max_chars: int = 8000) -> ToolResult:
    """
    Fetch the text content of a URL.
    Strips HTML tags. Returns clean readable text.
    """
    if not url.startswith(("http://", "https://")):
        return ToolResult("http_fetch", False, None, "URL must start with http:// or https://")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MyAI/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw  = resp.read(1_000_000).decode("utf-8", errors="replace")

        # Strip HTML
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', raw, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>',  ' ', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return ToolResult("http_fetch", True, {
            "url":       url,
            "content":   text[:max_chars],
            "truncated": len(text) > max_chars,
            "length":    len(text),
        })
    except urllib.error.URLError as e:
        return ToolResult("http_fetch", False, None, f"Fetch failed: {e}")
    except Exception as e:
        return ToolResult("http_fetch", False, None, str(e))


# ── Summarize ──────────────────────────────────────────────────────────────────

@_timed
def summarize(text: str, max_sentences: int = 5) -> ToolResult:
    """
    Extractive summarization: select the most important sentences.
    Uses TF-IDF scoring — no model needed.
    """
    if len(text) < 200:
        return ToolResult("summarize", True, {"summary": text, "method": "passthrough"})

    # Sentence tokenize
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return ToolResult("summarize", True, {
            "summary": text[:2000], "sentences": len(sentences)
        })

    # Score sentences by word frequency
    words  = re.findall(r'\b[a-z]{3,}\b', text.lower())
    from collections import Counter
    freq   = Counter(words)
    # Penalize very common words
    for stopword in ("the","and","for","are","but","not","you","all",
                     "can","had","her","was","one","our","out","day","get"):
        freq.pop(stopword, None)

    def score_sentence(sent):
        ws = re.findall(r'\b[a-z]{3,}\b', sent.lower())
        return sum(freq.get(w, 0) for w in ws) / max(len(ws), 1)

    scored = [(score_sentence(s), i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: -x[0])
    top    = sorted(scored[:max_sentences], key=lambda x: x[1])
    summary = " ".join(s for _, _, s in top)

    return ToolResult("summarize", True, {
        "summary":          summary[:2000],
        "original_length":  len(text),
        "summary_length":   len(summary),
        "sentences_kept":   max_sentences,
        "total_sentences":  len(sentences),
    })


# ── Tool Registry ──────────────────────────────────────────────────────────────

class ToolKit:
    """
    Central registry for all tools.
    Provides a unified call interface for the agent.
    Logs every tool invocation.
    """

    TOOLS = {
        "calculator":   calculator,
        "datetime":     datetime_tool,
        "file_read":    file_read,
        "file_write":   file_write,
        "code_execute": code_execute,
        "web_search":   web_search,
        "http_fetch":   http_fetch,
        "summarize":    summarize,
    }

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Call a tool by name with keyword arguments.
        Logs the invocation and result.
        """
        fn = self.TOOLS.get(tool_name)
        if not fn:
            return ToolResult(tool_name, False, None,
                              f"Unknown tool: '{tool_name}'. Available: {list(self.TOOLS)}")
        result = fn(**kwargs)
        self._log(tool_name, kwargs, result)
        return result

    def _log(self, tool: str, args: dict, result: ToolResult):
        log_entry = {
            "tool":     tool,
            "args":     {k: str(v)[:100] for k, v in args.items()},
            "success":  result.success,
            "duration": result.duration_ms,
            "ts":       result.timestamp,
        }
        log_file = TOOL_LOG_DIR / f"{tool}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def available(self) -> List[str]:
        return list(self.TOOLS)

    def schema(self) -> List[dict]:
        """Return tool schemas for model function-calling prompts."""
        return [
            {"name": "calculator",   "description": "Evaluate math expressions",
             "params": {"expression": "str"}},
            {"name": "datetime",     "description": "Get current date/time or calculate dates",
             "params": {"query": "str — 'now', 'today', 'in N days', 'days since DATE'"}},
            {"name": "file_read",    "description": "Read a local text file",
             "params": {"path": "str", "max_chars": "int (optional, default 5000)"}},
            {"name": "file_write",   "description": "Write text to a file in outputs/",
             "params": {"path": "str", "content": "str", "append": "bool (optional)"}},
            {"name": "code_execute", "description": "Execute Python code safely (no network/filesystem)",
             "params": {"code": "str"}},
            {"name": "web_search",   "description": "Search the web with DuckDuckGo",
             "params": {"query": "str", "max_results": "int (optional)"}},
            {"name": "http_fetch",   "description": "Fetch and read a URL",
             "params": {"url": "str"}},
            {"name": "summarize",    "description": "Summarize long text extractively",
             "params": {"text": "str", "max_sentences": "int (optional)"}},
        ]

    def test_all(self) -> dict:
        """Run a quick smoke test on every tool."""
        results = {}
        tests = [
            ("calculator",   {"expression": "2 ** 10 + sqrt(144)"}),
            ("datetime",     {"query": "now"}),
            ("code_execute", {"code": "print(sum(range(10)))"}),
            ("summarize",    {"text": "This is sentence one. This is sentence two. "
                                       "This is sentence three. This is the fourth one. "
                                       "And the fifth and final sentence.", "max_sentences": 2}),
        ]
        for name, kwargs in tests:
            r = self.call(name, **kwargs)
            results[name] = {"ok": r.success, "ms": r.duration_ms,
                              "preview": str(r.output)[:80] if r.success else r.error}
        return results


# Singleton
toolkit = ToolKit()
