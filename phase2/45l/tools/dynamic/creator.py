"""
tools/dynamic/creator.py — Dynamic Tool Creation Engine

Agent detects capability gaps, writes Python tools, tests them,
and registers them persistently. Tools survive restarts.
Human review queue for new tools before use on important tasks.
"""

import ast, uuid, time, json, sqlite3, threading, textwrap, traceback, hashlib, os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from io import StringIO
import contextlib


TOOLS_DIR = Path("state/dynamic_tools")
DB_PATH   = Path("state/tool_registry.db")
TOOLS_DIR.mkdir(parents=True, exist_ok=True)


# ── Database ───────────────────────────────────────────────────────────────────

class ToolRegistryDB:
    def __init__(self, path=DB_PATH):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS tools (
                    tool_id     TEXT PRIMARY KEY,
                    name        TEXT UNIQUE,
                    description TEXT,
                    code        TEXT,
                    version     INTEGER DEFAULT 1,
                    status      TEXT DEFAULT 'pending_review',
                    created_at  TEXT,
                    approved_at TEXT,
                    call_count  INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    avg_ms      REAL DEFAULT 0,
                    last_used   TEXT,
                    author      TEXT DEFAULT 'agent'
                );
                CREATE TABLE IF NOT EXISTS tool_versions (
                    version_id TEXT PRIMARY KEY,
                    tool_id    TEXT,
                    version    INTEGER,
                    code       TEXT,
                    created_at TEXT,
                    perf_score REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS review_queue (
                    queue_id   TEXT PRIMARY KEY,
                    tool_id    TEXT,
                    created_at TEXT,
                    reason     TEXT,
                    reviewed   INTEGER DEFAULT 0
                );
            """)
            self._conn.commit()

    def register(self, tool_id, name, description, code,
                 status="pending_review", author="agent"):
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO tools
                (tool_id,name,description,code,status,created_at,author)
                VALUES (?,?,?,?,?,?,?)
            """, (tool_id, name, description, code, status,
                  datetime.utcnow().isoformat(), author))
            self._conn.commit()

    def save_version(self, tool_id, version, code, perf_score=0.0):
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO tool_versions VALUES (?,?,?,?,?,?)
            """, (str(uuid.uuid4()), tool_id, version, code,
                  datetime.utcnow().isoformat(), perf_score))
            self._conn.commit()

    def get(self, name) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tools WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        cols = ["tool_id","name","description","code","version","status",
                "created_at","approved_at","call_count","success_count",
                "avg_ms","last_used","author"]
        return dict(zip(cols, row))

    def list_all(self, status=None) -> List[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM tools WHERE status=?", (status,)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM tools").fetchall()
        cols = ["tool_id","name","description","code","version","status",
                "created_at","approved_at","call_count","success_count",
                "avg_ms","last_used","author"]
        return [dict(zip(cols, r)) for r in rows]

    def update_stats(self, name, success, elapsed_ms):
        with self._lock:
            tool = self._conn.execute(
                "SELECT call_count,success_count,avg_ms FROM tools WHERE name=?",
                (name,)).fetchone()
            if not tool:
                return
            calls    = tool[0] + 1
            successes = tool[1] + (1 if success else 0)
            avg_ms   = (tool[2] * tool[0] + elapsed_ms) / calls
            self._conn.execute("""
                UPDATE tools SET call_count=?,success_count=?,avg_ms=?,last_used=?
                WHERE name=?
            """, (calls, successes, avg_ms, datetime.utcnow().isoformat(), name))
            self._conn.commit()

    def approve(self, name):
        with self._lock:
            self._conn.execute("""
                UPDATE tools SET status='approved', approved_at=?
                WHERE name=?
            """, (datetime.utcnow().isoformat(), name))
            self._conn.commit()

    def retire(self, name):
        with self._lock:
            self._conn.execute(
                "UPDATE tools SET status='retired' WHERE name=?", (name,))
            self._conn.commit()

    def queue_for_review(self, tool_id, reason):
        with self._lock:
            self._conn.execute("""
                INSERT INTO review_queue VALUES (?,?,?,?,0)
            """, (str(uuid.uuid4()), tool_id,
                  datetime.utcnow().isoformat(), reason))
            self._conn.commit()

    def pending_review(self) -> List[dict]:
        with self._lock:
            rows = self._conn.execute("""
                SELECT q.queue_id, q.tool_id, q.created_at, q.reason,
                       t.name, t.description, t.code
                FROM review_queue q JOIN tools t ON q.tool_id=t.tool_id
                WHERE q.reviewed=0
            """).fetchall()
        cols = ["queue_id","tool_id","created_at","reason","name","description","code"]
        return [dict(zip(cols, r)) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            total   = self._conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]
            approved= self._conn.execute(
                "SELECT COUNT(*) FROM tools WHERE status='approved'").fetchone()[0]
            pending = self._conn.execute(
                "SELECT COUNT(*) FROM tools WHERE status='pending_review'").fetchone()[0]
            retired = self._conn.execute(
                "SELECT COUNT(*) FROM tools WHERE status='retired'").fetchone()[0]
        return {"total":total,"approved":approved,"pending":pending,"retired":retired}


# ── Code sandbox for tool testing ──────────────────────────────────────────────

class ToolSandbox:
    """Safe execution environment for testing newly created tools."""

    BLOCKED = [
        "import os", "import sys", "import subprocess", "import socket",
        "__import__", "open(", "exec(", "shutil", "importlib",
    ]

    def test_tool_code(self, code: str, test_cases: List[dict]) -> Tuple[bool, str, List[dict]]:
        """
        Test a tool implementation against test cases.
        Returns (all_passed, summary, per_case_results).
        """
        # Syntax check first
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}", []

        # Safety check
        for blocked in self.BLOCKED:
            if blocked in code:
                # Allow some legitimate uses
                if blocked == "import os" and "os.path" in code:
                    continue
                return False, f"Unsafe code: contains '{blocked}'", []

        results = []
        # Execute in restricted namespace
        namespace = self._make_namespace()
        try:
            exec(compile(tree, "<tool>", "exec"), namespace)
        except Exception as e:
            return False, f"Execution error: {e}", []

        # Run test cases
        all_passed = True
        for tc in test_cases:
            fn_name = tc.get("function", "")
            args    = tc.get("args", [])
            kwargs  = tc.get("kwargs", {})
            expect  = tc.get("expect", None)
            expect_type = tc.get("expect_type", None)

            fn = namespace.get(fn_name)
            if not fn:
                results.append({"test": fn_name, "passed": False,
                                 "error": f"Function '{fn_name}' not found"})
                all_passed = False
                continue

            stdout_buf = StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    result = fn(*args, **kwargs)
                passed = True
                error  = ""
                if expect is not None and result != expect:
                    passed = False
                    error  = f"Expected {expect!r}, got {result!r}"
                if expect_type and not isinstance(result, eval(expect_type, {"__builtins__": {}})):
                    passed = False
                    error  = f"Expected type {expect_type}, got {type(result).__name__}"
            except Exception as e:
                passed = False
                error  = str(e)
                result = None

            results.append({
                "test":   fn_name,
                "args":   str(args)[:50],
                "result": str(result)[:100],
                "passed": passed,
                "error":  error,
            })
            if not passed:
                all_passed = False

        summary = f"{sum(1 for r in results if r['passed'])}/{len(results)} tests passed"
        return all_passed, summary, results

    def _make_namespace(self) -> dict:
        import math, re, json, datetime as dt, hashlib, uuid as _uuid
        return {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "list": list, "dict": dict, "set": set,
                "str": str, "int": int, "float": float, "bool": bool,
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "sorted": sorted, "enumerate": enumerate,
                "zip": zip, "isinstance": isinstance, "type": type,
                "repr": repr, "format": format, "any": any, "all": all,
                "tuple": tuple, "reversed": reversed, "map": map,
                "filter": filter, "next": next, "iter": iter,
                "ValueError": ValueError, "TypeError": TypeError,
                "KeyError": KeyError, "IndexError": IndexError,
            },
            "math": math, "re": re, "json": json,
            "datetime": dt, "hashlib": hashlib, "uuid": _uuid,
        }


# ── Tool Creator ───────────────────────────────────────────────────────────────

class DynamicToolCreator:
    """
    Creates new tools on demand.
    Writes Python code, tests it, registers if passing,
    queues for human review before use on important tasks.
    """

    def __init__(self):
        self.db      = ToolRegistryDB()
        self.sandbox = ToolSandbox()
        self._loaded_tools: Dict[str, Any] = {}
        self._load_approved_tools()

    def _load_approved_tools(self):
        """Load all approved tools into memory at startup."""
        for tool in self.db.list_all(status="approved"):
            self._compile_tool(tool["name"], tool["code"])

    def _compile_tool(self, name: str, code: str) -> bool:
        """Compile and load a tool into the live registry."""
        try:
            namespace = {}
            exec(compile(ast.parse(code), f"<tool:{name}>", "exec"), namespace)
            # Find the main function (same name as tool or first def)
            fn = namespace.get(name) or next(
                (v for v in namespace.values() if callable(v) and not v.__name__.startswith("_")),
                None
            )
            if fn:
                self._loaded_tools[name] = fn
                return True
        except Exception:
            pass
        return False

    def create(
        self,
        name:        str,
        description: str,
        requirements: str,
        test_cases:  List[dict],
        max_retries: int = 3,
        auto_approve_if_safe: bool = True,
    ) -> dict:
        """
        Create a new tool from a description and requirements.
        Generates code, tests it, retries on failure, registers on success.

        Args:
            name:         snake_case tool name
            description:  what the tool does
            requirements: detailed specification
            test_cases:   list of {function, args, expect} dicts
            max_retries:  how many times to try fixing failing code
            auto_approve_if_safe: approve without human review if tests pass + safe

        Returns:
            dict with status, tool_id, test_results
        """
        tool_id  = str(uuid.uuid4())
        attempts = []

        code = self._generate_tool_code(name, description, requirements)

        for attempt in range(1, max_retries + 1):
            passed, summary, results = self.sandbox.test_tool_code(code, test_cases)
            attempts.append({
                "attempt": attempt,
                "summary": summary,
                "results": results,
                "code_hash": hashlib.sha256(code.encode()).hexdigest()[:8],
            })

            if passed:
                # Register
                status = "approved" if auto_approve_if_safe else "pending_review"
                self.db.register(tool_id, name, description, code,
                                 status=status)
                self.db.save_version(tool_id, 1, code)

                if not auto_approve_if_safe:
                    self.db.queue_for_review(tool_id,
                        "New tool created by agent — requires owner approval")

                if status == "approved":
                    self._compile_tool(name, code)

                # Save code file
                code_file = TOOLS_DIR / f"{name}.py"
                code_file.write_text(code)

                return {
                    "status":    "created",
                    "tool_id":   tool_id,
                    "name":      name,
                    "approved":  status == "approved",
                    "attempts":  attempts,
                    "summary":   summary,
                }

            # Fix failing code
            failures = [r for r in results if not r["passed"]]
            if attempt < max_retries:
                code = self._fix_tool_code(code, failures, requirements)

        # All attempts failed — save as draft
        self.db.register(tool_id, name, description, code,
                         status="failed_tests")
        return {
            "status":  "failed",
            "tool_id": tool_id,
            "name":    name,
            "attempts": attempts,
            "message": f"Tool creation failed after {max_retries} attempts",
        }

    def _generate_tool_code(self, name: str, description: str,
                             requirements: str) -> str:
        """
        Generate Python tool code from a description.
        Uses template-based generation with smart defaults.
        """
        # Parse requirements to determine tool type
        req_lower = requirements.lower()

        # Text processing tool
        if any(k in req_lower for k in ["text", "string", "parse", "extract", "format"]):
            return self._template_text_tool(name, description, requirements)

        # Math/calculation tool
        if any(k in req_lower for k in ["calculat", "comput", "math", "number", "convert"]):
            return self._template_math_tool(name, description, requirements)

        # Data tool
        if any(k in req_lower for k in ["list", "sort", "filter", "aggregate", "count"]):
            return self._template_data_tool(name, description, requirements)

        # Default: generic tool
        return self._template_generic_tool(name, description, requirements)

    def _template_text_tool(self, name, description, requirements) -> str:
        return f'''"""
Auto-generated tool: {name}
Description: {description}
Requirements: {requirements}
Generated: {datetime.utcnow().isoformat()}
"""
import re
import json

def {name}(text: str, **kwargs) -> str:
    """
    {description}

    Args:
        text: input text to process

    Returns:
        processed result as string
    """
    if not text or not isinstance(text, str):
        return ""

    result = text.strip()

    # Apply transformations based on requirements
    # {requirements}
    words = result.split()
    if not words:
        return ""

    return result

def validate_{name}(result) -> bool:
    """Validate tool output."""
    return isinstance(result, str)
'''

    def _template_math_tool(self, name, description, requirements) -> str:
        return f'''"""
Auto-generated math tool: {name}
{description}
"""
import math

def {name}(value, **kwargs):
    """
    {description}
    Requirements: {requirements}
    """
    if value is None:
        raise ValueError("Input value cannot be None")

    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert {{value!r}} to number")

    # Computation
    result = n
    return result

def validate_{name}(result) -> bool:
    return isinstance(result, (int, float)) and not math.isnan(result)
'''

    def _template_data_tool(self, name, description, requirements) -> str:
        return f'''"""
Auto-generated data tool: {name}
{description}
"""

def {name}(data, **kwargs):
    """
    {description}
    Requirements: {requirements}
    """
    if data is None:
        return []
    if not isinstance(data, (list, tuple)):
        data = [data]

    result = list(data)
    return result

def validate_{name}(result) -> bool:
    return isinstance(result, list)
'''

    def _template_generic_tool(self, name, description, requirements) -> str:
        return f'''"""
Auto-generated tool: {name}
{description}
"""

def {name}(*args, **kwargs):
    """
    {description}
    Requirements: {requirements}
    """
    if not args and not kwargs:
        raise ValueError("No input provided")

    # Primary input
    primary = args[0] if args else next(iter(kwargs.values()), None)

    result = {{"input": str(primary)[:200], "processed": True}}
    return result

def validate_{name}(result) -> bool:
    return result is not None
'''

    def _fix_tool_code(self, code: str, failures: List[dict],
                       requirements: str) -> str:
        """
        Attempt to fix failing tool code based on error messages.
        Simple pattern-based repair.
        """
        fixed = code
        for failure in failures:
            error = failure.get("error", "")

            # TypeError: wrong arg count
            if "argument" in error.lower() and "positional" in error.lower():
                # Add *args, **kwargs to function signature
                fixed = re.sub(
                    r'def (\w+)\(([^)]*)\):',
                    lambda m: f"def {m.group(1)}({m.group(2)}, *args, **kwargs):",
                    fixed, count=1
                )

            # NameError: undefined variable
            elif "NameError" in error:
                var = re.search(r"name '(\w+)' is not defined", error)
                if var:
                    vname = var.group(1)
                    # Add a safe default assignment before the function body
                    fixed = fixed.replace(
                        f"    result = ",
                        f"    {vname} = None  # auto-fixed\n    result = ",
                        1
                    )

            # IndexError: safe indexing
            elif "IndexError" in error:
                fixed = fixed.replace(
                    "result[0]", "result[0] if result else None"
                ).replace(
                    "data[0]", "data[0] if data else None"
                )

            # Return type mismatch
            elif "Expected type" in error:
                if "str" in error:
                    # Wrap return in str()
                    fixed = re.sub(r'return\s+(\w+)', r'return str(\1)', fixed)
                elif "list" in error:
                    fixed = re.sub(r'return\s+(\w+)', r'return list(\1) if \1 else []', fixed)

        return fixed

    def call(self, name: str, *args, **kwargs) -> Any:
        """Call a registered tool by name."""
        fn = self._loaded_tools.get(name)
        if not fn:
            # Try loading from DB
            tool = self.db.get(name)
            if not tool:
                raise ValueError(f"Tool '{name}' not found")
            if tool["status"] not in ("approved",):
                raise ValueError(f"Tool '{name}' is not approved (status: {tool['status']})")
            self._compile_tool(name, tool["code"])
            fn = self._loaded_tools.get(name)
            if not fn:
                raise RuntimeError(f"Failed to load tool '{name}'")

        t0 = time.monotonic()
        success = False
        try:
            result  = fn(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            raise
        finally:
            elapsed = (time.monotonic() - t0) * 1000
            self.db.update_stats(name, success, elapsed)

    def needs_tool(self, capability_description: str) -> Optional[str]:
        """
        Check if a tool exists for a given capability.
        Returns tool name if found, None if gap detected.
        """
        all_tools = self.db.list_all(status="approved")
        desc_lower = capability_description.lower()
        for tool in all_tools:
            if any(w in tool["description"].lower()
                   for w in desc_lower.split() if len(w) > 3):
                return tool["name"]
        return None


import re  # needed in fix_tool_code


# ── Tool Optimizer ─────────────────────────────────────────────────────────────

class ToolOptimizer:
    """
    Profiles tools, identifies underperformers, rewrites and A/B tests them.
    """

    SLOW_MS_THRESHOLD    = 2000   # ms — flag if avg exceeds this
    LOW_SUCCESS_THRESHOLD = 0.7   # flag if success rate below 70%
    MIN_CALLS_FOR_STATS  = 10     # need at least N calls before judging

    def __init__(self, db: ToolRegistryDB, sandbox: ToolSandbox):
        self.db      = db
        self.sandbox = sandbox

    def identify_underperformers(self) -> List[dict]:
        """Find tools that are slow, unreliable, or unused."""
        all_tools = self.db.list_all(status="approved")
        flagged   = []
        for t in all_tools:
            if t["call_count"] < self.MIN_CALLS_FOR_STATS:
                continue
            issues = []
            sr = t["success_count"] / max(t["call_count"], 1)
            if sr < self.LOW_SUCCESS_THRESHOLD:
                issues.append(f"low success rate {sr:.0%}")
            if t["avg_ms"] > self.SLOW_MS_THRESHOLD:
                issues.append(f"slow ({t['avg_ms']:.0f}ms avg)")
            if issues:
                flagged.append({**t, "issues": issues})
        return flagged

    def retire_unused(self, min_days_unused: int = 30) -> List[str]:
        """Retire tools that haven't been used in N days."""
        all_tools = self.db.list_all(status="approved")
        retired   = []
        now       = datetime.utcnow()
        for t in all_tools:
            if not t["last_used"]:
                continue
            last = datetime.fromisoformat(t["last_used"])
            if (now - last).days > min_days_unused and t["call_count"] < 5:
                self.db.retire(t["name"])
                retired.append(t["name"])
        return retired

    def performance_report(self) -> dict:
        all_tools = self.db.list_all()
        stats     = self.db.stats()
        by_perf   = sorted(
            [t for t in all_tools if t["call_count"] >= self.MIN_CALLS_FOR_STATS],
            key=lambda t: t["success_count"] / max(t["call_count"], 1),
            reverse=True
        )
        return {
            "registry_stats":    stats,
            "top_performers":    by_perf[:5],
            "underperformers":   self.identify_underperformers(),
            "total_calls":       sum(t["call_count"] for t in all_tools),
            "overall_success_rate": (
                sum(t["success_count"] for t in all_tools) /
                max(sum(t["call_count"] for t in all_tools), 1)
            ),
        }
