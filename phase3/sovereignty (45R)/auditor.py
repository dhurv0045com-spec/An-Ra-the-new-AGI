"""
sovereignty/auditor.py
======================
Pass 1 of the nightly improvement pipeline: deep code audit via AST.

For every .py file in a target directory, computes per-function metrics:
  - Cyclomatic complexity (decision node count)
  - Cognitive complexity (nesting-weighted decision count)
  - Lines of code and comment ratio
  - Number of arguments and return statements
  - Docstring presence

Aggregates metrics project-wide, compares to the previous night's baseline
stored in audit_baseline.json, and produces:
  audit_YYYYMMDD.json      — full per-function detail
  audit_summary_YYYYMMDD.txt — human-readable with delta arrows

Relationship to other modules:
    improver.py calls AuditPass.run() during Pass 1.
    reporter.py reads audit_YYYYMMDD.json for the nightly report.
"""

import ast
import json
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)

# AST node types that count as decision branches for cyclomatic complexity
_CYCLOMATIC_NODES = (
    ast.If, ast.For, ast.While, ast.ExceptHandler,
    ast.With, ast.Assert, ast.comprehension,
)

# AST node types that also increment cognitive complexity (structurally complex)
_COGNITIVE_NODES = (
    ast.If, ast.For, ast.While, ast.ExceptHandler,
    ast.With, ast.comprehension,
)


def _cyclomatic_complexity(func_node: ast.FunctionDef) -> int:
    """
    Compute cyclomatic complexity of a function.

    Cyclomatic complexity = 1 + number of decision nodes (if/for/while/
    except/with/assert/comprehension).

    Parameters:
        func_node: AST FunctionDef or AsyncFunctionDef node.

    Returns:
        Integer complexity score (minimum 1).
    """
    count = 1
    for node in ast.walk(func_node):
        if isinstance(node, _CYCLOMATIC_NODES):
            count += 1
        elif isinstance(node, ast.BoolOp):
            # Each extra operand in and/or is a branch
            count += len(node.values) - 1
    return count


def _cognitive_complexity(func_node: ast.FunctionDef) -> int:
    """
    Compute cognitive complexity of a function (nesting-weighted).

    Each control structure increments by (1 + current_nesting_level).
    Boolean sequences add 1 flat each. Recursion adds 1.

    Parameters:
        func_node: AST FunctionDef or AsyncFunctionDef node.

    Returns:
        Integer cognitive score (minimum 0).
    """

    def _walk(node: ast.AST, depth: int) -> int:
        score = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _COGNITIVE_NODES):
                score += 1 + depth
                score += _walk(child, depth + 1)
            elif isinstance(child, ast.BoolOp):
                score += 1
                score += _walk(child, depth)
            else:
                score += _walk(child, depth)
        return score

    return _walk(func_node, 0)


def _count_lines(func_node: ast.FunctionDef, source_lines: List[str]) -> Tuple[int, int]:
    """
    Count total lines and comment lines in a function body.

    Parameters:
        func_node: The AST function node (must have lineno / end_lineno).
        source_lines: All lines of the source file (0-indexed).

    Returns:
        Tuple of (total_lines, comment_lines).
    """
    start = func_node.lineno - 1
    end = getattr(func_node, "end_lineno", func_node.lineno)
    body_lines = source_lines[start:end]
    total = len(body_lines)
    comments = sum(1 for ln in body_lines if ln.strip().startswith("#"))
    return total, comments


def _has_docstring(func_node: ast.FunctionDef) -> bool:
    """Return True if the function has a docstring as its first statement."""
    body = func_node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return isinstance(body[0].value.value, str)
    return False


def _count_returns(func_node: ast.FunctionDef) -> int:
    """Count the number of return statements in a function."""
    return sum(1 for n in ast.walk(func_node) if isinstance(n, ast.Return))


def _analyse_file(path: pathlib.Path) -> List[Dict]:
    """
    Parse a Python file and return per-function metrics.

    Parameters:
        path: Path to the .py file.

    Returns:
        List of dicts, one per function/method found in the file.
        Each dict contains: file, function, line, cyclomatic, cognitive,
        lines, comment_lines, comment_ratio, args, returns, has_docstring.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        source_lines = source.splitlines()
    except SyntaxError as exc:
        log.warning("Syntax error in %s: %s", path, exc)
        return []
    except Exception as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        total_lines, comment_lines = _count_lines(node, source_lines)
        args = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
        if node.args.vararg:
            args += 1
        if node.args.kwarg:
            args += 1

        results.append({
            "file": str(path),
            "function": node.name,
            "line": node.lineno,
            "cyclomatic": _cyclomatic_complexity(node),
            "cognitive": _cognitive_complexity(node),
            "lines": total_lines,
            "comment_lines": comment_lines,
            "comment_ratio": round(comment_lines / max(total_lines, 1), 3),
            "args": args,
            "returns": _count_returns(node),
            "has_docstring": _has_docstring(node),
        })
    return results


def _aggregate(functions: List[Dict]) -> Dict:
    """
    Compute project-level aggregate metrics from per-function data.

    Parameters:
        functions: List of per-function metric dicts.

    Returns:
        Aggregate dict with totals, averages, and extremes.
    """
    if not functions:
        return {
            "total_functions": 0,
            "avg_cyclomatic": 0.0,
            "max_cyclomatic": 0,
            "max_cyclomatic_fn": "",
            "avg_cognitive": 0.0,
            "max_cognitive": 0,
            "max_cognitive_fn": "",
            "pct_with_docstring": 0.0,
            "avg_lines": 0.0,
            "flagged_complexity": [],
            "flagged_no_docstring": [],
        }

    n = len(functions)
    avg_cyc = sum(f["cyclomatic"] for f in functions) / n
    max_cyc_fn = max(functions, key=lambda f: f["cyclomatic"])
    avg_cog = sum(f["cognitive"] for f in functions) / n
    max_cog_fn = max(functions, key=lambda f: f["cognitive"])
    pct_doc = sum(1 for f in functions if f["has_docstring"]) / n * 100

    return {
        "total_functions": n,
        "avg_cyclomatic": round(avg_cyc, 2),
        "max_cyclomatic": max_cyc_fn["cyclomatic"],
        "max_cyclomatic_fn": f"{max_cyc_fn['function']} ({max_cyc_fn['file']}:{max_cyc_fn['line']})",
        "avg_cognitive": round(avg_cog, 2),
        "max_cognitive": max_cog_fn["cognitive"],
        "max_cognitive_fn": f"{max_cog_fn['function']} ({max_cog_fn['file']}:{max_cog_fn['line']})",
        "pct_with_docstring": round(pct_doc, 1),
        "avg_lines": round(sum(f["lines"] for f in functions) / n, 1),
        "flagged_complexity": [
            f["function"]
            for f in functions
            if f["cyclomatic"] > 10 or f["cognitive"] > 15
        ],
        "flagged_no_docstring": [
            f["function"] for f in functions if not f["has_docstring"]
        ],
    }


class AuditPass:
    """
    Orchestrates Pass 1: scans all .py files, computes metrics, writes output.
    """

    def __init__(self, config: Config, target_dir: pathlib.Path) -> None:
        """
        Parameters:
            config: Active Config instance (for DATA_DIR, thresholds).
            target_dir: Directory of Python files to audit.
        """
        self._config = config
        self._target_dir = target_dir

    def run(self, date_str: Optional[str] = None) -> Dict:
        """
        Execute Pass 1 and write output files.

        Parameters:
            date_str: Date label for output files (YYYYMMDD). Defaults to today.

        Returns:
            Dict with 'aggregate' metrics and 'deltas' vs baseline.
        """
        date_str = date_str or datetime.now().strftime("%Y%m%d")
        log.info("Pass 1: Scanning %s for .py files", self._target_dir)

        py_files = list(self._target_dir.rglob("*.py"))
        log.info("Pass 1: Found %d Python files", len(py_files))

        all_functions: List[Dict] = []
        for path in py_files:
            all_functions.extend(_analyse_file(path))

        aggregate = _aggregate(all_functions)
        baseline = self._load_baseline()
        deltas = self._compute_deltas(aggregate, baseline)

        # Write detailed JSON
        out_json = self._config.DATA_DIR / f"audit_{date_str}.json"
        out_json.write_text(
            json.dumps({"functions": all_functions, "aggregate": aggregate, "deltas": deltas}, indent=2),
            encoding="utf-8",
        )

        # Write human-readable summary
        summary = self._format_summary(aggregate, deltas, date_str)
        out_txt = self._config.DATA_DIR / f"audit_summary_{date_str}.txt"
        out_txt.write_text(summary, encoding="utf-8")

        # Save new baseline
        self._save_baseline(aggregate)

        self._log_results(aggregate, deltas)
        return {"aggregate": aggregate, "deltas": deltas}

    def _load_baseline(self) -> Dict:
        """Load the previous night's aggregate metrics, or empty dict."""
        path = self._config.AUDIT_BASELINE_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_baseline(self, aggregate: Dict) -> None:
        """Write the current aggregate as the new baseline."""
        self._config.AUDIT_BASELINE_FILE.write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )

    def _compute_deltas(self, current: Dict, baseline: Dict) -> Dict:
        """
        Compute numeric deltas between current and baseline metrics.

        Returns:
            Dict mapping metric keys to delta values and direction labels.
        """
        numeric_keys = [
            "total_functions", "avg_cyclomatic", "max_cyclomatic",
            "avg_cognitive", "max_cognitive", "pct_with_docstring", "avg_lines",
        ]
        deltas = {}
        for key in numeric_keys:
            cur = current.get(key, 0)
            base = baseline.get(key, 0)
            diff = cur - base if baseline else 0
            deltas[key] = {
                "current": cur,
                "baseline": base,
                "delta": round(diff, 3),
                "direction": (
                    "→" if diff == 0 or not baseline
                    else ("↑" if diff > 0 else "↓")
                ),
            }
        return deltas

    def _format_summary(self, aggregate: Dict, deltas: Dict, date_str: str) -> str:
        """Format a human-readable summary with delta arrows."""
        lines = [
            f"Code Audit Summary — {date_str}",
            "=" * 50,
            "",
            f"{'Metric':<30} {'Value':>10} {'Delta':>10} {'Dir':>4}",
            "-" * 56,
        ]
        for key, info in deltas.items():
            lines.append(
                f"{key:<30} {str(info['current']):>10} {str(info['delta']):>10} {info['direction']:>4}"
            )
        lines += [
            "",
            "Flagged (high complexity):",
        ]
        for fn in aggregate.get("flagged_complexity", []):
            lines.append(f"  ⚠ {fn}")
        lines += ["", "Flagged (no docstring):"]
        for fn in aggregate.get("flagged_no_docstring", [])[:10]:
            lines.append(f"  - {fn}")
        return "\n".join(lines)

    def _log_results(self, aggregate: Dict, deltas: Dict) -> None:
        """Emit summary log lines for Pass 1."""
        log.info(
            "Pass 1 complete: %d functions, avg cyclomatic=%.2f, avg cognitive=%.2f, "
            "%.1f%% with docstrings",
            aggregate["total_functions"],
            aggregate["avg_cyclomatic"],
            aggregate["avg_cognitive"],
            aggregate["pct_with_docstring"],
        )
        for key, info in deltas.items():
            direction = {"↑": "improved", "↓": "regressed", "→": "unchanged"}.get(
                info["direction"], "unknown"
            )
            log.info("  %s: %s (delta %+.3f)", key, direction, info["delta"])
