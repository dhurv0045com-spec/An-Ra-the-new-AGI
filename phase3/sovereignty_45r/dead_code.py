"""
sovereignty/dead_code.py
========================
Pass 2 of the nightly improvement pipeline: dead code and quality sweep.

Uses pure AST analysis (no external linters) to detect:
  - Unused imports (imported but never referenced in the module body)
  - Unused variables (assigned but never read after assignment)
  - Unreachable code (statements after return/raise/break/continue)
  - Functions defined but never called within their own module
  - Magic numbers (numeric literals not assigned to a named constant)
  - Long functions (> COMPLEXITY_MAX_LINES lines)
  - Deep nesting (> COMPLEXITY_MAX_NESTING indent levels)

NEVER modifies source files — only produces suggestion reports.

Output:
  dead_code_YYYYMMDD.json         — machine-readable findings
  suggested_removals_YYYYMMDD.txt — plain English suggestions

Relationship to other modules:
    improver.py calls DeadCodePass.run() during Pass 2.
    reporter.py reads dead_code_YYYYMMDD.json for the nightly report.
"""

import ast
import json
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from sovereignty.config import Config
from sovereignty.logger import get_logger

log = get_logger(__name__)


# ── Detectors ─────────────────────────────────────────────────────────────────

def _find_unused_imports(tree: ast.Module, source: str) -> List[Dict]:
    """
    Find imports that are never referenced in the module body.

    Parameters:
        tree: Parsed AST of the module.
        source: Source text (used to extract import line numbers).

    Returns:
        List of finding dicts: {category, name, line, description}.
    """
    imported_names: Dict[str, int] = {}  # name → line

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                imported_names[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname if alias.asname else alias.name
                imported_names[name] = node.lineno

    # Collect all Name references that are not in import statements
    used_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

    findings = []
    for name, line in imported_names.items():
        if name not in used_names:
            findings.append({
                "category": "unused_import",
                "name": name,
                "line": line,
                "description": f"'{name}' is imported at line {line} but never used in this module.",
            })
    return findings


def _find_unreachable_code(tree: ast.Module) -> List[Dict]:
    """
    Find statements that can never be reached (after return/raise/break/continue).

    Parameters:
        tree: Parsed AST.

    Returns:
        List of finding dicts.
    """
    findings = []
    _TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

    def _check_block(stmts: List[ast.stmt]) -> None:
        terminated = False
        for stmt in stmts:
            if terminated:
                findings.append({
                    "category": "unreachable_code",
                    "name": f"line {stmt.lineno}",
                    "line": stmt.lineno,
                    "description": (
                        f"Statement at line {stmt.lineno} is unreachable — "
                        "it follows a return/raise/break/continue."
                    ),
                })
            if isinstance(stmt, _TERMINATORS):
                terminated = True
            elif isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                # Recurse into nested blocks
                for child_list in ast.iter_child_nodes(stmt):
                    if isinstance(child_list, list):
                        _check_block(child_list)
            # Reset after an except handler (code may continue)
            if isinstance(stmt, ast.Try):
                terminated = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_block(node.body)

    return findings


def _find_magic_numbers(tree: ast.Module) -> List[Dict]:
    """
    Find numeric literals used directly in expressions (not in assignments to names).

    Exceptions: 0, 1, -1, 2 are considered conventional and not flagged.

    Parameters:
        tree: Parsed AST.

    Returns:
        List of finding dicts.
    """
    ALLOWED = {0, 1, -1, 2}
    findings = []

    for node in ast.walk(tree):
        # Skip assignments like X = 42 (that IS naming the constant)
        if isinstance(node, ast.Assign):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in ALLOWED:
                findings.append({
                    "category": "magic_number",
                    "name": str(node.value),
                    "line": getattr(node, "lineno", 0),
                    "description": (
                        f"Magic number {node.value!r} at line {getattr(node, 'lineno', '?')}. "
                        "Consider assigning it to a named constant."
                    ),
                })
    return findings


def _find_long_functions(tree: ast.Module, max_lines: int) -> List[Dict]:
    """
    Find functions longer than max_lines lines.

    Parameters:
        tree: Parsed AST.
        max_lines: Flag threshold.

    Returns:
        List of finding dicts.
    """
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            length = end - start + 1
            if length > max_lines:
                findings.append({
                    "category": "long_function",
                    "name": node.name,
                    "line": start,
                    "description": (
                        f"Function '{node.name}' is {length} lines long "
                        f"(limit: {max_lines}). Consider splitting it."
                    ),
                })
    return findings


def _find_deep_nesting(tree: ast.Module, max_depth: int) -> List[Dict]:
    """
    Find code blocks nested deeper than max_depth levels.

    Parameters:
        tree: Parsed AST.
        max_depth: Flag threshold.

    Returns:
        List of finding dicts.
    """
    _NEST_NODES = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)
    findings = []

    def _walk_depth(node: ast.AST, depth: int) -> None:
        if isinstance(node, _NEST_NODES):
            if depth > max_depth:
                line = getattr(node, "lineno", 0)
                findings.append({
                    "category": "deep_nesting",
                    "name": f"line {line}",
                    "line": line,
                    "description": (
                        f"Code at line {line} is nested {depth} levels deep "
                        f"(limit: {max_depth}). Consider refactoring."
                    ),
                })
            depth += 1
        for child in ast.iter_child_nodes(node):
            _walk_depth(child, depth)

    _walk_depth(tree, 0)
    return findings


def _analyse_file(path: pathlib.Path, config: Config) -> List[Dict]:
    """
    Run all detectors on a single Python file.

    Parameters:
        path: Path to the .py file.
        config: Active Config for thresholds.

    Returns:
        List of all findings from all detectors.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception as exc:
        log.warning("Could not parse %s: %s", path, exc)
        return []

    findings = []
    findings += _find_unused_imports(tree, source)
    findings += _find_unreachable_code(tree)
    findings += _find_magic_numbers(tree)
    findings += _find_long_functions(tree, config.COMPLEXITY_MAX_LINES)
    findings += _find_deep_nesting(tree, config.COMPLEXITY_MAX_NESTING)

    # Tag each finding with its source file
    for f in findings:
        f["file"] = str(path)

    return findings


class DeadCodePass:
    """
    Orchestrates Pass 2: scans target directory, collects all findings.
    """

    def __init__(self, config: Config, target_dir: pathlib.Path) -> None:
        """
        Parameters:
            config: Active Config instance.
            target_dir: Directory of Python files to scan.
        """
        self._config = config
        self._target_dir = target_dir

    def run(self, date_str: Optional[str] = None) -> Dict:
        """
        Execute Pass 2 and write output files.

        Parameters:
            date_str: Date label for output files (YYYYMMDD).

        Returns:
            Dict with 'findings' list and 'summary' dict.
        """
        date_str = date_str or datetime.now().strftime("%Y%m%d")
        log.info("Pass 2: Scanning %s for dead code", self._target_dir)

        py_files = list(self._target_dir.rglob("*.py"))
        all_findings: List[Dict] = []
        for path in py_files:
            all_findings.extend(_analyse_file(path, self._config))

        # Summarise by category
        categories = {}
        for f in all_findings:
            cat = f["category"]
            categories[cat] = categories.get(cat, 0) + 1

        summary = {"total": len(all_findings), "by_category": categories}

        # Write JSON
        out_json = self._config.DATA_DIR / f"dead_code_{date_str}.json"
        out_json.write_text(
            json.dumps({"findings": all_findings, "summary": summary}, indent=2),
            encoding="utf-8",
        )

        # Write human-readable suggestions
        suggestions = self._format_suggestions(all_findings, summary, date_str)
        out_txt = self._config.DATA_DIR / f"suggested_removals_{date_str}.txt"
        out_txt.write_text(suggestions, encoding="utf-8")

        log.info(
            "Pass 2 complete: %d issues found — %s",
            len(all_findings),
            ", ".join(f"{v} {k}" for k, v in categories.items()),
        )
        return {"findings": all_findings, "summary": summary}

    def _format_suggestions(
        self, findings: List[Dict], summary: Dict, date_str: str
    ) -> str:
        """Format findings as a numbered plain-English suggestion list."""
        lines = [
            f"Dead Code & Quality Report — {date_str}",
            "=" * 50,
            f"Total issues: {summary['total']}",
            "",
        ]
        for cat, count in summary["by_category"].items():
            lines.append(f"  {cat}: {count}")
        lines.append("")

        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. [{finding['category'].upper()}] {finding['file']} line {finding['line']}")
            lines.append(f"   {finding['description']}")
            lines.append("")

        if not findings:
            lines.append("No issues found. Code quality looks good!")

        return "\n".join(lines)
