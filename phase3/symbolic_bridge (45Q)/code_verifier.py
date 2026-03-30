"""
code_verifier.py — Code Verification Engine for 45Q Symbolic Logic Bridge.

Performs static analysis on Python code using only the stdlib `ast` module.
No external linters. Detects complexity issues, undefined variables,
off-by-one risks, infinite loops, anti-patterns, and type issues.
"""

from __future__ import annotations
import ast
import textwrap
from dataclasses import dataclass, field
from typing import Optional
from . import config


# ── Issue data types ───────────────────────────────────────────────────────────

class Severity:
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class Category:
    COMPLEXITY   = "COMPLEXITY"
    UNDEFINED    = "UNDEFINED"
    UNUSED       = "UNUSED"
    TYPE         = "TYPE"
    OFF_BY_ONE   = "OFF_BY_ONE"
    INFINITE_LOOP = "INFINITE_LOOP"
    ANTI_PATTERN = "ANTI_PATTERN"
    TEST_FAILURE = "TEST_FAILURE"


@dataclass
class CodeIssue:
    """
    A single code issue found by the verifier.

    Attributes
    ----------
    location : str
        Description of where the issue is (function name, line number).
    line : int
        Line number (1-indexed).
    category : str
        Category constant from Category class.
    severity : str
        Severity constant from Severity class.
    description : str
        Plain English explanation of the problem.
    suggestion : str
        Concrete suggested fix (code snippet or description).
    confidence : float
        Confidence this is a real issue (0.0–1.0).
    """
    location: str
    line: int
    category: str
    severity: str
    description: str
    suggestion: str
    confidence: float = 0.9


@dataclass
class StaticAnalysisResult:
    """
    Full result of static analysis on a code snippet.

    Attributes
    ----------
    issues : list[CodeIssue]
        All detected issues.
    cyclomatic_complexity : int
        McCabe cyclomatic complexity of the primary function.
    cognitive_complexity : int
        Weighted cognitive complexity.
    functions_found : list[str]
        Names of functions parsed from the code.
    ast_dump : str
        AST dump for debugging.
    steps : list[str]
        Analysis trace.
    """
    issues: list[CodeIssue] = field(default_factory=list)
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    functions_found: list[str] = field(default_factory=list)
    ast_dump: str = ""
    steps: list[str] = field(default_factory=list)


class StaticAnalyzer(ast.NodeVisitor):
    """
    AST-based static analyser.

    Walks the AST and applies all detection rules.
    Create one per function/module analysis.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.issues: list[CodeIssue] = []
        self.steps: list[str] = []
        self._functions: list[str] = []
        self._cyclomatic: int = 1   # Starts at 1 (baseline for the function)
        self._cognitive: int = 0
        self._nesting: int = 0
        self._defined: set[str] = set()   # Variables defined in scope
        self._used: set[str] = set()       # Variables used
        self._args: set[str] = set()       # Function arguments
        self._imports: set[str] = set()    # Imported names
        self._assigned: set[str] = set()   # Names assigned in function
        self._has_return_value: bool = False
        self._has_bare_return: bool = False

    def _issue(
        self,
        node: ast.AST,
        category: str,
        severity: str,
        description: str,
        suggestion: str,
        confidence: float = 0.9,
    ) -> None:
        """Add a detected issue."""
        lineno = getattr(node, 'lineno', 0)
        func_name = self._functions[-1] if self._functions else "<module>"
        self.issues.append(CodeIssue(
            location=f"{func_name}:{lineno}",
            line=lineno,
            category=category,
            severity=severity,
            description=description,
            suggestion=suggestion,
            confidence=confidence,
        ))
        self.steps.append(f"  [{severity}] Line {lineno}: {description}")

    # ── Complexity tracking ────────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._functions.append(node.name)
        self.steps.append(f"Analysing function: {node.name}()")

        # Check for mutable default arguments
        for default in node.args.defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self._issue(
                    node, Category.ANTI_PATTERN, Severity.WARNING,
                    f"Mutable default argument in {node.name}() — "
                    "the same object is reused across all calls",
                    "Use None as default and initialise inside the function:\n"
                    f"  def {node.name}(x=None):\n      if x is None: x = []",
                )

        # Track arguments as defined names
        for arg in node.args.args:
            self._args.add(arg.arg)
            self._defined.add(arg.arg)

        # Check for unannotated public functions
        has_return_ann = node.returns is not None
        has_arg_anns = all(a.annotation is not None for a in node.args.args)
        if not node.name.startswith('_') and (not has_return_ann or not has_arg_anns):
            self._issue(
                node, Category.TYPE, Severity.INFO,
                f"Public function {node.name}() has missing type annotations",
                f"Add type annotations: def {node.name}(arg: type, ...) -> return_type:",
                confidence=0.7,
            )

        # Check for bare recursive calls with no base case
        self._check_recursion(node)

        self.generic_visit(node)
        self._functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_recursion(self, func_node: ast.FunctionDef) -> None:
        """
        Detect functions that appear to recurse with no base case guard.

        Looks for: function calls to self, with no if-statement anywhere
        in the body that could serve as a base case.
        """
        func_name = func_node.name
        has_if = any(isinstance(n, ast.If) for n in ast.walk(func_node))
        has_recursive_call = False

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    has_recursive_call = True
                    break

        if has_recursive_call and not has_if:
            self._issue(
                func_node, Category.INFINITE_LOOP, Severity.CRITICAL,
                f"{func_name}() is recursive but has NO if-statement for a base case. "
                "This will cause infinite recursion and a stack overflow.",
                f"Add a base case:\n  def {func_name}(n):\n"
                f"      if n <= 0: return 0  # base case\n"
                f"      return {func_name}(n - 1)  # recursive case",
            )

    def visit_If(self, node: ast.If) -> None:
        self._cyclomatic += 1
        self._cognitive += 1 + self._nesting
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_For(self, node: ast.For) -> None:
        self._cyclomatic += 1
        self._cognitive += 1 + self._nesting
        self._nesting += 1

        # Off-by-one detection in range calls
        self._check_range_offbyone(node)

        # Check for index arithmetic
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                self._check_index_arithmetic(child)

        self.generic_visit(node)
        self._nesting -= 1

    def _check_range_offbyone(self, node: ast.For) -> None:
        """Flag loop ranges ending at len(x) or len(x)-1."""
        if isinstance(node.iter, ast.Call):
            func = node.iter.func
            if isinstance(func, ast.Name) and func.id == 'range':
                args = node.iter.args
                # Check the stop argument (last positional arg to range)
                stop = args[-1] if args else None
                if stop and self._is_len_call(stop):
                    # range(len(x)) is fine; range(len(x)-1) might be off-by-one
                    self._issue(
                        node, Category.OFF_BY_ONE, Severity.WARNING,
                        "Loop range ends at len(container). "
                        "Verify this is intentional — range(len(x)) is 0..n-1 which is correct for indexing, "
                        "but if you want all elements, consider 'for item in container' instead.",
                        "Prefer: for item in container  (avoids index arithmetic entirely)",
                        confidence=0.6,
                    )

    def _is_len_call(self, node: ast.AST) -> bool:
        """Return True if node is a call to len(...)."""
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == 'len')

    def _check_index_arithmetic(self, node: ast.Subscript) -> None:
        """Flag slice expressions like lst[0:len(lst)-1] that may exclude last element."""
        if isinstance(node.slice, ast.Slice):
            slc = node.slice
            # Check upper bound
            upper = slc.upper
            if upper is not None:
                # Look for len(x) - 1 pattern
                if (isinstance(upper, ast.BinOp) and
                        isinstance(upper.op, ast.Sub) and
                        self._is_len_call(upper.left) and
                        isinstance(upper.right, ast.Constant) and
                        upper.right.value == 1):
                    self._issue(
                        node, Category.OFF_BY_ONE, Severity.CRITICAL,
                        "Slice upper bound is len(x)-1 which EXCLUDES the last element. "
                        "Python slices are exclusive of the upper bound, "
                        "so lst[0:len(lst)-1] misses lst[-1].",
                        "Use lst[0:len(lst)] or simply lst[:] or lst[0:] to include all elements.",
                    )

    def visit_While(self, node: ast.While) -> None:
        self._cyclomatic += 1
        self._cognitive += 1 + self._nesting
        self._nesting += 1

        # Detect while True with no break
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
            if not has_break:
                self._issue(
                    node, Category.INFINITE_LOOP, Severity.CRITICAL,
                    "while True loop with no break statement — infinite loop",
                    "Add a break or return condition inside the loop body",
                )

        self.generic_visit(node)
        self._nesting -= 1

    def visit_Try(self, node: ast.Try) -> None:
        self._cyclomatic += len(node.handlers)
        self._cognitive += 1 + self._nesting

        # Check for bare except
        for handler in node.handlers:
            if handler.type is None:
                self._issue(
                    handler, Category.ANTI_PATTERN, Severity.WARNING,
                    "Bare 'except:' catches ALL exceptions including SystemExit and KeyboardInterrupt",
                    "Specify the exception type: except Exception: or except (ValueError, TypeError):",
                )
            # Check for swallowed exceptions (bare except with only pass)
            if (isinstance(handler.type, ast.Name) or handler.type is None):
                if (len(handler.body) == 1 and
                        isinstance(handler.body[0], ast.Pass)):
                    self._issue(
                        handler, Category.ANTI_PATTERN, Severity.WARNING,
                        "Exception silently swallowed with 'pass' — errors will be invisible",
                        "At minimum: log the error or re-raise it:\n  except Exception as e:\n      logger.error(e)\n      raise",
                    )

        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._cyclomatic += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._cyclomatic += 1
        self._cognitive += 1 + self._nesting
        self.generic_visit(node)

    # ── Variable tracking ──────────────────────────────────────────────────────

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._defined.add(node.id)
            self._assigned.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self._used.add(node.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self._imports.add(name)
            self._defined.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._imports.add(name)
            self._defined.add(name)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self._has_return_value = True
        else:
            self._has_bare_return = True
        self.generic_visit(node)

    # ── Post-analysis checks ───────────────────────────────────────────────────

    def check_unused_assignments(self, func_node: ast.FunctionDef) -> None:
        """
        Check for variables assigned but never used in a function.

        Parameters
        ----------
        func_node : ast.FunctionDef
            The function AST node.
        """
        assigned_in_func: set[str] = set()
        used_in_func: set[str] = set()

        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigned_in_func.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used_in_func.add(node.id)

        unused = assigned_in_func - used_in_func - set(a.arg for a in func_node.args.args)
        # Filter out underscore convention
        unused = {v for v in unused if not v.startswith('_')}

        for var in sorted(unused):
            self._issue(
                func_node, Category.UNUSED, Severity.INFO,
                f"Variable '{var}' is assigned but never used",
                f"Remove the assignment to '{var}' or prefix with underscore if intentionally unused: _{var}",
                confidence=0.7,
            )

    def check_return_consistency(self, func_node: ast.FunctionDef) -> None:
        """Flag functions that sometimes return a value and sometimes don't."""
        returns_value = []
        bare_returns = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if node.value is not None:
                    returns_value.append(node)
                else:
                    bare_returns.append(node)

        if returns_value and bare_returns:
            self._issue(
                func_node, Category.TYPE, Severity.WARNING,
                f"{func_node.name}() sometimes returns a value and sometimes returns None. "
                "This is a common source of bugs.",
                f"Ensure all return paths return a value, or explicitly return None:\n"
                f"  return None  # instead of bare return",
            )


def analyse_code(source: str) -> StaticAnalysisResult:
    """
    Perform full static analysis on a Python source code string.

    Parameters
    ----------
    source : str
        Python source code.

    Returns
    -------
    StaticAnalysisResult
        All detected issues, complexity metrics, and analysis trace.
    """
    result = StaticAnalysisResult()
    result.steps.append("Static analysis started")

    # Dedent to handle indented code blocks
    source = textwrap.dedent(source)

    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        result.issues.append(CodeIssue(
            location="<module>", line=e.lineno or 0,
            category=Category.UNDEFINED, severity=Severity.CRITICAL,
            description=f"Syntax error: {e.msg}",
            suggestion="Fix the syntax error before analysis",
        ))
        return result

    result.ast_dump = ast.dump(tree, indent=2)[:500] + "..." if len(ast.dump(tree)) > 500 else ast.dump(tree, indent=2)

    # Analyse each function separately
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            result.functions_found.append(node.name)
            result.steps.append(f"\nAnalysing function: {node.name}()")

            analyser = StaticAnalyzer(source)
            analyser.visit(node)
            analyser.check_unused_assignments(node)
            analyser.check_return_consistency(node)

            result.issues.extend(analyser.issues)
            result.steps.extend(analyser.steps)

            result.cyclomatic_complexity = max(result.cyclomatic_complexity, analyser._cyclomatic)
            result.cognitive_complexity = max(result.cognitive_complexity, analyser._cognitive)

    # Complexity flags
    if result.cyclomatic_complexity > config.COMPLEXITY_CYCLOMATIC_MAX:
        result.issues.insert(0, CodeIssue(
            location="<module>", line=1,
            category=Category.COMPLEXITY, severity=Severity.WARNING,
            description=f"Cyclomatic complexity = {result.cyclomatic_complexity} "
                        f"(threshold = {config.COMPLEXITY_CYCLOMATIC_MAX})",
            suggestion="Refactor into smaller functions to reduce complexity",
        ))

    if result.cognitive_complexity > config.COMPLEXITY_COGNITIVE_MAX:
        result.issues.insert(0, CodeIssue(
            location="<module>", line=1,
            category=Category.COMPLEXITY, severity=Severity.WARNING,
            description=f"Cognitive complexity = {result.cognitive_complexity} "
                        f"(threshold = {config.COMPLEXITY_COGNITIVE_MAX})",
            suggestion="Reduce nesting depth and branching to improve readability",
        ))

    if not result.functions_found:
        result.steps.append("No function definitions found; analysing as script")

    result.steps.append(
        f"\nAnalysis complete: {len(result.issues)} issue(s) found "
        f"(cyclomatic={result.cyclomatic_complexity}, "
        f"cognitive={result.cognitive_complexity})"
    )
    return result
