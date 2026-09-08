"""Minimal static undefined-name check (ruff/pyflakes equivalent).

Walks each module's AST with simple scope tracking (imports, assignments,
function/class defs, arguments, comprehension scopes, walrus) and reports
names loaded but never bound, excluding builtins and attribute roots.
Conservative by design: dynamic machinery (getattr, globals, exec-heavy
patterns) can yield false negatives, never used to excuse a failure.
Exit nonzero on any finding. Used locally and in Colab CELL 0.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _check_path(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            # Never rely on the ambient __builtins__ binding: under pytest
            # and other harnesses it can be a partial dict (dir() on a
            # dict yields methods, not names). Import the module explicitly.
            import builtins as _builtins_module
            builtin_names = set(dir(_builtins_module)) | {
                "__builtins__", "__file__", "__name__", "__doc__", "__package__",
            }
            self.scopes: list[set[str]] = [builtin_names]
            # Module-level forward references are legal at runtime
            # (function bodies execute after the module loads).
            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.ClassDef)):
                    self.scopes[0].add(statement.name)
                elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                    for alias in statement.names:
                        if getattr(alias, "name", "") != "*":
                            self.scopes[0].add(
                                alias.asname or alias.name.split(".")[0])
                elif isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        for child in ast.walk(target):
                            if isinstance(child, ast.Name) and isinstance(
                                    child.ctx, ast.Store):
                                self.scopes[0].add(child.id)

        def _bind(self, name: str) -> None:
            self.scopes[-1].add(name.split(".")[0])

        def _bound(self, name: str) -> bool:
            return any(name in scope for scope in reversed(self.scopes))

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                self._bind(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name != "*":
                    self._bind(alias.asname or alias.name.split(".")[0])

        def _bind_target(self, target: ast.AST) -> None:
            for child in ast.walk(target):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    self._bind(child.id)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                        ast.ClassDef)):
                    self._bind(child.name)

        def visit_FunctionDef(self, node) -> None:
            self._bind(node.name)
            self.scopes.append(set())
            for arg in list(node.args.posonlyargs) + list(node.args.args) + \
                    list(node.args.kwonlyargs):
                self._bind(arg.arg)
            if node.args.vararg:
                self._bind(node.args.vararg.arg)
            if node.args.kwarg:
                self._bind(node.args.kwarg.arg)
            for default in list(node.args.defaults) + \
                    [d for d in node.args.kw_defaults if d is not None]:
                self.visit(default)
            for decorator in node.decorator_list:
                self.visit(decorator)
            for statement in node.body:
                self.visit(statement)
            self.scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node) -> None:
            self._bind(node.name)
            self.scopes.append(set())
            for base in node.bases:
                self.visit(base)
            for statement in node.body:
                self.visit(statement)
            self.scopes.pop()

        def visit_Lambda(self, node) -> None:
            self.scopes.append(set())
            for arg in list(node.args.posonlyargs) + list(node.args.args):
                self._bind(arg.arg)
            self.visit(node.body)
            self.scopes.pop()

        def visit_ListComp(self, node) -> None:
            self._handle_comp(node)

        def visit_SetComp(self, node) -> None:
            self._handle_comp(node)

        def visit_DictComp(self, node) -> None:
            self._handle_comp(node)

        def visit_GeneratorExp(self, node) -> None:
            self._handle_comp(node)

        def _handle_comp(self, node) -> None:
            self.scopes.append(set())
            for generator in node.generators:
                self.visit(generator.iter)
                self._bind_target(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
            self.scopes.pop()

        def visit_NamedExpr(self, node) -> None:
            self.visit(node.value)
            self._bind(node.target.id)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load) and not self._bound(node.id):
                findings.append(f"{path}:{node.lineno}: undefined name '{node.id}'")

        def visit_Assign(self, node: ast.Assign) -> None:
            self.visit(node.value)
            for target in node.targets:
                self._bind_target(target)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self.visit(node.value)
            self._bind_target(node.target)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.visit(node.value)
            self._bind_target(node.target)

        def visit_For(self, node) -> None:
            self.visit(node.iter)
            self._bind_target(node.target)
            for statement in node.body:
                self.visit(statement)
            for statement in node.orelse:
                self.visit(statement)

        visit_AsyncFor = visit_For

        def visit_With(self, node) -> None:
            for item in node.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars)
            for statement in node.body:
                self.visit(statement)

        visit_AsyncWith = visit_With

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.type is not None:
                self.visit(node.type)
            if node.name:
                self._bind(node.name)
            for statement in node.body:
                self.visit(statement)

        def visit_Global(self, node: ast.Global) -> None:
            for name in node.names:
                for scope in self.scopes:
                    scope.add(name)

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            for name in node.names:
                for scope in self.scopes:
                    scope.add(name)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            self.visit(node.value)

    Visitor().visit(tree)
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (argv or sys.argv[1:])]
    if not paths:
        print("usage: static_check.py <file> [...]", file=sys.stderr)
        return 2
    findings = [finding for path in paths for finding in _check_path(path)]
    for finding in findings:
        print(finding)
    print(f"{len(findings)} undefined-name findings")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
