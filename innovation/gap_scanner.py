from __future__ import annotations

import ast
import hashlib
import json
import re
import time
import tokenize
from pathlib import Path

from innovation.schema import CapabilityGap


_TOKENS = re.compile(r"\b(TODO|FIXME|MISSING|STUB|HACK)\b", re.IGNORECASE)
_SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", "checkpoints", "output"}
_SEVERITY_RANK = {"critical": 0, "moderate": 1, "minor": 2}


def _repo_root(repo_root: Path | None) -> Path:
    return (repo_root or Path.cwd()).resolve()


def _iter_py(root: Path):
    for path in root.rglob("*.py"):
        parts = set(path.relative_to(root).parts)
        if parts & _SKIP_DIRS:
            continue
        yield path


def _gap_id(description: str, detected_in: str) -> str:
    return hashlib.sha1(f"{detected_in}|{description}".encode("utf-8")).hexdigest()[:12]


def _add(gaps: list[CapabilityGap], description: str, path: Path, severity: str, evidence: list[str], root: Path) -> None:
    detected_in = str(path.relative_to(root)) if path.is_absolute() or path.exists() else str(path)
    gaps.append(
        CapabilityGap(
            gap_id=_gap_id(description, detected_in),
            description=description,
            detected_in=detected_in,
            severity=severity,
            evidence=evidence,
            detected_at=time.time(),
        )
    )


def _comment_gaps(path: Path, root: Path, gaps: list[CapabilityGap]) -> None:
    try:
        with path.open("rb") as f:
            tokens = tokenize.tokenize(f.readline)
            for tok in tokens:
                if tok.type != tokenize.COMMENT:
                    continue
                match = _TOKENS.search(tok.string)
                if not match:
                    continue
                word = match.group(1).upper()
                severity = "moderate" if word in {"FIXME", "MISSING", "STUB", "HACK"} else "minor"
                _add(
                    gaps,
                    f"{word} comment marks unfinished or fragile behavior",
                    path,
                    severity,
                    [f"line {tok.start[0]}: {tok.string.strip()}"],
                    root,
                )
    except Exception:
        return


class _AstGapVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, root: Path, gaps: list[CapabilityGap]) -> None:
        self.path = path
        self.root = root
        self.gaps = gaps

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        name = ""
        if isinstance(exc, ast.Call):
            func = exc.func
            name = getattr(func, "id", getattr(func, "attr", ""))
        elif exc is not None:
            name = getattr(exc, "id", getattr(exc, "attr", ""))
        if name == "NotImplementedError":
            _add(
                self.gaps,
                "Function raises NotImplementedError",
                self.path,
                "critical",
                [f"line {node.lineno}: raise NotImplementedError"],
                self.root,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_pytest_skip = (
            isinstance(func, ast.Attribute)
            and func.attr == "skip"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        )
        if is_pytest_skip and self.path.name.startswith("test"):
            _add(
                self.gaps,
                "Test contains pytest.skip, leaving behavior unverified in this environment",
                self.path,
                "moderate",
                [f"line {node.lineno}: pytest.skip(...)"],
                self.root,
            )
        self.generic_visit(node)


def _ast_gaps(path: Path, root: Path, gaps: list[CapabilityGap]) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except Exception:
        return
    _AstGapVisitor(path, root, gaps).visit(tree)


def _use_hal_flags(path: Path, root: Path, gaps: list[CapabilityGap]) -> None:
    if path.name.startswith("test") or "/tests/" in path.as_posix():
        return
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if "use_hal=False" in line.replace(" ", ""):
                _add(
                    gaps,
                    "HAL integration flag is still disabled by default",
                    path,
                    "critical",
                    [f"line {idx}: {line.strip()}"],
                    root,
                )
    except Exception:
        return


def _system_graph_gaps(root: Path, gaps: list[CapabilityGap]) -> None:
    path = root / "system_graph.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    nodes = data.get("nodes", data.get("components", []))
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name", node.get("id", "unknown")))
        status = node.get("status")
        ok = node.get("ok")
        import_status = str(node.get("import_status", ""))
        bad_status = status is not None and str(status).lower() != "ok"
        bad_ok = ok is False
        degraded = import_status.startswith("degraded")
        if bad_status or bad_ok or degraded:
            _add(
                gaps,
                f"System graph reports non-ok component: {name}",
                path,
                "moderate",
                [json.dumps({"name": name, "status": status, "ok": ok, "import_status": import_status}, sort_keys=True)],
                root,
            )


def _relative_import_exists(init_path: Path, module: str | None, level: int) -> bool:
    base = init_path.parent
    for _ in range(max(0, level - 1)):
        base = base.parent
    if not module:
        return True
    target = base.joinpath(*module.split("."))
    return target.with_suffix(".py").exists() or (target / "__init__.py").exists()


def _init_import_gaps(path: Path, root: Path, gaps: list[CapabilityGap]) -> None:
    if path.name != "__init__.py":
        return
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except Exception:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level > 0 and not _relative_import_exists(path, node.module, node.level):
            _add(
                gaps,
                f"__init__.py imports missing relative module {'.' * node.level}{node.module or ''}",
                path,
                "critical",
                [f"line {node.lineno}: from {'.' * node.level}{node.module or ''} import ..."],
                root,
            )


def scan(repo_root: Path | None = None) -> list[CapabilityGap]:
    """Scan the repository for concrete capability gaps."""
    root = _repo_root(repo_root)
    gaps: list[CapabilityGap] = []
    for path in _iter_py(root):
        _comment_gaps(path, root, gaps)
        _ast_gaps(path, root, gaps)
        _use_hal_flags(path, root, gaps)
        _init_import_gaps(path, root, gaps)
    _system_graph_gaps(root, gaps)
    gaps.sort(key=lambda gap: (_SEVERITY_RANK.get(gap.severity, 99), gap.description.lower(), gap.detected_in))
    return gaps
