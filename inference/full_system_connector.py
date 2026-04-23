from __future__ import annotations

"""
Repository-wide connector and capability graph for An-Ra.

Purpose:
- Traverse the full repository tree and summarize all subsystems.
- Expose phase-aware metadata (Phase 2/3/4 + core/turboquant, ouroboros, etc.).
- Provide lightweight health snapshots for orchestration and API diagnostics.
"""

import ast
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class FileNode:
    path: str
    size_bytes: int
    line_count: int
    has_python: bool
    classes: List[str]
    functions: List[str]


@dataclass
class PhaseSnapshot:
    name: str
    root: str
    file_count: int
    python_files: int
    total_lines: int
    notable_files: List[str]


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _python_symbols(path: Path) -> Tuple[List[str], List[str]]:
    if path.suffix != ".py":
        return [], []
    src = _safe_read_text(path)
    if not src.strip():
        return [], []
    try:
        tree = ast.parse(src)
    except Exception:
        return [], []
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return classes, funcs


def walk_repository(repo_root: Path) -> List[FileNode]:
    nodes: List[FileNode] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", "node_modules"} for part in path.parts):
            continue
        text = _safe_read_text(path)
        classes, funcs = _python_symbols(path)
        nodes.append(
            FileNode(
                path=str(path.relative_to(repo_root)),
                size_bytes=path.stat().st_size,
                line_count=text.count("\n") + (1 if text else 0),
                has_python=path.suffix == ".py",
                classes=classes[:25],
                functions=funcs[:50],
            )
        )
    return nodes


def phase_snapshots(repo_root: Path, nodes: List[FileNode]) -> List[PhaseSnapshot]:
    phases = [
        ("phase1_core", "core"),
        ("phase2", "phase2"),
        ("phase3", "phase3"),
        ("phase4", "phase4"),
        ("api", "."),
    ]

    snapshots: List[PhaseSnapshot] = []
    for name, root in phases:
        prefix = "" if root == "." else root + "/"
        scoped = [n for n in nodes if root == "." or n.path.startswith(prefix)]
        if root == ".":
            scoped = [n for n in nodes if n.path in {"anra.py", "app.py", "generate.py", "finetune_anra.py", "test_suite.py"}]
        py = [n for n in scoped if n.has_python]
        notable = [n.path for n in scoped if any(k in n.path.lower() for k in ["turboquant", "ouroboros", "symbolic", "sovereignty", "system", "app.py", "generate.py"])]
        snapshots.append(
            PhaseSnapshot(
                name=name,
                root=root,
                file_count=len(scoped),
                python_files=len(py),
                total_lines=sum(n.line_count for n in scoped),
                notable_files=notable[:30],
            )
        )
    return snapshots


def build_capability_graph(repo_root: Path) -> Dict[str, object]:
    nodes = walk_repository(repo_root)
    snapshots = phase_snapshots(repo_root, nodes)

    capabilities = {
        "turboquant": any("turboquant" in n.path.lower() for n in nodes),
        "ouroboros": any("ouroboros" in n.path.lower() for n in nodes),
        "symbolic_bridge": any("symbolic_bridge" in n.path.lower() for n in nodes),
        "sovereignty": any("sovereignty" in n.path.lower() for n in nodes),
        "web_ui": any(n.path.startswith("phase4/web/") for n in nodes),
        "fastapi": any(n.path == "app.py" for n in nodes),
        "integration_tests": any(n.path.startswith("tests/") for n in nodes) or any(n.path == "test_suite.py" for n in nodes),
    }

    graph = {
        "repo_root": str(repo_root),
        "file_count": len(nodes),
        "python_file_count": sum(1 for n in nodes if n.has_python),
        "total_lines": sum(n.line_count for n in nodes),
        "capabilities": capabilities,
        "phase_snapshots": [asdict(s) for s in snapshots],
    }
    return graph


def save_graph(repo_root: Path, output: Path) -> Dict[str, object]:
    graph = build_capability_graph(repo_root)
    output.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return graph


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    out = root / "system_graph.json"
    g = save_graph(root, out)
    print(json.dumps({"saved": str(out), "file_count": g["file_count"], "total_lines": g["total_lines"]}, indent=2))
