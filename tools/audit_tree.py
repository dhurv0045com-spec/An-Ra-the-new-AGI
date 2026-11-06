"""Inventory the immutable pre-V3 tree and audit the retained import closure."""

import argparse
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess


ARCHIVE = "930ca3bb723e0b5b191700eec3d5f450875a6e25"
ARCHIVE_REF = "elendril-pre-v3"
SHARED = {
    "v5_checkpointing.py",
    "v5_contracts/model_spec.py",
    "v5_data/bucket_cursor.py",
    "v5_evaluation/firewall.py",
    "v5_objectives/__init__.py",
    "v5_objectives/causal_lm.py",
    "v5_objectives/query_swap.py",
    "v5_training/checkpoint.py",
    "v5_training/checkpoint_lock.py",
    "v5_training/distributed_checkpoint.py",
    "v5_training/optimizer.py",
    "v5_training/persistent_store.py",
    "v5_training/state.py",
}
REIMPLEMENT = {
    ".github/workflows/elendril.yml",
    "README.md",
    "pyproject.toml",
    "anra_v5/__init__.py",
    "v5_contracts/__init__.py",
    "v5_data/__init__.py",
    "v5_evaluation/__init__.py",
    "v5_training/__init__.py",
    "tests/test_v5_durability_contracts.py",
}
REGRESSIONS = {"tests/test_v5_checkpointing.py", "tests/test_v5_model.py", "tests/test_v5_objectives.py"}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def classify(path: str) -> tuple[str, str]:
    if path in REIMPLEMENT:
        return "REIMPLEMENT", "replace legacy entry points or extract active regressions"
    if path.startswith("anra_v5/elendril/"):
        return "KEEP", "native Elendril implementation and replay entry points"
    if path.startswith("artifacts/elendril/"):
        return "KEEP", "unique raw evidence or checkpoint custody manifest"
    if path in SHARED or path.startswith("v5_model/"):
        return "KEEP", "mapped dense-core, objective, state or checkpoint dependency"
    if path.startswith("tests/test_elendril") or path in REGRESSIONS:
        return "KEEP", "active numerical, replay or mechanism regression"
    if path in {"LICENSE", ".gitattributes", ".gitignore"}:
        return "KEEP", "legal notice or repository hygiene"
    if path.startswith(("docs/", "artifacts/", "notebooks/", "experiments/", "blueprint/")):
        return "REFERENCE-BY-SHA", "historical research and operators remain in archive"
    if path == "tools/elendril_join_diagnostic.py":
        return "REFERENCE-BY-SHA", "original diagnostic operator is bound to archived receipt"
    return "REMOVE-FROM-ELENDRIL", "outside active dependency closure, retrievable from archive"


def archive_inventory() -> dict:
    entries = {}
    for item in git("ls-tree", "-r", "--full-tree", "-z", ARCHIVE).split(b"\0"):
        if not item:
            continue
        meta, path = item.split(b"\t", 1)
        mode, kind, object_id = meta.decode().split()
        name = path.decode("utf-8")
        action, why = classify(name)
        entries[name] = {"action": action, "reason": why, "mode": mode,
                         "object_type": kind, "git_object": object_id}
    return {"schema": "elendril-clean-tree-audit/v1", "archive_commit": ARCHIVE,
            "archive_ref": ARCHIVE_REF, "counts": dict(Counter(x["action"] for x in entries.values())),
            "files": entries}


def module_for(path: str) -> str:
    parts = PurePosixPath(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def dependencies(paths: list[str]) -> dict[str, list[str]]:
    py_files = {module_for(p): p for p in paths if p.endswith(".py")}
    graph = {}
    for module, path in sorted(py_files.items()):
        imports = set()
        package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(Path(path).read_bytes(), filename=path)):
            if isinstance(node, ast.Import):
                imports.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                origin = node.module or ""
                if node.level and package:
                    origin = importlib.util.resolve_name("." * node.level + origin, package)
                if origin:
                    imports.add(origin)
                    imports.update(f"{origin}.{item.name}" for item in node.names)
        graph[path] = sorted({py_files[name] for name in imports if name in py_files and name != module})
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-imports", action="store_true")
    args = parser.parse_args()
    if args.active_imports:
        paths = git("ls-files", "-z").decode().strip("\0").split("\0")
        value = dependencies(paths)
    else:
        value = archive_inventory()
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
