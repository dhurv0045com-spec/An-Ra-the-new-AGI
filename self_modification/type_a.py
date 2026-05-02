from __future__ import annotations

import ast
import importlib.util
import json
import time
from pathlib import Path


class ToolLibraryMutation:
    """Type A self-modification: add executable Python tools safely."""

    def __init__(self, registry_path: str | Path = "state/tool_library.json", tool_dir: str | Path = "workspace/tools") -> None:
        self.registry_path = Path(registry_path)
        self.tool_dir = Path(tool_dir)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.tool_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"tools": []}, indent=2), encoding="utf-8")

    def _tool_path(self, name: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip())
        if not safe:
            raise ValueError("[type_a] tool name cannot be empty")
        return self.tool_dir / f"{safe}.py"

    def _validate_python(self, path: Path, source: str) -> None:
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise SyntaxError(f"[type_a] syntax check failed for {path}: {exc}") from exc
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"[type_a] cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise RuntimeError(f"[type_a] import execution failed for {path}: {exc}") from exc

    def add_tool(self, name: str, source: str, description: str = "", entrypoint: str = "run") -> dict:
        if f"def {entrypoint}" not in source:
            raise ValueError(f"[type_a] source must define entrypoint function {entrypoint!r}")
        path = self._tool_path(name)
        tmp = path.with_name(f".{path.stem}.tmp.py")
        tmp.write_text(source, encoding="utf-8")
        self._validate_python(tmp, source)
        tmp.replace(path)

        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        tools = [tool for tool in payload.get("tools", []) if tool.get("name") != name]
        tool = {
            "name": name,
            "path": str(path),
            "entrypoint": entrypoint,
            "description": description,
            "added_at": time.time(),
        }
        tools.append(tool)
        payload["tools"] = tools
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return tool
