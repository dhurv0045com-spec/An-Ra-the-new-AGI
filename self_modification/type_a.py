from __future__ import annotations

from pathlib import Path
import ast
import importlib.util
import json
import time

from anra_paths import STATE_DIR, WORKSPACE_DIR

DEFAULT_TOOL_REGISTRY = STATE_DIR / "tool_library.json"
DEFAULT_TOOL_DIR = WORKSPACE_DIR / "tools"


class ToolLibraryMutation:
    """Type A self-modification: add executable Python tools safely."""

    def __init__(self, registry_path: str | Path | None = None, tool_dir: str | Path | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path is not None else DEFAULT_TOOL_REGISTRY
        self.tool_dir = Path(tool_dir) if tool_dir is not None else DEFAULT_TOOL_DIR
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.tool_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"tools": []}, indent=2), encoding="utf-8")

    def add_tool(self, name: str, entrypoint: str, description: str = "") -> dict:
        module_name = self._module_name(name)
        ast.parse(entrypoint, filename=f"{module_name}.py")
        path = self.tool_dir / f"{module_name}.py"
        path.write_text(entrypoint, encoding="utf-8")
        self._validate_import(module_name, path)

        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        tool = {
            "name": name,
            "module": module_name,
            "path": str(path),
            "description": description,
            "added_at": time.time(),
        }
        tools = [row for row in payload.get("tools", []) if row.get("name") != name]
        tools.append(tool)
        payload["tools"] = tools
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return tool

    def _module_name(self, name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip().lower())
        cleaned = cleaned.strip("_")
        if not cleaned:
            raise ValueError("tool name must contain at least one alphanumeric character")
        if cleaned[0].isdigit():
            cleaned = f"tool_{cleaned}"
        return cleaned

    def _validate_import(self, module_name: str, path: Path) -> None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
