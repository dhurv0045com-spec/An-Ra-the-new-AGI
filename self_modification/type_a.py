from __future__ import annotations

from pathlib import Path
import json
import time

from anra_paths import STATE_DIR, WORKSPACE_DIR

DEFAULT_TOOL_REGISTRY = STATE_DIR / "tool_library.json"
DEFAULT_TOOL_DIR = WORKSPACE_DIR / "tools"


class ToolLibraryMutation:
<<<<<<< HEAD
    def __init__(self, registry_path: str | Path = "state/tool_library.json") -> None:
        self.registry_path = Path(registry_path)
=======
    """Type A self-modification: add executable Python tools safely."""

    def __init__(self, registry_path: str | Path | None = None, tool_dir: str | Path | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path is not None else DEFAULT_TOOL_REGISTRY
        self.tool_dir = Path(tool_dir) if tool_dir is not None else DEFAULT_TOOL_DIR
>>>>>>> cf05483 (sing the moment)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"tools": []}, indent=2), encoding="utf-8")

    def add_tool(self, name: str, entrypoint: str, description: str = "") -> dict:
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        tool = {
            "name": name,
            "entrypoint": entrypoint,
            "description": description,
            "added_at": time.time(),
        }
        payload.setdefault("tools", []).append(tool)
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return tool
