from __future__ import annotations

from pathlib import Path
import json
import time


class ToolLibraryMutation:
    def __init__(self, registry_path: str | Path = "state/tool_library.json") -> None:
        self.registry_path = Path(registry_path)
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
