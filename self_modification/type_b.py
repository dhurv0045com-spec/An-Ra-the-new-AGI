from __future__ import annotations

from pathlib import Path
import difflib
import time


class AgentCodeMutation:
    def __init__(self, backup_dir: str | Path = "state/self_mod_backups") -> None:
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def mutate(self, file_path: str | Path, new_content: str, reason: str = "") -> dict:
        path = Path(file_path)
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        stamp = str(int(time.time()))
        backup = self.backup_dir / f"{path.name}.{stamp}.bak"
        backup.write_text(old, encoding="utf-8")
        path.write_text(new_content, encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(old.splitlines(), new_content.splitlines(), fromfile="old", tofile="new", lineterm=""))
        return {"file": str(path), "backup": str(backup), "reason": reason, "diff": diff[:12000]}
