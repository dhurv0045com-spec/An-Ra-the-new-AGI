from __future__ import annotations

from pathlib import Path
import difflib
import subprocess
import time


class AgentCodeMutation:
    def __init__(self, backup_dir: str | Path = "state/self_mod_backups") -> None:
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _rollback(self, path: Path, backup: Path) -> None:
        path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")

    def mutate(
        self,
        file_path: str | Path,
        new_content: str,
        reason: str = "",
        benchmark_cmd: list[str] | None = None,
        timeout: int = 120,
    ) -> dict:
        path = Path(file_path)
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        stamp = str(int(time.time()))
        backup = self.backup_dir / f"{path.name}.{stamp}.bak"
        backup.write_text(old, encoding="utf-8")
        path.write_text(new_content, encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(old.splitlines(), new_content.splitlines(), fromfile="old", tofile="new", lineterm=""))
        result = {"file": str(path), "backup": str(backup), "reason": reason, "diff": diff[:12000], "accepted": True}
        if benchmark_cmd:
            try:
                proc = subprocess.run(benchmark_cmd, capture_output=True, text=True, timeout=timeout, cwd=str(path.parent if path.parent != Path("") else Path.cwd()))
            except Exception as exc:
                self._rollback(path, backup)
                result.update({"accepted": False, "rolled_back": True, "benchmark_error": f"[type_b] benchmark failed to run: {exc}"})
                return result
            result["benchmark_return_code"] = int(proc.returncode)
            result["benchmark_output"] = (proc.stdout + proc.stderr)[-12000:]
            if proc.returncode != 0:
                self._rollback(path, backup)
                result.update({"accepted": False, "rolled_back": True})
        return result
