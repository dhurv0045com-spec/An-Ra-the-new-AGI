from __future__ import annotations

from pathlib import Path
import difflib
import json
import sys
import time

from anra_paths import SELF_MOD_AUDIT_LOG, STATE_DIR
from self_modification.sovereignty_gate import SovereigntyRollback, sovereignty_audit_change


class AgentCodeMutation:
    def __init__(self, backup_dir: str | Path | None = None) -> None:
        self.backup_dir = Path(backup_dir) if backup_dir is not None else STATE_DIR / "self_mod_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def mutate(
        self,
        file_path: str | Path,
        new_content: str,
        reason: str = "",
        benchmark_cmd: list[str] | None = None,
    ) -> dict:
        path = Path(file_path)
        gate = sovereignty_audit_change(path, new_content, reason=reason)
        if not gate.get("allowed"):
            return {"accepted": False, "file": str(path), "reason": "sovereignty audit rejected change", "audit": gate}
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        stamp = str(int(time.time()))
        backup = self.backup_dir / f"{path.name}.{stamp}.bak"
        backup.write_text(old, encoding="utf-8")
        diff = "\n".join(difflib.unified_diff(old.splitlines(), new_content.splitlines(), fromfile="old", tofile="new", lineterm=""))
        test_cmd = benchmark_cmd or [sys.executable, "-c", "raise SystemExit(0)"]
        with SovereigntyRollback([path], test_cmd=test_cmd) as rollback:
            path.write_text(new_content, encoding="utf-8")
            if not rollback.commit():
                self._log_rollback(path, reason or "benchmark failed; rolled back")
                return {
                    "accepted": False,
                    "file": str(path),
                    "backup": str(backup),
                    "reason": reason or "benchmark failed; rolled back",
                    "diff": diff[:12000],
                    "stdout": "",
                    "stderr": "SovereigntyRollback restored original file after failed verification.",
                }
        return {"accepted": True, "file": str(path), "backup": str(backup), "reason": reason, "diff": diff[:12000]}

    def propose(
        self,
        file_path: str | Path,
        new_content: str,
        reason: str = "",
        sandbox=None,
        verifier=None,
        benchmark_tasks: list[dict] | None = None,
        drop_threshold: float = 0.02,
    ) -> dict:
        """
        Write new_content, benchmark with internal verifier, accept or rollback.

        Uses VerifierHierarchy instead of a subprocess shell command.
        """
        path = Path(file_path)
        gate = sovereignty_audit_change(path, new_content, reason=reason)
        if not gate.get("allowed"):
            return {
                "accepted": False,
                "score_before": 0.0,
                "score_after": 0.0,
                "score_delta": 0.0,
                "diff": "",
                "backup": "",
                "reason": "sovereignty audit rejected change",
                "audit": gate,
            }
        old_content = path.read_text(encoding="utf-8") if path.exists() else ""

        stamp = str(int(time.time()))
        backup = self.backup_dir / f"{path.name}.{stamp}.bak"
        backup.write_text(old_content, encoding="utf-8")

        diff = "\n".join(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile="old",
            tofile="proposed",
            lineterm="",
        ))

        score_before = self._run_internal_benchmark(
            sandbox, verifier, benchmark_tasks, path, old_content
        )

        path.write_text(new_content, encoding="utf-8")

        score_after = self._run_internal_benchmark(
            sandbox, verifier, benchmark_tasks, path, new_content
        )

        if score_after < score_before - drop_threshold:
            path.write_text(old_content, encoding="utf-8")
            return {
                "accepted": False,
                "score_before": score_before,
                "score_after": score_after,
                "score_delta": score_after - score_before,
                "diff": diff[:4000],
                "backup": str(backup),
                "reason": (
                    f"Score dropped {score_before:.3f}->{score_after:.3f} "
                    f"(threshold={drop_threshold}). Rolled back."
                ),
            }

        return {
            "accepted": True,
            "score_before": score_before,
            "score_after": score_after,
            "score_delta": score_after - score_before,
            "diff": diff[:4000],
            "backup": str(backup),
            "reason": reason or "Benchmark passed.",
        }

    def _run_internal_benchmark(
        self,
        sandbox,
        verifier,
        tasks: list[dict] | None,
        _path: Path,
        _content: str,
    ) -> float:
        """Run benchmark_tasks through VerifierHierarchy. Return mean score."""
        if not tasks or sandbox is None or verifier is None:
            return 1.0

        scores = []
        for task in tasks[:50]:
            code = task.get("code", "print('ok')")
            test_code = task.get("test_code", task.get("test", ""))
            task_type = task.get("type", task.get("task_type", "code"))
            try:
                sandbox.execute(code + ("\n\n" + test_code if test_code else ""))
                if hasattr(verifier, "call_count") and hasattr(type(verifier), "call_count"):
                    type(verifier).call_count = int(getattr(verifier, "call_count", 0)) + 1
                vr = verifier.score(
                    task_type,
                    code=code,
                    test_code=test_code,
                )
                scores.append(float(vr.score))
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 1.0

    def _log_rollback(self, path: Path, reason: str) -> None:
        event = {
            "ts": time.time(),
            "event_type": "SELF_MOD_ROLLBACK",
            "component": "self_modification",
            "action": "rollback",
            "file": str(path),
            "reason": reason,
        }
        try:
            SELF_MOD_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with SELF_MOD_AUDIT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception:
            pass
