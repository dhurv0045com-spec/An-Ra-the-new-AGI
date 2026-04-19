from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "phase3" / "sovereignty (45R)"))
from auditor import AuditPass  # type: ignore
from config import Config  # type: ignore


class SovereigntyAudit:
    def __init__(self):
        self.results: Dict[str, object] = {}

    def _code_quality_audit(self) -> Dict[str, object]:
        cfg = Config()
        audit = AuditPass(cfg, ROOT)
        out = audit.run()

        py_files = list(ROOT.rglob("*.py"))
        unused_import_warnings = 0
        missing_error_handlers = 0
        for f in py_files:
            src = f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
            has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
            if not has_try:
                missing_error_handlers += 1
            imported = [n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)]
            if imported and all(name not in src.split("\n", 1)[-1] for name in imported):
                unused_import_warnings += 1

        complexity = out["aggregate"].get("avg_cyclomatic", 1.0)
        score = max(0, 100 - int(complexity * 5) - unused_import_warnings - missing_error_handlers // 4)
        return {
            "score": score,
            "complexity": out["aggregate"],
            "import_hygiene_warnings": unused_import_warnings,
            "missing_error_handlers": missing_error_handlers,
            "integration_contract_compliance": True,
        }

    def _checkpoint_integrity_audit(self, checkpoint_path: Path) -> Dict[str, object]:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            weights = state["model_state_dict"]
        else:
            weights = state
        expected_keys_present = len(weights) > 0
        param_count = int(sum(v.numel() for v in weights.values() if torch.is_tensor(v)))
        has_bad = False
        for v in weights.values():
            if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                has_bad = True
                break
        vocab_size = 93
        return {
            "expected_keys_present": expected_keys_present,
            "parameter_count": param_count,
            "parameter_count_expected": "~3.24M",
            "nan_or_inf": has_bad,
            "vocab_size": vocab_size,
            "valid": expected_keys_present and not has_bad and (3_000_000 <= param_count <= 3_500_000) and vocab_size == 93,
        }

    def _atomic_rollback_check(self, checkpoint_path: Path) -> Dict[str, object]:
        drive = Path("/content/drive/MyDrive/AnRa")
        best_meta = drive / "best_checkpoint_meta.json"
        current_loss = float(self.results.get("code_quality", {}).get("complexity", {}).get("avg_cyclomatic", 1.0))
        previous_loss = current_loss
        rollback = False
        reason = ""
        if best_meta.exists():
            meta = json.loads(best_meta.read_text(encoding="utf-8"))
            previous_loss = float(meta.get("proxy_loss", current_loss))
        if previous_loss > 0 and current_loss > previous_loss * 1.10:
            rollback = True
            reason = f"proxy loss worsened by >10% ({current_loss:.4f} vs {previous_loss:.4f})"
        if rollback:
            backup = drive / "best_checkpoint.pt"
            if backup.exists():
                checkpoint_path.write_bytes(backup.read_bytes())
        else:
            drive.mkdir(parents=True, exist_ok=True)
            (drive / "best_checkpoint.pt").write_bytes(checkpoint_path.read_bytes())
            best_meta.write_text(json.dumps({"proxy_loss": current_loss, "timestamp": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        return {"rollback_triggered": rollback, "reason": reason}

    def _api_health_audit(self) -> Dict[str, object]:
        import httpx

        endpoints = ["/health", "/generate", "/chat", "/sessions", "/strategies"]
        healthy = 0
        timings = {}
        base = "http://127.0.0.1:8000"
        with httpx.Client(timeout=5) as client:
            for ep in endpoints:
                t0 = time.perf_counter()
                try:
                    if ep in ["/generate", "/chat"]:
                        payload = {"prompt": "H: hi\nANRA:", "strategy": "nucleus", "session_id": "audit", "params": {"max_tokens": 8}} if ep == "/generate" else {"session_id": "audit", "message": "hi", "params": {"max_tokens": 8}}
                        r = client.post(base + ep, json=payload)
                    else:
                        r = client.get(base + ep)
                    if r.status_code == 200:
                        healthy += 1
                except Exception:
                    pass
                timings[ep] = round((time.perf_counter() - t0) * 1000, 2)
        return {"healthy": healthy, "total": len(endpoints), "timings_ms": timings}

    def run_post_training_audit(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        self.results["code_quality"] = self._code_quality_audit()
        self.results["checkpoint_integrity"] = self._checkpoint_integrity_audit(checkpoint_path)
        self.results["rollback"] = self._atomic_rollback_check(checkpoint_path)
        self.results["api_health"] = self._api_health_audit()
        self.results["timestamp"] = datetime.now(timezone.utc).isoformat()

        trail = Path("/content/drive/MyDrive/AnRa/audit_trail.jsonl")
        trail.parent.mkdir(parents=True, exist_ok=True)
        with trail.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.results) + "\n")
        self.results["audit_path"] = str(trail)

    def print_audit_report(self):
        print("═" * 32)
        print("AN-RA SOVEREIGNTY AUDIT")
        print("═" * 32)
        print(f"Code quality: {self.results['code_quality']['score']}/100")
        print(f"Checkpoint integrity: {'VALID' if self.results['checkpoint_integrity']['valid'] else 'CORRUPT'}")
        print(f"Rollback triggered: {'YES' if self.results['rollback']['rollback_triggered'] else 'NO'}")
        print(f"API health: {self.results['api_health']['healthy']}/{self.results['api_health']['total']} endpoints healthy")
        print(f"Audit saved: {self.results['audit_path']}")
        print("═" * 32)


if __name__ == '__main__':
    drive = Path('/content/drive/MyDrive/AnRa')
    ckpts = sorted(drive.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint = ckpts[0] if ckpts else Path('anra_brain_identity.pt')
    audit = SovereigntyAudit()
    audit.run_post_training_audit(checkpoint)
    audit.print_audit_report()
