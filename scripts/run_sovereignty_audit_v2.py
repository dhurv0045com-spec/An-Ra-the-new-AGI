from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anra_paths import DRIVE_V2_DIR, DRIVE_V2_CHECKPOINTS
from training.v2_runtime import canonical_v2_checkpoint, v2_report_path, write_json


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_sovereignty_audit_v2() -> dict[str, object]:
    eval_summary = _load_json(v2_report_path("eval_summary")) or {}
    improvement = _load_json(v2_report_path("improvement_report")) or {}
    current_score = float(eval_summary.get("overall_score", 0.0) or 0.0)
    previous = _load_json(v2_report_path("audit_report")) or {}
    best_score = max(current_score, float(previous.get("best_score", 0.0) or 0.0))
    checkpoint = canonical_v2_checkpoint("ouroboros")
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("identity")
    if not checkpoint.exists():
        checkpoint = canonical_v2_checkpoint("brain")

    promoted = current_score >= float(previous.get("best_score", 0.0) or 0.0)
    best_checkpoint = checkpoint.parent / "best_v2_checkpoint.pt"
    if promoted and checkpoint.exists():
        shutil.copy2(checkpoint, best_checkpoint)
        try:
            DRIVE_V2_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_checkpoint, DRIVE_V2_CHECKPOINTS / best_checkpoint.name)
        except Exception:
            pass

    report = {
        "generated_at": time.time(),
        "stage": "sovereignty_audit_v2",
        "checkpoint": str(checkpoint),
        "current_score": current_score,
        "best_score": best_score,
        "promoted": promoted,
        "best_checkpoint": str(best_checkpoint) if promoted else str(previous.get("best_checkpoint", best_checkpoint)),
        "decision": "promote" if promoted else "hold",
        "recommendations": improvement.get("recommendations", []),
    }
    write_json(v2_report_path("audit_report"), report)
    return report


if __name__ == "__main__":
    print(json.dumps(run_sovereignty_audit_v2(), indent=2))
