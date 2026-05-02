from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from anra_paths import OUTPUT_V2_DIR, STATE_DIR, V3_TOKENIZER_FILE, get_dataset_file, get_v2_checkpoint


def _file_info(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)) if path.exists() else None,
    }


def build_session_report() -> dict[str, object]:
    metrics_path = OUTPUT_V2_DIR / "reports" / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metrics = {"load_error": f"[session_dashboard] metrics load failed: {exc}"}
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dataset": _file_info(get_dataset_file()),
        "tokenizer": _file_info(V3_TOKENIZER_FILE),
        "checkpoint": _file_info(get_v2_checkpoint("brain")),
        "state_dir": _file_info(STATE_DIR),
        "latest_metrics": metrics,
    }


def print_session_dashboard() -> None:
    report = build_session_report()
    print("=" * 64)
    print("AN-RA SESSION DASHBOARD")
    print("=" * 64)
    print(json.dumps(report, indent=2))
    print("=" * 64)


if __name__ == "__main__":
    print_session_dashboard()
