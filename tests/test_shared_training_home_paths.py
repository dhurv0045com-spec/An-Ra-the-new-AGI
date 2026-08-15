from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_shared_training_home_routes_logs_and_sessions(monkeypatch, tmp_path) -> None:
    shared = tmp_path / "ANRA_T4_TRAINING_HOME"
    monkeypatch.setenv("ANRA_SHARED_CHECKPOINT_DIR", str(shared))
    module_path = Path(__file__).resolve().parents[1] / "anra" / "anra_paths.py"
    spec = importlib.util.spec_from_file_location("_test_anra_paths", module_path)
    assert spec is not None and spec.loader is not None
    paths = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = paths
    spec.loader.exec_module(paths)
    try:
        assert shared / "logs" == paths.DRIVE_LOGS
        assert shared / "logs" / "scorecards" == paths.DRIVE_SCORECARD
        assert shared / "sessions" == paths.DRIVE_SESSIONS
        assert shared != paths.DRIVE_DIR
    finally:
        monkeypatch.delenv("ANRA_SHARED_CHECKPOINT_DIR", raising=False)
        sys.modules.pop(spec.name, None)
