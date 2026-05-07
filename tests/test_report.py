import json

from engine import report
from runtime.system_registry import component_registry


def test_build_report_has_all_keys(monkeypatch):
    monkeypatch.setattr(report, "build_system_manifest", lambda: _manifest())

    data = report.build_report()

    assert {"generated_at", "system_health", "components", "performance", "recent_failures"} <= set(data)
    assert data["system_health"]["total_components"] == 1


def test_save_report_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "build_system_manifest", lambda: _manifest())

    path = report.save_report(tmp_path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["components"][0]["name"] == "brain"


def test_report_component_count_matches_registry():
    data = report.build_report()

    assert data["system_health"]["total_components"] == len(component_registry())


def _manifest():
    return {
        "components": [
            {
                "name": "brain",
                "source_ok": True,
                "import_status": "ok",
                "missing": [],
                "metric_hooks": ["latency_ms", "success", "error_type"],
            }
        ],
        "training_readiness": {},
        "artifacts": {},
    }
