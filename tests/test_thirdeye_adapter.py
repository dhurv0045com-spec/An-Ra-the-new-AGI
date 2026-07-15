from __future__ import annotations

import sqlite3

from evaluation.thirdeye_adapter import (
    PROJECT_ID,
    activation_snapshot,
    feature_specs,
    run_one_click,
)
from scripts.show_thirdeye_summary import render_summary
from training.v2_runtime import build_model_for_profile, model_summary


def test_reference_model_and_activation_probes() -> None:
    model = build_model_for_profile("anra-v4-180m")

    assert model_summary(model)["parameters"] == 181_132_071
    snapshot = activation_snapshot(model)
    assert snapshot["anra.esv"] is True
    assert snapshot["anra.rim"] is False
    assert snapshot["anra.dstp"] is False
    assert snapshot["anra.hal"] is False


def test_feature_registry_is_hierarchical() -> None:
    features = {feature.feature_id: feature for feature in feature_specs()}

    assert "anra.hal" in features
    assert features["anra.hal.attention_temperature"].parent_feature_id == "anra.hal"
    assert features["anra.hal.memory_threshold"].requires_retraining is False


def test_one_click_generates_report_bundle(tmp_path) -> None:
    result = run_one_click(profile="quick", home=tmp_path)

    assert result["project"]["project_id"] == PROJECT_ID
    assert len(result["features"]) >= 10
    assert len(result["recommended_experiments"]) > 0
    for report in result["report_paths"].values():
        assert __import__("pathlib").Path(report).exists()


def test_one_click_falls_back_for_colab_sqlite_unixepoch(monkeypatch, tmp_path) -> None:
    import evaluation.thirdeye_adapter as adapter

    def broken_register_project(home):
        del home
        raise sqlite3.OperationalError("unknown function: unixepoch()")

    monkeypatch.setattr(adapter, "register_project", broken_register_project)
    result = run_one_click(profile="quick", home=tmp_path)

    assert result["project"]["project_id"] == PROJECT_ID
    assert result["fallback"]["error_type"] == "OperationalError"
    assert "fallback" in result["report_paths"]
    assert __import__("pathlib").Path(result["report_paths"]["fallback"]).exists()


def test_colab_thirdeye_summary_is_visible(tmp_path) -> None:
    result = {
        "project": {"project_id": PROJECT_ID},
        "profile": "quick",
        "features": [{"feature_id": "anra.optimizer"}, {"feature_id": "anra.hal"}],
        "recommended_experiments": [
            {
                "feature_id": "anra.hal",
                "protocol": "system_audit",
                "reason": "No current activation evidence.",
            }
        ],
        "activation_snapshot": {"anra.optimizer": True, "anra.hal": False},
        "report_paths": {"fallback": str(tmp_path / "one_click_fallback.json")},
    }

    text = render_summary(result, intelligence_path=tmp_path / "missing_intelligence.json")

    assert "THIRD EYE EVIDENCE DASHBOARD" in text
    assert "Feature Activation" in text
    assert "OK   anra.optimizer" in text
    assert "MISS anra.hal" in text
    assert "Subsystem Intelligence" in text
    assert "not found yet" in text
