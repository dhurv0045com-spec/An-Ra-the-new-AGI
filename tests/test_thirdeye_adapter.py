from __future__ import annotations

import sqlite3

from evaluation.thirdeye_adapter import (
    PROJECT_ID,
    activation_snapshot,
    feature_specs,
    run_one_click,
)
from training.v2_runtime import build_frontier_model, model_summary


def test_reference_model_and_activation_probes() -> None:
    model = build_frontier_model()

    assert 850_000_000 <= model_summary(model)["parameters"] <= 1_000_000_000
    snapshot = activation_snapshot(model)
    assert snapshot["anra.esv"] is True
    assert snapshot["anra.rim"] is True
    assert snapshot["anra.dstp"] is True
    assert snapshot["anra.hal"] is True


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
