from __future__ import annotations

import json

import pytest

from training.train_unified import (
    checkpoint_resume_path,
    resolve_campaign_inventory,
    run_report_path,
    stage_plan_for_mode,
)
from training.v2_runtime import v2_report_path


def test_train_mode_runs_base_before_milestone_layers() -> None:
    assert stage_plan_for_mode("train") == [
        "base",
        "evaluation",
        "sovereignty_audit",
        "tests",
    ]


def test_session_mode_stays_daily_base_only() -> None:
    assert stage_plan_for_mode("session") == ["base"]


def test_eval_mode_runs_eval_only() -> None:
    assert stage_plan_for_mode("eval") == ["eval"]


def test_scratch_launch_never_becomes_a_resume_path() -> None:
    assert checkpoint_resume_path("scratch") is None
    assert checkpoint_resume_path(" SCRATCH ") is None
    assert checkpoint_resume_path("") is None
    assert checkpoint_resume_path("checkpoints/base.pt") == "checkpoints/base.pt"


def test_signed_worker_report_is_artifact_local(tmp_path) -> None:
    artifact = tmp_path / "cells" / "p050-baseline" / "seed-1301.pt"
    report = {"launch_manifest": {"artifact_path": str(artifact)}}

    assert run_report_path(report) == artifact.with_suffix(".run.json")


def test_signed_worker_metrics_can_be_isolated(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANRA_RUN_REPORT_DIR", str(tmp_path / "seed-1301.reports"))

    assert v2_report_path("metrics") == (
        tmp_path / "seed-1301.reports" / "v2_session_train_metrics.json"
    )


def test_signed_pilot_inventory_is_bound_to_its_train_manifest(tmp_path) -> None:
    signed_train = tmp_path / "manifest.json"
    signed_train.write_text(json.dumps({"total_tokens": 1234}), encoding="utf-8")
    signed_validation = tmp_path / "validation.json"
    signed_validation.write_text(json.dumps({"total_tokens": 99}), encoding="utf-8")
    global_inventory = tmp_path / "global.json"
    global_inventory.write_text(
        json.dumps({"licensed_tokens": 9999}), encoding="utf-8"
    )

    inventory = resolve_campaign_inventory(
        {
            "data_manifests": [str(signed_validation), str(signed_train)],
            "data_manifest_roles": {
                str(signed_validation): "validation",
                str(signed_train): "train",
            },
        },
        "anra-v4-180m",
        global_inventory,
    )

    assert inventory is not None
    assert inventory["licensed_tokens"] == 1234
    assert inventory["manifest"] == str(signed_train)
    assert inventory["validation_manifest"] == str(signed_validation.resolve())


def test_signed_pilot_inventory_fails_closed_on_empty_manifest(tmp_path) -> None:
    signed_train = tmp_path / "manifest.json"
    signed_train.write_text(json.dumps({"total_tokens": 0}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="has no tokens"):
        resolve_campaign_inventory(
            {
                "data_manifests": [str(signed_train), str(tmp_path / "validation.json")],
                "data_manifest_roles": {
                    str(signed_train): "train",
                    str(tmp_path / "validation.json"): "validation",
                },
            },
            "anra-v4-180m",
            tmp_path / "unused.json",
        )
