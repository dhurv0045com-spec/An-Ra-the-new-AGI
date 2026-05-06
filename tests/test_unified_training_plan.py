from __future__ import annotations

from training.train_unified import stage_plan_for_mode


def test_train_mode_runs_base_before_milestone_layers() -> None:
    assert stage_plan_for_mode("train") == [
        "base",
        "identity",
        "ouroboros",
        "self_improvement",
        "sovereignty_audit",
        "tests",
    ]


def test_session_mode_stays_daily_base_only() -> None:
    assert stage_plan_for_mode("session") == ["base"]
