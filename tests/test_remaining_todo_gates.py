from __future__ import annotations

from training.moonshot_pilots import evaluate_moonshot_pilot
from training.post_training_ablations import evaluate_post_training_ablations
from training.recovery_drill import run_kill_recovery_drill


def test_kill_recovery_drill_restores_a_checkpoint_boundary(tmp_path) -> None:
    report = run_kill_recovery_drill(tmp_path / "recovery")
    assert report["passed"] is True
    assert report["gates"]["recovered_step"] is True  # type: ignore[index]


def test_post_training_ablation_gate_requires_all_five_methods() -> None:
    reports = {
        stage: {
            "baseline_score": 0.5,
            "candidate_score": 0.6,
            "protected_regression": 0.01,
            "ablation_score": 0.5,
        }
        for stage in ("sft", "rlvr", "star", "dpo", "self_distillation")
    }
    assert evaluate_post_training_ablations(reports)["passed"] is True
    assert evaluate_post_training_ablations({})["passed"] is False
    reports["dpo"]["ablation_score"] = 0.7
    rejected = evaluate_post_training_ablations(reports)
    assert rejected["passed"] is False
    assert rejected["stages"]["dpo"]["ablation_supports_method"] is False  # type: ignore[index]


def test_moonshot_pilots_are_fail_closed_on_missing_or_bad_metrics() -> None:
    passed = evaluate_moonshot_pilot(
        "m1",
        {
            "short_context_ratio": 0.99,
            "long_context_speedup": 1.6,
            "model_parameters": 150_000_000,
            "seed_count": 3,
        },
    )
    assert passed["passed"] is True
    assert evaluate_moonshot_pilot("m5", {})["passed"] is False
    assert evaluate_moonshot_pilot(
        "m3",
        {
            "reasoning_score_ratio": 1.05,
            "inference_flops_ratio": 1.0,
            "model_parameters": 150_000_000,
            "seed_count": 3,
        },
    )["passed"] is False
    assert evaluate_moonshot_pilot(
        "m5", {"training_pairs": 19_999, "recall_at_5_gain": 1.0}
    )["passed"] is False
    assert evaluate_moonshot_pilot(
        "m6", {"proof_cases": 100, "deterministic_pass_rate": float("inf")}
    )["passed"] is False
