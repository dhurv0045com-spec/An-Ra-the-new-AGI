from __future__ import annotations

from training.foundation_campaign import (
    ARCHITECTURE_PILOT_TOKENS,
    evaluate_foundation_milestone,
    plan_foundation_window,
    compare_architecture_pilot,
)


def test_window_is_bounded_and_stops_at_milestone() -> None:
    window = plan_foundation_window(
        tokens_seen=190_000_000,
        tokens_per_second=10_000,
        session_budget_minutes=600,
    )
    assert window is not None
    assert window.start_token == 190_000_000
    assert window.end_token == 200_000_000

    regular = plan_foundation_window(
        tokens_seen=200_000_000,
        tokens_per_second=100_000,
        session_budget_minutes=600,
    )
    assert regular is not None
    assert regular.target_tokens == 170_000_000


def test_milestone_requires_durability_and_each_source() -> None:
    evidence = {
        "tokens_seen": 200_000_000,
        "durability_state": "protected",
        "numerically_stable": True,
        "duplicate_windows": 0,
        "validation": {
            "validation_identity": "abc",
            "domain_losses": {"prose": {"loss": 2.0}, "code": {"loss": 2.2}},
        },
        "behavior": {
            key: {"score": 0.1}
            for key in (
                "generation_noncollapse",
                "copy",
                "uncertainty",
                "reasoning",
                "math",
                "code",
                "context_use",
            )
        },
    }
    assert evaluate_foundation_milestone(evidence, target_tokens=200_000_000)["passed"]
    del evidence["validation"]["domain_losses"]
    assert not evaluate_foundation_milestone(evidence, target_tokens=200_000_000)["passed"]


def test_architecture_pilot_promotes_only_a_matched_useful_win() -> None:
    shared = {
        "parent_checkpoint_sha256": "parent",
        "window_id": "window",
        "seed": 1301,
        "training_tokens": ARCHITECTURE_PILOT_TOKENS,
    }
    baseline = {**shared, "capability_score": 0.50, "tokens_per_second": 1000.0}
    candidate = {
        **shared,
        "capability_score": 0.52,
        "tokens_per_second": 900.0,
        "domain_regressions": {"prose": 0.0, "code": 0.01},
        "numerically_stable": True,
        "oom": False,
    }
    assert compare_architecture_pilot(baseline, candidate)["decision"] == "promote"
    candidate["capability_score"] = 0.505
    assert compare_architecture_pilot(baseline, candidate)["decision"] == "replicate_once"

