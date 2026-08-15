from __future__ import annotations

from inference.reasoning_budget import (
    AdaptiveReasoningPolicy,
    ReasoningSignals,
    estimate_prompt_difficulty,
    plan_for_prompt,
)


def test_easy_high_competence_request_uses_direct_budget() -> None:
    plan = AdaptiveReasoningPolicy().plan(
        ReasoningSignals("general", 8, 0.05, 0.95, True, True)
    )
    assert plan.mode == "direct"
    assert plan.candidate_count == 1
    assert plan.verifier_calls == 0
    assert plan.maximum_extra_tokens == 128


def test_hard_low_competence_request_allocates_bounded_search() -> None:
    plan = plan_for_prompt(
        "Prove and verify this multi-step counterfactual constraint -> result",
        competence=0.1,
        verifier_available=True,
        retrieval_available=True,
        owner_token_cap=300,
    )
    assert plan.mode in {"retrieve_decompose", "search_verify"}
    assert 1 <= plan.candidate_count <= 4
    assert plan.maximum_extra_tokens <= 300
    assert not plan.blocked_requirements


def test_missing_services_are_explicit_and_irreversible_work_is_blocked() -> None:
    plan = AdaptiveReasoningPolicy().plan(
        ReasoningSignals("deployment", 100, 0.9, 0.1, False, False, True)
    )
    assert plan.verifier_calls == 0
    assert plan.revision_count == 0
    assert "verifier_unavailable" in plan.blocked_requirements
    assert "irreversible_action_has_no_verifier" in plan.blocked_requirements


def test_difficulty_estimate_is_monotonic_for_explicit_reasoning_signals() -> None:
    easy = estimate_prompt_difficulty("Say hello")
    hard = estimate_prompt_difficulty("Prove, derive, compare, debug, and verify this theorem")
    assert hard > easy
