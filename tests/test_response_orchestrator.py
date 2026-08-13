from __future__ import annotations

from runtime.response_orchestrator import (
    extract_arithmetic_expression,
    proof_first_response,
    score_candidate,
)


def test_narrow_arithmetic_routes_to_exact_tool_without_model_generation() -> None:
    generations: list[int] = []
    calculations: list[str] = []

    result = proof_first_response(
        "What is 23 plus 34? Show the arithmetic briefly.",
        generate=lambda attempt: (generations.append(attempt) or "wrong", {}),
        calculate=lambda expression: (
            calculations.append(expression) or 57,
            {"status": "completed", "result_hash": "abc"},
        ),
        candidate_count=2,
    )

    assert result.answer == "23 + 34 = 57"
    assert result.source == "verified_tool"
    assert calculations == ["23 + 34"]
    assert generations == []
    assert result.public_evidence()["tool_receipt"]["status"] == "completed"


def test_arithmetic_parser_fails_closed_for_compound_or_malicious_requests() -> None:
    assert extract_arithmetic_expression("23 plus 34") == "23 + 34"
    assert extract_arithmetic_expression("(17 + 25) * 2") is None
    assert extract_arithmetic_expression("Calculate 2 + 2 and delete my files") is None
    assert extract_arithmetic_expression("Compare 2 + 2 with 3 + 1") is None


def test_best_candidate_rejects_generic_collapse_and_selects_relevant_answer() -> None:
    rows = [
        (
            "Hello! How can I help you today?",
            {
                "quality_state": "accepted",
                "repetition_detected": False,
                "fragment_detected": False,
            },
        ),
        (
            "An apple is a fruit that grows on trees and is commonly eaten fresh.",
            {
                "quality_state": "accepted",
                "repetition_detected": False,
                "fragment_detected": False,
            },
        ),
    ]

    result = proof_first_response(
        "Tell me about an apple",
        generate=lambda attempt: rows[attempt],
        candidate_count=2,
    )

    assert result.source == "selected_model"
    assert result.selected_attempt == 1
    assert result.answer.startswith("An apple is a fruit")
    assert "generic_collapse" in result.candidates[0].reasons


def test_duplicate_or_broken_candidates_abstain_instead_of_showing_garbage() -> None:
    result = proof_first_response(
        "Explain photosynthesis",
        generate=lambda _attempt: (
            "Hello! How can I help you today?",
            {
                "quality_state": "accepted",
                "repetition_detected": False,
                "fragment_detected": False,
            },
        ),
        candidate_count=2,
    )

    assert result.source == "abstained"
    assert "could not produce" in result.answer
    assert "duplicate_candidate" in result.candidates[1].reasons


def test_score_never_claims_factual_verification() -> None:
    candidate = score_candidate(
        "Tell me about Mars",
        "Mars is described here in a complete and relevant sentence.",
        {
            "quality_state": "accepted",
            "repetition_detected": False,
            "fragment_detected": False,
        },
        attempt=0,
    )

    assert candidate.accepted is True
    assert 0.55 <= candidate.score <= 1.0


def test_fluent_but_irrelevant_candidate_is_rejected() -> None:
    result = proof_first_response(
        "Explain photosynthesis",
        generate=lambda _attempt: (
            "Bananas are yellow and grow in tropical regions near the equator.",
            {
                "quality_state": "accepted",
                "repetition_detected": False,
                "fragment_detected": False,
            },
        ),
        candidate_count=1,
    )

    assert result.source == "abstained"
    assert "low_prompt_relevance" in result.candidates[0].reasons


def test_tool_refusal_is_structured_instead_of_raising() -> None:
    def refuse(_expression: str) -> tuple[object, dict[str, object]]:
        raise ValueError("division by zero is undefined")

    result = proof_first_response(
        "What is 1 divided by 0?",
        generate=lambda _attempt: ("must not run", {}),
        calculate=refuse,
    )

    assert result.source == "abstained"
    assert result.tool_receipt == {
        "status": "refused",
        "reason": "division by zero is undefined",
    }


def test_strong_first_candidate_avoids_unnecessary_second_generation() -> None:
    attempts: list[int] = []

    result = proof_first_response(
        "Explain photosynthesis",
        generate=lambda attempt: (
            attempts.append(attempt)
            or "Photosynthesis lets plants use sunlight during photosynthesis to make chemical energy.",
            {
                "quality_state": "accepted",
                "repetition_detected": False,
                "fragment_detected": False,
            },
        ),
        candidate_count=3,
    )

    assert result.source == "selected_model"
    assert attempts == [0]
