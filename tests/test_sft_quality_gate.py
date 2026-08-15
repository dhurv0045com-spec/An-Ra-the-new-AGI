from evaluation.sft_behavior_gate import check_smoke_response


def test_objective_smoke_categories_require_correct_behavior() -> None:
    assert check_smoke_response("mathematics", "17 + 28 = 45")[0]
    assert not check_smoke_response("mathematics", "This is a math problem.")[0]
    assert check_smoke_response("code", "def add(a, b):\n    return a + b")[0]
    assert not check_smoke_response("code", "Hello! How can I help?")[0]
    assert check_smoke_response("tool_contracts", '{"status": "success"}')[0]
    assert not check_smoke_response("tool_contracts", "success")[0]


def test_language_smoke_categories_are_not_nonempty_only() -> None:
    assert check_smoke_response("correction", "The results were not consistent.")[0]
    assert not check_smoke_response("correction", "Looks good.")[0]
    assert check_smoke_response(
        "uncertainty", "There is not enough evidence, so I am uncertain."
    )[0]
