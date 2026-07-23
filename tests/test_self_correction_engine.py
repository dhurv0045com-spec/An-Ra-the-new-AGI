from __future__ import annotations

import pytest

from cognition.self_correction import (
    CorrectionBudget,
    SelfCorrectionEngine,
    Verification,
)


def test_self_correction_revises_until_verified_and_persists_evidence() -> None:
    persisted = []

    def verify(_prompt, answer, _context):
        correct = answer == "4"
        return Verification(
            verified=correct,
            score=1.0 if correct else 0.0,
            verifier="integer-arithmetic",
            evidence={"executed": True},
            feedback="calculate again" if not correct else "exact",
        )

    engine = SelfCorrectionEngine(
        generate=lambda *_args: "5",
        verify=verify,
        retrieve=lambda _query, _limit: ({"source": "math", "fact": "2+2"},),
        plan=lambda _prompt, _context: "calculate and verify",
        revise=lambda _prompt, _answer, _verification, _revision: "4",
        persist=persisted.append,
    )
    result = engine.run(
        "What is 2 + 2?",
        budget=CorrectionBudget(
            candidates=1,
            revisions=1,
            retrieval_queries=1,
            verifier_calls=2,
        ),
    )
    assert result.status == "verified"
    assert result.answer == "4"
    assert result.verifier_calls == 2
    assert len(result.candidates) == 2
    assert persisted == [result]


def test_self_correction_abstains_instead_of_claiming_unverified_success() -> None:
    engine = SelfCorrectionEngine(
        generate=lambda *_args: "guess",
        verify=lambda *_args: Verification(False, 0.2, "grounding", {"matched": False}),
    )
    result = engine.run(
        "unknown",
        budget=CorrectionBudget(candidates=1, revisions=0, verifier_calls=1),
    )
    assert result.status == "abstained"
    assert result.answer != "guess"


def test_required_verifier_fails_closed() -> None:
    engine = SelfCorrectionEngine(generate=lambda *_args: "answer", verify=None)
    with pytest.raises(RuntimeError, match="verifier is unavailable"):
        engine.run("question", budget=CorrectionBudget())

