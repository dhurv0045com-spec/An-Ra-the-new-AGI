from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from cognition.deliberation import (
    DELIBERATION_SCHEMA,
    DeliberationBudget,
    GenerationArtifact,
    VerificationDecision,
    VerifiedDeliberationController,
)


def test_controller_runs_full_sequence_and_persists_scoped_evidence() -> None:
    persisted = []
    generated = []

    def generate(_prompt, _understanding, _plan, _retrieval, ordinal, previous):
        generated.append((ordinal, previous is not None))
        return GenerationArtifact("5" if previous is None else "4", 1)

    controller = VerifiedDeliberationController(
        generate=generate,
        retrieve=lambda _query, _limit: ({"content": "2 + 2 is 4"},),
        verify=lambda _prompt, _understanding, artifact, _retrieval: VerificationDecision(
            artifact.text == "4",
            1.0 if artifact.text == "4" else 0.0,
            "exact_math",
            "exact arithmetic",
            "recalculate",
        ),
        persist=persisted.append,
    )
    result = controller.run(
        "What is 2 + 2?",
        budget=DeliberationBudget(revisions=1, verifier_calls=2),
    )

    assert result.schema == DELIBERATION_SCHEMA
    assert result.status == "accepted"
    assert result.answer == "4"
    assert generated == [(0, False), (1, True)]
    assert result.stages_completed == (
        "understand",
        "retrieve",
        "plan",
        "candidate",
        "verify",
        "revise",
        "select",
        "persist",
    )
    # The sink receives the complete decision first; only its return value can
    # establish whether the public trace may add the `persist` stage.
    assert persisted == [
        replace(result, stages_completed=result.stages_completed[:-1])
    ]


def test_controller_abstains_when_only_integrity_scope_fails() -> None:
    controller = VerifiedDeliberationController(
        generate=lambda *_args: GenerationArtifact("broken", 2),
        verify=lambda *_args: VerificationDecision(
            False,
            0.1,
            "integrity",
            "coherence only; not factual truth",
        ),
    )
    result = controller.run(
        "Explain something",
        budget=DeliberationBudget(revisions=0, verifier_calls=1),
    )
    assert result.status == "abstained"
    assert result.answer != "broken"
    assert result.public_evidence()["verification"]["scope"].endswith(
        "not factual truth"
    )


def test_hard_off_switch_runs_no_model_or_verifier() -> None:
    calls = []
    controller = VerifiedDeliberationController(
        generate=lambda *_args: calls.append("generate"),  # type: ignore[arg-type]
        verify=lambda *_args: calls.append("verify"),  # type: ignore[arg-type]
        enabled=False,
    )
    result = controller.run("hello", budget=DeliberationBudget())
    assert result.status == "disabled"
    assert calls == []
    assert result.generated_tokens == 0


def test_generated_token_budget_stops_extra_candidates() -> None:
    controller = VerifiedDeliberationController(
        generate=lambda *_args: GenerationArtifact("draft", 8),
        verify=lambda *_args: VerificationDecision(False, 0.0, "check", "test"),
    )
    result = controller.run(
        "hello",
        budget=DeliberationBudget(
            candidates=3,
            revisions=2,
            verifier_calls=5,
            max_generated_tokens=8,
        ),
    )
    assert len(result.candidates) == 1
    assert result.generated_tokens == 8


def test_local_runtime_retrieval_is_bounded_and_provenance_labelled() -> None:
    from runtime.sft_prototype import PrototypeRuntime

    runtime = PrototypeRuntime()
    runtime.add_turn("s", "The project codename is An-Ra.", "Understood.")
    runtime.add_turn("s", "Weather is unrelated.", "Okay.")
    hits = runtime.retrieve_session("s", "What is the project codename?", 1)

    assert len(hits) == 1
    assert hits[0]["source"] == "local_session_memory"
    assert hits[0]["trust"] == "user_provided_session_context"
    assert "An-Ra" in str(hits[0]["content"])


def test_sft_runtime_adapter_uses_symbolic_evidence_and_existing_ledger(
    monkeypatch,
) -> None:
    import runtime.sft_prototype as prototype

    recorded = []

    def fake_generate(*_args, **_kwargs):
        return SimpleNamespace(
            output="4",
            tokens_generated=1,
            prompt_tokens=12,
            time_ms=2.0,
            stopped_by="eos",
            quality_state="accepted",
            repeated_ngrams_detected=False,
            language_fragment_detected=False,
            entropy_curve=[0.2],
            subsystem_trace={
                "symbolic_verifier": {
                    "score": 1.0,
                    "reason": "symbolic_output_matched",
                }
            },
        )

    monkeypatch.setattr(prototype, "generate_traced", fake_generate)
    monkeypatch.setattr(
        prototype,
        "record_experience",
        lambda **kwargs: recorded.append(kwargs) or (kwargs["trace_id"], True),
    )
    body = prototype.ChatRequest(
        message="What is 2 + 2?",
        deliberation=prototype.DeliberationControls(mode="verified", revisions=0),
    )
    result = prototype._run_verified_deliberation(prototype.PrototypeRuntime(), body)

    assert result.status == "accepted"
    assert result.answer == "4"
    assert recorded[0]["kind"] == "verified_deliberation"
    assert recorded[0]["verifier_verdicts"][0]["scope"] == "exact symbolic answer"


def test_symbolic_evidence_cannot_approve_a_non_arithmetic_request() -> None:
    from runtime.sft_prototype import _verify_deliberation_artifact

    decision = _verify_deliberation_artifact(
        "What is the project codename?",
        SimpleNamespace(task_type="factual", needs_retrieval=True),
        GenerationArtifact(
            "The codename is unknown.",
            4,
            evidence={
                "trace": {
                    "quality_state": "accepted",
                    "repetition_detected": False,
                    "fragment_detected": False,
                },
                "symbolic": {"score": 1.0, "reason": "unrelated math matched"},
            },
        ),
        (),
    )

    assert decision.verifier == "session_retrieval_overlap"
    assert decision.passed is False


def test_failed_persistence_is_not_reported_as_persisted() -> None:
    controller = VerifiedDeliberationController(
        generate=lambda *_args: GenerationArtifact("ok", 1),
        verify=lambda *_args: VerificationDecision(True, 1.0, "test", "test scope"),
        persist=lambda _result: False,
    )
    result = controller.run("hello", budget=DeliberationBudget())

    assert "persist" not in result.stages_completed
    assert result.stages_completed[-1] == "persistence_failed"
