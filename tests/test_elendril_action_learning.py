"""The positive controls qualify supervision/episode mechanics, not cognition."""

from dataclasses import replace

import pytest

from anra_v5.elendril.action_learning import (
    ACTION_GRAMMAR, CHANNEL, DEFAULT_BUDGET, KINDS, ActionDataset, ControlPolicy, DecodedAction,
    NeuralActionPolicy, action_fixture, action_tokens, assert_disjoint,
    curriculum_indices, evaluate_policy, render_action, supervision,
)
from anra_v5.elendril.codec import BOS, CODEC_SHA, EOS, encode
from anra_v5.elendril.cognition import (
    Budget, CognitiveState, ExperienceEvent, PublicEnvironment, Truth, run_episode,
)


@pytest.fixture(scope="module")
def datasets():
    return tuple(action_fixture(split, worlds=10) for split in ("train", "validation", "test"))


def test_action_worlds_are_deterministic_disjoint_and_legacy_framed(datasets):
    train, validation, test = datasets
    assert_disjoint(*datasets)
    assert train == action_fixture("train", worlds=10)
    assert train.identity == action_fixture("train", worlds=10).identity
    assert len({data.identity for data in datasets}) == 3
    for data in datasets:
        assert data.manifest["tokenizer_sha256"] == CODEC_SHA
        assert data.manifest["action_grammar"] == ACTION_GRAMMAR
        assert data.manifest["kinds"] == dict.fromkeys(KINDS, 2)
        assert data.manifest["max_sequence_tokens"] <= 64
        for row in data.examples:
            start, end = row.target_span
            assert row.tokens == (BOS, *encode(row.prompt), *row.target_tokens)
            assert row.tokens[-1] == EOS
            assert not any(row.eligible[:start])
            assert all(row.eligible[start:end]) and sum(row.eligible) == 2
            assert row.world_id not in row.prompt
            assert row.kind not in row.prompt
    with pytest.raises(ValueError, match="contamination"):
        assert_disjoint(train, train)
    duplicate = ActionDataset(train.split, train.seed, train.cases,
                              (*train.examples, train.examples[0]))
    with pytest.raises(ValueError, match="contamination"):
        assert_disjoint(duplicate)
    for options in ({"split": "sealed"}, {"worlds": 0}, {"worlds": True}, {"worlds": 11}, {"seed": -1}):
        with pytest.raises(ValueError):
            action_fixture(**options)


def test_truth_and_unobserved_query_payload_do_not_enter_supervision(datasets):
    case = next(case for case in datasets[0].cases if case.kind == "missing")
    rows = supervision(case)
    assert rows[0].target_tokens == action_tokens("QUERY", CHANNEL)
    state = CognitiveState(case.goal).observe(case.environment.initial)
    prompt = render_action(state, tuple(case.environment.channels))
    assert prompt == rows[0].prompt
    # A malicious evaluation label changes neither teacher input nor labels.
    lie = replace(case, truth=Truth(case.goal.episode, "P", ()))
    assert supervision(lie) == rows
    # A different channel outcome is hidden until observation, then honestly
    # becomes a different public answer target and prompt.
    answer = "P" if case.truth.expected != "P" else "A"
    updated = tuple(replace(event, value=answer)
                    for event in case.environment.channels[CHANNEL])
    counterfactual = replace(case, environment=PublicEnvironment(
        case.environment.initial, {CHANNEL: updated}))
    counterfactual_rows = supervision(counterfactual)
    assert counterfactual_rows[0] == rows[0]
    assert counterfactual_rows[1].target_tokens == action_tokens("ANSWER", answer)
    assert counterfactual_rows[1].prompt != rows[1].prompt
    assert counterfactual_rows[1].action == "ANSWER"


def test_fixed_oracle_and_negative_controls_score_complete_costed_episodes(datasets):
    data = datasets[1]
    fixed = evaluate_policy(data, ControlPolicy())
    always = evaluate_policy(data, ControlPolicy("always-query"))
    noquery = evaluate_policy(data, ControlPolicy("no-query"))
    oracle = evaluate_policy(data, "oracle")
    assert fixed["totals"]["all"]["correct"] == 10
    assert fixed["totals"]["all"]["requests"] == 4
    assert fixed["totals"]["all"]["utility"] == pytest.approx(9.6)
    assert always["totals"]["all"]["correct"] == 10
    assert always["totals"]["all"]["requests"] == 8
    assert always["totals"]["all"]["utility"] == pytest.approx(9.2)
    assert noquery["totals"]["all"]["correct"] < 10
    assert noquery["totals"]["all"]["requests"] == 0
    assert oracle["totals"]["all"]["correct"] == 10
    assert oracle["totals"]["all"]["requests"] == 0
    assert all(step["origin"] == "privileged_oracle"
               for row in oracle["episodes"] for step in row["trace"]["decisions"])
    for mode in ("random", "query-blind", "recency"):
        result = evaluate_policy(data, ControlPolicy(mode))
        assert result == evaluate_policy(data, ControlPolicy(mode))
        assert result["totals"]["all"]["correct"] < 10
        assert len(result["episodes"]) == 10
    with pytest.raises(ValueError, match="privileged"):
        ControlPolicy("fixed", oracle=data.cases[0].truth)


def test_available_query_is_legal_but_costly_on_known_and_null_worlds(datasets):
    for kind, operation in (("known", "ANSWER"), ("null", "ABSTAIN")):
        case = next(case for case in datasets[0].cases if case.kind == kind)
        direct = run_episode(case.goal, case.environment, ControlPolicy())
        redundant = run_episode(case.goal, case.environment, ControlPolicy("always-query"))
        assert direct.decisions[0].operation == operation
        assert direct.requests == 0
        assert redundant.decisions[0].operation == "QUERY"
        assert redundant.requests == 1
        assert direct.answer == redundant.answer


def test_raw_neural_adapter_requires_eos_and_preserves_invalid_attempts(datasets):
    case = next(case for case in datasets[0].cases if case.kind == "known")
    expected_prompt = supervision(case)[0].prompt
    returned = action_tokens("ANSWER", case.truth.expected)
    def decode(prompt, remaining):
        assert prompt == expected_prompt
        assert remaining == 384
        return DecodedAction(returned, 2 * (len(encode(prompt)) + 1) + 1, 2, 0.75)
    policy = NeuralActionPolicy(decode, "test-only-decoder-stub")
    trace = run_episode(case.goal, case.environment, policy, budget=DEFAULT_BUDGET)
    assert trace.answer == case.truth.expected
    assert trace.decisions[0].origin == "model"
    assert trace.decisions[0].raw_tokens == returned
    # Stub here verifies the adapter only; it is never reported as a trained result.
    malformed = NeuralActionPolicy(lambda *_: DecodedAction(returned[:-1], 1, 1), "stub")
    rejected = run_episode(case.goal, case.environment, malformed)
    assert rejected.status == "invalid_action"
    assert rejected.decisions[0].raw_tokens == returned[:-1]
    unavailable = next(case for case in datasets[0].cases if case.kind == "unavailable")
    querying = NeuralActionPolicy(lambda *_: DecodedAction(action_tokens("QUERY", CHANNEL), 1, 1), "stub")
    denied = run_episode(unavailable.goal, unavailable.environment, querying)
    assert denied.status == "invalid_action" and denied.requests == 0
    overspending = NeuralActionPolicy(lambda *_: DecodedAction(returned, 2, 1), "stub")
    exhausted = run_episode(case.goal, case.environment, overspending, budget=Budget(model_positions=1))
    assert exhausted.status == "invalid_action"


def test_channel_metadata_and_event_order_controls_cannot_rewrite_truth(datasets):
    case = next(case for case in datasets[0].cases if case.kind == "known")
    state = CognitiveState(case.goal).observe(case.environment.initial)
    backwards = CognitiveState(case.goal).observe(tuple(reversed(case.environment.initial)))
    assert render_action(state, ()) == render_action(backwards, ())
    assert render_action(state, ()) != render_action(state, (CHANNEL,))
    with pytest.raises(ValueError, match="declared"):
        render_action(state, ("secret-answer-channel",))
    with pytest.raises(ValueError):
        action_tokens("ANSWER", None)
    with pytest.raises(ValueError):
        action_tokens("ANSWER", "AB")
    with pytest.raises(ValueError):
        action_tokens("QUERY", "wrong")


def test_curriculum_has_identical_multiset_and_prospective_stage(datasets):
    data = datasets[0]
    fixed = curriculum_indices(data, seed=7, arm="fixed-mixture")
    candidate = curriculum_indices(data, seed=7, arm="uncertainty-first")
    assert sorted(fixed) == sorted(candidate) == list(range(len(data.examples)))
    uncertain = [data.examples[index].kind in {"missing", "conflict"} for index in candidate]
    assert uncertain == sorted(uncertain, reverse=True)
    assert curriculum_indices(data, seed=7, arm="fixed-mixture", epoch=1) == curriculum_indices(
        data, seed=7, arm="uncertainty-first", epoch=1)
    with pytest.raises(ValueError):
        curriculum_indices(data, seed=7, arm="posthoc-best")
