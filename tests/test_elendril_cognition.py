import json
from dataclasses import asdict, replace

import pytest

from anra_v5.elendril.cognition import (
    Budget, CognitiveState, ExperienceEvent, MemoryVersion, OperationCandidate,
    PublicEnvironment, belief, run_episode, verify,
)
from anra_v5.elendril.cognitive_demo import FixedPolicy, revision_cases, route_cases, run_demo


def test_public_evidence_preserves_time_missing_null_and_conflicts():
    case = revision_cases()[2]
    state = CognitiveState(case.goal)
    assert belief(state.events, "A").status == "missing"
    state = state.observe(case.environment.initial * 2)
    assert len(state.events) == len(case.environment.initial)
    assert belief(state.events, "A").status == "contradicted"
    revised = state.observe(case.environment.channels["request-current-status"])
    assert belief(revised.events, "A").status == "null"
    assert belief(state.events, "A").status == "contradicted"
    foreign = ExperienceEvent("other-world", "A", "operator", 10, "B")
    with pytest.raises(ValueError, match="outside goal custody"):
        revised.observe((foreign,))
    orphan = ExperienceEvent(case.goal.scope, "A", "operator", 10, "B", ("a"*64,))
    with pytest.raises(ValueError, match="ancestor"):
        revised.observe((orphan,))


def test_memory_admission_is_bound_and_never_serializes_outcome_labels():
    prior, target = route_cases()[0]
    trace = run_episode(prior.goal, prior.environment, FixedPolicy())
    outcome = verify(trace, prior.truth)
    empty = MemoryVersion()
    with pytest.raises(ValueError, match="does not bind"):
        empty.consolidate(trace, replace(outcome, trace_sha256="b"*64))
    no_admission = empty.consolidate(trace, replace(outcome, admitted_public=()))
    assert not no_admission.events
    saved = empty.consolidate(trace, outcome)
    assert len(saved.events) == 1
    record = json.loads(json.dumps(asdict(saved)))
    assert set(record) == {"events", "parent", "capacity"}
    assert set(record["events"][0]) == {"scope", "subject", "source", "tick", "value", "parents"}
    restored = MemoryVersion.from_record(record, saved.identity)
    assert restored.recall(target.goal) == restored.events
    assert not restored.recall(route_cases()[1][1].goal)
    with pytest.raises(ValueError, match="identity"):
        MemoryVersion.from_record(record, "0"*64)


def test_fixed_positive_control_two_families_and_baselines():
    first = run_demo(FixedPolicy())
    assert first == run_demo(FixedPolicy())
    for arm, expected in (("route/memory", 4), ("route/no-memory", 0),
                          ("route/same-token-extra-context", 0), ("revision/public-query", 4)):
        assert first["totals"][arm]["episodes"] == 4
        assert first["totals"][arm]["correct"] == expected
    for row in first["episodes"]:
        assert all(step["origin"] == "fixed_baseline" for step in row["trace"]["decisions"])


def test_invalid_query_or_budget_cannot_bypass_controller():
    case = revision_cases()[0]
    trace = run_episode(case.goal, case.environment, FixedPolicy(), budget=Budget(decisions=0))
    assert trace.status == "decision_budget" and not trace.decisions
    assert not verify(trace, case.truth).correct

    class Forbidden(FixedPolicy):
        def propose(self, state, legal, channels, positions):
            assert not hasattr(state, "truth")
            assert channels == ("request-current-status",)
            return OperationCandidate("QUERY", "unavailable")

    denied = run_episode(case.goal, case.environment, Forbidden())
    assert denied.status == "invalid_action" and denied.requests == 0
    assert denied.state.events == case.environment.initial
    with pytest.raises(ValueError, match="episode identity"):
        verify(denied, replace(case.truth, episode="different"))
