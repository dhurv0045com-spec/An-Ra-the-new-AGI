"""V7.15 matched stale-memory correction and removals; developer-only diagnostics."""
from dataclasses import replace

from .codec import canonical, digest
from .cognition import CognitiveState, MemoryVersion, Truth
from .v6_data import SYMBOLS, _supervision, case_to_record, public_action
from .v71_data import build_suite as v71_suite, _admitted, assert_training_disjoint

VARIANTS = (
    "single", "consistent-two", "stale-corrected",
    "reversed-stale-corrected", "removed", "irrelevant",
)


def build_suite(*, seed=17071):
    baseline = tuple(c for c in v71_suite(seed=seed)["base"] if c.kind == "memory")
    if len(baseline) != 16:
        raise ValueError("expected sixteen original memory development worlds")
    suite = {name: [] for name in VARIANTS}
    for case in baseline:
        if case.memory is None or len(case.memory.events) != 1:
            raise ValueError("unexpected original memory case structure")
        e = case.memory.events[0]
        wrong = next(x for x in SYMBOLS if x != e.value)
        unused = next(x for x in SYMBOLS if x not in case.goal.slots)
        original = case.memory
        repeated = MemoryVersion((replace(e, value=e.value), replace(e, tick=e.tick + 1, value=e.value)))
        corrected = MemoryVersion((replace(e, value=wrong), replace(e, tick=e.tick + 1, value=e.value)))
        reversed_rows = MemoryVersion(tuple(reversed(corrected.events)))
        irrelevant = MemoryVersion((replace(e, subject=unused),))
        for name, memory, expected in (
            ("single", original, case.truth.expected),
            ("consistent-two", repeated, case.truth.expected),
            ("stale-corrected", corrected, case.truth.expected),
            ("reversed-stale-corrected", reversed_rows, case.truth.expected),
            ("removed", None, None),
            ("irrelevant", irrelevant, None),
        ):
            ident = digest(["v715-memory", case.case_id, name])
            admitted = _admitted(case.environment, memory)
            row = replace(case,
                case_id=ident, goal=replace(case.goal, episode=ident),
                memory=memory, truth=Truth(ident, expected, admitted),
                pair_id=digest(["v715-memory-pair", case.world_id]),
                panel="v715-memory-" + name)
            suite[name].append(row)
    return {k: tuple(v) for k,v in suite.items()}


def audit(suite, *, seed=17071, train=None):
    rebuilt = build_suite(seed=seed)
    if set(suite) != set(rebuilt) or any(
        canonical([case_to_record(c) for c in suite[k]]) != canonical([case_to_record(c) for c in rebuilt[k]])
        for k in rebuilt
    ):
        raise ValueError("memory intervention identity mismatch")
    for i in range(16):
        original = suite["single"][i]
        same = suite["consistent-two"][i]
        corrected = suite["stale-corrected"][i]
        reverse = suite["reversed-stale-corrected"][i]
        if len(same.memory.events) != len(corrected.memory.events):
            raise ValueError("memory context length differs")
        if [(e.subject,e.source,e.tick) for e in same.memory.events] != [
            (e.subject,e.source,e.tick) for e in corrected.memory.events
        ]:
            raise ValueError("memory position/tick/source confound")
        if tuple(reversed(corrected.memory.events)) != reverse.memory.events:
            raise ValueError("reverse correction variant not aligned")
        if same.memory.events[1].value != corrected.memory.events[1].value:
            raise ValueError("corrected memory final value differs from control")
        for label in VARIANTS:
            c = suite[label][i]
            state = CognitiveState(c.goal).observe((*c.environment.initial,
                                                     *(c.memory.events if c.memory else ())))
            expected = ("ANSWER", original.truth.expected) if c.truth.expected is not None else ("ABSTAIN", None)
            if public_action(state, (), c.surface, False) != expected:
                raise ValueError("public memory oracle cannot solve expected variant")
            if set(c.truth.admitted_public) != set(e.identity for e in
                    (*c.environment.initial, *(c.memory.events if c.memory else ()))):
                raise ValueError("private evaluator admission mismatch")
    disjoint = (
        {"status": "NOT_RUN", "reason": "actual train unavailable"} if train is None
        else assert_training_disjoint(train, suite)
    )
    return {"schema": "elendril-v715-memory-audit-v1", "status": "PASS",
            "worlds": 16, "cohorts": len(suite),
            "case_records": sum(map(len, suite.values())),
            "training_disjoint": disjoint, "scientific_status": "CAUSE UNRESOLVED",
            "qualification": "Matched public memory evidence contrasts; goal-filtered retrieval must be checked in frozen traces."}
