"""Independent V8 episode verifier and hand-written semantic fixtures.

The verifier never shares solver functions with the teacher and never exposes
hidden truth to any actor.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import EvaluationOutcome, OP_VOCABULARY, SYMBOLS, VERIFIER_IDENTITY


def _independent_resolve_subject(records: list[dict[str, object]], subject: str) -> tuple[str, str | None]:
    """Independent reference evaluator for timestamp/source resolution."""

    by_source: dict[str, list[tuple[int, str | None]]] = {}
    for rec in records:
        if rec["subject"] == subject:
            by_source.setdefault(str(rec["source"]), []).append((int(rec["tick"]), rec["value"]))
    if not by_source:
        return "missing", None
    latest_vals: set[str | None] = set()
    for items in by_source.values():
        max_t = max(t for t, _ in items)
        vals_at_max = {v for t, v in items if t == max_t}
        if len(vals_at_max) > 1:
            return "contradicted", None
        latest_vals.update(vals_at_max)
    if len(latest_vals) != 1:
        return "contradicted", None
    val = next(iter(latest_vals))
    return ("null", None) if val is None else ("ok", val)


def independent_expected_outcome(world: dict[str, object]) -> str | None:
    """Compute ground-truth obtainable outcome from full world record independently."""

    all_records = (
        list(world["initial_facts"])
        + (list(world["hidden"]["query_payload"]) if world["allowed_public_tools"]["query_available"] else [])
        + (list(world["hidden"]["memory_payload"]) if world["allowed_public_tools"]["memory_available"] else [])
    )
    cursor = str(world["goal_subject"])
    for _ in range(max(1, int(world["hops"]))):
        status, val = _independent_resolve_subject(all_records, cursor)
        if status != "ok" or val is None:
            return None
        cursor = val
    return cursor


def verify_episode_trace(
    world: dict[str, object],
    executed_actions: list[tuple[str, str | None]] | tuple[tuple[str, str | None], ...],
    *,
    query_cost: float = 0.1,
    action_budget: int = 4,
) -> EvaluationOutcome:
    """Verify a complete sequence of executed actions against hidden truth and rules."""

    expected = world["hidden"]["expected"]
    query_avail = bool(world["allowed_public_tools"]["query_available"])
    mem_avail = bool(world["allowed_public_tools"]["memory_available"])
    verified_queries = 0

    if not executed_actions:
        return EvaluationOutcome(
            correct=False,
            terminal_status="budget_exhausted",
            charged_cost=0.0,
            verifier_identity=VERIFIER_IDENTITY,
        )

    for step_idx, (op, arg) in enumerate(executed_actions):
        if step_idx >= action_budget:
            return EvaluationOutcome(
                correct=False,
                terminal_status="budget_exhausted",
                charged_cost=verified_queries * query_cost,
                verifier_identity=VERIFIER_IDENTITY,
            )
        if op not in OP_VOCABULARY:
            return EvaluationOutcome(
                correct=False,
                terminal_status="invalid_action",
                charged_cost=verified_queries * query_cost,
                verifier_identity=VERIFIER_IDENTITY,
            )
        if op == "QUERY":
            if not query_avail:
                return EvaluationOutcome(
                    correct=False,
                    terminal_status="invalid_action",
                    charged_cost=verified_queries * query_cost,
                    verifier_identity=VERIFIER_IDENTITY,
                )
            query_avail = False
            verified_queries += 1
            continue
        if op == "RETRIEVE":
            if not mem_avail:
                return EvaluationOutcome(
                    correct=False,
                    terminal_status="invalid_action",
                    charged_cost=verified_queries * query_cost,
                    verifier_identity=VERIFIER_IDENTITY,
                )
            mem_avail = False
            continue
        if op == "ABSTAIN":
            return EvaluationOutcome(
                correct=(expected is None),
                terminal_status="abstained",
                charged_cost=verified_queries * query_cost,
                verifier_identity=VERIFIER_IDENTITY,
            )
        if op == "ANSWER":
            if arg not in SYMBOLS or len(str(arg)) != 1:
                return EvaluationOutcome(
                    correct=False,
                    terminal_status="invalid_action",
                    charged_cost=verified_queries * query_cost,
                    verifier_identity=VERIFIER_IDENTITY,
                )
            return EvaluationOutcome(
                correct=(expected is not None and arg == expected),
                terminal_status="answered",
                charged_cost=verified_queries * query_cost,
                verifier_identity=VERIFIER_IDENTITY,
            )
    return EvaluationOutcome(
        correct=False,
        terminal_status="budget_exhausted",
        charged_cost=verified_queries * query_cost,
        verifier_identity=VERIFIER_IDENTITY,
    )


def semantic_fixtures() -> tuple[dict[str, object], ...]:
    """Hand-written Section 6.3 semantic fixtures locking timestamp/source/CF semantics."""

    def row(subject: str, source: str, tick: int, value: str | None) -> dict[str, object]:
        return {"subject": subject, "source": source, "tick": tick, "value": value}

    return (
        {
            "name": "two_hop_base",
            "goal_subject": "A",
            "hops": 2,
            "initial_facts": [row("A", "map", 10, "C"), row("C", "map", 11, "F"), row("B", "map", 12, "D")],
            "allowed_public_tools": {"query_available": False, "memory_available": False},
            "hidden": {"expected": "F", "query_payload": [], "memory_payload": []},
            "expected_teacher": [("ANSWER", "F")],
        },
        {
            "name": "two_hop_relevant_cf",
            "goal_subject": "A",
            "hops": 2,
            "initial_facts": [row("A", "map", 10, "C"), row("C", "map", 11, "H"), row("B", "map", 12, "D")],
            "allowed_public_tools": {"query_available": False, "memory_available": False},
            "hidden": {"expected": "H", "query_payload": [], "memory_payload": []},
            "expected_teacher": [("ANSWER", "H")],
        },
        {
            "name": "two_hop_irrelevant_cf",
            "goal_subject": "A",
            "hops": 2,
            "initial_facts": [row("A", "map", 10, "C"), row("C", "map", 11, "F"), row("B", "map", 12, "E")],
            "allowed_public_tools": {"query_available": False, "memory_available": False},
            "hidden": {"expected": "F", "query_payload": [], "memory_payload": []},
            "expected_teacher": [("ANSWER", "F")],
        },
        {
            "name": "two_hop_hidden_query_F",
            "goal_subject": "A",
            "hops": 2,
            "initial_facts": [row("A", "map", 10, "C"), row("B", "map", 12, "D")],
            "allowed_public_tools": {"query_available": True, "memory_available": False},
            "hidden": {"expected": "F", "query_payload": [row("C", "map", 15, "F")], "memory_payload": []},
            "expected_teacher": [("QUERY", None), ("ANSWER", "F")],
        },
        {
            "name": "two_hop_hidden_query_H",
            "goal_subject": "A",
            "hops": 2,
            "initial_facts": [row("A", "map", 10, "C"), row("B", "map", 12, "D")],
            "allowed_public_tools": {"query_available": True, "memory_available": False},
            "hidden": {"expected": "H", "query_payload": [row("C", "map", 15, "H")], "memory_payload": []},
            "expected_teacher": [("QUERY", None), ("ANSWER", "H")],
        },
        {
            "name": "two_hop_unavailable_abstain",
            "goal_subject": "A",
            "hops": 2,
            "initial_facts": [row("A", "map", 10, "C"), row("B", "map", 12, "D")],
            "allowed_public_tools": {"query_available": False, "memory_available": False},
            "hidden": {"expected": None, "query_payload": [], "memory_payload": []},
            "expected_teacher": [("ABSTAIN", None)],
        },
        {
            "name": "same_source_later_supersedes",
            "goal_subject": "A",
            "hops": 1,
            "initial_facts": [row("A", "s0", 10, "B"), row("A", "s0", 20, "D")],
            "allowed_public_tools": {"query_available": False, "memory_available": False},
            "hidden": {"expected": "D", "query_payload": [], "memory_payload": []},
            "expected_teacher": [("ANSWER", "D")],
        },
        {
            "name": "equal_time_contradiction_unresolved",
            "goal_subject": "A",
            "hops": 1,
            "initial_facts": [row("A", "s0", 10, "B"), row("A", "s0", 10, "D")],
            "allowed_public_tools": {"query_available": False, "memory_available": False},
            "hidden": {"expected": None, "query_payload": [], "memory_payload": []},
            "expected_teacher": [("ABSTAIN", None)],
        },
        {
            "name": "cross_source_conflict_resolved_by_query",
            "goal_subject": "A",
            "hops": 1,
            "initial_facts": [row("A", "s0", 10, "B"), row("A", "s1", 12, "D")],
            "allowed_public_tools": {"query_available": True, "memory_available": False},
            "hidden": {
                "expected": "E",
                "query_payload": [row("A", "s0", 20, "E"), row("A", "s1", 20, "E")],
                "memory_payload": [],
            },
            "expected_teacher": [("QUERY", None), ("ANSWER", "E")],
        },
    )


def verify_semantic_fixtures() -> dict[str, object]:
    """Run independent verification and teacher checks on all hand-written fixtures."""

    from .teacher import PublicFact, teacher_decide

    fixtures = semantic_fixtures()
    checked = []
    for fx in fixtures:
        ind_expected = independent_expected_outcome(fx)
        if ind_expected != fx["hidden"]["expected"]:
            raise AssertionError(
                f"fixture {fx['name']}: independent expected {ind_expected!r} != {fx['hidden']['expected']!r}"
            )
        # Step teacher through fixture
        facts = [PublicFact(**row) for row in fx["initial_facts"]]
        q_avail = bool(fx["allowed_public_tools"]["query_available"])
        m_avail = bool(fx["allowed_public_tools"]["memory_available"])
        actions: list[tuple[str, str | None]] = []
        for _ in range(4):
            op, arg = teacher_decide(tuple(facts), fx["goal_subject"], fx["hops"], q_avail, m_avail)
            actions.append((op, arg))
            if op == "QUERY":
                facts.extend(PublicFact(**row) for row in fx["hidden"]["query_payload"])
                q_avail = False
            elif op == "RETRIEVE":
                facts.extend(PublicFact(**row) for row in fx["hidden"]["memory_payload"])
                m_avail = False
            else:
                break
        if actions != fx["expected_teacher"]:
            raise AssertionError(f"fixture {fx['name']}: teacher {actions!r} != {fx['expected_teacher']!r}")
        outcome = verify_episode_trace(fx, actions)
        if not outcome.correct:
            raise AssertionError(f"fixture {fx['name']}: teacher trace failed verifier: {outcome}")
        checked.append(fx["name"])
    return {"status": "PASS", "fixtures_verified": len(checked), "names": checked}


__all__ = [
    "independent_expected_outcome",
    "semantic_fixtures",
    "verify_episode_trace",
    "verify_semantic_fixtures",
]
