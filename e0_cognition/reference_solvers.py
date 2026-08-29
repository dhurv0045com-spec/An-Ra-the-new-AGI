"""Independent surface-text solvers used to validate E0 generator semantics."""

from __future__ import annotations

import re
from functools import lru_cache

from .contracts import CausalCase, EvaluationSuite


def _match(pattern: str, text: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"reference solver could not parse: {text!r}")
    return match


def solve_case(case: CausalCase) -> str:
    """Solve from serialized facts/query only; never read answer or hidden truth."""

    family = case.family
    if family == "exact_contextual_copy":
        return _match(r"reads exactly \[([^]]+)\]", case.context()).group(1)

    if family == "nonce_identifier_retrieval":
        for fact in case.facts:
            entity, value = fact.split(" :: ", 1)
            if entity in case.query:
                return value

    if family == "entity_value_binding":
        for fact in case.facts:
            parsed = _match(r"The tag for (.+) is (.+)\.$", fact)
            if parsed.group(1) in case.query:
                return parsed.group(2)

    if family == "state_overwrite":
        query = _match(r"For (.+), what value is in force at time (\d+)", case.query)
        target, cutoff = query.group(1), int(query.group(2))
        events: list[tuple[int, int, str, str, int | None]] = []
        for fact in case.facts:
            literal = re.search(
                r"Event time=(\d+) priority=(\d+): (.+) := ([A-Z0-9-]+)\.", fact
            )
            rollback = re.search(
                r"Event time=(\d+) priority=(\d+): (.+) := value@time=(\d+)\.", fact
            )
            if rollback:
                events.append((int(rollback.group(1)), int(rollback.group(2)), rollback.group(3), "", int(rollback.group(4))))
            elif literal:
                events.append((int(literal.group(1)), int(literal.group(2)), literal.group(3), literal.group(4), None))
            else:
                raise ValueError(f"could not parse state event: {fact!r}")
        events.sort(key=lambda event: (event[0], event[1]))

        @lru_cache(maxsize=None)
        def value_at(variable: str, time: int) -> str:
            value: str | None = None
            for event_time, _, event_variable, event_value, rollback_time in events:
                if event_time > time:
                    break
                if event_variable != variable:
                    continue
                value = value_at(variable, rollback_time) if rollback_time is not None else event_value
            if value is None:
                raise ValueError(f"no state for {variable!r} at time {time}")
            return value

        return value_at(target, cutoff)

    if family == "matched_direct_retrieval":
        direct = next(fact for fact in case.facts if fact.startswith("The direct result for "))
        return _match(r"direct result for .+ is (.+)\.$", direct).group(1)

    if family.startswith("relation_"):
        query = _match(
            r"Starting at (.+), follow in order: (.+)\. Which symbol is reached\?",
            case.query,
        )
        node = query.group(1)
        relations = query.group(2).split(" then ")
        edges: dict[tuple[str, str], str] = {}
        for fact in case.facts:
            source, relation, target = fact[:-1].split(" ", 2)
            edges[(source, relation)] = target
        for relation in relations:
            node = edges[(node, relation)]
        return node

    if family == "missing_information":
        known_entities = {
            _match(r"registered tag for (.+) is .+\.$", fact).group(1) for fact in case.facts
        }
        queried = _match(r"registered tag for (.+)\?", case.query).group(1)
        return "<MISSING>" if queried not in known_entities else "<UNEXPECTED-KNOWN>"

    if family == "counterfactual_premise":
        return "NO" if case.facts[0].startswith("In this hypothetical world, no ") else "YES"

    if family == "rule_induction":
        pair = _match(r"unseen pair \(([^,]+), ([^)]+)\)", case.query)
        left, right = pair.group(1), pair.group(2)
        structure: tuple[int, ...] | None = None
        for fact in case.facts:
            demo = _match(r"pair \(([^,]+), ([^)]+)\) returns (.+)\.$", fact)
            demo_left, demo_right, output = demo.group(1), demo.group(2), demo.group(3)
            output_parts = output.split("|")
            inferred = tuple(0 if part == demo_left else 1 if part == demo_right else -1 for part in output_parts)
            if -1 in inferred or not inferred:
                raise ValueError(f"rule output is not an operand structure: {fact!r}")
            if structure is None:
                structure = inferred
            elif structure != inferred:
                raise ValueError("rule demonstrations disagree")
        if structure is None:
            raise ValueError("rule requires at least one demonstration")
        return "|".join((left, right)[position] for position in structure)

    if family == "natural_binding_analogue":
        assay = next(fact for fact in case.facts if fact.startswith("The assay summary "))
        return _match(r"-B carries ([^.]+)\.", assay).group(1)

    if family == "natural_state_analogue":
        query = _match(r"node (.+) at minute (\d+),", case.query)
        target, cutoff = query.group(1), int(query.group(2))
        events: list[tuple[int, int, str, str, int | None]] = []
        for fact in case.facts:
            literal = re.search(
                r"At minute (\d+) \(priority (\d+)\), the router configuration for node (.+) was set to ([A-Z0-9-]+)\.",
                fact,
            )
            rollback = re.search(
                r"At minute (\d+) \(priority (\d+)\), an approved rollback restored node (.+) to its value at minute (\d+)\.",
                fact,
            )
            if rollback:
                events.append((int(rollback.group(1)), int(rollback.group(2)), rollback.group(3), "", int(rollback.group(4))))
            elif literal:
                events.append((int(literal.group(1)), int(literal.group(2)), literal.group(3), literal.group(4), None))
            else:
                raise ValueError(f"could not parse natural state event: {fact!r}")
        events.sort(key=lambda event: (event[0], event[1]))

        @lru_cache(maxsize=None)
        def value_at(variable: str, time: int) -> str:
            value: str | None = None
            for event_time, _, event_variable, event_value, rollback_time in events:
                if event_time > time:
                    break
                if event_variable != variable:
                    continue
                value = value_at(variable, rollback_time) if rollback_time is not None else event_value
            if value is None:
                raise ValueError(f"no natural state for {variable!r} at time {time}")
            return value

        return value_at(target, cutoff)

    if family == "natural_composition_analogue":
        start_fact = next(fact for fact in case.facts if fact.startswith("Module "))
        start = _match(r"Module (.+) writes buffer (.+)\.$", start_fact)
        buffer_name = start.group(2)
        buffer_fact = next(fact for fact in case.facts if fact.startswith(f"Buffer {buffer_name} "))
        converter = _match(
            rf"Buffer {re.escape(buffer_name)} feeds converter (.+)\.$", buffer_fact
        ).group(1)
        converter_fact = next(
            fact for fact in case.facts if fact.startswith(f"Converter {converter} ")
        )
        return _match(
            rf"Converter {re.escape(converter)} emits channel (.+)\.$", converter_fact
        ).group(1)

    raise ValueError(f"no reference solver for family {family!r}")


def assert_reference_solver_agreement(suite: EvaluationSuite) -> None:
    failures = [
        (case.case_id, solve_case(case), case.answer)
        for case in suite.cases
        if solve_case(case) != case.answer
    ]
    if failures:
        raise AssertionError(f"reference solver disagreements: {failures[:5]}")
