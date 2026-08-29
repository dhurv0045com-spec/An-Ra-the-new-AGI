"""Independent surface-text solvers used to validate E0 generator semantics."""

from __future__ import annotations

import re

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
        return _match(r" became (.+)\.$", case.facts[-1]).group(1)

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
        return f"{pair.group(2)}|{pair.group(1)}"

    if family == "natural_binding_analogue":
        assay = next(fact for fact in case.facts if fact.startswith("The assay summary "))
        return _match(r"-B carries ([^.]+)\.", assay).group(1)

    if family == "natural_state_analogue":
        return _match(r"set it to (.+)\.$", case.facts[-1]).group(1)

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
