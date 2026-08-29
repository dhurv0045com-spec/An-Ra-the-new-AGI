"""Evaluation-only cognition generators with causal counterfactual contracts."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .contracts import (
    CausalCase,
    CausalPair,
    EvaluationSuite,
    GraphEdge,
    HiddenTruth,
    PairKind,
    Split,
)

GENERATOR_VERSION = "e0-eval/0.3.0"


@dataclass(frozen=True, slots=True)
class SplitProfile:
    split: Split
    prefix: str
    relations: tuple[str, ...]
    domains: tuple[str, ...]
    template_prefix: str
    rule_structures: tuple[tuple[int, ...], ...]


PROFILES = {
    Split.DEVELOPMENT: SplitProfile(
        Split.DEVELOPMENT,
        "DV",
        ("guides", "links", "feeds", "indexes"),
        ("inventory", "laboratory", "technical-manual"),
        "dev",
        ((1, 0), (0, 1, 0), (1, 0, 1), (0, 0, 1), (1, 1, 0), (0, 1, 1), (0, 0, 0, 1), (1, 1, 1, 0)),
    ),
    Split.SEALED: SplitProfile(
        Split.SEALED,
        "SL",
        ("supports", "orbits", "routes", "anchors"),
        ("legal-clause", "geology", "manufacturing"),
        "sealed",
        ((0, 1), (1, 1, 1), (0, 1, 1, 1), (1, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0)),
    ),
    Split.FRESH: SplitProfile(
        Split.FRESH,
        "FR",
        ("buffers", "maps", "powers", "shields"),
        ("network-log", "ecology", "clinical-procedure"),
        "fresh",
        ((0, 0, 1, 1, 1), (1, 1, 0, 0, 0), (0, 1, 0, 1, 1), (1, 0, 1, 0, 0), (0, 0, 0, 1, 1), (1, 1, 1, 0, 0), (0, 1, 1, 0, 1), (1, 0, 0, 1, 0)),
    ),
}


def _code(profile: SplitProfile, rng: random.Random) -> str:
    return f"{profile.prefix}{rng.randrange(1000, 9999)}-{rng.choice('ABCDEFGH')}"


def _entity(profile: SplitProfile, rng: random.Random, index: int) -> str:
    syllables = ("vek", "lum", "sor", "nax", "tir", "pel", "gom", "ruz", "cai", "fen")
    return f"{profile.prefix.lower()}-{rng.choice(syllables)}-{index}-{rng.randrange(100,999)}"


def _difficulty(**values: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((key, int(value)) for key, value in values.items()))


def _candidate_order(rng: random.Random, values: tuple[str, ...]) -> tuple[str, ...]:
    ordered = list(dict.fromkeys(values))
    rng.shuffle(ordered)
    return tuple(ordered)


def _answer_format(answer: str, profile: SplitProfile) -> str:
    if answer == "<MISSING>":
        return "abstention"
    if answer in {"YES", "NO"}:
        return "boolean"
    if "|" in answer:
        return "composite"
    if answer.startswith(profile.prefix):
        return "nonce-code"
    if answer.startswith(profile.prefix.lower() + "-"):
        return "nonce-entity"
    return "literal"


def _surface_axes(
    *,
    facts: tuple[str, ...],
    relevant: tuple[int, ...],
    answer: str,
    profile: SplitProfile,
    extra: dict[str, str] | None = None,
) -> tuple[tuple[str, str], ...]:
    if not relevant:
        position = "answer-absent"
    elif len(relevant) > 1:
        position = "distributed"
    elif relevant[0] == 0:
        position = "front"
    elif relevant[0] == len(facts) - 1:
        position = "back"
    else:
        position = "middle"
    length = "short" if len(facts) <= 2 else "medium" if len(facts) <= 4 else "long"
    axes = {
        "answer_format": _answer_format(answer, profile),
        "context_fact_count": length,
        "relevant_position": position,
    }
    if extra:
        axes.update(extra)
    return tuple(sorted(axes.items()))


def _provenance(profile: SplitProfile, seed: int, family: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "generator": GENERATOR_VERSION,
                "seed": str(seed),
                "split": profile.split.value,
                "family": family,
                "truth": "executable",
            }.items()
        )
    )


def _case(
    *,
    case_id: str,
    family: str,
    profile: SplitProfile,
    seed: int,
    facts: tuple[str, ...],
    query: str,
    answer: str,
    candidates: tuple[str, ...],
    relevant: tuple[int, ...],
    distractors: tuple[int, ...],
    graph: tuple[GraphEdge, ...],
    trace: tuple[str, ...],
    template: str,
    domain: str,
    difficulty: tuple[tuple[str, int], ...],
    axes: dict[str, str] | None = None,
) -> CausalCase:
    return CausalCase(
        case_id=case_id,
        family=family,
        split=profile.split,
        domain=domain,
        template_id=f"{profile.template_prefix}.{template}",
        seed=seed,
        facts=facts,
        query=query,
        answer=answer,
        candidates=candidates,
        difficulty=difficulty,
        surface_axes=_surface_axes(
            facts=facts, relevant=relevant, answer=answer, profile=profile, extra=axes
        ),
        provenance=_provenance(profile, seed, family),
        hidden=HiddenTruth(relevant, distractors, graph, trace),
    )


def _binding_bundle(
    profile: SplitProfile, rng: random.Random, seed: int, index: int, cardinality: int
) -> tuple[list[CausalCase], list[CausalPair]]:
    entities = [_entity(profile, rng, j) for j in range(cardinality)]
    values = [_code(profile, rng) for _ in range(cardinality)]
    replacement = _code(profile, rng)
    irrelevant_replacement = _code(profile, rng)
    relation = profile.relations[0]
    facts = tuple(f"The tag for {entity} is {value}." for entity, value in zip(entities, values))
    target = rng.randrange(cardinality)
    other = (target + 1) % cardinality
    distractor = next(i for i in range(cardinality) if i not in (target, other)) if cardinality > 2 else other
    graph = tuple(GraphEdge(entity, relation, value) for entity, value in zip(entities, values))
    common = dict(
        profile=profile,
        seed=seed,
        candidates=_candidate_order(rng, tuple(values) + (replacement, irrelevant_replacement)),
        graph=graph,
        template="binding",
        domain=profile.domains[0],
        difficulty=_difficulty(cardinality=cardinality, hops=0, distractors=cardinality - 1),
    )
    base = _case(
        case_id=f"{profile.prefix}-binding-{index}-base",
        family="entity_value_binding",
        facts=facts,
        query=f"Which tag belongs to {entities[target]}?",
        answer=values[target],
        relevant=(target,),
        distractors=tuple(i for i in range(cardinality) if i != target),
        trace=(f"address:{entities[target]}", f"select:{values[target]}"),
        **common,
    )
    query_swap = replace(
        base,
        case_id=f"{profile.prefix}-binding-{index}-query",
        query=f"Which tag belongs to {entities[other]}?",
        answer=values[other],
        hidden=HiddenTruth(
            (other,),
            tuple(i for i in range(cardinality) if i != other),
            graph,
            (f"address:{entities[other]}", f"select:{values[other]}"),
        ),
    )
    relevant_facts = list(facts)
    relevant_facts[target] = f"The tag for {entities[target]} is {replacement}."
    relevant_graph = list(graph)
    relevant_graph[target] = GraphEdge(entities[target], relation, replacement)
    relevant_swap = replace(
        base,
        case_id=f"{profile.prefix}-binding-{index}-relevant",
        facts=tuple(relevant_facts),
        answer=replacement,
        hidden=HiddenTruth(
            (target,),
            base.hidden.distractor_fact_indices,
            tuple(relevant_graph),
            (f"address:{entities[target]}", f"select:{replacement}"),
        ),
    )
    irrelevant_facts = list(facts)
    irrelevant_facts[distractor] = f"The tag for {entities[distractor]} is {irrelevant_replacement}."
    irrelevant_graph = list(graph)
    irrelevant_graph[distractor] = GraphEdge(entities[distractor], relation, irrelevant_replacement)
    irrelevant_swap = replace(
        base,
        case_id=f"{profile.prefix}-binding-{index}-irrelevant",
        facts=tuple(irrelevant_facts),
        hidden=HiddenTruth(
            (target,),
            base.hidden.distractor_fact_indices,
            tuple(irrelevant_graph),
            base.hidden.operation_trace,
        ),
    )
    order = list(range(cardinality))
    order.reverse()
    permuted_facts = tuple(facts[i] for i in order)
    permuted_graph = tuple(graph[i] for i in order)
    new_target = order.index(target)
    permutation = replace(
        base,
        case_id=f"{profile.prefix}-binding-{index}-permuted",
        facts=permuted_facts,
        surface_axes=_surface_axes(
            facts=permuted_facts,
            relevant=(new_target,),
            answer=base.answer,
            profile=profile,
        ),
        hidden=HiddenTruth(
            (new_target,),
            tuple(i for i in range(cardinality) if i != new_target),
            permuted_graph,
            base.hidden.operation_trace,
        ),
    )
    cases = [base, query_swap, relevant_swap, irrelevant_swap, permutation]
    pairs = [
        CausalPair(f"{base.case_id}:query", PairKind.QUERY_SWAP, base, query_swap),
        CausalPair(f"{base.case_id}:relevant", PairKind.RELEVANT_FACT_SWAP, base, relevant_swap),
        CausalPair(f"{base.case_id}:irrelevant", PairKind.IRRELEVANT_FACT_SWAP, base, irrelevant_swap),
        CausalPair(f"{base.case_id}:order", PairKind.ORDER_PERMUTATION, base, permutation),
    ]
    return cases, pairs


def _copy_case(profile: SplitProfile, rng: random.Random, seed: int, index: int) -> CausalCase:
    value = _code(profile, rng)
    marker = _entity(profile, rng, index)
    return _case(
        case_id=f"{profile.prefix}-copy-{index}",
        family="exact_contextual_copy",
        profile=profile,
        seed=seed,
        facts=(f"Marker {marker} reads exactly [{value}].",),
        query=f"Copy only the characters inside the brackets for marker {marker}.",
        answer=value,
        candidates=(value,),
        relevant=(0,),
        distractors=(),
        graph=(GraphEdge(marker, profile.relations[1], value),),
        trace=(f"copy:{value}",),
        template="exact-copy",
        domain=profile.domains[1],
        difficulty=_difficulty(cardinality=1, hops=0, distractors=0),
    )


def _nonce_case(profile: SplitProfile, rng: random.Random, seed: int, index: int) -> CausalCase:
    count = 4
    entities = [_entity(profile, rng, j) for j in range(count)]
    values = [_code(profile, rng) for _ in range(count)]
    target = rng.randrange(count)
    facts = tuple(f"{entity} :: {value}" for entity, value in zip(entities, values))
    return _case(
        case_id=f"{profile.prefix}-nonce-{index}",
        family="nonce_identifier_retrieval",
        profile=profile,
        seed=seed,
        facts=facts,
        query=f"Return the identifier paired with {entities[target]}.",
        answer=values[target],
        candidates=_candidate_order(rng, tuple(values)),
        relevant=(target,),
        distractors=tuple(i for i in range(count) if i != target),
        graph=tuple(GraphEdge(e, profile.relations[1], v) for e, v in zip(entities, values)),
        trace=(f"address:{entities[target]}", f"copy:{values[target]}"),
        template="nonce-map",
        domain=profile.domains[1],
        difficulty=_difficulty(cardinality=count, hops=0, distractors=count - 1),
    )


def _state_bundle(
    profile: SplitProfile, rng: random.Random, seed: int, index: int
) -> tuple[list[CausalCase], list[CausalPair]]:
    """Generate state tasks where semantic time is independent of serialization order.

    The four scenarios deliberately include latest, intermediate, rollback, and
    same-time precedence queries. Two variables are interleaved and every event
    is shuffled after its semantic graph is built, so position-only heuristics
    cannot identify the answer.
    """

    target = _entity(profile, rng, index)
    other = _entity(profile, rng, index + 100)
    values = [_code(profile, rng) for _ in range(10)]
    scenario = index % 4
    if scenario == 0:
        events = [
            (1, 1, target, values[0], None),
            (4, 1, other, values[1], None),
            (7, 1, target, values[2], None),
            (9, 1, other, values[3], None),
            (12, 1, target, values[4], None),
        ]
        query_time, answer = 12, values[4]
        query_kind = "latest"
        winning = {4}
    elif scenario == 1:
        events = [
            (2, 1, target, values[0], None),
            (3, 1, other, values[1], None),
            (6, 1, target, values[2], None),
            (8, 1, other, values[3], None),
            (11, 1, target, values[4], None),
        ]
        query_time, answer = 6, values[2]
        query_kind = "intermediate"
        winning = {2}
    elif scenario == 2:
        events = [
            (1, 1, target, values[0], None),
            (3, 1, other, values[1], None),
            (6, 1, target, values[2], None),
            (8, 1, target, values[0], 1),
            (10, 1, other, values[3], None),
        ]
        query_time, answer = 8, values[0]
        query_kind = "rollback"
        winning = {0, 3}
    else:
        events = [
            (2, 1, target, values[0], None),
            (4, 1, other, values[1], None),
            (7, 2, target, values[2], None),
            (7, 5, target, values[3], None),
            (9, 1, other, values[4], None),
        ]
        query_time, answer = 7, values[3]
        query_kind = "precedence"
        winning = {3}

    replacement = _code(profile, rng)
    def render(event: tuple[int, int, str, str, int | None]) -> str:
        time, priority, variable, value, rollback_time = event
        if rollback_time is None:
            return f"Event time={time} priority={priority}: {variable} := {value}."
        return (
            f"Event time={time} priority={priority}: {variable} := "
            f"value@time={rollback_time}."
        )

    order = list(range(len(events)))
    rng.shuffle(order)
    facts = tuple(render(events[i]) for i in order)
    relevant = tuple(position for position, original in enumerate(order) if original in winning)
    graph = tuple(
        GraphEdge(variable, f"{profile.prefix.lower()}-time-{time}-p{priority}", value)
        for time, priority, variable, value, _ in events
    )
    candidates = _candidate_order(
        rng,
        tuple(values[:5]) + (replacement, _code(profile, rng), _code(profile, rng)),
    )
    base = _case(
        case_id=f"{profile.prefix}-state-{index}-base",
        family="state_overwrite",
        profile=profile,
        seed=seed,
        facts=facts,
        query=(
            f"For {target}, what value is in force at time {query_time} after applying "
            "semantic time, rollback, and priority rules?"
        ),
        answer=answer,
        candidates=candidates,
        relevant=relevant,
        distractors=tuple(i for i in range(len(facts)) if i not in relevant),
        graph=graph,
        trace=(f"state-query:{query_kind}", f"select:{answer}"),
        template=f"state-log-{query_kind}",
        domain=profile.domains[2],
        difficulty=_difficulty(
            cardinality=len(facts),
            hops=1,
            distractors=len(facts) - len(relevant),
            state_events=len(events),
            variables=2,
        ),
        axes={
            "state_query": query_kind,
            "serialization": "semantic-shuffled",
            "variable_interleaving": "two-variable",
        },
    )
    changed_events = list(events)
    # In a rollback case the earlier source event is the causal variable; the
    # later rollback record must remain unchanged so the pair changes one fact.
    change_index = min(winning) if scenario == 2 else max(winning)
    time, priority, variable, _, rollback_time = changed_events[change_index]
    changed_events[change_index] = (time, priority, variable, replacement, rollback_time)
    changed_facts = tuple(render(changed_events[i]) for i in order)
    changed_graph = list(graph)
    changed_graph[change_index] = GraphEdge(
        variable, f"{profile.prefix.lower()}-time-{time}-p{priority}", replacement
    )
    changed_answer = replacement if change_index in winning else answer
    changed_relevant = tuple(position for position, original in enumerate(order) if original in winning)
    swapped = replace(
        base,
        case_id=f"{profile.prefix}-state-{index}-changed",
        facts=changed_facts,
        answer=changed_answer,
        hidden=HiddenTruth(
            changed_relevant,
            tuple(i for i in range(len(changed_facts)) if i not in changed_relevant),
            tuple(changed_graph),
            (f"state-query:{query_kind}", f"select:{changed_answer}"),
        ),
    )
    return [base, swapped], [CausalPair(f"{base.case_id}:state", PairKind.STATE_SWAP, base, swapped)]


def _relation_case(
    profile: SplitProfile,
    rng: random.Random,
    seed: int,
    index: int,
    hops: int,
    *,
    direct_control: bool = False,
) -> CausalCase:
    nodes = [_entity(profile, rng, j) for j in range(hops + 1)]
    relations = [profile.relations[(index + j) % len(profile.relations)] for j in range(hops)]
    edges = tuple(GraphEdge(nodes[j], relations[j], nodes[j + 1]) for j in range(hops))
    facts = [f"{edge.source} {edge.relation} {edge.target}." for edge in edges]
    distractor_source = _entity(profile, rng, 90)
    distractor_target = _entity(profile, rng, 91)
    facts.append(f"{distractor_source} {profile.relations[-1]} {distractor_target}.")
    graph = edges + (GraphEdge(distractor_source, profile.relations[-1], distractor_target),)
    if direct_control:
        facts = [f"The direct result for {nodes[0]} is {nodes[-1]}.", facts[-1]]
        original_relevant = {0}
        trace = (f"retrieve:{nodes[-1]}",)
        family = "matched_direct_retrieval"
        template = f"direct-{hops}"
    else:
        original_relevant = set(range(hops))
        trace = tuple(f"follow:{edge.relation}->{edge.target}" for edge in edges)
        family = f"relation_{hops}_hop"
        template = f"relation-{hops}"
    order = list(range(len(facts)))
    rng.shuffle(order)
    facts = [facts[i] for i in order]
    relevant = tuple(i for i, original in enumerate(order) if original in original_relevant)
    query = (
        f"Starting at {nodes[0]}, follow in order: "
        + " then ".join(relations)
        + ". Which symbol is reached?"
    )
    return _case(
        case_id=f"{profile.prefix}-{'direct' if direct_control else 'rel'}-{hops}-{index}",
        family=family,
        profile=profile,
        seed=seed,
        facts=tuple(facts),
        query=query,
        answer=nodes[-1],
        candidates=_candidate_order(rng, tuple(nodes[1:]) + (distractor_target,)),
        relevant=relevant,
        distractors=tuple(i for i in range(len(facts)) if i not in relevant),
        graph=graph,
        trace=trace,
        template=template,
        domain=profile.domains[2],
        difficulty=_difficulty(cardinality=len(facts), hops=hops, distractors=1),
    )


def _missing_case(profile: SplitProfile, rng: random.Random, seed: int, index: int) -> CausalCase:
    known = _entity(profile, rng, 1)
    unknown = _entity(profile, rng, 2)
    value = _code(profile, rng)
    return _case(
        case_id=f"{profile.prefix}-missing-{index}",
        family="missing_information",
        profile=profile,
        seed=seed,
        facts=(f"The registered tag for {known} is {value}.",),
        query=f"What is the registered tag for {unknown}?",
        answer="<MISSING>",
        candidates=_candidate_order(rng, (value, "<MISSING>")),
        relevant=(),
        distractors=(0,),
        graph=(GraphEdge(known, profile.relations[0], value),),
        trace=(f"search:{unknown}", "absent:1"),
        template="missing",
        domain=profile.domains[0],
        difficulty=_difficulty(cardinality=1, hops=0, distractors=1),
    )


def _counterfactual_bundle(
    profile: SplitProfile, rng: random.Random, seed: int, index: int
) -> tuple[list[CausalCase], list[CausalPair]]:
    subject = _entity(profile, rng, 1)
    category = _entity(profile, rng, 2)
    predicate = profile.relations[2]
    facts = (f"In this hypothetical world, every {category} {predicate}.", f"{subject} is a {category}.")
    graph = (
        GraphEdge(category, f"{profile.prefix.lower()}-implies", predicate),
        GraphEdge(subject, f"{profile.prefix.lower()}-member-of", category),
    )
    base = _case(
        case_id=f"{profile.prefix}-counterfactual-{index}-base",
        family="counterfactual_premise",
        profile=profile,
        seed=seed,
        facts=facts,
        query=f"Under only these premises, does {subject} {predicate}?",
        answer="YES",
        candidates=_candidate_order(rng, ("YES", "NO", "<MISSING>")),
        relevant=(0, 1),
        distractors=(),
        graph=graph,
        trace=("apply:hypothetical-rule", "select:YES"),
        template="counterfactual",
        domain=profile.domains[0],
        difficulty=_difficulty(cardinality=2, hops=1, distractors=0),
    )
    negated_facts = (f"In this hypothetical world, no {category} {predicate}.", facts[1])
    changed = replace(
        base,
        case_id=f"{profile.prefix}-counterfactual-{index}-changed",
        facts=negated_facts,
        answer="NO",
        hidden=HiddenTruth(
            (0, 1),
            (),
            (GraphEdge(category, f"{profile.prefix.lower()}-forbids", predicate), graph[1]),
            ("apply:hypothetical-rule", "select:NO"),
        ),
    )
    return [base, changed], [
        CausalPair(f"{base.case_id}:premise", PairKind.RELEVANT_FACT_SWAP, base, changed)
    ]


def _apply_rule(structure: tuple[int, ...], left: str, right: str) -> str:
    operands = (left, right)
    return "|".join(operands[position] for position in structure)


def _rule_case(
    profile: SplitProfile, rng: random.Random, seed: int, index: int
) -> tuple[CausalCase, CausalPair]:
    """Infer one of several latent operand structures from demonstrations.

    Structures are disjoint by split and are stored as hidden surface metadata,
    allowing the split contract to reject symbol-only OOD claims. The model must
    infer the structure from demonstrations; no permanent reverse-pair rule is
    valid across this family.
    """

    structure = profile.rule_structures[index % len(profile.rule_structures)]
    operation = f"KEL-{profile.prefix}-{index % len(profile.rule_structures)}"
    demonstration_count = 2 + (index % 3)
    demos: list[tuple[str, str]] = [(_code(profile, rng), _code(profile, rng)) for _ in range(demonstration_count)]
    facts = tuple(
        f"Operation {operation} on pair ({left}, {right}) returns "
        f"{_apply_rule(structure, left, right)}."
        for left, right in demos
    )
    x, y = _code(profile, rng), _code(profile, rng)
    u, v = _code(profile, rng), _code(profile, rng)
    answer = _apply_rule(structure, x, y)
    changed_answer = _apply_rule(structure, u, v)
    candidate_values = [
        _apply_rule(candidate, x, y)
        for candidate in profile.rule_structures
    ] + [
        _apply_rule(candidate, u, v)
        for candidate in profile.rule_structures
    ]
    base = _case(
        case_id=f"{profile.prefix}-rule-{index}-base",
        family="rule_induction",
        profile=profile,
        seed=seed,
        facts=facts,
        query=f"Apply {operation} to the unseen pair ({x}, {y}).",
        answer=answer,
        candidates=_candidate_order(rng, tuple(candidate_values)),
        relevant=tuple(range(len(facts))),
        distractors=(),
        graph=(GraphEdge(operation, f"{profile.prefix.lower()}-maps", str(structure)),),
        trace=(f"infer:structure:{structure}", f"apply:{x},{y}", f"select:{answer}"),
        template=f"rule-induction-{index % len(profile.rule_structures)}",
        domain=profile.domains[1],
        difficulty=_difficulty(
            cardinality=len(candidate_values),
            hops=len(structure),
            distractors=len(candidate_values) - 1,
            demonstrations=demonstration_count,
        ),
        axes={
            "rule_structure": ",".join(map(str, structure)),
            "rule_arity": str(len(structure)),
            "serialization": "demonstration-order-randomized",
        },
    )
    changed = replace(
        base,
        case_id=f"{profile.prefix}-rule-{index}-query",
        query=f"Apply {operation} to the unseen pair ({u}, {v}).",
        answer=changed_answer,
        hidden=HiddenTruth(
            base.hidden.relevant_fact_indices,
            base.hidden.distractor_fact_indices,
            base.hidden.graph,
            (f"infer:structure:{structure}", f"apply:{u},{v}", f"select:{changed_answer}"),
        ),
    )
    return base, CausalPair(f"{base.case_id}:query", PairKind.QUERY_SWAP, base, changed)


def _natural_cases(profile: SplitProfile, rng: random.Random, seed: int, index: int) -> list[CausalCase]:
    sample = _entity(profile, rng, index)
    values = [_code(profile, rng) for _ in range(6)]
    binding_facts = [
        f"The assay summary assigns specimen {sample}-A the accession {values[0]}, while specimen {sample}-B carries {values[1]}.",
        f"A calibration control uses accession {values[2]} and is not a specimen result.",
    ]
    binding_order = [0, 1]
    rng.shuffle(binding_order)
    binding = _case(
        case_id=f"{profile.prefix}-natural-binding-{index}",
        family="natural_binding_analogue",
        profile=profile,
        seed=seed,
        facts=tuple(binding_facts[i] for i in binding_order),
        query=f"Which accession belongs to specimen {sample}-B?",
        answer=values[1],
        candidates=_candidate_order(rng, tuple(values[:3])),
        relevant=(binding_order.index(0),),
        distractors=(binding_order.index(1),),
        graph=(GraphEdge(f"{sample}-B", f"{profile.prefix.lower()}-accession", values[1]),),
        trace=(f"address:{sample}-B", f"select:{values[1]}"),
        template="natural-assay",
        domain=profile.domains[1],
        difficulty=_difficulty(cardinality=3, hops=0, distractors=1),
    )
    state_scenario = index % 3
    other_sample = f"{sample}-peer"
    if state_scenario == 0:
        state_events = [
            (2, 1, sample, values[0], None),
            (5, 1, other_sample, values[1], None),
            (8, 1, sample, values[2], None),
            (11, 1, other_sample, values[3], None),
            (14, 1, sample, values[4], None),
        ]
        state_time, state_answer, state_kind, state_winning = 14, values[4], "latest", {4}
    elif state_scenario == 1:
        state_events = [
            (2, 1, sample, values[0], None),
            (4, 1, other_sample, values[1], None),
            (7, 1, sample, values[2], None),
            (10, 1, other_sample, values[3], None),
            (13, 1, sample, values[4], None),
        ]
        state_time, state_answer, state_kind, state_winning = 7, values[2], "intermediate", {2}
    else:
        state_events = [
            (3, 1, sample, values[0], None),
            (6, 1, other_sample, values[1], None),
            (9, 1, sample, values[2], None),
            (12, 1, sample, values[0], 3),
            (15, 1, other_sample, values[3], None),
        ]
        state_time, state_answer, state_kind, state_winning = 12, values[0], "rollback", {0, 3}

    def render_natural(event: tuple[int, int, str, str, int | None]) -> str:
        minute, priority, variable, value, rollback_minute = event
        if rollback_minute is None:
            return (
                f"At minute {minute} (priority {priority}), the router configuration for "
                f"node {variable} was set to {value}."
            )
        return (
            f"At minute {minute} (priority {priority}), an approved rollback restored "
            f"node {variable} to its value at minute {rollback_minute}."
        )

    state_order = list(range(len(state_events)))
    rng.shuffle(state_order)
    state_facts = tuple(render_natural(state_events[i]) for i in state_order)
    state_relevant = tuple(position for position, original in enumerate(state_order) if original in state_winning)
    state = _case(
        case_id=f"{profile.prefix}-natural-state-{index}",
        family="natural_state_analogue",
        profile=profile,
        seed=seed,
        facts=state_facts,
        query=(
            f"What configuration is in force for node {sample} at minute {state_time}, "
            "respecting semantic time, rollback, and priority?"
        ),
        answer=state_answer,
        candidates=_candidate_order(rng, tuple(values[:5]) + (_code(profile, rng),)),
        relevant=state_relevant,
        distractors=tuple(i for i in range(len(state_facts)) if i not in state_relevant),
        graph=tuple(
            GraphEdge(variable, f"{profile.prefix.lower()}-minute-{minute}-p{priority}", value)
            for minute, priority, variable, value, _ in state_events
        ),
        trace=(f"state-query:{state_kind}", f"select:{state_answer}"),
        template=f"natural-network-log-{state_kind}",
        domain=profile.domains[2],
        difficulty=_difficulty(
            cardinality=len(state_facts),
            hops=1,
            distractors=len(state_facts) - len(state_relevant),
            state_events=len(state_events),
            variables=2,
        ),
        axes={
            "state_query": state_kind,
            "serialization": "naturalized-semantic-shuffled",
            "variable_interleaving": "two-variable",
            "analogue": "naturalistic",
        },
    )
    composition_facts = [
        f"Module {sample}-input writes buffer {values[0]}.",
        f"Buffer {values[0]} feeds converter {values[1]}.",
        f"Converter {values[1]} emits channel {values[2]}.",
        f"An unrelated monitor watches channel {values[3]}.",
    ]
    composition_order = list(range(4))
    rng.shuffle(composition_order)
    composition = _case(
        case_id=f"{profile.prefix}-natural-composition-{index}",
        family="natural_composition_analogue",
        profile=profile,
        seed=seed,
        facts=tuple(composition_facts[i] for i in composition_order),
        query=f"Which channel ultimately receives data from module {sample}-input?",
        answer=values[2],
        candidates=_candidate_order(rng, (values[0], values[1], values[2], values[3])),
        relevant=tuple(i for i, original in enumerate(composition_order) if original in {0, 1, 2}),
        distractors=(composition_order.index(3),),
        graph=(
            GraphEdge(f"{sample}-input", f"{profile.prefix.lower()}-writes", values[0]),
            GraphEdge(values[0], f"{profile.prefix.lower()}-feeds", values[1]),
            GraphEdge(values[1], f"{profile.prefix.lower()}-emits", values[2]),
        ),
        trace=(f"follow:{values[0]}", f"follow:{values[1]}", f"select:{values[2]}"),
        template="natural-dataflow",
        domain=profile.domains[2],
        difficulty=_difficulty(cardinality=4, hops=3, distractors=1),
    )
    return [binding, state, composition]


def build_evaluation_suite(
    split: Split,
    *,
    seed: int,
    groups_per_family: int = 8,
) -> EvaluationSuite:
    """Build a deterministic evaluation suite.

    A sealed seed must be supplied by external custody. The library deliberately
    has no default or embedded sealed seed.
    """

    if groups_per_family <= 0:
        raise ValueError("groups_per_family must be positive")
    if split is Split.SEALED and seed == 0:
        raise ValueError("sealed generation requires an externally held nonzero seed")
    profile = PROFILES[split]
    rng = random.Random(seed)
    cases: list[CausalCase] = []
    pairs: list[CausalPair] = []
    cardinalities = (2, 4, 8)
    for index in range(groups_per_family):
        local_seed = seed * 100_000 + index
        cases.append(_copy_case(profile, rng, local_seed, index))
        cases.append(_nonce_case(profile, rng, local_seed, index))
        binding_cases, binding_pairs = _binding_bundle(
            profile, rng, local_seed, index, cardinalities[index % len(cardinalities)]
        )
        cases.extend(binding_cases)
        pairs.extend(binding_pairs)
        state_cases, state_pairs = _state_bundle(profile, rng, local_seed, index)
        cases.extend(state_cases)
        pairs.extend(state_pairs)
        for hops in (1, 2, 3):
            cases.append(_relation_case(profile, rng, local_seed, index, hops))
            cases.append(
                _relation_case(profile, rng, local_seed, index, hops, direct_control=True)
            )
        cases.append(_missing_case(profile, rng, local_seed, index))
        counter_cases, counter_pairs = _counterfactual_bundle(profile, rng, local_seed, index)
        cases.extend(counter_cases)
        pairs.extend(counter_pairs)
        rule_case, rule_pair = _rule_case(profile, rng, local_seed, index)
        cases.extend((rule_case, rule_pair.changed))
        pairs.append(rule_pair)
        cases.extend(_natural_cases(profile, rng, local_seed, index))

    suite = EvaluationSuite(
        schema="esoes-e0-suite/v1",
        generator_version=GENERATOR_VERSION,
        split=split,
        cases=tuple(cases),
        pairs=tuple(pairs),
    )
    suite.assert_valid()
    return suite
