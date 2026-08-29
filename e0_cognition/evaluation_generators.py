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

GENERATOR_VERSION = "e0-eval/0.2.0"


@dataclass(frozen=True, slots=True)
class SplitProfile:
    split: Split
    prefix: str
    relations: tuple[str, ...]
    domains: tuple[str, ...]
    template_prefix: str


PROFILES = {
    Split.DEVELOPMENT: SplitProfile(
        Split.DEVELOPMENT,
        "DV",
        ("guides", "links", "feeds", "indexes"),
        ("inventory", "laboratory", "technical-manual"),
        "dev",
    ),
    Split.SEALED: SplitProfile(
        Split.SEALED,
        "SL",
        ("supports", "orbits", "routes", "anchors"),
        ("legal-clause", "geology", "manufacturing"),
        "sealed",
    ),
    Split.FRESH: SplitProfile(
        Split.FRESH,
        "FR",
        ("buffers", "maps", "powers", "shields"),
        ("network-log", "ecology", "clinical-procedure"),
        "fresh",
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
    variable = _entity(profile, rng, index)
    values = [_code(profile, rng) for _ in range(3)]
    facts = tuple(f"At time {i + 1}, {variable} became {value}." for i, value in enumerate(values))
    state_relation = lambda i: f"{profile.prefix.lower()}-state-at-{i + 1}"
    graph = tuple(GraphEdge(variable, state_relation(i), value) for i, value in enumerate(values))
    replacement = _code(profile, rng)
    base = _case(
        case_id=f"{profile.prefix}-state-{index}-base",
        family="state_overwrite",
        profile=profile,
        seed=seed,
        facts=facts,
        query=f"What is the newest state of {variable}?",
        answer=values[-1],
        candidates=_candidate_order(rng, tuple(values) + (replacement,)),
        relevant=(2,),
        distractors=(0, 1),
        graph=graph,
        trace=("order:time", f"overwrite:{values[-1]}"),
        template="state-log",
        domain=profile.domains[2],
        difficulty=_difficulty(cardinality=3, hops=1, distractors=2),
    )
    changed_facts = list(facts)
    changed_facts[-1] = f"At time 3, {variable} became {replacement}."
    changed_graph = list(graph)
    changed_graph[-1] = GraphEdge(variable, state_relation(2), replacement)
    swapped = replace(
        base,
        case_id=f"{profile.prefix}-state-{index}-changed",
        facts=tuple(changed_facts),
        answer=replacement,
        hidden=HiddenTruth((2,), (0, 1), tuple(changed_graph), ("order:time", f"overwrite:{replacement}")),
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
        relevant = (0,)
        trace = (f"retrieve:{nodes[-1]}",)
        family = "matched_direct_retrieval"
        template = f"direct-{hops}"
    else:
        relevant = tuple(range(hops))
        trace = tuple(f"follow:{edge.relation}->{edge.target}" for edge in edges)
        family = f"relation_{hops}_hop"
        template = f"relation-{hops}"
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


def _rule_case(profile: SplitProfile, rng: random.Random, seed: int, index: int) -> CausalCase:
    a, b, c, d = (_code(profile, rng) for _ in range(4))
    operation = f"KEL-{profile.prefix}"
    facts = (
        f"Operation {operation} on pair ({a}, {b}) returns {b}|{a}.",
        f"Operation {operation} on pair ({c}, {d}) returns {d}|{c}.",
    )
    x, y = _code(profile, rng), _code(profile, rng)
    answer = f"{y}|{x}"
    return _case(
        case_id=f"{profile.prefix}-rule-{index}",
        family="rule_induction",
        profile=profile,
        seed=seed,
        facts=facts,
        query=f"Apply {operation} to the unseen pair ({x}, {y}).",
        answer=answer,
        candidates=_candidate_order(rng, (answer, f"{x}|{y}", x, y)),
        relevant=(0, 1),
        distractors=(),
        graph=(
            GraphEdge(
                operation,
                f"{profile.prefix.lower()}-maps",
                f"{profile.prefix.lower()}-reverse-pair",
            ),
        ),
        trace=("infer:reverse-pair", f"apply:{x},{y}", f"select:{answer}"),
        template="rule-induction",
        domain=profile.domains[1],
        difficulty=_difficulty(cardinality=2, hops=2, distractors=0),
    )


def _natural_cases(profile: SplitProfile, rng: random.Random, seed: int, index: int) -> list[CausalCase]:
    sample = _entity(profile, rng, index)
    values = [_code(profile, rng) for _ in range(4)]
    binding = _case(
        case_id=f"{profile.prefix}-natural-binding-{index}",
        family="natural_binding_analogue",
        profile=profile,
        seed=seed,
        facts=(
            f"The assay summary assigns specimen {sample}-A the accession {values[0]}, while specimen {sample}-B carries {values[1]}.",
            f"A calibration control uses accession {values[2]} and is not a specimen result.",
        ),
        query=f"Which accession belongs to specimen {sample}-B?",
        answer=values[1],
        candidates=_candidate_order(rng, tuple(values[:3])),
        relevant=(0,),
        distractors=(1,),
        graph=(GraphEdge(f"{sample}-B", f"{profile.prefix.lower()}-accession", values[1]),),
        trace=(f"address:{sample}-B", f"select:{values[1]}"),
        template="natural-assay",
        domain=profile.domains[1],
        difficulty=_difficulty(cardinality=3, hops=0, distractors=1),
    )
    state = _case(
        case_id=f"{profile.prefix}-natural-state-{index}",
        family="natural_state_analogue",
        profile=profile,
        seed=seed,
        facts=(
            f"At 09:00 the router configuration for node {sample} was {values[0]}.",
            f"At 11:30 maintenance changed it to {values[1]}.",
            f"At 14:10 the approved rollback set it to {values[2]}.",
        ),
        query=f"What configuration is current for node {sample} after the log?",
        answer=values[2],
        candidates=_candidate_order(rng, tuple(values[:3])),
        relevant=(2,),
        distractors=(0, 1),
        graph=tuple(
            GraphEdge(sample, f"{profile.prefix.lower()}-time-{j}", v)
            for j, v in enumerate(values[:3])
        ),
        trace=("order:timestamps", f"select-latest:{values[2]}"),
        template="natural-network-log",
        domain=profile.domains[2],
        difficulty=_difficulty(cardinality=3, hops=1, distractors=2),
    )
    composition = _case(
        case_id=f"{profile.prefix}-natural-composition-{index}",
        family="natural_composition_analogue",
        profile=profile,
        seed=seed,
        facts=(
            f"Module {sample}-input writes buffer {values[0]}.",
            f"Buffer {values[0]} feeds converter {values[1]}.",
            f"Converter {values[1]} emits channel {values[2]}.",
            f"An unrelated monitor watches channel {values[3]}.",
        ),
        query=f"Which channel ultimately receives data from module {sample}-input?",
        answer=values[2],
        candidates=_candidate_order(rng, (values[0], values[1], values[2], values[3])),
        relevant=(0, 1, 2),
        distractors=(3,),
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
        cases.append(_rule_case(profile, rng, local_seed, index))
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
