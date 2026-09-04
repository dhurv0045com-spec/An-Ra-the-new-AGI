"""Binding generator v2: pair-preserving counterfactual cognition data.

V1 failed scientifically (not mechanically): one surface template, the
query entity sharing exactly one sentence with its answer, unattested
distractor candidates, and no pair-destroying controls, so bag-of-words
reaches 100% and raw accuracy cannot gate selection. V2 separates latent
semantic instances (pairings) from surface renderers (five grammars),
generates query-swap groups, pair-destroyed controls, value-swap variants,
counterbalanced positions, and split-disjoint lexicons. The unit of
evidence is the PAIR and the GROUP, never a lone raw accuracy.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


GENERATOR_ID = "anra-binding-v2"
GENERATOR_VERSION = "2.0.0"

GRAMMARS = ("s1-declarative", "s2-assignment", "s3-record", "s4-table", "s5-natural")

# Interference grammars: versioned registrations where sentence retrieval
# alone cannot select among an entity's competing values.
IGRAMMARS = ("i1-register-update", "i2-record-revise")

# Structural Fresh uses grammars and value formats unseen in train/dev.
FRESH_GRAMMARS = ("s4-table", "s5-natural")
FRESH_VALUE_PREFIX = "FX"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_fact(grammar: str, entity: str, value: str) -> str:
    """Render one entity/value pairing through an independent grammar."""

    if grammar == "s1-declarative":
        return f"The tag for {entity} is {value}."
    if grammar == "s2-assignment":
        return f"Code assigned to {entity}: {value}."
    if grammar == "s3-record":
        return f"Record({entity}) = {value}."
    if grammar == "s4-table":
        return f"| {entity} | {value} |"
    if grammar == "s5-natural":
        return f"We registered {entity} under code {value}."
    raise ValueError(f"unknown binding grammar: {grammar}")


def render_query(grammar: str, entity: str) -> str:
    """Render the query for one entity through the same grammar family."""

    if grammar == "s1-declarative":
        return f"Which tag belongs to {entity}?"
    if grammar == "s2-assignment":
        return f"Which code was assigned to {entity}?"
    if grammar == "s3-record":
        return f"What is Record({entity})?"
    if grammar == "s4-table":
        return f"Which code sits beside {entity} in the table?"
    if grammar == "s5-natural":
        return f"Under which code did we register {entity}?"
    raise ValueError(f"unknown binding grammar: {grammar}")


def render_registration(grammar: str, entity: str, value: str, *, initial: bool) -> str:
    """Render one versioned registration event (interference mode)."""

    if grammar == "i1-register-update":
        verb = "Registered" if initial else "Updated"
        return f"{verb} {entity} as {value}."
    if grammar == "i2-record-revise":
        verb = "Record" if initial else "Revise"
        return f"{verb}({entity}) = {value}."
    raise ValueError(f"unknown interference grammar: {grammar}")


def render_selector(grammar: str, entity: str, *, which: str) -> str:
    """Render a version-selecting query (current vs first)."""

    if which not in {"current", "first"}:
        raise ValueError("selector must be current or first")
    if grammar == "i1-register-update":
        return (
            f"What is the current code for {entity}?"
            if which == "current"
            else f"What was the first code for {entity}?"
        )
    if grammar == "i2-record-revise":
        return (
            f"What does Record({entity}) hold now?"
            if which == "current"
            else f"What did Record({entity}) hold first?"
        )
    raise ValueError(f"unknown interference grammar: {grammar}")


@dataclass(frozen=True, slots=True)
class LatentInstance:
    pairing: tuple[tuple[str, str], ...]
    queried: int
    grammar: str
    split: str

    def cluster_id(self) -> str:
        return _sha256_hex(
            _canonical_json({"pairing": [list(pair) for pair in self.pairing]})
        )

    def answer(self) -> str:
        return self.pairing[self.queried][1]


@dataclass(frozen=True, slots=True)
class BindingCase:
    case_id: str
    family: str
    split: str
    cluster_id: str
    control_of: str | None
    control_kind: str | None
    grammar: str
    facts: tuple[str, ...]
    query: str
    candidates: tuple[str, ...]
    gold: str
    target_position: int
    difficulty: tuple[tuple[str, int], ...]

    def prompt(self) -> str:
        return "\n".join(self.facts) + "\n" + self.query

    def model_text(self) -> str:
        return self.prompt() + "\n" + self.gold


def _lexicon(rng: random.Random, prefix: str, count: int, digits: int = 3) -> list[str]:
    pool = [f"{prefix}-{rng.randrange(10**digits):0{digits}d}" for _ in range(count * 8)]
    seen: list[str] = []
    for token in pool:
        if token not in seen:
            seen.append(token)
        if len(seen) == count:
            break
    if len(seen) != count:
        raise ValueError("lexicon pool exhausted")
    return seen


def generate_group(
    *,
    seed: int,
    group_index: int,
    cardinality: int,
    split: str,
    structural_fresh: bool = False,
    mode: str = "clean",
) -> tuple[list[BindingCase], dict[str, object]]:
    """Generate one query group: same facts, every entity queried once.

    Returns cases plus an aux record ``{"pairing": ..., "histories": ...}``
    (kept out of model inputs). Mode ``clean`` is single-version
    calibration (documented as ungated for selection: sentence
    co-occurrence solves it). Mode ``interference`` registers two versions
    per entity with current/first selectors; sentence retrieval alone
    cannot disambiguate.
    """

    if cardinality < 2:
        raise ValueError("binding groups need at least two mappings")
    if mode == "interference":
        return _generate_interference(
            seed=seed, group_index=group_index, cardinality=cardinality,
            split=split, structural_fresh=structural_fresh,
        )
    if mode != "clean":
        raise ValueError(f"unknown binding mode: {mode}")
    rng = random.Random(hashlib.sha256(f"{seed}/{group_index}".encode()).hexdigest())
    value_prefix = FRESH_VALUE_PREFIX if structural_fresh else "VX"
    entity_prefix = "EZ" if structural_fresh else "EN"
    entities = _lexicon(rng, entity_prefix, cardinality)
    values = _lexicon(rng, value_prefix, cardinality)
    pairing = tuple(zip(entities, values))
    grammars = FRESH_GRAMMARS if structural_fresh else tuple(g for g in GRAMMARS if g not in FRESH_GRAMMARS)
    grammar = grammars[group_index % len(grammars)]
    rotation = group_index % cardinality
    ordered = pairing[rotation:] + pairing[:rotation]
    cases = []
    for position, (entity, value) in enumerate(ordered):
        queried = pairing.index((entity, value))
        instance = LatentInstance(pairing=pairing, queried=queried, grammar=grammar, split=split)
        facts = tuple(render_fact(grammar, entity, value) for entity, value in ordered)
        candidates = tuple(sorted(value for _, value in pairing))
        case_id = f"bindv2-{split}-g{group_index}-{grammar}-q{position}"
        cases.append(
            BindingCase(
                case_id=case_id, family="entity_value_binding_v2", split=split,
                cluster_id=instance.cluster_id(), control_of=None, control_kind=None,
                grammar=grammar, facts=facts, query=render_query(grammar, entity),
                candidates=candidates, gold=value, target_position=position,
                difficulty=(("cardinality", cardinality), ("table_grammar", int(grammar == "s4-table"))),
            )
        )
    return cases, {"pairing": pairing, "histories": None}


def _generate_interference(
    *, seed: int, group_index: int, cardinality: int, split: str,
    structural_fresh: bool,
) -> tuple[list[BindingCase], tuple[tuple[str, str], ...]]:
    """Versioned registrations: each entity maps to first AND current values.

    The pairing returned binds entities to CURRENT values (the default
    query); first-value queries resolve against per-entity histories kept in
    the case cluster record. Fact order interleaves versions so no entity's
    versions cluster adjacently by default.
    """

    rng = random.Random(hashlib.sha256(f"{seed}/i{group_index}".encode()).hexdigest())
    value_prefix = FRESH_VALUE_PREFIX if structural_fresh else "VX"
    entity_prefix = "EZ" if structural_fresh else "EN"
    entities = _lexicon(rng, entity_prefix, cardinality)
    all_values = _lexicon(rng, value_prefix, 2 * cardinality)
    first_values, current_values = all_values[:cardinality], all_values[cardinality:]
    pairing = tuple(zip(entities, current_values))
    histories = {
        entity: (first, current)
        for entity, first, current in zip(entities, first_values, current_values)
    }
    grammars = ("i1-register-update", "i2-record-revise")
    grammar = grammars[group_index % len(grammars)]
    selectors = ("current", "first")
    facts: list[str] = []
    for entity in entities:
        facts.append(render_registration(grammar, entity, histories[entity][0], initial=True))
    for entity in reversed(entities):
        facts.append(render_registration(grammar, entity, histories[entity][1], initial=False))
    order_seed = random.Random(hashlib.sha256(f"{seed}/o{group_index}".encode()).hexdigest())
    order_seed.shuffle(facts)
    candidates = tuple(sorted(set(first_values) | set(current_values)))
    cases = []
    sequence = [(entity, which) for entity in entities for which in selectors]
    for position, (entity, which) in enumerate(sequence):
        gold = histories[entity][0] if which == "first" else histories[entity][1]
        case_id = f"bindv2-{split}-ig{group_index}-{grammar}-{which}-{position}"
        cluster_id = _sha256_hex(
            _canonical_json(
                {
                    "histories": [[entity, first, current] for entity, (first, current) in sorted(histories.items())],
                    "queried": entity,
                    "which": which,
                }
            )
        )
        target_position = next(
            index for index, fact in enumerate(facts)
            if entity in fact and gold in fact
        )
        cases.append(
            BindingCase(
                case_id=case_id, family="entity_value_binding_v2", split=split,
                cluster_id=cluster_id, control_of=None, control_kind=None,
                grammar=grammar, facts=tuple(facts),
                query=render_selector(grammar, entity, which=which),
                candidates=candidates, gold=gold,
                target_position=target_position,
                difficulty=(
                    ("cardinality", cardinality), ("versions", 2),
                    ("selector", 0 if which == "current" else 1),
                ),
            )
        )
    return cases, {"pairing": pairing, "histories": histories}


def interference_pair_control(
    case: BindingCase, *, histories: dict[str, tuple[str, str]]
) -> BindingCase:
    """Permute current values across entities, preserving the token multiset.

    First-version registrations stay fixed; current values rotate to new
    entities. The queried entity's correct current value changes while every
    bag feature stays near-identical.
    """

    entities = sorted(histories)
    currents = [histories[entity][1] for entity in entities]
    rotated = currents[1:] + currents[:1]
    swapped = {entity: value for entity, value in zip(entities, rotated)}
    query_entity = next(
        entity for entity in entities if entity in case.query
    )
    new_gold = swapped[query_entity]
    if new_gold == case.gold:
        rotated = currents[2:] + currents[:2]
        swapped = {entity: value for entity, value in zip(entities, rotated)}
        new_gold = swapped[query_entity]
    facts = []
    for fact in case.facts:
        replaced = fact
        for entity in entities:
            if entity in fact and histories[entity][1] in fact:
                replaced = fact.replace(histories[entity][1], swapped[entity])
                break
        facts.append(replaced)
    return BindingCase(
        case_id=case.case_id + "-pairdestroyed", family=case.family, split=case.split,
        cluster_id=_sha256_hex(_canonical_json({"interference_control": case.case_id})),
        control_of=case.cluster_id, control_kind="pair_destroyed",
        grammar=case.grammar, facts=tuple(facts), query=case.query,
        candidates=case.candidates, gold=new_gold,
        target_position=case.target_position,
        difficulty=case.difficulty,
    )


def pair_destroyed_control(case: BindingCase, *, pairing: tuple[tuple[str, str], ...]) -> BindingCase:
    """Re-permute pair assignments keeping the exact token multiset.

    Same entities, same values, same lengths and frequencies; the queried
    entity now maps elsewhere so the correct answer changes. A
    bag-of-words model receives near-identical features and must fail the
    pair unless it uses pair structure. Clean-mode grammars only; use
    interference_pair_control for versioned histories.
    """

    if case.grammar.startswith("i"):
        raise ValueError("use interference_pair_control for interference histories")

    entities = [entity for entity, _ in pairing]
    values = [value for _, value in pairing]
    query_entity = queried_entity(case, pairing)
    deranged = _derange(entities, values, query_entity, case.gold)
    facts = tuple(render_fact(case.grammar, entity, value) for entity, value in deranged)
    new_gold = dict(deranged)[query_entity]
    if new_gold == case.gold:
        raise ValueError("derangement preserved the queried pairing")
    return BindingCase(
        case_id=case.case_id + "-pairdestroyed", family=case.family, split=case.split,
        cluster_id=_sha256_hex(_canonical_json({"deranged": [list(p) for p in deranged]})),
        control_of=case.cluster_id, control_kind="pair_destroyed",
        grammar=case.grammar, facts=facts, query=case.query,
        candidates=case.candidates, gold=new_gold,
        target_position=next(i for i, (entity, _) in enumerate(deranged) if entity == query_entity),
        difficulty=case.difficulty,
    )


def value_swap_control(case: BindingCase, *, pairing: tuple[tuple[str, str], ...]) -> BindingCase:
    """Swap the queried value with another, preserving skeleton and query."""

    query_entity = queried_entity(case, pairing)
    query_index = [entity for entity, _ in pairing].index(query_entity)
    other_index = (query_index + 1) % len(pairing)
    swapped = list(pairing)
    swapped[query_index], swapped[other_index] = (
        (swapped[query_index][0], swapped[other_index][1]),
        (swapped[other_index][0], swapped[query_index][1]),
    )
    swapped = tuple(swapped)
    resolved = dict(swapped)
    if resolved[query_entity] == case.gold:
        raise ValueError("value swap preserved the queried pairing")
    facts = tuple(render_fact(case.grammar, entity, value) for entity, value in swapped)
    return BindingCase(
        case_id=case.case_id + "-valueswap", family=case.family, split=case.split,
        cluster_id=_sha256_hex(_canonical_json({"valueswap": [list(p) for p in swapped]})),
        control_of=case.cluster_id, control_kind="value_swap",
        grammar=case.grammar, facts=facts, query=case.query,
        candidates=case.candidates, gold=resolved[query_entity],
        target_position=next(i for i, (entity, _) in enumerate(swapped) if entity == query_entity),
        difficulty=case.difficulty,
    )


def queried_entity(case: BindingCase, pairing: tuple[tuple[str, str], ...]) -> str:
    """Recover the queried entity by matching query content against the pairing."""

    for entity, _ in pairing:
        if entity in case.query:
            return entity
    raise ValueError("cannot recover the queried entity")


def _query_entity(_case: BindingCase) -> str:
    raise ValueError("use queried_entity(case, pairing)")


def _derange(
    entities: list[str], values: list[str], query_entity: str, forbidden_gold: str
) -> tuple[tuple[str, str], ...]:
    query_index = entities.index(query_entity)
    rotation = [values[(i + 1) % len(values)] for i in range(len(values))]
    if rotation[query_index] == forbidden_gold:
        rotation = [values[(i + 2) % len(values)] for i in range(len(values))]
    return tuple(zip(entities, rotation))


def truth_solver(
    case: BindingCase, *, pairing: tuple[tuple[str, str], ...],
    histories: dict[str, tuple[str, str]] | None = None,
) -> str:
    """Perfect solver from latent structure (never exposed to the model).

    Clean cases resolve against the pairing; interference cases resolve the
    version selector against per-entity histories.
    """

    entity = queried_entity(case, pairing)
    if histories is not None:
        which = dict(case.difficulty).get("selector", 0)
        first, current = histories[entity]
        return first if which == 1 else current
    return dict(pairing)[entity]


__all__ = [
    "FRESH_GRAMMARS",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "GRAMMARS",
    "BindingCase",
    "LatentInstance",
    "generate_group",
    "interference_pair_control",
    "pair_destroyed_control",
    "queried_entity",
    "truth_solver",
    "render_fact",
    "render_query",
    "render_registration",
    "render_selector",
    "value_swap_control",
]
