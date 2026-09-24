"""Training-only generators. Their language namespace is disjoint from E0 evaluation."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping


TRAINING_GENERATOR_VERSION = "e0-train/0.4.0"
TRAINING_TEMPLATE_PREFIX = "train.causal."
TRAINING_COGNITION_FAMILIES = (
    "identity_copy",
    "query_binding",
    "semantic_state",
    "interference_retrieval",
    "relational_composition",
    "counterfactual_sensitivity",
    "heldout_rule_induction",
    "missing_information",
    "faithful_realization",
)
DEFAULT_DIFFICULTY_FRACTIONS = {"easy": 0.34, "medium": 0.355, "hard": 0.305}


@dataclass(frozen=True, slots=True)
class TrainingExample:
    example_id: str
    template_id: str
    context: str
    query: str
    answer: str
    causal_graph: tuple[tuple[str, str, str], ...]
    distractor_spans: tuple[str, ...]
    relevant_variables: tuple[str, ...]
    counterfactual_query: str
    counterfactual_answer: str
    difficulty: tuple[tuple[str, int], ...]
    surface_axes: tuple[tuple[str, str], ...]
    generator_version: str
    seed: int
    provenance: tuple[tuple[str, str], ...]
    split_identity: str
    relevant_span: str
    contrast_answer: str
    family: str
    surface: str
    difficulty_band: str

    def model_view(self) -> dict[str, str]:
        return {"context": self.context, "query": self.query}


def _apportion(count: int, fractions: Mapping[str, float], order: tuple[str, ...]) -> dict[str, int]:
    """Largest-remainder allocation with a stable tie break and exact total."""

    if set(fractions) != set(order) or not order:
        raise ValueError("allocation fractions must cover the declared categories exactly")
    raw_values = [fractions[name] for name in order]
    if any(isinstance(value, bool) for value in raw_values):
        raise ValueError("allocation fractions must be numeric weights, not booleans")
    try:
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("allocation fractions must be finite numeric weights") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("allocation fractions must be finite numeric weights")
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("allocation fractions must sum to one") from exc
    if any(value < 0 for value in values) or not math.isclose(
        total, 1.0, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("allocation fractions must be nonnegative and sum to one")
    quotas = [count * value for value in values]
    result = [int(quota) for quota in quotas]
    remainder = count - sum(result)
    ranked = sorted(range(len(order)), key=lambda i: (-(quotas[i] - result[i]), i))
    for index in ranked[:remainder]:
        result[index] += 1
    return dict(zip(order, result))


def _resolve_training_state(
    entity: str,
    cutoff: int,
    events: list[tuple[int, int, str, str, int | None]],
) -> tuple[str, set[tuple[int, int, str]]]:
    """Resolve one entity's state and the assignments/rollback that determine it."""

    eligible = sorted(
        (
            event
            for event in events
            if event[2] == entity and event[0] <= cutoff
        ),
        key=lambda event: (event[0], event[1]),
    )
    if not eligible:
        raise ValueError(f"no training state exists for {entity!r} at minute {cutoff}")
    minute, priority, _event_entity, value, rollback_time = eligible[-1]
    event_key = (minute, priority, entity)
    if rollback_time is None:
        return value, {event_key}
    restored, causal = _resolve_training_state(entity, rollback_time, events)
    return restored, causal | {event_key}


def build_training_examples(
    *,
    seed: int,
    count: int = 256,
    family_fractions: Mapping[str, float] | None = None,
    difficulty_fractions: Mapping[str, float] | None = None,
) -> tuple[TrainingExample, ...]:
    """Build executable examples across the nine frozen cognition families.

    This is a causal-LM training generator, not an evaluator: no evaluation
    template is imported, and model_view() exposes only context and query.
    Hidden graph truth and counterfactual answers remain metadata.
    """

    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    families = tuple(TRAINING_COGNITION_FAMILIES)
    family_mix = (
        family_fractions
        if family_fractions is not None
        else {name: 1.0 / len(families) for name in families}
    )
    if count < len(families):
        raise ValueError(f"count must be at least {len(families)} to cover every cognition family")
    family_base = {name: 1 for name in families}
    family_extra = _apportion(count - len(families), family_mix, families)
    family_counts = {name: family_base[name] + family_extra[name] for name in families}
    difficulty_mix = (
        difficulty_fractions
        if difficulty_fractions is not None
        else DEFAULT_DIFFICULTY_FRACTIONS
    )
    difficulty_order = ("easy", "medium", "hard")

    schedule: list[tuple[str, str]] = []
    for family in families:
        for band, amount in _apportion(family_counts[family], difficulty_mix, difficulty_order).items():
            schedule.extend((family, band) for _ in range(amount))
    schedule_rng = random.Random(seed ^ 0x5A17C0DE)
    schedule_rng.shuffle(schedule)

    rng = random.Random(seed)
    family_seen = {name: 0 for name in families}
    family_band_seen = {name: Counter() for name in families}
    examples: list[TrainingExample] = []
    for index, (family, band) in enumerate(schedule):
        local_index = family_seen[family]
        family_seen[family] += 1
        band_index = family_band_seen[family][band]
        family_band_seen[family][band] += 1
        # At least one quarter of every sufficiently represented family uses
        # natural or semi-natural language, crossed within the family.
        surface = ("natural", "semi_natural", "formal", "formal")[local_index % 4]
        keys = [f"R{value}" for value in rng.sample(range(10_000, 99_999), 40)]
        values = [f"TX{value}" for value in rng.sample(range(100_000, 999_999), 40)]
        facts: list[str] = []
        graph: list[tuple[str, str, str]] = []
        relevant_indices: list[int] = []
        answer = ""
        query = ""
        counterfactual_query = ""
        counterfactual_answer = ""
        relevant_variables: tuple[str, ...] = ()
        surface_axes: tuple[tuple[str, str], ...] = ()
        difficulty: tuple[tuple[str, int], ...] = (
            ("difficulty_level", {"easy": 1, "medium": 2, "hard": 3}[band]),
        )

        def record(subject: str, relation: str, obj: str) -> str:
            graph.append((subject, relation, obj))
            if surface == "natural":
                return f"The field log says {subject} {relation.replace('-', ' ')} {obj}."
            if surface == "semi_natural":
                return f"According to the record, {subject} {relation.replace('-', ' ')} {obj}."
            return f"Record: {subject} | {relation} | {obj}"

        if family == "identity_copy":
            cardinality = {"easy": 2, "medium": 4, "hard": 8}[band]
            target = rng.randrange(cardinality)
            for item in range(cardinality):
                facts.append(record(f"note-{item}", "contains-code", values[item]))
            answer = values[target]
            query = f"Copy the exact code written in note-{target}."
            counterfactual_query = f"Copy the exact code written in note-{(target + 1) % cardinality}."
            counterfactual_answer = values[(target + 1) % cardinality]
            relevant_indices = [target]
            relevant_variables = (f"note-{target}", answer)
            difficulty += (("cardinality", cardinality), ("hops", 0))

        elif family == "query_binding":
            cardinality = (
                2 if band == "easy"
                else (4 if local_index % 2 == 0 else 8) if band == "medium"
                else 16
            )
            target = rng.randrange(cardinality)
            for item in range(cardinality):
                facts.append(record(f"account-{keys[item]}", "has-payload", values[item]))
            answer = values[target]
            query = f"Which payload belongs to account-{keys[target]}?"
            counterfactual_query = f"Which payload belongs to account-{keys[(target + 1) % cardinality]}?"
            counterfactual_answer = values[(target + 1) % cardinality]
            relevant_indices = [target]
            relevant_variables = (keys[target], answer)
            difficulty += (("cardinality", cardinality), ("hops", 0))

        elif family == "semantic_state":
            entity_count = {"easy": 1, "medium": 2, "hard": 4}[band]
            update_count = {"easy": 2, "medium": 4, "hard": 8}[band]
            entities = [f"state-{keys[item]}" for item in range(entity_count)]
            target_entity = entities[-1]
            events: list[tuple[int, int, str, str, int | None]] = []
            for revision in range(update_count):
                entity = entities[revision % entity_count]
                events.append((revision * 3 + 1, 1, entity, values[revision], None))

            target_events = [event for event in events if event[2] == target_entity]
            query_kind = ("latest", "intermediate", "rollback", "precedence")[local_index % 4]
            if query_kind == "latest":
                cutoff = max(event[0] for event in events) + 1
                query = f"What value is active for {target_entity} after minute {cutoff}?"
            elif query_kind == "intermediate":
                cutoff = target_events[0][0] + 1
                query = (
                    f"What value was active for {target_entity} at minute {cutoff}, "
                    "between its recorded updates?"
                )
            elif query_kind == "rollback":
                rollback_event = target_events[-1]
                rollback_time = target_events[0][0]
                events[events.index(rollback_event)] = (
                    rollback_event[0], rollback_event[1], target_entity, "", rollback_time
                )
                cutoff = rollback_event[0] + 1
                query = (
                    f"After the approved rollback, what value was active for "
                    f"{target_entity} at minute {cutoff}?"
                )
            else:
                first_event, last_event = target_events[0], target_events[-1]
                precedence_minute = first_event[0]
                events.remove(first_event)
                events.remove(last_event)
                events.extend(
                    (
                        (precedence_minute, 1, target_entity, first_event[3], None),
                        (precedence_minute, 5, target_entity, last_event[3], None),
                    )
                )
                cutoff = precedence_minute
                query = (
                    f"After the same-minute updates at minute {cutoff}, which priority takes "
                    f"precedence for {target_entity}?"
                )

            answer, winning_events = _resolve_training_state(target_entity, cutoff, events)
            event_rows: list[tuple[tuple[int, int, str], str]] = []
            graph_rows: list[tuple[int, int, str, tuple[str, str, str]]] = []
            for minute, priority, entity, value, rollback_time in events:
                event_key = (minute, priority, entity)
                if rollback_time is None:
                    if surface == "natural":
                        fact = (
                            f"At minute {minute} (priority {priority}), the field log says "
                            f"{entity} was set to {value}."
                        )
                    elif surface == "semi_natural":
                        fact = (
                            f"The record lists {entity} as {value} at minute {minute} "
                            f"with priority {priority}."
                        )
                    else:
                        fact = f"Event time={minute} priority={priority}: {entity} := {value}."
                    edge = (f"{entity}@minute-{minute}-priority-{priority}", "sets-value", value)
                else:
                    if surface == "natural":
                        fact = (
                            f"At minute {minute} (priority {priority}), an approved rollback "
                            f"restored {entity} to its value at minute {rollback_time}."
                        )
                    elif surface == "semi_natural":
                        fact = (
                            f"A rollback returned {entity} to its minute {rollback_time} "
                            f"value at minute {minute} with priority {priority}."
                        )
                    else:
                        fact = (
                            f"Event time={minute} priority={priority}: {entity} := "
                            f"value@time={rollback_time}."
                        )
                    edge = (
                        f"{entity}@minute-{minute}-priority-{priority}",
                        f"rolls-back-to-minute-{rollback_time}",
                        "historical-value",
                    )
                event_rows.append((event_key, fact))
                graph_rows.append((minute, priority, entity, edge))

            serialization = list(range(len(event_rows)))
            rng.shuffle(serialization)
            facts = [event_rows[position][1] for position in serialization]
            relevant_indices = [
                position
                for position, event_row_index in enumerate(serialization)
                if event_rows[event_row_index][0] in winning_events
            ]
            graph = [row[3] for row in graph_rows]
            first_target_time = min(event[0] for event in events if event[2] == target_entity)
            counterfactual_query = (
                f"What value was active for {target_entity} immediately after "
                f"minute {first_target_time}?"
            )
            counterfactual_answer, _ = _resolve_training_state(
                target_entity, first_target_time, events
            )
            relevant_variables = (target_entity, answer)
            surface_axes = (("state_query", query_kind), ("serialization", "shuffled"))
            difficulty += (("state_variables", entity_count), ("state_updates", update_count))

        elif family == "interference_retrieval":
            distractor_options = {
                "easy": (0, 2),
                "medium": (4, 8),
                "hard": (16, 32),
            }[band]
            distractor_count = distractor_options[band_index % 2]
            target_key = keys[0]
            target_fact = record(f"shipment-{target_key}-north", "contains-payload", values[0])
            distractors = []
            if distractor_count:
                distractors.append(
                    record(f"shipment-{target_key}-south", "contains-payload", values[1])
                )
            distractors.extend(
                record(f"shipment-{keys[i + 1]}-north", "contains-payload", values[i + 1])
                for i in range(max(0, distractor_count - len(distractors)))
            )
            available_quartiles = [
                quartile
                for quartile in range(1, 5)
                if any(
                    min(4, (position * 4) // (len(distractors) + 1) + 1) == quartile
                    for position in range(len(distractors) + 1)
                )
            ]
            position_cycle = band_index // 2
            position_quartile = available_quartiles[position_cycle % len(available_quartiles)]
            position_support = [
                position
                for position in range(len(distractors) + 1)
                if min(4, (position * 4) // (len(distractors) + 1) + 1) == position_quartile
            ]
            position_repeat = position_cycle // len(available_quartiles)
            insert_at = position_support[position_repeat % len(position_support)]
            facts = distractors[:insert_at] + [target_fact] + distractors[insert_at:]
            answer = values[0]
            query = f"Find the payload for shipment-{target_key}-north."
            counterfactual_query = f"Find the payload for shipment-{target_key}-south."
            counterfactual_answer = values[1] if distractor_count else "<MISSING>"
            relevant_indices = [insert_at]
            relevant_variables = (target_key, "north", answer)
            position_quartile = min(
                4, (insert_at * 4) // (len(distractors) + 1) + 1
            )
            difficulty += (("distractors", distractor_count), ("context_position_quartile", position_quartile))
            surface_axes = (
                ("distractor_dose", str(distractor_count)),
                ("context_position_quartile", str(position_quartile)),
                ("dose_position_cell", f"{distractor_count}:{position_quartile}"),
            )

        elif family == "relational_composition":
            hops = {"easy": 1, "medium": 2, "hard": 3}[band]
            path = [keys[i] for i in range(hops + 1)]
            for source, target in zip(path, path[1:]):
                facts.append(record(source, "routes-to", target))
            answer = values[hops]
            facts.append(record(path[-1], "stores-payload", answer))
            facts.append(record(keys[10], "stores-payload", values[10]))
            query = f"Follow the route beginning at {path[0]}. What payload is stored at its destination?"
            counterfactual_query = f"What payload is stored directly at {keys[10]}?"
            counterfactual_answer = values[10]
            relevant_indices = list(range(hops + 1))
            relevant_variables = tuple(path) + (answer,)
            difficulty += (("hops", hops),)

        elif family == "counterfactual_sensitivity":
            original = values[0]
            intervention = values[1]
            facts.append(record(keys[0], "has-baseline-value", original))
            if surface == "natural":
                intervention_fact = f"The instruction replaces the value for {keys[0]} with {intervention}."
            elif surface == "semi_natural":
                intervention_fact = f"Apply this change: {keys[0]} now has value {intervention}."
            else:
                intervention_fact = f"Intervention: replace {keys[0]} with {intervention}."
            facts.append(intervention_fact)
            graph.append((keys[0], "intervention-sets-value", intervention))
            facts.append(record(keys[1], "has-value", values[2]))
            answer = intervention
            query = f"After applying the intervention, what value does {keys[0]} have?"
            counterfactual_query = f"Without applying the intervention, what value does {keys[0]} have?"
            counterfactual_answer = original
            relevant_indices = [0, 1]
            relevant_variables = (keys[0], original, intervention)
            difficulty += (("intervention", 1), ("distractors", 0 if band == "easy" else 2 if band == "medium" else 4))

        elif family == "heldout_rule_induction":
            step = rng.randrange(1, 10)
            demo_count = {"easy": 2, "medium": 4, "hard": 8}[band]
            inputs = rng.sample(range(10), demo_count + 1)
            for i, digit in enumerate(inputs[:-1]):
                output = (digit + step) % 10
                if surface == "natural":
                    facts.append(f"In one example, {digit} becomes {output}.")
                elif surface == "semi_natural":
                    facts.append(f"The recorded change maps {digit} to {output}.")
                else:
                    facts.append(f"Rule example {i + 1}: {digit} becomes {output}.")
                graph.append((str(digit), f"latent-add-{step}-mod-10", str(output)))
            answer = str((inputs[-1] + step) % 10)
            query = f"Apply the demonstrated rule to {inputs[-1]}. What number results?"
            counterfactual_query = f"Apply the demonstrated rule to {(inputs[-1] + 1) % 10}. What number results?"
            counterfactual_answer = str((inputs[-1] + 1 + step) % 10)
            relevant_indices = list(range(demo_count))
            relevant_variables = tuple(str(item) for item in inputs)
            difficulty += (("rule_demonstrations", demo_count),)

        elif family == "missing_information":
            cardinality = {"easy": 2, "medium": 4, "hard": 8}[band]
            for item in range(cardinality):
                facts.append(record(f"observed-{keys[item]}", "has-value", values[item]))
            absent_key = f"absent-{keys[cardinality]}"
            answer = "<MISSING>"
            query = f"What value is recorded for {absent_key}? If no value is present, answer <MISSING>."
            counterfactual_query = f"What value is recorded for observed-{keys[0]}?"
            counterfactual_answer = values[0]
            relevant_indices = []
            relevant_variables = (absent_key, answer)
            difficulty += (("cardinality", cardinality),)

        elif family == "faithful_realization":
            revision_count = {"easy": 2, "medium": 4, "hard": 8}[band]
            target = rng.randrange(revision_count)
            for revision in range(revision_count):
                facts.append(record(f"item-{keys[revision]} revision-{revision + 1}", "has-payload", values[revision]))
            answer = f"payload={values[target]}; revision={target + 1}"
            query = (
                f"Report the payload and its revision for item-{keys[target]} using exactly "
                "payload=<code>; revision=<number>."
            )
            counterfactual_query = (
                f"Report the payload and revision for item-{keys[(target + 1) % revision_count]} "
                "using the same format."
            )
            counterfactual_answer = (
                f"payload={values[(target + 1) % revision_count]}; revision={(target + 1) % revision_count + 1}"
            )
            relevant_indices = [target]
            relevant_variables = (keys[target], values[target], str(target + 1))
            difficulty += (("cardinality", revision_count), ("format_constraints", 2))
        else:  # pragma: no cover - the immutable family schedule makes this unreachable
            raise AssertionError(f"unimplemented cognition family: {family}")

        relevant_span = "\n".join(facts[item] for item in relevant_indices)
        distractor_spans = tuple(fact for i, fact in enumerate(facts) if i not in set(relevant_indices))
        examples.append(
            TrainingExample(
                example_id=f"train-{seed}-{index}",
                template_id=f"{TRAINING_TEMPLATE_PREFIX}{family}",
                context="\n".join(facts),
                query=query,
                answer=answer,
                causal_graph=tuple(graph),
                distractor_spans=distractor_spans,
                relevant_variables=relevant_variables,
                counterfactual_query=counterfactual_query,
                counterfactual_answer=counterfactual_answer,
                difficulty=difficulty,
                surface_axes=surface_axes,
                generator_version=TRAINING_GENERATOR_VERSION,
                seed=seed * 100_000 + index,
                provenance=(
                    ("generator", TRAINING_GENERATOR_VERSION),
                    ("source", "executable"),
                    ("family", family),
                    ("surface", surface),
                ),
                split_identity="training",
                relevant_span=relevant_span,
                contrast_answer=counterfactual_answer,
                family=family,
                surface=surface,
                difficulty_band=band,
            )
        )
    if len(examples) != count or any(not example.answer for example in examples):
        raise AssertionError("training generator violated its exact-count or target contract")
    return tuple(examples)


def assert_training_eval_disjoint(training: tuple[TrainingExample, ...], eval_template_ids: set[str]) -> None:
    collisions = {example.template_id for example in training} & eval_template_ids
    if collisions:
        raise AssertionError(f"training/evaluation template collision: {sorted(collisions)}")
    if any(not example.template_id.startswith(TRAINING_TEMPLATE_PREFIX) for example in training):
        raise AssertionError("training example escaped its reserved namespace")
