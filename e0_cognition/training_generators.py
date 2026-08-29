"""Training-only generators. Their language namespace is disjoint from E0 evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass


TRAINING_GENERATOR_VERSION = "e0-train/0.2.0"
TRAINING_TEMPLATE_PREFIX = "train.causal."


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
    generator_version: str
    seed: int
    provenance: tuple[tuple[str, str], ...]
    split_identity: str
    relevant_span: str
    contrast_answer: str

    def model_view(self) -> dict[str, str]:
        return {"context": self.context, "query": self.query}


def build_training_examples(*, seed: int, count: int = 256) -> tuple[TrainingExample, ...]:
    """Create deterministic causal examples without importing evaluation templates."""

    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    examples: list[TrainingExample] = []
    for index in range(count):
        keys = [f"tr-{rng.randrange(10_000, 99_999)}-{j}" for j in range(4)]
        values = [f"TX{rng.randrange(100_000, 999_999)}" for _ in range(4)]
        target = rng.randrange(4)
        mode = index % 3
        if mode == 0:
            facts = [f"Registry entry {key} carries payload {value}." for key, value in zip(keys, values)]
            query = f"Return the payload assigned to registry entry {keys[target]}."
            relevant = facts[target]
            template = f"{TRAINING_TEMPLATE_PREFIX}registry"
            graph = tuple((key, "train-payload", value) for key, value in zip(keys, values))
            distractors = tuple(fact for i, fact in enumerate(facts) if i != target)
            counterfactual_query = f"Return the payload assigned to registry entry {keys[(target + 1) % 4]}."
            counterfactual_answer = values[(target + 1) % 4]
            difficulty = (("cardinality", 4), ("hops", 0))
            relevant_variables = (keys[target], values[target])
        elif mode == 1:
            facts = [f"Revision {j + 1} sets {keys[0]} to {value}." for j, value in enumerate(values)]
            target = 3
            query = f"After all revisions, what is the value of {keys[0]}?"
            relevant = facts[-1]
            template = f"{TRAINING_TEMPLATE_PREFIX}revision"
            graph = tuple((keys[0], f"train-revision-{j + 1}", value) for j, value in enumerate(values))
            distractors = tuple(facts[:-1])
            counterfactual_query = f"What was the value of {keys[0]} after revision 1 only?"
            counterfactual_answer = values[0]
            difficulty = (("cardinality", 4), ("hops", 1))
            relevant_variables = (keys[0], values[target])
        else:
            facts = [
                f"{keys[0]} transfers to {keys[1]}.",
                f"{keys[1]} transfers to {keys[2]}.",
                f"{keys[2]} stores payload {values[2]}.",
                f"Unrelated {keys[3]} stores payload {values[3]}.",
            ]
            target = 2
            query = f"Which payload is reached by following transfers from {keys[0]}?"
            relevant = "\n".join(facts[:3])
            template = f"{TRAINING_TEMPLATE_PREFIX}transfer"
            graph = (
                (keys[0], "train-transfers", keys[1]),
                (keys[1], "train-transfers", keys[2]),
                (keys[2], "train-stores", values[2]),
                (keys[3], "train-stores", values[3]),
            )
            distractors = (facts[3],)
            counterfactual_query = f"Which payload is stored by unrelated {keys[3]}?"
            counterfactual_answer = values[3]
            difficulty = (("cardinality", 4), ("hops", 3))
            relevant_variables = (keys[0], keys[1], keys[2], values[target])
        examples.append(
            TrainingExample(
                example_id=f"train-{seed}-{index}",
                template_id=template,
                context="\n".join(facts),
                query=query,
                answer=values[target],
                causal_graph=graph,
                distractor_spans=distractors,
                relevant_variables=relevant_variables,
                counterfactual_query=counterfactual_query,
                counterfactual_answer=counterfactual_answer,
                difficulty=difficulty,
                generator_version=TRAINING_GENERATOR_VERSION,
                seed=seed * 100_000 + index,
                provenance=(("generator", TRAINING_GENERATOR_VERSION), ("source", "executable")),
                split_identity="training",
                relevant_span=relevant,
                contrast_answer=values[(target + 1) % 4],
            )
        )
    return tuple(examples)


def assert_training_eval_disjoint(training: tuple[TrainingExample, ...], eval_template_ids: set[str]) -> None:
    collisions = {example.template_id for example in training} & eval_template_ids
    if collisions:
        raise AssertionError(f"training/evaluation template collision: {sorted(collisions)}")
    if any(not example.template_id.startswith(TRAINING_TEMPLATE_PREFIX) for example in training):
        raise AssertionError("training example escaped its reserved namespace")
