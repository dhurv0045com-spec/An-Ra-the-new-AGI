"""World-disjoint, variable-render relational retrieval development fixture."""

from dataclasses import dataclass
from itertools import combinations
import random

from v5_evaluation.firewall import EvaluatorTruth, VisibleTask

from .benchmark import Example, VALUES
from .codec import Prediction, answer_tokens, digest


KNOWN_KEYS = "IJKLMNOPQRST"
NOVEL_KEYS = "UVWXYZ"
ROBUST_PANELS = ("train", "alias", "two-facts", "five-facts", "query-neutral")


@dataclass(frozen=True, slots=True)
class World:
    targets: tuple[str, str]
    distractors: tuple[str, ...]
    key_indices: tuple[int, ...]
    order: tuple[int, ...]

    @property
    def cluster(self) -> str:
        return digest(["relational-v1", sorted((*self.targets, self.distractors[0]))])


def worlds(split: str) -> tuple[World, ...]:
    if split not in ("train", "development"):
        raise ValueError("relational fixtures are development only")
    result = []
    seen = set()
    for index, values in enumerate(combinations(VALUES, 3)):
        if (index % 5 == 0) != (split == "development"):
            continue
        rng = random.Random(int(digest(["relational-v1", values]), 16))
        # Four views per latent value set. Keep the same worlds across stress panels.
        for _ in range(4):
            while True:
                first, second, distractor = rng.sample(values, 3)
                extra = rng.sample([value for value in VALUES if value not in values], 2)
                indices = tuple(rng.sample(range(len(KNOWN_KEYS)), 5))
                order = tuple(rng.sample(range(5), 5))
                signature = tuple(sorted(zip(indices[:3], (first, second, distractor))))
                if signature not in seen:
                    seen.add(signature)
                    break
            result.append(World((first, second), (distractor, *extra), indices, order))
    return tuple(result)


def relational_fixture(split: str, panel: str = "base") -> tuple[Example, ...]:
    if panel not in ("base", *ROBUST_PANELS):
        raise ValueError("unknown relational panel")
    if panel == "train" and split != "train":
        raise ValueError("training panel requires training partition")
    if split == "train" and panel not in ("base", "train"):
        raise ValueError("stress panels require development partition")
    result = []
    for world_index, world in enumerate(worlds(split)):
        count = {"two-facts": 2, "five-facts": 5}.get(panel, 3)
        keys = tuple(KNOWN_KEYS[index] for index in world.key_indices)
        separator = "#" if world_index % 2 else ";"
        if panel == "alias":
            rng = random.Random(int(digest(["alias", world_index]), 16))
            keys = tuple(rng.sample(NOVEL_KEYS, 5))
            separator = "-"  # An entirely new separator and all new entity symbols.
        values = (*world.targets, *world.distractors)
        facts = separator.join(f"{keys[index]}:{values[index]}"
                               for index in world.order if index < count)
        for query_index in range(2):
            task_id = f"{world.cluster}/{world_index}/{query_index}/{panel}"
            task = VisibleTask(task_id, world.cluster, "binding", split, "relational-v1",
                               f"B:{facts}?{keys[query_index]}>", ())
            result.append(Example(task, EvaluatorTruth(task_id, world.targets[query_index])))
    return tuple(result)


def positional_control(rows: tuple[Example, ...], *, last: bool) -> tuple[Prediction, ...]:
    """Ignore query and emit the first or last factual value."""
    result = []
    for row in rows:
        facts = row.visible.prompt.partition("?")[0][2:]
        values = [facts[index + 1] for index, token in enumerate(facts) if token == ":"]
        result.append(Prediction(row.visible.task_id, answer_tokens(values[-1 if last else 0])))
    return tuple(result)
