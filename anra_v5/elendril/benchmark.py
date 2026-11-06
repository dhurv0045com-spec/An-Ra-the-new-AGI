"""Deterministic synthetic train and development surfaces and shortcut controls."""

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations, permutations, product
import random

from v5_evaluation.firewall import EvaluatorTruth, VisibleTask

from .codec import EOS, Prediction, answer_tokens, digest, encode


FAMILIES = ("copy", "binding", "abstention", "state", "composition", "primitive")
VALUES = "ABCDEFGH"
OP_SYMBOLS = "JKLM"


@dataclass(frozen=True, slots=True)
class Example:
    visible: VisibleTask
    truth: EvaluatorTruth


def apply(op: str, value: int) -> int:
    return {"J": value + 1, "K": value * 2, "L": -value, "M": value + 3}[op] % 7


def fixture(family: str, split: str) -> tuple[Example, ...]:
    """Partition by world (or unordered operation pair), then expand views."""
    if split not in ("train", "development"):
        raise ValueError("only train and development fixtures exist")
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    rows = []

    def add(world: tuple, view: str, prompt: str, gold: str) -> None:
        cluster = digest([family, world])
        task_id = f"{cluster}/{view}"
        visible = VisibleTask(task_id, cluster, family, split, "v1", prompt, ())
        rows.append(Example(visible, EvaluatorTruth(task_id, gold)))

    if family == "copy":
        worlds = list(product(VALUES, repeat=3))
        random.Random(718).shuffle(worlds)
        for index, world in enumerate(worlds):
            if (index % 4 == 0) == (split == "development"):
                text = "".join(world)
                add(world, "copy", f"C:{text}>", text)
    elif family in ("binding", "abstention"):
        for index, world in enumerate(combinations(VALUES, 2)):
            if (index % 4 == 0) != (split == "development"):
                continue
            for x, y in permutations(world):
                facts = (f"X:{x}", f"Y:{y}")
                for order, (left, right) in enumerate(permutations(facts)):
                    for query in ("XY" if family == "binding" else "XYZ"):
                        prefix = "B" if family == "binding" else "M"
                        gold = {"X": x, "Y": y, "Z": "!"}[query]
                        add(world, f"{x}{y}{order}{query}", f"{prefix}:{left};{right}?{query}>", gold)
    elif family == "primitive":
        for op in OP_SYMBOLS:
            for initial in range(7):
                if (initial in (2, 5)) == (split == "development"):
                    add((op, initial), "primitive", f"U:{initial};{op}>", str(apply(op, initial)))
    else:
        for op_a, op_b in combinations(OP_SYMBOLS, 2):
            pair = (op_a, op_b)
            for initial in range(7):
                if family == "composition":
                    held_out = pair == ("J", "L")
                    world = pair
                else:
                    held_out = initial in (2, 5)
                    world = (*pair, initial)
                if held_out != (split == "development"):
                    continue
                forward = apply(op_b, apply(op_a, initial))
                reverse = apply(op_a, apply(op_b, initial))
                if forward == reverse:
                    continue
                prefix = "P" if family == "composition" else "S"
                add(world, f"{initial}{op_a}{op_b}", f"{prefix}:{initial};{op_a}{op_b}>", str(forward))
                add(world, f"{initial}{op_b}{op_a}", f"{prefix}:{initial};{op_b}{op_a}>", str(reverse))
    return tuple(rows)


def corrupt_binding(rows: tuple[Example, ...], seed: int) -> tuple[Example, ...]:
    """Permute the two labels by a fair coin per latent world, across all views."""
    rng = random.Random(seed)
    values: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.visible.family != "binding" or row.visible.split != "train":
            raise ValueError("label control requires binding training rows")
        values[row.visible.cluster_id].append(row.truth.gold)
    mappings = {}
    for cluster, answers in sorted(values.items()):
        choices = sorted(set(answers))
        if len(choices) != 2:
            raise ValueError("binding world must contain two distinct labels")
        mappings[cluster] = dict(zip(choices, reversed(choices) if rng.randrange(2) else choices))
    return tuple(Example(row.visible, EvaluatorTruth(
        row.truth.task_id, mappings[row.visible.cluster_id][row.truth.gold]
    )) for row in rows)


def shortcut(task: VisibleTask, policy: str, rng: random.Random) -> Prediction:
    """Prompt-only controls use declared format and vocabulary, with no gold."""
    prompt = task.prompt
    if policy == "deliberate_failure":
        tokens = (EOS,)
    elif policy == "length":
        tokens = answer_tokens("AAA" if task.family == "copy" else "A")
    elif policy == "fixed":
        tokens = answer_tokens("0" if task.family in ("state", "composition", "primitive") else "A")
    elif policy == "format_copy":
        tokens = (*encode(prompt[2:-1]), EOS)
    elif policy == "recency":
        text = prompt.split("?")[0][-1] if "?" in prompt else prompt[-2]
        tokens = answer_tokens(text)
    elif policy == "query_blind":
        tokens = answer_tokens(prompt[4] if "?" in prompt else prompt[2])
    elif policy == "random":
        if "?" in prompt:
            choices = prompt[4] + prompt[8]
            if task.family == "abstention":
                choices += "!"
            text = rng.choice(choices)
        elif task.family == "copy":
            text = "".join(rng.choices(VALUES, k=3))
        else:
            text = rng.choice("0123456")
        tokens = answer_tokens(text)
    else:
        raise ValueError(f"unknown shortcut {policy!r}")
    return Prediction(task.task_id, tokens)


POLICIES = ("deliberate_failure", "length", "fixed", "format_copy", "recency",
            "query_blind", "random")


BINDING_PANELS = ("train", "alias", "distractor", "query-neutral")


def binding_panel(panel: str) -> tuple[Example, ...]:
    """Development stress views reuse latent worlds and keep their original truth."""
    if panel == "train":
        return fixture("binding", "train")
    rows = fixture("binding", "development")
    if panel == "query-neutral":
        return rows  # The model-facing query is replaced after projecting the visible rows.
    if panel not in ("alias", "distractor"):
        raise ValueError("unknown binding stress panel")
    result = []
    for row in rows:
        task = row.visible
        if panel == "alias":
            prompt = task.prompt.translate(str.maketrans({"X": "U", "Y": "V", ";": "#"}))
        else:
            prefix, query = task.prompt.split("?")
            left, right = prefix[2:].split(";")
            extra = next(value for value in VALUES if value not in (left[-1], right[-1]))
            # Vary the distractor's location without changing the matching query pair.
            facts = [left, right]
            facts.insert(int(digest(prefix), 16) % 3, f"Z:{extra}")
            prompt = "B:" + ";".join(facts) + "?" + query
        task_id = f"{task.task_id}/{panel}"
        result.append(Example(replace(task, task_id=task_id, prompt=prompt),
                              replace(row.truth, task_id=task_id)))
    return tuple(result)


def binding_panel_prompts(panel: str, rows: tuple[Example, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((row.visible.task_id, row.visible.prompt[:-2] + "Z>"
                  if panel == "query-neutral" else row.visible.prompt) for row in rows)
