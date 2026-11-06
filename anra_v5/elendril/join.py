"""World-disjoint two-hop development surface and deterministic adversaries."""

from dataclasses import dataclass
from itertools import combinations
import random

from v5_evaluation.firewall import EvaluatorTruth, VisibleTask

from .benchmark import Example
from .codec import Prediction, answer_tokens, digest


KEYS = "ABCDEFGHIJKLMNOP"
ALIASES = "QRSTUVWXYZ"
JOIN_PANELS = ("train", "alias", "rewire", "more-distractors", "depth-three", "query-neutral")
SPLIT_SEEDS = {"train": 8041, "development": 9907}


@dataclass(frozen=True, slots=True)
class Graph:
    roots: tuple[str, str]
    paths: tuple[tuple[str, ...], tuple[str, ...]]
    decoys: tuple[tuple[str, str], ...]
    cluster: str
    ordinal: int


def solve(edges: tuple[tuple[str, str], ...], root: str, depth: int) -> str:
    """Return abstention on a missing or ambiguous outgoing relation."""
    for _ in range(depth):
        targets = {target for source, target in edges if source == root}
        if len(targets) != 1:
            return "!"
        root = targets.pop()
    return root


def _graphs(split: str) -> tuple[Graph, ...]:
    if split not in SPLIT_SEEDS:
        raise ValueError("join fixtures are development only")
    root_sets = list(combinations(KEYS, 2))
    random.Random(611).shuffle(root_sets)
    root_sets = root_sets[:32] if split == "train" else root_sets[32:48]
    worlds = []
    for roots in root_sets:
        cluster = digest(["join-v1", roots])
        rng = random.Random(int(digest([SPLIT_SEEDS[split], roots]), 16))
        seen = set()
        for _ in range(4):
            while True:
                nodes = rng.sample([key for key in KEYS if key not in roots], 10)
                if tuple(nodes[:4]) not in seen:
                    seen.add(tuple(nodes[:4]))
                    break
            paths = ((roots[0], nodes[0], nodes[1], nodes[4]),
                     (roots[1], nodes[2], nodes[3], nodes[5]))
            decoy_target = rng.choice([nodes[6], nodes[0], nodes[3]])
            worlds.append(Graph(roots, paths, ((nodes[7], decoy_target), (nodes[8], nodes[9])),
                                cluster, len(worlds)))
    return tuple(worlds)


def join_fixture(split: str, panel: str = "base") -> tuple[Example, ...]:
    if panel not in ("base", *JOIN_PANELS) or (split == "train" and panel not in ("base", "train")):
        raise ValueError("invalid join fixture panel")
    if panel == "train" and split != "train":
        raise ValueError("training panel requires the training split")
    result = []
    for graph in _graphs(split):
        depth = 3 if panel == "depth-three" else 2
        paths = [list(path[:depth + 1]) for path in graph.paths]
        if panel == "rewire":
            paths[0][2], paths[1][2] = paths[1][2], paths[0][2]
        edges = [edge for path in paths for edge in zip(path, path[1:])]
        if depth == 2:
            edges.extend(graph.decoys[:2 if panel == "more-distractors" else 1])
        roots = graph.roots
        if panel == "alias":
            symbols = sorted({symbol for edge in edges for symbol in edge})
            aliases = random.Random(graph.ordinal + 771).sample(ALIASES, len(symbols))
            mapping = dict(zip(symbols, aliases))
            edges = [(mapping[a], mapping[b]) for a, b in edges]
            roots = tuple(mapping[root] for root in roots)
        # Matched rewire keeps the same edge slot order as the base panel.
        rng = random.Random(int(digest(["edge-order", SPLIT_SEEDS[split], graph.ordinal]), 16))
        rng.shuffle(edges)
        separator = "-" if panel == "alias" else (";" if graph.ordinal % 2 else "#")
        facts = separator.join(f"{left}:{right}" for left, right in edges)
        for view, root in enumerate(roots):
            task_id = f"{graph.cluster}/{graph.ordinal}/{view}/{panel}"
            task = VisibleTask(task_id, graph.cluster, "binding", split, "join-v1",
                               f"B:{facts}?{root}>", ())
            result.append(Example(task, EvaluatorTruth(task_id, solve(tuple(edges), root, depth))))
    return tuple(result)


def parse(prompt: str) -> tuple[tuple[tuple[str, str], ...], str]:
    facts, query = prompt.split("?")
    facts = facts[2:].translate(str.maketrans({"#": ";", "-": ";"}))
    return tuple(tuple(edge.split(":")) for edge in facts.split(";")), query[0]


def controls(rows: tuple[Example, ...], depth: int = 2) -> dict:
    from .scoring import score

    policies = {
        "oracle": lambda edges, root: solve(edges, root, depth),
        "one_step": lambda edges, root: solve(edges, root, 1),
        "first": lambda edges, root: edges[0][1],
        "last": lambda edges, root: edges[-1][1],
        "frequency": lambda edges, root: max(sorted({b for _, b in edges}),
                                             key=lambda value: sum(b == value for _, b in edges)),
    }
    return {name: score(rows, (Prediction(row.visible.task_id, answer_tokens(
        policy(*parse(row.visible.prompt)))) for row in rows)) for name, policy in policies.items()}
