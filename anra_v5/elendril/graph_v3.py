"""Public V3 graph fixtures, topology splits, and independent shortcut audit."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import combinations, permutations
import random

from v5_evaluation.firewall import EvaluatorTruth, VisibleTask

from .benchmark import Example
from .codec import BOS, TOKEN_IDS, Prediction, answer_tokens, digest
from .scoring import score


SURFACES = ("graph-v3", "graph-v3-replication")
PANELS = ("depth-one", "depth-three", "depth-four", "alias-symbol", "alias-separator",
          "alias-combined", "order", "rewire", "absent", "ambiguous", "query-neutral")
TRANSFORMS = ("raw", "augment", "canonical")
RENDER_TRANSFORM = {"recurrent-aug": "augment", "recurrent-canon": "canonical"}
GRAPH_ONLY_METHODS = frozenset({"three-read", "recurrent-read", "recurrent-dense", *RENDER_TRANSFORM})
_TRAIN_KEYS, _ALIAS_KEYS = "ABCDEFGHIJKLMNOP", "QRSTUVWXYZ"
_BASE = ((0, 1), (1, 2), (3, 4), (4, 5))
_SWAP = (3, 4, 5, 0, 1, 2)


@dataclass(frozen=True, slots=True)
class Graph:
    topology: tuple[tuple[int, int], ...]
    labels: tuple[str, ...]
    ordinal: int
    query_depth: int

    @property
    def cluster(self) -> str:
        return digest(["graph-v3-topology", self.topology])


def solve(edges, root: str, depth: int) -> str:
    """Resolve only paths with exactly one distinct outgoing edge per step."""
    for _ in range(depth):
        destinations = {target for source, target in edges if source == root}
        if len(destinations) != 1:
            return "!"
        root = next(iter(destinations))
    return root


def canonical_topology(nuisance: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Canonicalize the two marked-root paths and the four unmarked slots."""
    best = None
    for chain in (tuple(range(6)), _SWAP):
        for extra in permutations(range(6, 10)):
            mapping = (*chain, *extra)
            candidate = tuple(sorted((mapping[a], mapping[b]) for a, b in nuisance))
            if best is None or candidate < best:
                best = candidate
    return best


def _rewire_group(nuisance: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Keep a latent class and its terminal-rewired counterpart in one split."""
    terminals = {2: 5, 5: 2}
    swapped = tuple((terminals.get(a, a), terminals.get(b, b)) for a, b in nuisance)
    return min(canonical_topology(nuisance), canonical_topology(swapped))


@lru_cache(maxsize=1)
def topology_partitions() -> dict[str, tuple]:
    edges = [(source, target) for source in range(6, 10)
             for target in range(10) if source != target]
    classes = {
        _rewire_group((first, second)) for first, second in combinations(edges, 2)
        if first != second[::-1]
    }
    ordered = sorted(classes)
    random.Random(3007).shuffle(ordered)
    if len(ordered) < 28:
        raise RuntimeError("insufficient graph isomorphism classes")
    return {"train": tuple(ordered[:16]), "screen": tuple(ordered[16:22]),
            "replication": tuple(ordered[22:28])}


@lru_cache(maxsize=3)
def _worlds(partition: str) -> tuple[Graph, ...]:
    seed = {"train": 710318, "screen": 499079, "replication": 3109223}[partition]
    count = 16 if partition == "replication" else 8
    rng = random.Random(seed)
    worlds = []
    for topology in topology_partitions()[partition]:
        samples = set()
        while len(samples) < count:
            labels = tuple(rng.sample(_TRAIN_KEYS, 14))
            if labels[:10] in samples:
                continue
            samples.add(labels[:10])
            ordinal = len(worlds)
            worlds.append(Graph(topology, labels, ordinal, 1 + ordinal % 2 if partition == "train" else 2))
    return tuple(worlds)


def parse(prompt: str) -> tuple[int, tuple[tuple[str, str], ...], str]:
    facts, query = prompt.split("?")
    depth, edges = facts.split(":", 1)
    edges = edges.translate(str.maketrans({"#": ";", "-": ";"}))
    if depth not in ("1", "2", "3", "4") or len(query) != 2 or query[1] != ">" or not "A" <= query[0] <= "Z":
        raise ValueError("invalid graph query syntax")
    pairs = tuple(tuple(edge.split(":")) for edge in edges.split(";"))
    if any(len(pair) != 2 or any(len(token) != 1 or not "A" <= token <= "Z" for token in pair)
           for pair in pairs):
        raise ValueError("invalid graph edge syntax")
    return int(depth), pairs, query[0]


def _prompt(depth, edges, root, separator) -> str:
    return str(depth) + ":" + separator.join(f"{a}:{b}" for a, b in edges) + f"?{root}>"


def normalize(prompt: str) -> tuple[str, dict[str, str]]:
    """Surface-only first-occurrence symbol renaming and punctuation normalization."""
    depth, edges, query = parse(prompt)
    mapping = {}
    for source, target in edges:
        for token in (source, target):
            mapping.setdefault(token, chr(65 + len(mapping)))
    mapping.setdefault(query, chr(65 + len(mapping)))
    return _prompt(depth, ((mapping[a], mapping[b]) for a, b in edges), mapping[query], ";"), mapping


def normalize_training(row: Example) -> Example:
    prompt, mapping = normalize(row.visible.prompt)
    gold = mapping.get(row.truth.gold, row.truth.gold)
    return replace(row, visible=replace(row.visible, prompt=prompt),
                   truth=replace(row.truth, gold=gold))


def inference_inputs(rows: tuple[Example, ...], panel: str, method: str):
    prompts, inverse = [], {}
    for row in rows:
        task_id = row.visible.task_id
        prompt = row.visible.prompt
        if panel == "query-neutral":
            prompt = prompt[:-2] + "Z>"
        if RENDER_TRANSFORM.get(method) == "canonical":
            prompt, mapping = normalize(prompt)
            inverse[task_id] = {TOKEN_IDS[target]: TOKEN_IDS[source] for source, target in mapping.items()}
        prompts.append((task_id, prompt))
    return tuple(prompts), inverse


def restore_predictions(predictions: tuple[Prediction, ...], inverse: dict) -> tuple[Prediction, ...]:
    """Decode canonical outputs using the input-only inverse map, rejecting unknown symbols."""
    return tuple(Prediction(prediction.task_id, tuple(
        inverse[prediction.task_id].get(token, BOS) if TOKEN_IDS["A"] <= token <= TOKEN_IDS["Z"] else token
        for token in prediction.tokens
    )) for prediction in predictions)


@lru_cache(maxsize=80)
def graph_fixture(surface: str, split: str, panel: str = "base", transform: str = "raw") -> tuple[Example, ...]:
    if surface not in SURFACES or split not in ("train", "development"):
        raise ValueError("graph fixture is limited to the registered public splits")
    if panel not in ("base", "train", *PANELS) or (split == "train" and panel not in ("base", "train")):
        raise ValueError("invalid graph panel")
    if panel == "train" and split != "train":
        raise ValueError("training panel requires the training split")
    if transform not in TRANSFORMS:
        raise ValueError("unknown graph rendering transform")
    if split == "development" and transform != "raw":
        raise ValueError("score against immutable raw development examples")
    partition = "train" if split == "train" else ("screen" if surface == SURFACES[0] else "replication")
    rows = []
    for graph in _worlds(partition):
        depth = {"depth-one": 1, "depth-three": 3, "depth-four": 4}.get(panel, graph.query_depth)
        edges = [*_BASE, *graph.topology]
        for step in range(2, depth):
            left = 2 if step == 2 else 10
            right = 5 if step == 2 else 12
            edges.extend(((left, 10 + step - 2), (right, 12 + step - 2)))
        if panel == "rewire":
            edges[1], edges[3] = (1, 5), (4, 2)
        elif panel == "absent":
            edges[1] = (10, 2)
        elif panel == "ambiguous":
            edges.append((1, 5))
        order_seed = digest(["graph-v3-order", partition, graph.ordinal,
                             "alternate" if panel == "order" or (transform == "augment" and graph.ordinal % 2)
                             else "normal"])
        random.Random(int(order_seed, 16)).shuffle(edges)
        if panel in ("alias-symbol", "alias-combined"):
            aliases = random.Random(graph.ordinal + 3910).sample(_ALIAS_KEYS, 10)
            labels = (*aliases, *graph.labels[10:])
        elif transform == "augment" and graph.ordinal % 2:
            # Re-label a fixed half of the same latent worlds, staying inside the train alphabet.
            permutation = list(_TRAIN_KEYS)
            random.Random(int(digest(["graph-v3-rerender", graph.ordinal]), 16)).shuffle(permutation)
            mapping = dict(zip(_TRAIN_KEYS, permutation))
            labels = tuple(mapping[token] for token in graph.labels)
        else:
            labels = graph.labels
        separator = "-" if panel in ("alias-separator", "alias-combined") else (
            "#" if transform == "augment" and graph.ordinal % 2 else ";"
        )
        labelled = tuple((labels[source], labels[target]) for source, target in edges)
        for view, root in enumerate((0, 3)):
            task_id = f"{graph.cluster}/{graph.ordinal}/{view}/{panel}"
            prompt = _prompt(depth, labelled, labels[root], separator)
            task = VisibleTask(task_id, graph.cluster, "binding", split, "graph-v3", prompt, ())
            row = Example(task, EvaluatorTruth(task_id, solve(labelled, labels[root], depth)))
            rows.append(normalize_training(row) if transform == "canonical" else row)
    result = tuple(rows)
    if len({row.visible.prompt for row in result}) != len(result):
        raise ValueError(f"duplicate rendered prompts in {partition}/{panel}/{transform}")
    return result


def graph_audit(surface: str) -> dict:
    train = graph_fixture(surface, "train")
    dev = graph_fixture(surface, "development")
    train_clusters = {row.visible.cluster_id for row in train}
    dev_clusters = {row.visible.cluster_id for row in dev}
    if train_clusters & dev_clusters:
        raise ValueError("graph topology split overlap")
    train_views = set()
    for transform in TRANSFORMS:
        train_views.update(normalize(row.visible.prompt)[0] for row in graph_fixture(
            surface, "train", transform=transform))
    for panel in ("base", *PANELS):
        dev_views = {normalize(row.visible.prompt)[0] for row in graph_fixture(surface, "development", panel)}
        if train_views & dev_views:
            raise ValueError(f"canonical rendered split overlap in {panel}")
    # Independent set propagation. The generator follows each unique path sequentially.
    def closure(edges, root, depth):
        frontier = {root}
        for _ in range(depth):
            outgoing = {node: {b for a, b in edges if a == node} for node in frontier}
            if any(len(targets) != 1 for targets in outgoing.values()):
                return "!"
            frontier = set().union(*outgoing.values())
        return next(iter(frontier)) if len(frontier) == 1 else "!"

    policies = {
        "oracle": closure,
        "one-edge": lambda edges, root, depth: closure(edges, root, 1),
        "first": lambda edges, root, depth: edges[0][1],
        "last": lambda edges, root, depth: edges[-1][1],
        "frequency": lambda edges, root, depth: min(Counter(b for _, b in edges).items(),
                                                    key=lambda x: (-x[1], x[0]))[0],
        "always-abstain": lambda edges, root, depth: "!",
    }

    def evaluate(name, rows):
        scores = {}
        for policy, fn in policies.items():
            predictions = []
            for row in rows:
                depth, edges, root = parse(row.visible.prompt)
                predictions.append(Prediction(row.visible.task_id, answer_tokens(fn(edges, root, depth))))
            scores[policy] = score(rows, predictions)
        if scores["oracle"]["exact_eos"] != [len(rows), len(rows)]:
            raise ValueError(f"independent oracle disagrees on {name}")
        return {"rows": len(rows), "fixture_sha256": digest([asdict(row) for row in rows]),
                "controls": scores}

    panels = {"base": evaluate("base", dev), "train": evaluate("train", graph_fixture(surface, "train", "train"))}
    for name in PANELS:
        panels[name] = evaluate(name, graph_fixture(surface, "development", name))
    neutral_rows = graph_fixture(surface, "development", "query-neutral")
    neutral_prompts, _ = inference_inputs(neutral_rows, "query-neutral", "recurrent-read")
    neutral_predictions = []
    for task_id, prompt in neutral_prompts:
        depth, edges, root = parse(prompt)
        neutral_predictions.append(Prediction(task_id, answer_tokens(closure(edges, root, depth))))
    neutral = score(neutral_rows, neutral_predictions)
    if neutral["query_swap_both_correct"][0]:
        raise ValueError("query-neutral oracle fails negative-control gate")
    panels["query-neutral"]["controls"]["neutralized-oracle"] = neutral
    base = panels["base"]["controls"]
    if base["one-edge"]["exact_eos"][0] or any(
        base[policy]["query_swap_both_correct"][0] for policy in ("first", "last", "frequency")
    ):
        raise ValueError("base graph shortcut gate failed")
    training = {
        method: digest([(asdict(row.visible), asdict(row.truth))
                        for row in graph_fixture(surface, "train", transform=method)])
        for method in TRANSFORMS
    }
    return {"schema": "elendril-graph-v3-audit/v1", "surface": surface,
            "training_rows": len(train), "development_rows": len(dev),
            "training_clusters": len(train_clusters), "development_clusters": len(dev_clusters),
            "training_fixture_sha256": training, "panels": panels}
