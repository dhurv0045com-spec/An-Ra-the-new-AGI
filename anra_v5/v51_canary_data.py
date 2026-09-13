"""V5.1 canary data instrument: deterministic, screened, group-split.

NOT a production corpus and NOT cognition training data: an integration
instrument that exercises the real tokenizer -> manifest -> pack -> stream
plane with mechanically verifiable families. Every family has a reference
solver. Splits are cut at the LATENT-WORLD level (all renderings of one
causal world stay in one split). The sealed test split is generated first
and hashed before any training begins. Shortcut baselines and contamination
screens run before launch and fail closed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

GENERATOR_VERSION = "v51-canary-data/v1"
FAMILIES = ("identity", "binding", "state_order", "composition", "termination", "missing_info")
SPLIT_ORDER = ("sealed", "development", "training")  # sealed generated FIRST

_COLORS = (
    "crimson", "blue", "green", "amber", "violet", "scarlet",
    "teal", "coral", "ivory", "jade", "indigo", "rose",
)
_OBJECTS = (
    "zibble", "woggle", "flint", "quark", "moppet", "drayle", "snorpt",
    "vandle", "pilcrow", "gimbal", "thrixte", "lumpen", "bantle", "yorple",
)
_NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight"}
ABSTENTION_ANSWER = "none"


@dataclass(frozen=True)
class Example:
    example_id: str
    group_id: str          # latent-world id: split boundary
    family: str
    difficulty: str
    split: str
    prompt: str
    answer: str
    template_id: str


def _rng(seed: int, domain: str):
    """Deterministic stream: hash(seed:domain:counter) -> draws."""
    state = {"counter": 0, "seed": seed, "domain": domain}

    def draw() -> int:
        state["counter"] += 1
        payload = f"{state['seed']}:{state['domain']}:{state['counter']}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    return draw


def _draw_world(family: str, index: int, seed: int, attempt: int) -> dict:
    """Draw one latent world; ``attempt`` diversifies on content collision."""
    draw = _rng(seed, f"worlds:{family}:{attempt}")
    # per-world counterbalance domain: independent of the draw stream and of
    # the hash-sort used for splitting; the freshness check sees FINAL content
    world_id = f"{family}-w{index:04d}a{attempt:03d}"
    balance = int(hashlib.sha256(f"balance:{world_id}".encode()).hexdigest()[:8], 16)
    if family == "identity":
        length = 3 + balance % 6  # 3..8 words: content space must exceed the dataset size
        seq = [_OBJECTS[draw() % len(_OBJECTS)] for _ in range(length)]
        return {"world_id": world_id, "sequence": seq}
    elif family == "binding":
        # 4..5 facts, DISTINCT colors, query rank counterbalanced per world:
        # the positional baseline "first color in prompt" is pinned near 1/4.
        n = 4 + balance % 2
        entities: list[str] = []
        guard = 0
        while len(entities) < n and guard < 100:
            candidate = _OBJECTS[draw() % len(_OBJECTS)]
            if candidate not in entities:
                entities.append(candidate)
            guard += 1
        n = len(entities)
        color_pick = draw()
        colors = [_COLORS[(color_pick + 5 * k) % len(_COLORS)] for k in range(n)]
        facts = dict(sorted(zip(entities, colors)))
        query = sorted(facts)[balance % n]
        return {"world_id": world_id, "facts": facts,
                "query": query, "known": True}
    elif family == "state_order":
        # 4 distinct objects placed, one removed: content space (16P4 x 4)
        # comfortably exceeds the dataset size
        order: list[str] = []
        guard = 0
        while len(order) < 4 and guard < 100:
            candidate = _OBJECTS[draw() % len(_OBJECTS)]
            if candidate not in order:
                order.append(candidate)
            guard += 1
        removed = order.pop(draw() % len(order))
        return {"world_id": world_id, "events": order,
                "removed": removed}
    elif family == "composition":
        # transitive 4-object chain with SHUFFLED mention order: the answer
        # is the heaviest object; no positional heuristic can exceed ~1/4
        picks = []
        while len(picks) < 4:
            candidate = _OBJECTS[draw() % len(_OBJECTS)]
            if candidate not in picks:
                picks.append(candidate)
        weight_order = picks[:]
        mention_order = picks[:]
        # deterministic derangement of mention order vs weight order
        rot = 1 + draw() % 3
        mention_order = mention_order[rot:] + mention_order[:rot]
        return {"world_id": world_id,
                "weight_order": weight_order,
                "mention_order": mention_order}
    elif family == "termination":
        # 3..8 items counterbalanced per world: no answer exceeds ~1/6
        length = 3 + balance % 6
        seq = [_COLORS[draw() % len(_COLORS)] for _ in range(length)]
        return {"world_id": world_id, "sequence": seq}
    elif family == "missing_info":
        n = 4 + balance % 2
        entities: list[str] = []
        guard = 0
        while len(entities) < n and guard < 100:
            candidate = _OBJECTS[draw() % len(_OBJECTS)]
            if candidate not in entities:
                entities.append(candidate)
            guard += 1
        n = len(entities)
        color_pick = draw()
        colors = [_COLORS[(color_pick + 5 * k) % len(_COLORS)] for k in range(n)]
        facts = dict(sorted(zip(entities, colors)))
        unknown = _OBJECTS[(draw() + 7) % len(_OBJECTS)]
        while unknown in facts:
            unknown = _OBJECTS[(_OBJECTS.index(unknown) + 1) % len(_OBJECTS)]
        abstain = balance % 5 == 0  # ~20% abstention rate
        query = sorted(facts)[balance % n] if not abstain else unknown
        return {"world_id": world_id, "facts": facts,
                "unknown": unknown, "query": query, "known": not abstain}
    raise RuntimeError(f"unreachable: family {family}")


def render(family: str, world: dict) -> list[tuple[str, str, str]]:
    """Render one latent world into (template_id, prompt, answer) variants.
    All variants of a world share the world's group id."""
    if family == "identity":
        seq = world["sequence"]
        return [("copy-list",
                 "Copy these words: " + " ".join(seq) + ".",
                 " ".join(seq))]
    if family == "binding":
        facts = world["facts"]
        fact_text = " ".join(f"The {e} is {c}." for e, c in sorted(facts.items()))
        query = world["query"]
        prompt = f"{fact_text} What color is the {query}?"
        answer = facts.get(query) if query in facts else ABSTENTION_ANSWER
        return [("bind-query", prompt, answer)]
    if family == "state_order":
        events = world["events"]
        narrative = " ".join(f"Put the {item} in the box." for item in events)
        narrative += f" Take the {world['removed']} out of the box."
        remaining = [item for item in events if item != world["removed"]]
        prompt = f"The box starts empty. {narrative} What is in the box?"
        return [("state-recency", prompt, " and ".join(remaining))]
    if family == "composition":
        weight_order = world["weight_order"]
        mention_order = world["mention_order"]
        rank = {obj: i for i, obj in enumerate(weight_order)}  # 0 = heaviest
        sentences = []
        for i in range(len(mention_order) - 1):
            a, b = mention_order[i], mention_order[i + 1]
            if rank[a] < rank[b]:
                sentences.append(f"The {a} is heavier than the {b}.")
            else:
                sentences.append(f"The {b} is heavier than the {a}.")
        prompt = " ".join(sentences) + " Which object is the heaviest?"
        return [("transitive-chain", prompt, weight_order[0])]
    if family == "termination":
        seq = world["sequence"]
        return [("count-list",
                 f"Count these colors: " + " ".join(seq) + ". How many?",
                 _NUMBERS[len(seq)])]
    if family == "missing_info":
        facts = world["facts"]
        fact_text = " ".join(f"The {e} is {c}." for e, c in sorted(facts.items()))
        query = world["query"]
        prompt = f"{fact_text} What color is the {query}?"
        answer = facts.get(query, ABSTENTION_ANSWER)
        return [("abstain-or-bind", prompt, answer)]
    raise ValueError(f"unknown family {family}")


def solve(family: str, world: dict) -> list[str]:
    """Reference-solver answers for every rendering of the world (order matches render)."""
    return [answer for _, _, answer in render(family, world)]


def _normalized_renderings(family: str, world: dict) -> set[str]:
    return {
        " ".join(prompt.lower().split()) + " => " + answer.lower()
        for _, prompt, answer in render(family, world)
    }


def generate_split_plan(*, seed: int, worlds_per_family: int,
                        dev_fraction: float = 0.2, sealed_fraction: float = 0.2) -> dict:
    """Assign whole latent worlds to splits; sealed FIRST, then development,
    then training. Group-level: a world lives in exactly one split. Worlds
    whose normalized renderings collide with ANY already-accepted world are
    redrawn (deterministic attempt counter) — the contamination screen can
    then never fire on content collisions."""
    plan: dict[str, list[tuple[str, dict]]] = {s: [] for s in ("training", "development", "sealed")}
    seen: set[str] = set()  # GLOBAL across families: the contamination screen is global
    for family in FAMILIES:
        worlds: list[dict] = []
        index = 0
        attempts = 0
        while len(worlds) < worlds_per_family:
            world = _draw_world(family, index, seed, attempts)
            index += 1
            attempts += 1
            fresh = not (_normalized_renderings(family, world) & seen)
            if fresh:
                seen |= _normalized_renderings(family, world)
                worlds.append(world)
            if attempts > 100_000:
                raise RuntimeError("generator cannot produce fresh worlds")
        worlds.sort(key=lambda w: hashlib.sha256(w["world_id"].encode()).hexdigest())
        n_sealed = max(1, int(round(worlds_per_family * sealed_fraction)))
        n_dev = max(1, int(round(worlds_per_family * dev_fraction)))
        cut = {"sealed": worlds[:n_sealed],
               "development": worlds[n_sealed:n_sealed + n_dev],
               "training": worlds[n_sealed + n_dev:]}
        # stratified counterbalance per split: query rank / abstention / answer
        # length cycle exactly, pinning trivial baselines by construction
        for split, group in cut.items():
            plan[split].extend((family, world) for world in group)
    return plan


def build_dataset(*, seed: int, worlds_per_family: int) -> dict:
    """Generate the full dataset. Sealed split is produced and hashed first."""
    plan = generate_split_plan(seed=seed, worlds_per_family=worlds_per_family)
    splits: dict[str, list[Example]] = {s: [] for s in ("training", "development", "sealed")}
    split_hashes: dict[str, str] = {}
    for split in SPLIT_ORDER:  # sealed first
        rows = []
        for family, world in plan[split]:
            for template_id, prompt, answer in render(family, world):
                example_id = hashlib.sha256(
                    f"{world['world_id']}|{template_id}".encode()).hexdigest()[:16]
                rows.append(Example(example_id, world["world_id"], family, "easy",
                                    split, prompt, answer, template_id))
        rows.sort(key=lambda e: e.example_id)
        splits[split] = rows
        payload = "\n".join(json.dumps(e.__dict__, sort_keys=True) for e in rows)
        split_hashes[split] = hashlib.sha256(payload.encode()).hexdigest()
    return {"splits": splits, "split_hashes": split_hashes,
            "generator_version": GENERATOR_VERSION, "seed": seed,
            "worlds_per_family": worlds_per_family}


# ---------------------------------------------------------------- screens --

def contamination_screen(dataset: dict) -> dict:
    """Exact and normalized cross-split duplicate screens plus latent-group
    collisions. REQUIRED: zero collisions. Fail closed by caller."""
    collisions: dict[str, list[str]] = {"exact": [], "normalized": [], "group": []}
    normalized = {
        split: {" ".join(e.prompt.lower().split()) + " => " + e.answer.lower()
                for e in rows}
        for split, rows in dataset["splits"].items()
    }
    groups = {
        split: {e.group_id for e in rows}
        for split, rows in dataset["splits"].items()
    }
    names = ("training", "development", "sealed")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            exact = {e.example_id for e in dataset["splits"][a]} & {
                e.example_id for e in dataset["splits"][b]}
            for example_id in sorted(exact):
                collisions["exact"].append(f"{a}:{b}:{example_id}")
            for row in sorted(normalized[a] & normalized[b]):
                collisions["normalized"].append(f"{a}:{b}:{row[:80]}")
            for group in sorted(groups[a] & groups[b]):
                collisions["group"].append(f"{a}:{b}:{group}")
    return {"collisions": collisions, "clean": not any(collisions.values())}


def shortcut_baselines(dataset: dict, split: str = "development") -> dict:
    """Dumb baselines scored per family as exact-match rate. A family a trivial
    heuristic solves is a compromised family: repair before training."""
    rows = dataset["splits"][split]
    per_family: dict[str, dict[str, list[int]]] = {}
    for e in rows:
        scores = per_family.setdefault(e.family, {})
        words = e.prompt.replace("?", "").replace(".", "").lower().split()
        answers = [o.answer.lower() for o in rows]
        freq_answer = max(set(answers), key=answers.count)
        color_hits = [c for c in _COLORS if c in words]
        scores.setdefault("constant", []).append(int(e.answer.lower() == freq_answer))
        scores.setdefault("latest_position", []).append(
            int(e.answer.lower() == words[-1]))
        scores.setdefault("answer_frequency", []).append(
            int(e.answer.lower() == freq_answer))
        scores.setdefault("first_color_in_prompt", []).append(
            int(bool(color_hits) and e.answer.lower() == color_hits[0]))
        scores.setdefault("answer_is_fixed_none", []).append(
            int(e.answer == ABSTENTION_ANSWER))
    result = {
        family: {name: sum(v) / len(v) for name, v in scores.items()}
        for family, scores in per_family.items()
    }
    return result


def generator_receipt(dataset: dict, tokenizer_identity: dict) -> dict:
    payload = {
        "schema": "anra-v51-canary-data-receipt/v1",
        "generator_version": dataset["generator_version"],
        "generator_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "seed": dataset["seed"],
        "worlds_per_family": dataset["worlds_per_family"],
        "split_hashes": dataset["split_hashes"],
        "split_sizes": {s: len(rows) for s, rows in dataset["splits"].items()},
        "families": list(FAMILIES),
        "tokenizer_identity": tokenizer_identity,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload
