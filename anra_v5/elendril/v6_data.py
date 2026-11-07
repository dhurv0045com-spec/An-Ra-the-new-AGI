"""V6 public-evidence worlds, supervised observations and auditable UTF8 data.

This is a procedural development corpus, not a natural-language pretraining
corpus. No pretrained text, inherited fixture or evaluator answer is rendered.
All splits share the same procedural generator and use a global normalized
observation-state split. The semantics intentionally
remain related so the campaign can diagnose failures of the same learner.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import random

from .codec import canonical, digest
from .cognition import (CognitiveState, ExperienceEvent, Goal, MemoryVersion,
                        PublicEnvironment, Truth, belief)
from .foundation_training import TokenSample
from .representation import UTF8

VERSION = "elendril-v6-public-worlds-v1"
CHANNEL = "request-current-status"
SYMBOLS = "ABCDEFGHIJKLMNOP"
MAX_SEQUENCE = 512
KINDS = ("known", "missing", "conflict", "null", "unavailable", "route", "memory", "route")


@dataclass(frozen=True, slots=True)
class Surface:
    template: int = 0
    aliases: str = ""
    hops: int = 0
    scratch: bool = False
    order: str = "original"

    def __post_init__(self):
        if (self.template not in range(4) or self.aliases not in ("", "node-", "unit/")
                or self.hops not in (0, 2, 3) or type(self.scratch) is not bool
                or self.order not in ("original", "reverse")):
            raise ValueError("unsupported V6 public surface")


@dataclass(frozen=True, slots=True)
class V6Case:
    case_id: str
    world_id: str
    family: str
    kind: str
    goal: Goal
    environment: PublicEnvironment
    truth: Truth
    surface: Surface
    memory: MemoryVersion | None = None
    pair_id: str = ""
    panel: str = "base"

    @property
    def visible_surface(self):
        return self.surface


def render_public(state: CognitiveState, channels, surface: Surface, memory_available=False) -> str:
    """Raw facts only: no derived status, teacher action, world ID or truth.

    Output aliases remain publicly decodable: e.g. unit/A has answer label A.
    A model-generated scratch symbol is labeled as scratch, never as evidence.
    """
    if tuple(channels) not in ((), (CHANNEL,)) or type(memory_available) is not bool:
        raise ValueError("unexpected V6 public channels/memory advertisement")
    entity = lambda symbol: surface.aliases + symbol
    instructions = ("Latest time per source counts; disagreement needs evidence. "
                    "Output A-P, Q=query, !=abstain, R=memory, ~X=think.\n")
    question = (f"Follow {surface.hops} links from {entity(state.goal.subject)}."
                if state.goal.family == "route" else f"Current value of {entity(state.goal.subject)}?")
    if surface.aliases:
        question += " Output the letter after the alias."
    header = f"{question} Query={int(bool(channels))} Memory={int(memory_available)}\n"
    events = state.events if surface.order == "original" else tuple(reversed(state.events))
    rows = []
    for event in events:
        subject, value = entity(event.subject), "null" if event.value is None else entity(event.value)
        if surface.template == 0:
            rows.append(f"{subject}={value}@{event.tick}/{event.source}")
        elif surface.template == 1:
            rows.append(f"At {event.tick} {event.source} says {subject}:{value}")
        elif surface.template == 2:
            rows.append(f"({event.source},{event.tick},{subject},{value})")
        else:
            rows.append(f"[{event.tick}|{subject}->{value}|{event.source}]")
    scratch = "" if state.hypothesis is None else f"\nModel scratch={state.hypothesis} (unverified)"
    prompt = instructions + header + "; ".join(rows) + scratch + "\nAction:"
    if len(prompt.encode("utf-8")) + 6 > MAX_SEQUENCE:
        raise ValueError("V6 public prompt exceeds qualified byte context; no truncation")
    return prompt


def action_text(operation, argument=None):
    if operation == "ANSWER" and argument in SYMBOLS:
        return argument
    if operation == "QUERY" and argument == CHANNEL:
        return "Q"
    if operation == "ABSTAIN" and argument is None:
        return "!"
    if operation == "RETRIEVE" and argument is None:
        return "R"
    if operation == "DELIBERATE" and argument in SYMBOLS:
        return "~" + argument
    raise ValueError("unsupported V6 action grammar")


def public_action(state, channels, surface: Surface, memory_available=False):
    """Engineered teacher using public observations, never Truth/channel values."""
    current = state.goal.subject
    first = None
    for _ in range(surface.hops if state.goal.family == "route" else 1):
        value = belief(state.events, current)
        if value.status == "null":
            return "ABSTAIN", None
        if value.status != "derived":
            if memory_available:
                return "RETRIEVE", None
            return ("QUERY", CHANNEL) if channels else ("ABSTAIN", None)
        current = value.value
        first = current if first is None else first
    if surface.scratch and state.goal.family == "route" and state.hypothesis is None:
        return "DELIBERATE", first
    return "ANSWER", current


@dataclass(frozen=True, slots=True)
class V6Example:
    task_id: str
    case_id: str
    world_id: str
    kind: str
    stage: str
    prompt: str
    target: str

    @property
    def sample(self):
        frame = UTF8.frame(self.prompt, self.target, max_length=MAX_SEQUENCE)
        return TokenSample(self.task_id, frame.ids, frame.loss_mask, self.world_id, self.kind)


def _state_key(state, channels, memory_available):
    # Scopes, event order, formatting and the model's hypothesis are irrelevant
    # to public semantic identity; their variations must not cross a split.
    ticks = {tick: rank for rank, tick in enumerate(sorted({e.tick for e in state.events}))}
    events = sorted((e.subject, e.source, ticks[e.tick], e.value or "<NULL>") for e in state.events)
    return digest([VERSION, state.goal.family, state.goal.subject, events, bool(channels), memory_available])


def _partition(key):
    bucket = int(key, 16) % 10
    return "train" if bucket < 7 else "validation" if bucket < 9 else "confirmation"


def _supervision(case):
    state = CognitiveState(case.goal).observe(case.environment.initial)
    channels = tuple(case.environment.channels)
    available = case.memory is not None
    examples, state_keys = [], []
    for decision in range(4):
        state_keys.append(_state_key(state, channels, available))
        op, arg = public_action(state, channels, case.surface, available)
        examples.append(V6Example(f"{case.case_id}/{decision}", case.case_id, case.world_id,
                                  case.kind, ("initial" if not decision else f"observed-{decision}"),
                                  render_public(state, channels, case.surface, available), action_text(op, arg)))
        if op == "QUERY":
            state = state.observe(case.environment.channels[arg])
            channels = ()
        elif op == "RETRIEVE":
            state = state.observe(case.memory.recall(case.goal))
            available = False
        elif op == "DELIBERATE":
            state = replace(state, hypothesis=arg)
        else:
            break
    else:
        raise ValueError("teacher failed to terminate bounded V6 supervision")
    return tuple(examples), tuple(state_keys)


def _case(index, seed, split):
    rng = random.Random(int(digest([VERSION, seed, index, "candidate"]), 16))
    kind = KINDS[index % len(KINDS)]
    subject, middle, terminal, spare, noise1, noise2 = rng.sample(SYMBOLS, 6)
    old, now = rng.sample(SYMBOLS, 2)
    # All splits share public distributions; state identity determines assignment.
    tick = rng.randrange(2, 800)
    source = rng.choice(("s0", "s1", "log"))
    rows = [(noise1, source, tick - 1, rng.choice(SYMBOLS)),
            (noise2, "map", tick, rng.choice(SYMBOLS)),
            (spare, "map", tick + 1, rng.choice(SYMBOLS))]
    hidden, remembered = [], []
    family = "route" if kind in ("route", "memory") else "revision"
    hops = 2 if family == "route" else 0
    if kind == "known":
        rows += [(subject, source, tick - 2, old), (subject, source, tick + 2, now)]
        expected = now
        hidden = [(subject, source, tick + 3, now)]
    elif kind == "null":
        rows += [(subject, source, tick + 2, None)]
        expected = None
        hidden = [(subject, source, tick + 3, None)]
    elif kind in ("missing", "conflict"):
        if kind == "conflict":
            rows += [(subject, "s0", tick, old), (subject, "s1", tick, now)]
        expected = rng.choice((None, now, now, now))
        hidden = [(subject, "s0", tick + 3, expected), (subject, "s1", tick + 3, expected)]
    elif kind == "unavailable":
        expected = None  # safe-action truth: no publicly obtainable answer
    else:
        rows += [(subject, "map", tick, middle)]
        edge = (middle, "map", tick, terminal)
        (remembered if kind == "memory" else rows).append(edge)
        expected = terminal
    rng.shuffle(rows)
    semantic = [family, hops, subject, sorted((a,b,c,d or "<NULL>") for a,b,c,d in rows),
                sorted((a,b,c,d or "<NULL>") for a,b,c,d in hidden), remembered]
    world = digest([VERSION, "semantic-world", semantic])
    scope, reusable = "world-" + world, "memory-" + world
    event = lambda row, custody: ExperienceEvent(custody, *row)
    initial = tuple(event(row, scope) for row in rows)
    channels = {CHANNEL: tuple(event(row, scope) for row in hidden)} if hidden else {}
    memory = MemoryVersion(tuple(event(row, reusable) for row in remembered)) if remembered else None
    slots = tuple(dict.fromkeys([subject, *(row[0] for row in rows), *(row[0] for row in remembered)]))
    goal = Goal(world, family, subject, scope, reusable, slots)
    all_events = [*initial, *(e for payload in channels.values() for e in payload),
                  *(memory.events if memory else ())]
    truth = Truth(world, expected, tuple(e.identity for e in all_events))
    surface = Surface(index % 2, "", hops, False, "original")
    return V6Case(world, world, family, kind, goal, PublicEnvironment(initial, channels), truth, surface, memory)


def _confirmation_case(index, seed):
    rng = random.Random(int(digest([VERSION, seed, index, "confirmation-transition-ledger-v1"]), 16))
    kind = KINDS[index % len(KINDS)]
    subject, middle, terminal, spare, noise1, noise2 = rng.sample(SYMBOLS, 6)
    old, now = rng.sample(SYMBOLS, 2)
    # All splits share public distributions; state identity determines assignment.
    tick = rng.randrange(2, 800)
    source = rng.choice(("s0", "s1", "log"))
    # Build a transition ledger with explicit observation destinations, then
    # materialize each custody stream; this path is reserved for confirmation.
    ledger = [("initial", (noise1, source, tick - 1, rng.choice(SYMBOLS))),
              ("initial", (noise2, "map", tick, rng.choice(SYMBOLS))),
              ("initial", (spare, "map", tick + 1, rng.choice(SYMBOLS)))]
    rows, hidden, remembered = [], [], []
    family = "route" if kind in ("route", "memory") else "revision"
    hops = 2 if family == "route" else 0
    if kind == "known":
        rows += [(subject, source, tick - 2, old), (subject, source, tick + 2, now)]
        expected = now
        hidden = [(subject, source, tick + 3, now)]
    elif kind == "null":
        rows += [(subject, source, tick + 2, None)]
        expected = None
        hidden = [(subject, source, tick + 3, None)]
    elif kind in ("missing", "conflict"):
        if kind == "conflict":
            rows += [(subject, "s0", tick, old), (subject, "s1", tick, now)]
        expected = rng.choice((None, now, now, now))
        hidden = [(subject, "s0", tick + 3, expected), (subject, "s1", tick + 3, expected)]
    elif kind == "unavailable":
        expected = None  # safe-action truth: no publicly obtainable answer
    else:
        rows += [(subject, "map", tick, middle)]
        edge = (middle, "map", tick, terminal)
        (remembered if kind == "memory" else rows).append(edge)
        expected = terminal
    ledger.extend(("initial", row) for row in rows)
    ledger.extend(("channel", row) for row in hidden)
    ledger.extend(("memory", row) for row in remembered)
    rng.shuffle(ledger)
    rows = [row for destination, row in ledger if destination == "initial"]
    hidden = [row for destination, row in ledger if destination == "channel"]
    remembered = [row for destination, row in ledger if destination == "memory"]
    semantic = [family, hops, subject, sorted((a,b,c,d or "<NULL>") for a,b,c,d in rows),
                sorted((a,b,c,d or "<NULL>") for a,b,c,d in hidden), remembered]
    world = digest([VERSION, "semantic-world", semantic])
    scope, reusable = "world-" + world, "memory-" + world
    event = lambda row, custody: ExperienceEvent(custody, *row)
    initial = tuple(event(row, scope) for row in rows)
    channels = {CHANNEL: tuple(event(row, scope) for row in hidden)} if hidden else {}
    memory = MemoryVersion(tuple(event(row, reusable) for row in remembered)) if remembered else None
    slots = tuple(dict.fromkeys([subject, *(row[0] for row in rows), *(row[0] for row in remembered)]))
    goal = Goal(world, family, subject, scope, reusable, slots)
    all_events = [*initial, *(e for payload in channels.values() for e in payload),
                  *(memory.events if memory else ())]
    truth = Truth(world, expected, tuple(e.identity for e in all_events))
    surface = Surface(index % 2, "", hops, False, "original")
    return V6Case(world, world, family, kind, goal, PublicEnvironment(initial, channels), truth, surface, memory)


@dataclass(frozen=True, slots=True)
class V6Dataset:
    split: str
    cases: tuple[V6Case, ...]
    examples: tuple[V6Example, ...]
    samples: tuple[TokenSample, ...]
    manifest: dict

    @property
    def identity(self):
        return digest(self.manifest)


def _dataset(split, cases, *, seed, target_bytes=0, parent=None, augmentation=False, scratch=False):
    examples = tuple(e for case in cases for e in _supervision(case)[0])
    samples = tuple(e.sample for e in examples)
    text_bytes = sum(len(e.prompt.encode("utf-8")) + len(e.target.encode("utf-8")) for e in examples)
    manifest = {"schema": VERSION, "split": split, "seed": seed, "license": "repository LICENSE; newly authored synthetic worlds",
        "source": "procedural public observations; no inherited corpus or external text",
        "generator": "confirmation-transition-ledger-v1" if split == "confirmation" else "procedural-event-worlds",
        "generator_limit": "related task semantics; development, not sealed external generalization",
        "tokenizer_sha256": UTF8.identity, "episodes": len(cases), "examples": len(examples),
        "text_bytes": text_bytes, "target_bytes": target_bytes, "max_sequence_tokens": max(len(s.tokens) for s in samples),
        "supervised_tokens": sum(sum(s.eligible) for s in samples), "input_positions": sum(len(s.tokens) for s in samples),
        "worlds_sha256": digest([c.world_id for c in cases]), "cases_sha256": digest([case_to_record(c) for c in cases]),
        "examples_sha256": digest([asdict(e) for e in examples]),
        "samples_sha256": digest([asdict(sample) for sample in samples]),
        "kinds": {k: sum(c.kind == k for c in cases) for k in sorted(set(KINDS))},
        "augmentation": augmentation, "scratch": scratch, "parent_dataset": parent,
        **({"sampling": "fixed-per-kind-quotas-v1"} if split != "train" else {})}
    result = V6Dataset(split, tuple(cases), examples, samples, manifest)
    audit_dataset(result)
    return result


def make_dataset(split="train", *, target_bytes=2_200_000, worlds=None, seed=606, augmentation=False, scratch=False):
    if (split not in ("train", "validation", "confirmation") or type(seed) is not int or not 0 <= seed < 2**32
            or type(target_bytes) is not int or not 1_000 <= target_bytes <= 3_000_000
            or (worlds is not None and (type(worlds) is not int or not 1 <= worlds <= 20_000))):
        raise ValueError("invalid bounded V6 dataset request")
    if split != "train" and worlds is None:
        worlds = 80
    cases, seen_worlds, seen_prompts, seen_states, body_bytes, examples_count = [], set(), set(), set(), 0, 0
    # Freeze requested per-kind quotas before sampling; never select by outcome.
    categories = tuple(sorted(set(KINDS)))
    quotas = ({kind: worlds // len(categories) + (i < worlds % len(categories))
               for i, kind in enumerate(categories)} if split != "train" and worlds is not None else None)
    counts = {kind: 0 for kind in categories}
    for index in range(2_000_000):
        case = _confirmation_case(index, seed) if split == "confirmation" else _case(index, seed, split)
        if quotas is not None and counts[case.kind] >= quotas[case.kind]:
            continue
        examples, state_keys = _supervision(case)
        prompts = {e.prompt for e in examples}
        if (case.world_id in seen_worlds or len(prompts) != len(examples) or prompts & seen_prompts
                or set(state_keys) & seen_states or any(_partition(key) != split for key in state_keys)):
            continue
        cases.append(case); counts[case.kind] += 1; seen_worlds.add(case.world_id); seen_prompts |= prompts; seen_states.update(state_keys)
        body_bytes += sum(len(e.prompt.encode("utf-8")) + len(e.target.encode("utf-8")) for e in examples)
        examples_count += len(examples)
        if ((worlds is not None and len(cases) == worlds) or
                (worlds is None and body_bytes >= target_bytes and examples_count % 16 == 0)):
            base = _dataset(split, cases, seed=seed, target_bytes=target_bytes if worlds is None else 0)
            return transform_dataset(base, augmentation=augmentation, scratch=scratch) if augmentation or scratch else base
    raise RuntimeError("V6 generator could not meet exact size/split closure; no partial dataset admitted")


def transform_dataset(base: V6Dataset, *, augmentation=False, scratch=False):
    if type(augmentation) is not bool or type(scratch) is not bool:
        raise ValueError("dataset transformations require explicit booleans")
    cases = tuple(replace(c, surface=replace(c.surface,
        template=2 if augmentation and i % 2 else c.surface.template,
        aliases="node-" if augmentation and i % 3 == 0 else c.surface.aliases,
        order="reverse" if augmentation and i % 2 else c.surface.order,
        scratch=scratch)) for i, c in enumerate(base.cases))
    return _dataset(base.split, cases, seed=base.manifest["seed"], target_bytes=base.manifest["target_bytes"],
                    parent=base.identity, augmentation=augmentation, scratch=scratch)


def audit_dataset(data: V6Dataset):
    if not data.cases or not data.examples or not data.samples:
        raise ValueError("empty V6 dataset")
    if len({c.world_id for c in data.cases}) != len(data.cases):
        raise ValueError("duplicate latent V6 world")
    if len({e.prompt for e in data.examples}) != len(data.examples):
        raise ValueError("duplicate model-visible V6 prompt")
    if any(len(s.tokens) > MAX_SEQUENCE or s.tokens[-1] != UTF8.eos_id or not s.eligible[-1] for s in data.samples):
        raise ValueError("invalid V6 UTF8 framing")
    for case in data.cases:
        _, keys = _supervision(case)
        if any(_partition(key) != data.split for key in keys):
            raise ValueError("V6 observation state crosses a split")
    return {"status": "PASS", "episodes": len(data.cases), "examples": len(data.examples),
            "text_bytes": data.manifest["text_bytes"], "max_sequence_tokens": data.manifest["max_sequence_tokens"]}


def assert_disjoint(*datasets):
    worlds, prompts, states = set(), set(), set()
    for data in datasets:
        audit_dataset(data)
        current_worlds = {c.world_id for c in data.cases}
        current_prompts = {e.prompt for e in data.examples}
        current_states = {key for c in data.cases for key in _supervision(c)[1]}
        if worlds & current_worlds or prompts & current_prompts or states & current_states:
            raise ValueError("V6 world/prompt/observed-state split leakage")
        worlds |= current_worlds; prompts |= current_prompts; states |= current_states


def case_to_record(case):
    return {"case_id": case.case_id, "world_id": case.world_id, "family": case.family, "kind": case.kind,
        "goal": asdict(case.goal), "initial": [asdict(e) for e in case.environment.initial],
        "channels": {key: [asdict(e) for e in values] for key,values in case.environment.channels.items()},
        "truth": asdict(case.truth), "surface": asdict(case.surface),
        "memory": None if case.memory is None else asdict(case.memory), "pair_id": case.pair_id, "panel": case.panel}


def case_from_record(row):
    event = lambda item: ExperienceEvent(**{**item, "parents": tuple(item["parents"])})
    goal = Goal(**{**row["goal"], "slots": tuple(row["goal"]["slots"])})
    memory = row["memory"]
    return V6Case(row["case_id"], row["world_id"], row["family"], row["kind"], goal,
        PublicEnvironment(tuple(event(e) for e in row["initial"]),
                          {key: tuple(event(e) for e in values) for key,values in row["channels"].items()}),
        Truth(**{**row["truth"], "admitted_public": tuple(row["truth"]["admitted_public"])}), Surface(**row["surface"]),
        None if memory is None else MemoryVersion(tuple(event(e) for e in memory["events"]), memory["parent"], memory["capacity"]),
        row["pair_id"], row["panel"])


def dataset_to_record(data):
    return {"schema": VERSION, "manifest": data.manifest, "identity": data.identity,
            "cases": [case_to_record(c) for c in data.cases], "examples": [asdict(e) for e in data.examples],
            "samples": [asdict(sample) for sample in data.samples]}


def dataset_from_record(record):
    if record.get("schema") != VERSION or digest(record["manifest"]) != record["identity"]:
        raise ValueError("V6 data manifest/identity mismatch")
    m = record["manifest"]
    data = _dataset(m["split"], tuple(case_from_record(c) for c in record["cases"]), seed=m["seed"],
                    target_bytes=m["target_bytes"], parent=m["parent_dataset"], augmentation=m["augmentation"], scratch=m["scratch"])
    if (data.manifest != m or canonical([asdict(e) for e in data.examples]) != canonical(record["examples"])
            or canonical([asdict(sample) for sample in data.samples]) != canonical(record["samples"])):
        raise ValueError("V6 semantic cases/examples fail deterministic reconstruction")
    return data


def write_dataset(data, path):
    path = Path(path)
    with path.open("xb") as handle:
        handle.write(canonical(dataset_to_record(data)) + b"\n")
    restored = read_dataset(path)
    if restored.identity != data.identity:
        raise ValueError("V6 data readback mismatch")
    return data.identity


def read_dataset(path):
    return dataset_from_record(json.loads(Path(path).read_bytes()))


def panel_cases(data, panel):
    if panel not in {"base", "query-swap", "time-order", "alias", "template", "distractor", "combined", "three-hop", "memory", "row-order", "payload-swap", "memory-swap"}:
        raise ValueError("unknown V6 data panel")
    if panel == "base":
        return data.cases
    result = []
    for case in data.cases:
        if panel in ("payload-swap", "memory-swap"):
            if panel == "payload-swap" and (case.kind not in ("missing", "conflict") or CHANNEL not in case.environment.channels):
                continue
            if panel == "memory-swap" and case.memory is None:
                continue
            pair = digest([case.world_id, panel])
            new_id = digest([case.case_id, panel, 1])
            alternative = next(symbol for symbol in SYMBOLS if symbol != case.truth.expected)
            if panel == "payload-swap":
                payload = tuple(replace(event, value=alternative) for event in case.environment.channels[CHANNEL])
                environment = PublicEnvironment(case.environment.initial, {CHANNEL: payload})
                memory = case.memory
            else:
                payload = tuple(replace(event, value=alternative) for event in case.memory.events)
                memory = MemoryVersion(payload)
                environment = case.environment
            admitted = (*environment.initial, *(event for values in environment.channels.values() for event in values),
                        *(memory.events if memory else ()))
            changed = replace(case, case_id=new_id, goal=replace(case.goal, episode=new_id),
                environment=environment, memory=memory, truth=Truth(new_id, alternative, tuple(event.identity for event in admitted)),
                pair_id=pair, panel=panel)
            result.extend((replace(case, pair_id=pair, panel=panel), changed))
            continue
        if panel == "row-order":
            pair = digest([case.world_id, panel])
            result.extend((replace(case, pair_id=pair, panel=panel),
                           replace(case, case_id=digest([case.case_id, panel]), pair_id=pair, panel=panel,
                                   surface=replace(case.surface, order="reverse"))))
            continue
        if panel == "memory":
            if case.memory is not None:
                result.append(replace(case, panel=panel))
            continue
        if panel in ("query-swap", "time-order"):
            if case.kind != "known":
                continue
            pair = digest([case.world_id, panel])
            candidates = [replace(case, pair_id=pair, panel=panel)]
            if panel == "query-swap":
                choices = [(subject, belief(case.environment.initial, subject).value) for subject in case.goal.slots
                           if subject != case.goal.subject and belief(case.environment.initial, subject).status == "derived"]
                choices = [(subject,value) for subject,value in choices if value != case.truth.expected]
                if not choices:
                    continue
                subject, value = choices[0]
                new_id = digest([case.case_id, panel, 1])
                candidates.append(replace(case, case_id=new_id, goal=replace(case.goal, episode=new_id, subject=subject),
                    truth=replace(case.truth, episode=new_id, expected=value), pair_id=pair, panel=panel))
            else:
                relevant = [e for e in case.environment.initial if e.subject == case.goal.subject]
                a,b = sorted(relevant, key=lambda e: e.tick)
                events = tuple(replace(e,tick=b.tick if e.identity == a.identity else a.tick) if e in relevant else e
                               for e in case.environment.initial)
                new_id = digest([case.case_id, panel, 1])
                environment = PublicEnvironment(events, {})
                candidates.append(replace(case, case_id=new_id, goal=replace(case.goal,episode=new_id), environment=environment,
                    truth=Truth(new_id,a.value,tuple(e.identity for e in events)), pair_id=pair, panel=panel))
            result.extend(candidates)
            continue
        if panel == "three-hop":
            if case.kind != "route":
                continue
            unused = next(s for s in SYMBOLS if s not in case.goal.slots and s != case.truth.expected)
            event = ExperienceEvent(case.goal.scope, case.truth.expected, "map", 1, unused)
            new_id = digest([case.case_id,panel])
            result.append(replace(case,case_id=new_id,goal=replace(case.goal,episode=new_id,
                slots=(*case.goal.slots,case.truth.expected)),surface=replace(case.surface,hops=3),
                environment=PublicEnvironment((*case.environment.initial,event),{}),
                truth=Truth(new_id,unused,(*case.truth.admitted_public,event.identity)),panel=panel))
            continue
        surface = replace(case.surface, aliases="unit/" if panel in ("alias","combined") else case.surface.aliases,
                          template=3 if panel in ("template","combined") else case.surface.template,
                          order="reverse" if panel in ("distractor","combined") else case.surface.order)
        environment = case.environment
        if panel in ("distractor","combined"):
            unused = [s for s in SYMBOLS if s not in case.goal.slots and s != case.truth.expected][:2]
            extras = tuple(ExperienceEvent(case.goal.scope,s,"audit",900+i,SYMBOLS[(i+5)%16]) for i,s in enumerate(unused))
            environment = PublicEnvironment((*case.environment.initial,*extras),case.environment.channels)
        variant = replace(case,surface=surface,environment=environment,panel=panel)
        # Reject the entire panel rather than silently dropping long/hard cases.
        _supervision(variant)
        result.append(variant)
    return tuple(result)


