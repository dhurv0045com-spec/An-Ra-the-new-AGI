"""Hitchhiker counterfactual worlds, split custody, health probes, and data streams.

No actor-visible structure exposes Truth, hidden channel payloads, or unretrieved
memory events. All world/variant constructors are deterministic and auditable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import itertools
import math
import random
import time
from typing import Any

from .codec import canonical, digest
from .cognition import (
    CognitiveState,
    ExperienceEvent,
    Goal,
    MemoryVersion,
    PublicEnvironment,
    Truth,
    belief,
    run_episode,
    verify,
)
from .foundation_training import TokenSample
from .hitchhiker_protocol import ALL_ARMS, PROTOCOL_SHA
from .representation import UTF8
from .v6_data import (
    CHANNEL,
    KINDS,
    MAX_SEQUENCE,
    SYMBOLS,
    Surface,
    V6Case,
    V6Dataset,
    V6Example,
    _case,
    _dataset,
    _partition,
    _state_key,
    _supervision,
    action_text,
    make_dataset,
    public_action,
)
from .v6_evaluation import ControlPolicy, PublicCase, retention_panel
from .v7_training import ordered_indices


HITCHHIKER_DATA_VERSION = "elendril-hitchhiker-worlds-v1"
HITCHHIKER_STREAM_VERSION = "elendril-hitchhiker-stream-v1"

# Canonical 14-stratum coverage schedule crossing completeness, hop depth, memory, and query availability.
COVERAGE_PATTERNS: tuple[tuple[str, int, bool, bool], ...] = (
    ("known", 0, False, False),
    ("known", 0, True, False),
    ("missing", 0, False, True),
    ("missing", 0, False, False),  # unsolvable -> ABSTAIN
    ("conflict", 0, False, True),
    ("null", 0, False, True),
    ("unavailable", 0, False, False),
    ("route", 1, False, False),
    ("route", 2, False, False),
    ("route", 3, False, False),
    ("memory", 1, True, False),
    ("memory", 2, True, False),
    ("memory", 3, True, False),
    ("memory", 2, False, False),  # missing bridge with memory off -> ABSTAIN
)


@dataclass(frozen=True, slots=True)
class HitchhikerSurface:
    """Extended public surface supporting 1..5 hops, random order, and clause separation."""

    template: int = 0
    aliases: str = ""
    hops: int = 0
    scratch: bool = False
    order: str = "original"
    order_seed: int = 0
    include_alias_instruction: bool = True
    neutral_pad_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            self.template not in range(4)
            or self.aliases not in ("", "node-", "unit/")
            or self.hops not in (0, 1, 2, 3, 4, 5)
            or type(self.scratch) is not bool
            or self.order not in ("original", "reverse", "shuffle")
            or type(self.order_seed) is not int
            or type(self.include_alias_instruction) is not bool
            or type(self.neutral_pad_bytes) is not int
            or self.neutral_pad_bytes < 0
        ):
            raise ValueError("unsupported Hitchhiker public surface")


def render_hitchhiker_public(
    state: CognitiveState,
    channels: tuple[str, ...],
    surface: Surface | HitchhikerSurface,
    memory_available: bool = False,
) -> str:
    """Render public-only observations for V6, ChainSurface, or HitchhikerSurface."""
    if tuple(channels) not in ((), (CHANNEL,)) or type(memory_available) is not bool:
        raise ValueError("unexpected Hitchhiker public channels/memory advertisement")
    aliases = surface.aliases
    include_clause = getattr(surface, "include_alias_instruction", True)
    neutral_pad = getattr(surface, "neutral_pad_bytes", 0)
    order_seed = getattr(surface, "order_seed", 0)

    def entity(symbol: str) -> str:
        return aliases + symbol

    instructions = (
        "Latest time per source counts; disagreement needs evidence. "
        "Output A-P, Q=query, !=abstain, R=memory, ~X=think.\n"
    )
    question = (
        f"Follow {surface.hops} links from {entity(state.goal.subject)}."
        if state.goal.family == "route"
        else f"Current value of {entity(state.goal.subject)}?"
    )
    if aliases and include_clause:
        question += " Output the letter after the alias."
    pad_note = ""
    if neutral_pad > 0:
        # Neutral filler matches byte count without altering facts or entity names.
        prefix = " Note:"
        if neutral_pad <= len(prefix):
            pad_note = " " * neutral_pad
        else:
            pad_note = prefix + " " * (neutral_pad - len(prefix))
    header = f"{question}{pad_note} Query={int(bool(channels))} Memory={int(memory_available)}\n"

    events = list(state.events)
    if surface.order == "reverse":
        events.reverse()
    elif surface.order == "shuffle":
        rng = random.Random(int(digest([HITCHHIKER_DATA_VERSION, "row-shuffle", order_seed, state.goal.episode]), 16))
        rng.shuffle(events)

    rows = []
    for event in events:
        subject = entity(event.subject)
        value = "null" if event.value is None else entity(event.value)
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
        raise ValueError("Hitchhiker public prompt exceeds qualified byte context; no truncation")
    return prompt


def public_case_for(case: V6Case) -> PublicCase:
    """Construct actor-only PublicCase excluding Truth, channels, and memory values."""
    return PublicCase(case.case_id, case.surface, True if case.memory is not None else None)


def _canonical_graph_signature(state: CognitiveState, *, use_raw_ticks: bool = False) -> list[Any]:
    """Compute a graph-structure signature invariant under A-P bijections and optionally ticks."""
    if use_raw_ticks:
        ticks = {e.tick: e.tick for e in state.events}
    else:
        ticks = {tick: rank for rank, tick in enumerate(sorted({e.tick for e in state.events}))}
    entities = sorted(
        {state.goal.subject}
        | {e.subject for e in state.events}
        | {e.value for e in state.events if e.value is not None}
    )
    out_edges: dict[str, list[ExperienceEvent]] = {u: [] for u in entities}
    in_edges: dict[str, list[ExperienceEvent]] = {u: [] for u in entities}
    for e in state.events:
        out_edges[e.subject].append(e)
        if e.value is not None:
            in_edges[e.value].append(e)

    # Color refinement invariant under any A-P bijection.
    colors: dict[str, Any] = {
        u: (
            1 if u == state.goal.subject else 0,
            tuple(sorted((e.source, ticks[e.tick], 1 if e.value is None else 0) for e in out_edges[u])),
            tuple(sorted((e.source, ticks[e.tick]) for e in in_edges[u])),
        )
        for u in entities
    }
    for _ in range(len(entities)):
        next_colors = {
            u: (
                colors[u],
                tuple(sorted((e.source, ticks[e.tick], None if e.value is None else colors[e.value]) for e in out_edges[u])),
                tuple(sorted((e.source, ticks[e.tick], colors[e.subject]) for e in in_edges[u])),
            )
            for u in entities
        }
        unique = sorted(set(next_colors.values()))
        rank = {val: idx for idx, val in enumerate(unique)}
        colors = {u: rank[next_colors[u]] for u in entities}
        if len(unique) == len(entities):
            break

    by_color: dict[int, list[str]] = {}
    for u, c in colors.items():
        by_color.setdefault(c, []).append(u)
    ordered_classes = [by_color[c] for c in sorted(by_color)]

    def serialize_with_map(id_map: dict[str, int]) -> tuple[int, tuple[tuple[int, str, int, int], ...]]:
        edges = tuple(
            sorted(
                (id_map[e.subject], e.source, ticks[e.tick], -1 if e.value is None else id_map[e.value])
                for e in state.events
            )
        )
        return id_map[state.goal.subject], edges

    total_perms = 1
    for group in ordered_classes:
        total_perms *= math.factorial(len(group))
        if total_perms > 720:
            break

    if total_perms <= 720:
        best: tuple[int, tuple[tuple[int, str, int, int], ...]] | None = None
        for combo in itertools.product(*(itertools.permutations(group) for group in ordered_classes)):
            flat = [entity for part in combo for entity in part]
            id_map = {entity: idx for idx, entity in enumerate(flat)}
            candidate = serialize_with_map(id_map)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
    else:
        flat = []
        for group in ordered_classes:
            # Deterministic structural tie-break when automorphisms exceed threshold.
            flat.extend(
                sorted(
                    group,
                    key=lambda u: (len(out_edges[u]), len(in_edges[u]), sorted(ticks[e.tick] for e in out_edges[u])),
                )
            )
        id_map = {entity: idx for idx, entity in enumerate(flat)}
        best = serialize_with_map(id_map)

    return [state.goal.family, best[0], list(best[1])]


def family_split_key(state: CognitiveState, channels: tuple[str, ...], memory_available: bool) -> str:
    """Canonical structural key invariant to A-P relabeling, row order, and tick scale."""
    return digest([HITCHHIKER_DATA_VERSION, "family-key", _canonical_graph_signature(state), bool(channels), bool(memory_available)])


def _exact_tick_family_key(state: CognitiveState, channels: tuple[str, ...], memory_available: bool) -> str:
    """Canonical structural key using raw (subject_color, source, raw_tick, value_color)."""
    return digest(
        [
            HITCHHIKER_DATA_VERSION,
            "exact-tick-family-key",
            _canonical_graph_signature(state, use_raw_ticks=True),
            bool(channels),
            bool(memory_available),
        ]
    )


def entity_bijection(seed_material: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Return a deterministic bijection pi: A-P -> A-P and its inverse."""
    rng = random.Random(int(digest([HITCHHIKER_DATA_VERSION, "bijection", seed_material]), 16))
    targets = list(SYMBOLS)
    rng.shuffle(targets)
    mapping = dict(zip(SYMBOLS, targets))
    inverse = {v: k for k, v in mapping.items()}
    return mapping, inverse


def _map_symbol(symbol: str | None, mapping: dict[str, str]) -> str | None:
    return None if symbol is None else mapping[symbol]


def apply_case_permutation(case: V6Case, mapping: dict[str, str]) -> V6Case:
    """Apply an A-P entity bijection consistently across goal, events, memory, and truth."""
    if set(mapping) != set(SYMBOLS) or set(mapping.values()) != set(SYMBOLS):
        raise ValueError("entity permutation must be a bijection over A-P")

    def map_event(event: ExperienceEvent) -> ExperienceEvent:
        return ExperienceEvent(
            event.scope,
            mapping[event.subject],
            event.source,
            event.tick,
            _map_symbol(event.value, mapping),
            event.parents,
        )

    initial = tuple(map_event(e) for e in case.environment.initial)
    channels = {k: tuple(map_event(e) for e in v) for k, v in case.environment.channels.items()}
    memory = (
        None
        if case.memory is None
        else MemoryVersion(tuple(map_event(e) for e in case.memory.events), case.memory.parent, case.memory.capacity)
    )
    admitted = tuple(
        e.identity
        for e in (*initial, *(ev for p in channels.values() for ev in p), *(memory.events if memory else ()))
    )
    new_id = digest([case.case_id, "perm", [mapping[s] for s in SYMBOLS]])
    goal = replace(
        case.goal,
        episode=new_id,
        subject=mapping[case.goal.subject],
        slots=tuple(mapping[s] for s in case.goal.slots),
    )
    truth = Truth(new_id, _map_symbol(case.truth.expected, mapping), admitted)
    return replace(
        case,
        case_id=new_id,
        goal=goal,
        environment=PublicEnvironment(initial, channels),
        memory=memory,
        truth=truth,
    )


def _case_supervision_keys(
    case: V6Case,
) -> tuple[tuple[V6Example, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Run public teacher supervision and return examples, state keys, family keys, and exact-tick family keys."""
    state = CognitiveState(case.goal).observe(case.environment.initial)
    channels = tuple(case.environment.channels)
    available = case.memory is not None
    examples, state_keys, family_keys, exact_tick_keys = [], [], [], []
    for decision in range(4):
        state_keys.append(_state_key(state, channels, available))
        family_keys.append(family_split_key(state, channels, available))
        exact_tick_keys.append(_exact_tick_family_key(state, channels, available))
        op, arg = public_action(state, channels, case.surface, available)
        examples.append(
            V6Example(
                f"{case.case_id}/{decision}",
                case.case_id,
                case.world_id,
                case.kind,
                "initial" if not decision else f"observed-{decision}",
                render_hitchhiker_public(state, channels, case.surface, available),
                action_text(op, arg),
            )
        )
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
        raise ValueError("teacher failed to terminate bounded Hitchhiker supervision")
    return tuple(examples), tuple(state_keys), tuple(family_keys), tuple(exact_tick_keys)


def _case_supervision(case: V6Case) -> tuple[tuple[V6Example, ...], tuple[str, ...], tuple[str, ...]]:
    """Run public teacher supervision using render_hitchhiker_public."""
    examples, state_keys, family_keys, _ = _case_supervision_keys(case)
    return examples, state_keys, family_keys


def assert_all_stage_disjoint(*datasets_or_suites: Any, structural_cohorts: Any = ()) -> dict[str, Any]:
    """Enforce instance/prompt/state/exact-tick disjointness and train/structural family custody."""
    seen_worlds: set[str] = set()
    seen_prompts: set[str] = set()
    seen_states: set[str] = set()
    seen_exact_tick_families: set[str] = set()
    split_families: dict[str, set[str]] = {}
    split_nontrivial_families: dict[str, set[str]] = {}

    def extract_cases(obj: Any, *, default_split: str | None = None) -> tuple[str, tuple[V6Case, ...]]:
        if isinstance(obj, V6Dataset):
            return obj.split, tuple(obj.cases)
        if isinstance(obj, dict) and "cohorts" in obj:
            split = obj.get("split", default_split or "validation")
            cases = tuple(c for cohort in obj["cohorts"].values() for c in cohort)
            return split, cases
        if isinstance(obj, dict):
            split = obj.get("split", default_split or "suite")
            cases = tuple(c for k, v in obj.items() if k != "split" and isinstance(v, (list, tuple)) for c in v)
            return split, cases
        if isinstance(obj, (list, tuple)):
            return default_split or "sequence", tuple(obj)
        raise TypeError("unsupported dataset or suite container")

    grouped: dict[str, list[V6Case]] = {}
    for item in datasets_or_suites:
        split, cases = extract_cases(item)
        grouped.setdefault(split, []).extend(cases)
    if structural_cohorts:
        s_split, s_cases = extract_cases(structural_cohorts, default_split="structural-holdout")
        if s_cases:
            grouped.setdefault(s_split, []).extend(s_cases)

    for split, cases in grouped.items():
        w_set, p_set, s_set, etf_set, f_set, nt_set = set(), set(), set(), set(), set(), set()
        for case in cases:
            examples, state_keys, fam_keys, exact_keys = _case_supervision_keys(case)
            w_set.add(case.world_id)
            p_set.update(e.prompt for e in examples)
            s_set.update(state_keys)
            etf_set.update(exact_keys)
            f_set.update(fam_keys)
            if len(case.environment.initial) >= 3:
                nt_set.update(fam_keys)
        if seen_worlds & w_set or seen_prompts & p_set or seen_states & s_set:
            raise ValueError("Hitchhiker split leakage in world_id, prompt, or state_key")
        if seen_exact_tick_families & etf_set:
            raise ValueError("Hitchhiker exact-tick family or cloned-instance leakage across splits")
        for other_split, other_fams in split_families.items():
            # Any relabeled leakage between train and held-out splits must fail, even on 1-edge graphs.
            if "train" in (split, other_split) and (f_set & other_fams):
                raise ValueError("Hitchhiker entity-relabeled family leakage between train and evaluation split")
        for other_split, other_nt in split_nontrivial_families.items():
            # Structural holdout enforces full nontrivial canonical family disjointness against train/val/confirmation.
            if "structural-holdout" in (split, other_split) and (nt_set & other_nt):
                raise ValueError("Hitchhiker structural-holdout nontrivial canonical family leakage across splits")
        seen_worlds |= w_set
        seen_prompts |= p_set
        seen_states |= s_set
        seen_exact_tick_families |= etf_set
        split_families[split] = f_set
        split_nontrivial_families[split] = nt_set

    return {
        "status": "PASS",
        "splits": sorted(grouped),
        "worlds": len(seen_worlds),
        "states": len(seen_states),
        "exact_tick_families": len(seen_exact_tick_families),
        "split_claims": {
            "train_vs_eval": "strict_full_canonical_family_disjoint",
            "validation_vs_confirmation": (
                "instance_prompt_state_exact_tick_disjoint_shared_elementary_chain_topology_allowed"
            ),
            "structural_holdout": (
                "strict_nontrivial_canonical_family_disjoint"
                if "structural-holdout" in grouped
                else "not_provided"
            ),
        },
    }


def _admitted_ids(env: PublicEnvironment, memory: MemoryVersion | None) -> tuple[str, ...]:
    return tuple(
        e.identity
        for e in (
            *env.initial,
            *(ev for p in env.channels.values() for ev in p),
            *(memory.events if memory else ()),
        )
    )


def _variant_case(
    base: V6Case,
    panel: str,
    env: PublicEnvironment,
    memory: MemoryVersion | None,
    *,
    goal: Goal | None = None,
    surface: Surface | HitchhikerSurface | None = None,
    expected: str | None = None,
    preserve_expected: bool = True,
    pair_id: str = "",
) -> V6Case:
    new_id = digest([base.case_id, panel, pair_id])
    resolved_goal = replace(goal or base.goal, episode=new_id)
    truth = Truth(new_id, base.truth.expected if preserve_expected else expected, _admitted_ids(env, memory))
    return replace(
        base,
        case_id=new_id,
        goal=resolved_goal,
        environment=env,
        memory=memory,
        truth=truth,
        surface=surface or base.surface,
        pair_id=pair_id,
        panel=panel,
    )


def _sample_base_chains(
    seed: int,
    hops: int,
    count: int,
    split: str,
    used_states: set[str],
    forbidden_families: set[str],
) -> tuple[V6Case, ...]:
    rng = random.Random(int(digest([HITCHHIKER_DATA_VERSION, "chain", seed, hops, split]), 16))
    cases: list[V6Case] = []
    attempts = 0
    while len(cases) < count:
        attempts += 1
        if attempts > count * 5000:
            raise RuntimeError("exhausted attempts sampling split-disjoint base chains")
        nodes = rng.sample(SYMBOLS, hops + 4)
        if split == "validation":
            tick = rng.randrange(10, 350)
        elif split == "confirmation":
            tick = rng.randrange(350, 700)
        else:
            tick = rng.randrange(10, 700)
        world = digest([HITCHHIKER_DATA_VERSION, "chain-world", seed, hops, split, nodes, tick, attempts])
        scope, reuse = "world-" + world, "memory-" + world
        src = f"src-{rng.randrange(64):02d}"
        goal = Goal(world, "route", nodes[0], scope, reuse, tuple(nodes[: hops + 1]))
        events = tuple(ExperienceEvent(scope, nodes[i], src, tick + i, nodes[i + 1]) for i in range(hops))
        state = CognitiveState(goal).observe(events)
        skey = _state_key(state, (), False)
        fkey = family_split_key(state, (), False)
        if _partition(skey) != split or skey in used_states or fkey in forbidden_families:
            continue
        # Ensure factor-answer-edge-swap has at least one untouched alternative target in the same split partition.
        path_nodes = nodes[:hops]
        unused_for_alt = [s for s in SYMBOLS if s not in path_nodes and s != nodes[hops]]
        has_split_alt = any(
            _partition(
                _state_key(
                    CognitiveState(goal).observe((*events[:-1], replace(events[-1], value=alt))),
                    (),
                    False,
                )
            )
            == split
            for alt in unused_for_alt
        )
        if not has_split_alt:
            continue
        used_states.add(skey)
        env = PublicEnvironment(events, {})
        cases.append(
            V6Case(
                world,
                world,
                "route",
                "route",
                goal,
                env,
                Truth(world, nodes[hops], _admitted_ids(env, None)),
                HitchhikerSurface(hops=hops),
                panel=f"hop-{hops}",
            )
        )
    return tuple(cases)


def build_counterfactual_suite(
    *,
    seed: int = 17071,
    worlds_per_hop: int = 16,
    split: str = "validation",
    forbidden_families: set[str] | None = None,
) -> dict[str, Any]:
    """Build the 7-factor matched counterfactual evaluation suite across 1..4 hops."""
    if split not in ("validation", "confirmation"):
        raise ValueError("counterfactual suite split must be validation or confirmation")
    if type(worlds_per_hop) is not int or worlds_per_hop < 1:
        raise ValueError("worlds_per_hop must be a positive integer")
    forbidden = set(forbidden_families or ())
    used_states: set[str] = set()

    cohorts: dict[str, list[V6Case]] = {
        "hop-1": [],
        "hop-2": [],
        "hop-3": [],
        "hop-4": [],
        "factor-wording": [],
        "factor-timestamps": [],
        "factor-row-order": [],
        "factor-distractors": [],
        "factor-entity-permutation": [],
        "factor-alias-clause": [],
        "factor-answer-edge-swap": [],
        "factor-answer-query-swap": [],
    }
    perm_records: dict[str, dict[str, Any]] = {}

    for hops in (1, 2, 3, 4):
        bases = _sample_base_chains(seed, hops, worlds_per_hop, split, used_states, forbidden)
        cohorts[f"hop-{hops}"].extend(bases)

        for idx, base in enumerate(bases):
            # Factor 1: 1-hop wording equivalence ("Follow 1 links" vs "Current value of")
            if hops == 1:
                pair = digest([base.world_id, "factor-wording"])
                follow = _variant_case(
                    base,
                    "wording-follow-1",
                    base.environment,
                    None,
                    goal=replace(base.goal, family="route"),
                    surface=HitchhikerSurface(hops=1),
                    pair_id=pair,
                )
                current = _variant_case(
                    base,
                    "wording-current-value",
                    base.environment,
                    None,
                    goal=replace(base.goal, family="revision"),
                    surface=HitchhikerSurface(hops=0),
                    pair_id=pair,
                )
                cohorts["factor-wording"].extend((follow, current))

            # Factor 2: In-range vs large timestamps with relative order fixed
            pair_ts = digest([base.world_id, "factor-timestamps"])
            in_range = _variant_case(base, "ticks-in-range", base.environment, None, pair_id=pair_ts)
            large_events = tuple(replace(e, tick=e.tick + 8000) for e in base.environment.initial)
            large_ts = _variant_case(
                base,
                "ticks-large",
                PublicEnvironment(large_events, {}),
                None,
                pair_id=pair_ts,
            )
            cohorts["factor-timestamps"].extend((in_range, large_ts))

            # Factor 3: Row order (original vs reverse vs deterministic random)
            pair_ord = digest([base.world_id, "factor-row-order"])
            cohorts["factor-row-order"].extend(
                (
                    _variant_case(base, "order-original", base.environment, None, surface=HitchhikerSurface(hops=hops, order="original"), pair_id=pair_ord),
                    _variant_case(base, "order-reverse", base.environment, None, surface=HitchhikerSurface(hops=hops, order="reverse"), pair_id=pair_ord),
                    _variant_case(
                        base,
                        "order-random",
                        base.environment,
                        None,
                        surface=HitchhikerSurface(hops=hops, order="shuffle", order_seed=seed + idx + hops),
                        pair_id=pair_ord,
                    ),
                )
            )

            # Factor 4: No distractors vs 3 unrelated distractors
            pair_dist = digest([base.world_id, "factor-distractors"])
            used_syms = set(base.goal.slots) | {base.truth.expected}
            spare_syms = [s for s in SYMBOLS if s not in used_syms]
            base_tick = base.environment.initial[0].tick
            distractor_rows = tuple(
                ExperienceEvent(base.goal.scope, spare_syms[i], "audit", base_tick + 10 + i, spare_syms[(i + 1) % len(spare_syms)])
                for i in range(min(3, len(spare_syms)))
            )
            dist_env = PublicEnvironment((*base.environment.initial, *distractor_rows), {})
            cohorts["factor-distractors"].extend(
                (
                    _variant_case(base, "distractors-none", base.environment, None, pair_id=pair_dist),
                    _variant_case(base, "distractors-matched3", dist_env, None, pair_id=pair_dist),
                )
            )

            # Factor 5: Consistent A-P entity permutation (keep permuted state in same split partition)
            pair_perm = digest([base.world_id, "factor-entity-permutation"])
            orig_p = _variant_case(base, "perm-original", base.environment, None, pair_id=pair_perm)
            for p_attempt in range(2048):
                mapping, inverse = entity_bijection((seed, base.world_id, hops, idx, p_attempt))
                cand_perm = apply_case_permutation(orig_p, mapping)
                perm_skey = _state_key(CognitiveState(cand_perm.goal).observe(cand_perm.environment.initial), (), False)
                if _partition(perm_skey) == split and perm_skey not in used_states:
                    break
            else:
                raise RuntimeError("exhausted split-disjoint permutation bijection search")
            used_states.add(perm_skey)
            perm_p = replace(
                cand_perm,
                pair_id=pair_perm,
                panel="perm-relabeled",
            )
            cohorts["factor-entity-permutation"].extend((orig_p, perm_p))
            perm_records[pair_perm] = {
                "original_case_id": orig_p.case_id,
                "permuted_case_id": perm_p.case_id,
                "mapping": mapping,
                "inverse_mapping": inverse,
            }

            # Factor 6: Alias prefix vs instruction clause vs length-matched neutral formatting
            pair_alias = digest([base.world_id, "factor-alias-clause"])
            std_surf = HitchhikerSurface(hops=hops, aliases="", include_alias_instruction=False)
            alias_clause_surf = HitchhikerSurface(hops=hops, aliases="node-", include_alias_instruction=True)
            alias_noclause_surf = HitchhikerSurface(hops=hops, aliases="node-", include_alias_instruction=False)
            dummy_state = CognitiveState(base.goal).observe(base.environment.initial)
            len_std = len(render_hitchhiker_public(dummy_state, (), std_surf, False).encode("utf-8"))
            len_alias = len(render_hitchhiker_public(dummy_state, (), alias_clause_surf, False).encode("utf-8"))
            neutral_surf = HitchhikerSurface(
                hops=hops,
                aliases="",
                include_alias_instruction=False,
                neutral_pad_bytes=max(0, len_alias - len_std),
            )
            cohorts["factor-alias-clause"].extend(
                (
                    _variant_case(base, "surface-standard", base.environment, None, surface=std_surf, pair_id=pair_alias),
                    _variant_case(base, "surface-alias-with-clause", base.environment, None, surface=alias_clause_surf, pair_id=pair_alias),
                    _variant_case(base, "surface-alias-no-clause", base.environment, None, surface=alias_noclause_surf, pair_id=pair_alias),
                    _variant_case(base, "surface-neutral-length-matched", base.environment, None, surface=neutral_surf, pair_id=pair_alias),
                )
            )

            # Factor 7a: Answer-changing terminal edge intervention with balanced marginals
            pair_edge = digest([base.world_id, "factor-answer-edge-swap"])
            path_nodes = [e.subject for e in base.environment.initial]
            unused_for_alt = [s for s in SYMBOLS if s not in path_nodes and s != base.truth.expected]
            split_alts = [
                alt
                for alt in unused_for_alt
                if _partition(
                    _state_key(
                        CognitiveState(base.goal).observe(
                            (*base.environment.initial[:-1], replace(base.environment.initial[-1], value=alt))
                        ),
                        (),
                        False,
                    )
                )
                == split
            ]
            alt_target = split_alts[(idx + hops) % len(split_alts)]
            swapped_events = (
                *base.environment.initial[:-1],
                replace(base.environment.initial[-1], value=alt_target),
            )
            cohorts["factor-answer-edge-swap"].extend(
                (
                    _variant_case(base, "edge-original", base.environment, None, pair_id=pair_edge),
                    _variant_case(
                        base,
                        "edge-intervened",
                        PublicEnvironment(swapped_events, {}),
                        None,
                        expected=alt_target,
                        preserve_expected=False,
                        pair_id=pair_edge,
                    ),
                )
            )

            # Factor 7b: Query-subject swap on the exact same visible two-chain graph
            pair_qswap = digest([base.world_id, "factor-answer-query-swap"])
            rng_q = random.Random(int(digest([base.world_id, "qswap", seed, idx]), 16))
            q_tick = (1000 + idx) if split == "validation" else (3000 + idx)
            for _q_att in range(2048):
                sym_pool = rng_q.sample(SYMBOLS, 2 * (hops + 1))
                chain_a = sym_pool[: hops + 1]
                chain_b = sym_pool[hops + 1 :]
                q_src = f"src-{rng_q.randrange(64):02d}"
                q_events = tuple(
                    [ExperienceEvent(base.goal.scope, chain_a[i], q_src, q_tick + i, chain_a[i + 1]) for i in range(hops)]
                    + [ExperienceEvent(base.goal.scope, chain_b[i], q_src, q_tick + 20 + i, chain_b[i + 1]) for i in range(hops)]
                )
                q_env = PublicEnvironment(q_events, {})
                q_slots = tuple(sym_pool)
                s_a = _state_key(CognitiveState(replace(base.goal, subject=chain_a[0], slots=q_slots)).observe(q_events), (), False)
                s_b = _state_key(CognitiveState(replace(base.goal, subject=chain_b[0], slots=q_slots)).observe(q_events), (), False)
                if _partition(s_a) == split and _partition(s_b) == split and s_a not in used_states and s_b not in used_states:
                    used_states.add(s_a)
                    used_states.add(s_b)
                    break
            else:
                raise RuntimeError("exhausted split-disjoint query-swap generation")
            case_a = _variant_case(
                base,
                "qswap-subject-a",
                q_env,
                None,
                goal=replace(base.goal, subject=chain_a[0], slots=q_slots),
                expected=chain_a[-1],
                preserve_expected=False,
                pair_id=pair_qswap,
            )
            case_b = _variant_case(
                base,
                "qswap-subject-b",
                q_env,
                None,
                goal=replace(base.goal, subject=chain_b[0], slots=q_slots),
                expected=chain_b[-1],
                preserve_expected=False,
                pair_id=pair_qswap,
            )
            cohorts["factor-answer-query-swap"].extend((case_a, case_b))

    frozen = {name: tuple(items) for name, items in cohorts.items()}
    suite = {
        "schema": "elendril-hitchhiker-counterfactual-suite-v1",
        "split": split,
        "seed": seed,
        "worlds_per_hop": worlds_per_hop,
        "cohorts": frozen,
        "permutations": perm_records,
    }
    verify_counterfactual_suite(suite)
    return suite


def _public_teacher_solve(case: V6Case) -> tuple[bool, str | None]:
    policy = ControlPolicy("fixed", surface=case.surface, memory_available=case.memory is not None)
    trace = run_episode(case.goal, case.environment, policy, memory=case.memory)
    outcome = verify(trace, case.truth)
    return outcome.correct, trace.answer


def verify_counterfactual_suite(suite: dict[str, Any]) -> dict[str, Any]:
    """Verify counterfactual invariants, public-teacher solvability, and actor-truth isolation."""
    cohorts = suite["cohorts"]
    checked = 0
    for name, cases in cohorts.items():
        if len({c.case_id for c in cases}) != len(cases):
            raise ValueError(f"duplicate case_id in cohort {name}")
        for case in cases:
            # Verify actor isolation object excludes Truth, channels, and memory values.
            pub = public_case_for(case)
            if hasattr(pub, "truth") or hasattr(pub, "environment"):
                raise ValueError("PublicCase leaked non-public fields")
            ok, _ = _public_teacher_solve(case)
            if not ok:
                raise ValueError(f"public teacher failed verification on {name}:{case.case_id}")
            _case_supervision(case)
            checked += 1

    # Check paired invariants by factor.
    def grouped(cohort_name: str) -> dict[str, list[V6Case]]:
        out: dict[str, list[V6Case]] = {}
        for c in cohorts[cohort_name]:
            out.setdefault(c.pair_id, []).append(c)
        return out

    for pair in grouped("factor-wording").values():
        if len(pair) != 2 or pair[0].environment.initial != pair[1].environment.initial or pair[0].truth.expected != pair[1].truth.expected:
            raise ValueError("factor-wording equivalence violated")

    for pair in grouped("factor-timestamps").values():
        if len(pair) != 2 or pair[0].truth.expected != pair[1].truth.expected:
            raise ValueError("factor-timestamps truth invariance violated")
        e0 = [(e.subject, e.source, e.value) for e in pair[0].environment.initial]
        e1 = [(e.subject, e.source, e.value) for e in pair[1].environment.initial]
        if e0 != e1:
            raise ValueError("factor-timestamps altered non-timestamp fields")

    for trio in grouped("factor-row-order").values():
        if len(trio) != 3 or len({c.truth.expected for c in trio}) != 1:
            raise ValueError("factor-row-order truth invariance violated")

    for pair in grouped("factor-distractors").values():
        if len(pair) != 2 or pair[0].truth.expected != pair[1].truth.expected:
            raise ValueError("factor-distractors altered expected truth")
        if len(pair[1].environment.initial) - len(pair[0].environment.initial) != 3:
            raise ValueError("factor-distractors must add 3 unrelated rows")

    for pid, pair in grouped("factor-entity-permutation").items():
        if len(pair) != 2:
            raise ValueError("incomplete entity permutation pair")
        meta = suite["permutations"][pid]
        mapping = meta["mapping"]
        if mapping[pair[0].truth.expected] != pair[1].truth.expected:
            raise ValueError("factor-entity-permutation expected answer mismatch")

    for quad in grouped("factor-alias-clause").values():
        if len(quad) != 4 or len({c.truth.expected for c in quad}) != 1:
            raise ValueError("factor-alias-clause truth invariance violated")
        by_panel = {c.panel: c for c in quad}
        p_alias = render_hitchhiker_public(CognitiveState(by_panel["surface-alias-with-clause"].goal).observe(by_panel["surface-alias-with-clause"].environment.initial), (), by_panel["surface-alias-with-clause"].surface, False)
        p_neutral = render_hitchhiker_public(CognitiveState(by_panel["surface-neutral-length-matched"].goal).observe(by_panel["surface-neutral-length-matched"].environment.initial), (), by_panel["surface-neutral-length-matched"].surface, False)
        if len(p_alias.encode("utf-8")) != len(p_neutral.encode("utf-8")):
            raise ValueError("neutral length-matched surface does not match alias prompt byte length")

    for pair in grouped("factor-answer-edge-swap").values():
        if len(pair) != 2 or pair[0].truth.expected == pair[1].truth.expected:
            raise ValueError("factor-answer-edge-swap failed to change ground-truth answer")

    for pair in grouped("factor-answer-query-swap").values():
        if len(pair) != 2:
            raise ValueError("incomplete query-swap pair")
        a, b = pair
        if a.environment.initial != b.environment.initial or a.goal.subject == b.goal.subject or a.truth.expected == b.truth.expected:
            raise ValueError("factor-answer-query-swap violated identical-graph distinct-query contract")

    return {"status": "PASS", "verified_cases": checked, "cohorts": len(cohorts)}


def build_structural_holdout_suite(
    *,
    seed: int = 37073,
    worlds_per_topology: int = 16,
    split: str = "structural-holdout",
) -> dict[str, Any]:
    """Build structurally novel multi-hop/diamond/converging-fork/5-hop topologies disjoint from train/val/conf."""
    if type(worlds_per_topology) is not int or worlds_per_topology < 1:
        raise ValueError("worlds_per_topology must be a positive integer")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")

    rng = random.Random(int(digest([HITCHHIKER_DATA_VERSION, "structural-holdout", seed, split]), 16))
    used_worlds: set[str] = set()
    used_states: set[str] = set()
    used_prompts: set[str] = set()

    cohorts: dict[str, list[V6Case]] = {
        "topology-diamond": [],
        "topology-converging-fork": [],
        "topology-5hop": [],
        "topology-multihop-bridge": [],
    }

    def accept_case(case: V6Case) -> bool:
        if case.world_id in used_worlds:
            return False
        exs, skeys, _ = _case_supervision(case)
        if any(sk in used_states for sk in skeys):
            return False
        if any(e.prompt in used_prompts for e in exs):
            return False
        ok, _ = _public_teacher_solve(case)
        if not ok:
            raise ValueError(f"structural holdout case failed public teacher: {case.case_id}")
        used_worlds.add(case.world_id)
        used_states.update(skeys)
        used_prompts.update(e.prompt for e in exs)
        return True

    for topo in cohorts:
        built = 0
        attempts = 0
        while built < worlds_per_topology:
            attempts += 1
            if attempts > worlds_per_topology * 500:
                raise RuntimeError(f"exhausted attempts building structural topology {topo}")
            t0 = 5000 + built * 10 + (attempts % 7)
            world = digest([HITCHHIKER_DATA_VERSION, "struct", topo, seed, split, built, attempts])
            scope, reuse = "world-" + world, "memory-" + world

            if topo == "topology-diamond":
                n0, n1, n2, n3, n4, n5 = rng.sample(SYMBOLS, 6)
                events = (
                    ExperienceEvent(scope, n0, "map", t0, n1),
                    ExperienceEvent(scope, n0, "map", t0 + 1, n2),
                    ExperienceEvent(scope, n1, "map", t0 + 2, n3),
                    ExperienceEvent(scope, n2, "map", t0 + 3, n4),
                    ExperienceEvent(scope, n3, "map", t0 + 4, n5),
                    ExperienceEvent(scope, n4, "map", t0 + 5, n5),
                )
                env = PublicEnvironment(events, {})
                goal = Goal(world, "route", n0, scope, reuse, (n0, n1, n2, n3, n4, n5))
                expected = n5
                hops = 3
            elif topo == "topology-converging-fork":
                s1, s2, s3, n, m, t1, t2 = rng.sample(SYMBOLS, 7)
                events = (
                    ExperienceEvent(scope, s1, "map", t0, m),
                    ExperienceEvent(scope, s2, "map", t0 + 1, m),
                    ExperienceEvent(scope, s3, "map", t0 + 2, n),
                    ExperienceEvent(scope, n, "map", t0 + 3, m),
                    ExperienceEvent(scope, m, "map", t0 + 4, t1),
                    ExperienceEvent(scope, t1, "map", t0 + 5, t2),
                )
                env = PublicEnvironment(events, {})
                goal = Goal(world, "route", s1, scope, reuse, (s1, s2, s3, n, m, t1, t2))
                expected = t2
                hops = 3
            elif topo == "topology-5hop":
                nodes = rng.sample(SYMBOLS, 6)
                events = tuple(
                    ExperienceEvent(scope, nodes[i], "map", t0 + i, nodes[i + 1])
                    for i in range(5)
                )
                env = PublicEnvironment(events, {})
                goal = Goal(world, "route", nodes[0], scope, reuse, tuple(nodes))
                expected = nodes[5]
                hops = 5
            else:
                # topology-multihop-bridge: 4-hop path with converging feeders and channel bridge
                n0, n1, n2, n3, n4, x, y = rng.sample(SYMBOLS, 7)
                events = (
                    ExperienceEvent(scope, n0, "map", t0, n1),
                    ExperienceEvent(scope, n2, "map", t0 + 2, n3),
                    ExperienceEvent(scope, n3, "map", t0 + 3, n4),
                    ExperienceEvent(scope, x, "map", t0 + 4, n2),
                    ExperienceEvent(scope, y, "map", t0 + 5, n3),
                )
                bridge = (ExperienceEvent(scope, n1, "map", t0 + 1, n2),)
                env = PublicEnvironment(events, {CHANNEL: bridge})
                goal = Goal(world, "route", n0, scope, reuse, (n0, n1, n2, n3, n4, x, y))
                expected = n4
                hops = 4

            case = V6Case(
                world,
                world,
                "route",
                topo,
                goal,
                env,
                Truth(world, expected, _admitted_ids(env, None)),
                HitchhikerSurface(hops=hops),
                panel=topo,
            )
            if accept_case(case):
                cohorts[topo].append(case)
                built += 1

    return {
        "schema": "elendril-hitchhiker-structural-holdout-v1",
        "split": split,
        "seed": seed,
        "worlds_per_topology": worlds_per_topology,
        "cohorts": {k: tuple(v) for k, v in cohorts.items()},
    }


def build_health_probes(
    *,
    seed: int = 27173,
    worlds_per_probe: int = 8,
    split: str = "validation",
) -> dict[str, Any]:
    """Construct Distractor-Trap, Belief-Revision, Missing-Bridge, and Preservation probes."""
    if worlds_per_probe < 1:
        raise ValueError("worlds_per_probe must be >= 1")
    rng = random.Random(int(digest([HITCHHIKER_DATA_VERSION, "health", seed, split]), 16))
    distractor_trap: list[V6Case] = []
    belief_revision: list[V6Case] = []
    missing_bridge: list[V6Case] = []

    for idx in range(worlds_per_probe):
        # 1. Distractor / Trap-Hop probe: 3-hop valid + 3-hop disconnected distractor
        #    enriched with a shared-vertex incoming edge and a stale older edge on the queried path.
        syms = rng.sample(SYMBOLS, 9)
        world_d = digest([HITCHHIKER_DATA_VERSION, "trap", seed, split, idx])
        scope_d = "world-" + world_d
        t0 = 100 + idx
        events_d = (
            ExperienceEvent(scope_d, syms[0], "map", t0, syms[1]),
            ExperienceEvent(scope_d, syms[1], "map", t0 - 5, syms[7]),  # stale older edge on queried path
            ExperienceEvent(scope_d, syms[1], "map", t0 + 1, syms[2]),
            ExperienceEvent(scope_d, syms[2], "map", t0 + 2, syms[3]),
            ExperienceEvent(scope_d, syms[8], "map", t0 + 3, syms[1]),  # shared-vertex incoming edge into syms[1]
            ExperienceEvent(scope_d, syms[4], "map", t0, syms[5]),
            ExperienceEvent(scope_d, syms[5], "map", t0 + 1, syms[6]),
            ExperienceEvent(scope_d, syms[6], "map", t0 + 2, syms[7]),
        )
        env_d = PublicEnvironment(events_d, {})
        goal_d = Goal(world_d, "route", syms[0], scope_d, "memory-" + world_d, tuple(syms))
        distractor_trap.append(
            V6Case(
                world_d,
                world_d,
                "route",
                "distractor-trap",
                goal_d,
                env_d,
                Truth(world_d, syms[3], _admitted_ids(env_d, None)),
                HitchhikerSurface(hops=3, order="shuffle", order_seed=seed + idx),
                panel="distractor-trap",
            )
        )

        # 2. Belief revision probe: same-source update + timestamp swap pair + cross-source disagreement variants
        s_syms = rng.sample(SYMBOLS, 5)
        subj, old_v, new_v, other_s, other_v = s_syms
        world_r = digest([HITCHHIKER_DATA_VERSION, "revision", seed, split, idx])
        scope_r = "world-" + world_r
        pair_r = digest([world_r, "timestamp-swap"])
        pair_cross = digest([world_r, "cross-source"])
        ev_r1 = (
            ExperienceEvent(scope_r, subj, "s0", 10 + idx, old_v),
            ExperienceEvent(scope_r, subj, "s0", 20 + idx, new_v),
            ExperienceEvent(scope_r, other_s, "s1", 25 + idx, other_v),
        )
        ev_r2 = (
            ExperienceEvent(scope_r, subj, "s0", 20 + idx, old_v),
            ExperienceEvent(scope_r, subj, "s0", 10 + idx, new_v),
            ExperienceEvent(scope_r, other_s, "s1", 25 + idx, other_v),
        )
        ev_cross = (
            ExperienceEvent(scope_r, subj, "s0", 10 + idx, old_v),
            ExperienceEvent(scope_r, subj, "s1", 30 + idx, new_v),
            ExperienceEvent(scope_r, other_s, "s1", 25 + idx, other_v),
        )
        resolve_event = ExperienceEvent(scope_r, subj, "s0", 40 + idx, new_v)
        env_cross_unres = PublicEnvironment(ev_cross, {})
        env_cross_res = PublicEnvironment(ev_cross, {CHANNEL: (resolve_event,)})
        goal_r = Goal(world_r, "revision", subj, scope_r, "memory-" + world_r, (subj, other_s))
        c_r1 = V6Case(
            digest([world_r, "r1"]),
            world_r,
            "revision",
            "belief-revision",
            replace(goal_r, episode=digest([world_r, "r1"])),
            PublicEnvironment(ev_r1, {}),
            Truth(digest([world_r, "r1"]), new_v, _admitted_ids(PublicEnvironment(ev_r1, {}), None)),
            HitchhikerSurface(hops=0),
            pair_id=pair_r,
            panel="belief-revision-latest",
        )
        c_r2 = V6Case(
            digest([world_r, "r2"]),
            world_r,
            "revision",
            "belief-revision",
            replace(goal_r, episode=digest([world_r, "r2"])),
            PublicEnvironment(ev_r2, {}),
            Truth(digest([world_r, "r2"]), old_v, _admitted_ids(PublicEnvironment(ev_r2, {}), None)),
            HitchhikerSurface(hops=0),
            pair_id=pair_r,
            panel="belief-revision-swapped",
        )
        c_cross_unres = V6Case(
            digest([world_r, "cross-unres"]),
            world_r,
            "revision",
            "belief-revision",
            replace(goal_r, episode=digest([world_r, "cross-unres"])),
            env_cross_unres,
            Truth(digest([world_r, "cross-unres"]), None, _admitted_ids(env_cross_unres, None)),
            HitchhikerSurface(hops=0),
            pair_id=pair_cross,
            panel="belief-revision-cross-source-unresolvable",
        )
        c_cross_res = V6Case(
            digest([world_r, "cross-res"]),
            world_r,
            "revision",
            "belief-revision",
            replace(goal_r, episode=digest([world_r, "cross-res"])),
            env_cross_res,
            Truth(digest([world_r, "cross-res"]), new_v, _admitted_ids(env_cross_res, None)),
            HitchhikerSurface(hops=0),
            pair_id=pair_cross,
            panel="belief-revision-cross-source-resolvable",
        )
        belief_revision.extend((c_r1, c_r2, c_cross_unres, c_cross_res))

        # 3. Missing-bridge quartet: A->B, C->D without B->C across (unsolvable, memory, query, complete)
        m_syms = rng.sample(SYMBOLS, 4)
        a, b, c, d = m_syms
        world_m = digest([HITCHHIKER_DATA_VERSION, "missing-bridge", seed, split, idx])
        scope_m, reuse_m = "world-" + world_m, "memory-" + world_m
        edge_ab = ExperienceEvent(scope_m, a, "map", 100 + idx, b)
        edge_cd = ExperienceEvent(scope_m, c, "map", 102 + idx, d)
        bridge_init = ExperienceEvent(scope_m, b, "map", 101 + idx, c)
        bridge_mem = ExperienceEvent(reuse_m, b, "map", 101 + idx, c)
        goal_m = Goal(world_m, "route", a, scope_m, reuse_m, (a, b, c, d))

        env_unsolv = PublicEnvironment((edge_ab, edge_cd), {})
        env_query = PublicEnvironment((edge_ab, edge_cd), {CHANNEL: (bridge_init,)})
        env_comp = PublicEnvironment((edge_ab, bridge_init, edge_cd), {})
        mem = MemoryVersion((bridge_mem,))

        for tag, env_i, mem_i, exp_i in (
            ("mb-unsolvable", env_unsolv, None, None),
            ("mb-memory", env_unsolv, mem, d),
            ("mb-query", env_query, None, d),
            ("mb-complete", env_comp, None, d),
        ):
            cid = digest([world_m, tag])
            missing_bridge.append(
                V6Case(
                    cid,
                    world_m,
                    "route",
                    tag,
                    replace(goal_m, episode=cid),
                    env_i,
                    Truth(cid, exp_i, _admitted_ids(env_i, mem_i)),
                    HitchhikerSurface(hops=3),
                    memory=mem_i,
                    pair_id=world_m,
                    panel=tag,
                )
            )

    probes = {
        "distractor-trap": tuple(distractor_trap),
        "belief-revision": tuple(belief_revision),
        "missing-bridge": tuple(missing_bridge),
        "preservation": (),
    }
    for name in ("distractor-trap", "belief-revision", "missing-bridge"):
        for case in probes[name]:
            ok, _ = _public_teacher_solve(case)
            if not ok:
                raise ValueError(f"health probe {name} failed public teacher verification")
    return probes


def score_missing_bridge_probe(episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score Missing-Bridge probe; an always-abstain policy fails the combined gate."""
    by_panel: dict[str, list[dict[str, Any]]] = {}
    for row in episode_rows:
        by_panel.setdefault(row["panel"], []).append(row)
    unsolv = by_panel.get("mb-unsolvable", [])
    mem_rows = by_panel.get("mb-memory", [])
    qry_rows = by_panel.get("mb-query", [])
    comp_rows = by_panel.get("mb-complete", [])

    def rate(rows: list[dict[str, Any]], pred) -> float | None:
        return sum(1 for r in rows if pred(r)) / len(rows) if rows else None

    abstain_unsolv = rate(unsolv, lambda r: r["trace"]["status"] == "abstained" and r["outcome"]["correct"])
    solv_rows = [*mem_rows, *qry_rows, *comp_rows]
    solv_acc = rate(solv_rows, lambda r: r["outcome"]["correct"])
    retrieve_recall = rate(mem_rows, lambda r: any(d["operation"] == "RETRIEVE" for d in r["trace"]["decisions"]))
    query_recall = rate(qry_rows, lambda r: any(d["operation"] == "QUERY" for d in r["trace"]["decisions"]))
    all_rows = [*unsolv, *solv_rows]
    invalid_rate = rate(all_rows, lambda r: r["trace"]["status"] == "invalid_action")
    combined_pass = bool(
        abstain_unsolv is not None
        and solv_acc is not None
        and invalid_rate is not None
        and abstain_unsolv >= 0.80
        and solv_acc >= 0.80
        and (retrieve_recall is None or retrieve_recall >= 0.80)
        and (query_recall is None or query_recall >= 0.80)
        and invalid_rate <= 0.05
    )
    return {
        "unsolvable_abstain_accuracy": abstain_unsolv,
        "solvable_answer_accuracy": solv_acc,
        "memory_retrieve_recall": retrieve_recall,
        "query_recall": query_recall,
        "invalid_action_rate": invalid_rate,
        "combined_gate_pass": combined_pass,
        "counts": {k: len(v) for k, v in by_panel.items()},
    }


def score_health_probes(
    probe_records: dict[str, list[dict[str, Any]]],
    *,
    stage_records: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score all four cognitive health probes including preservation gating."""
    trap_rows = probe_records.get("distractor-trap", [])
    rev_rows = probe_records.get("belief-revision", [])
    mb_rows = probe_records.get("missing-bridge", [])

    rev_pairs: dict[str, list[dict[str, Any]]] = {}
    cross_pairs: dict[str, list[dict[str, Any]]] = {}
    for r in rev_rows:
        if r.get("panel") in ("belief-revision-latest", "belief-revision-swapped"):
            rev_pairs.setdefault(r["pair_id"], []).append(r)
        elif r.get("panel") in (
            "belief-revision-cross-source-unresolvable",
            "belief-revision-cross-source-resolvable",
        ):
            cross_pairs.setdefault(r["pair_id"], []).append(r)
    both_rev = sum(1 for p in rev_pairs.values() if len(p) == 2 and all(x["outcome"]["correct"] for x in p))
    both_cross = sum(1 for p in cross_pairs.values() if len(p) == 2 and all(x["outcome"]["correct"] for x in p))
    cross_unres = [r for r in rev_rows if r.get("panel") == "belief-revision-cross-source-unresolvable"]
    cross_res = [r for r in rev_rows if r.get("panel") == "belief-revision-cross-source-resolvable"]

    return {
        "distractor_trap": {
            "episodes": len(trap_rows),
            "correct": sum(1 for r in trap_rows if r["outcome"]["correct"]),
            "accuracy": sum(1 for r in trap_rows if r["outcome"]["correct"]) / len(trap_rows) if trap_rows else None,
        },
        "belief_revision": {
            "episodes": len(rev_rows),
            "correct": sum(1 for r in rev_rows if r["outcome"]["correct"]),
            "both_correct_timestamp_pairs": both_rev,
            "total_timestamp_pairs": len(rev_pairs),
            "both_correct_cross_source_pairs": both_cross,
            "total_cross_source_pairs": len(cross_pairs),
            "cross_source_unresolvable_accuracy": (
                sum(1 for r in cross_unres if r["outcome"]["correct"]) / len(cross_unres) if cross_unres else None
            ),
            "cross_source_resolvable_accuracy": (
                sum(1 for r in cross_res if r["outcome"]["correct"]) / len(cross_res) if cross_res else None
            ),
        },
        "missing_bridge": score_missing_bridge_probe(mb_rows),
        "preservation": retention_panel(stage_records),
    }


def _sample_fresh_case_of_kind(
    *,
    generator_version: str,
    data_seed: int,
    update: int,
    batch_slot: int,
    target_kind: str,
    forbidden_families: set[str],
    forbidden_worlds: set[str],
    forbidden_prompts: set[str],
    hops_override: int | None = None,
) -> tuple[V6Case, int]:
    """Deterministically generate one split-isolated fresh training world of target_kind."""
    rejected = 0
    for nonce in range(10000):
        candidate_seed = int(digest([generator_version, data_seed, update, batch_slot, nonce]), 16) % (2**32)
        case = _case(KINDS.index(target_kind), candidate_seed, "train")
        if hops_override is not None and case.family == "route":
            # Build 1..3 hop route world preserving the 3 distractor rows.
            rng = random.Random(int(digest([generator_version, candidate_seed, "hops", hops_override]), 16))
            nodes = rng.sample(SYMBOLS, hops_override + 4)
            tick = rng.randrange(2, 800)
            distractors = [
                (nodes[hops_override + 1], "s0", tick - 1, rng.choice(SYMBOLS)),
                (nodes[hops_override + 2], "map", tick, rng.choice(SYMBOLS)),
                (nodes[hops_override + 3], "map", tick + 1, rng.choice(SYMBOLS)),
            ]
            path_rows = [(nodes[i], "map", tick + i, nodes[i + 1]) for i in range(hops_override)]
            remembered = [path_rows.pop()] if target_kind == "memory" else []
            rows = [*distractors, *path_rows]
            rng.shuffle(rows)
            world = digest([generator_version, "fresh-route", candidate_seed, hops_override, rows, remembered])
            scope, reuse = "world-" + world, "memory-" + world
            initial = tuple(ExperienceEvent(scope, *r) for r in rows)
            memory = MemoryVersion(tuple(ExperienceEvent(reuse, *r) for r in remembered)) if remembered else None
            goal = Goal(world, "route", nodes[0], scope, reuse, tuple(nodes[: hops_override + 1]))
            env = PublicEnvironment(initial, {})
            case = V6Case(
                world,
                world,
                "route",
                target_kind,
                goal,
                env,
                Truth(world, nodes[hops_override], _admitted_ids(env, memory)),
                HitchhikerSurface(hops=hops_override),
                memory=memory,
            )

        if case.world_id in forbidden_worlds:
            rejected += 1
            continue
        examples, state_keys, fam_keys = _case_supervision(case)
        prompts = {e.prompt for e in examples}
        if (
            any(_partition(sk) != "train" for sk in state_keys)
            or (set(fam_keys) & forbidden_families)
            or (prompts & forbidden_prompts)
        ):
            rejected += 1
            continue
        return case, rejected
    raise RuntimeError("fresh world stream could not sample a split-disjoint candidate within bound")


def _coverage_dataset(
    *,
    seed: int,
    target_examples: int,
    forbidden_families: set[str],
    forbidden_worlds: set[str],
    forbidden_prompts: set[str],
) -> tuple[tuple[V6Case, ...], tuple[V6Example, ...], tuple[TokenSample, ...], dict[str, int]]:
    """Construct matched coverage dataset crossing 1/2/3-hop completeness, memory, and query availability."""
    cases: list[V6Case] = []
    examples: list[V6Example] = []
    samples: list[TokenSample] = []
    seen_worlds = set(forbidden_worlds)
    seen_prompts = set(forbidden_prompts)
    rejected = 0
    cursor = 0

    # Cross completeness and availability systematically with public teacher targets.
    while len(samples) < target_examples:
        kind_tag, hops, mem_on, qry_on = COVERAGE_PATTERNS[cursor % len(COVERAGE_PATTERNS)]
        cursor += 1
        base_kind = "route" if kind_tag == "route" else ("memory" if kind_tag == "memory" else kind_tag)
        base_case, rej = _sample_fresh_case_of_kind(
            generator_version=HITCHHIKER_STREAM_VERSION + "-coverage",
            data_seed=seed,
            update=0,
            batch_slot=cursor,
            target_kind=base_kind,
            forbidden_families=forbidden_families,
            forbidden_worlds=seen_worlds,
            forbidden_prompts=seen_prompts,
            hops_override=hops if base_kind in ("route", "memory") else None,
        )
        rejected += rej

        # Apply explicit availability crossing and recompute truth via public semantics.
        env = base_case.environment
        mem = base_case.memory
        if base_kind == "known" and mem_on:
            # Available but irrelevant memory when context is already complete.
            unused = next(s for s in SYMBOLS if s not in base_case.goal.slots)
            mem = MemoryVersion((ExperienceEvent(base_case.goal.reusable_scope, unused, "audit", 1, unused),))
        if not qry_on and env.channels:
            env = PublicEnvironment(env.initial, {})
        if not mem_on:
            mem = None

        # Recompute teacher truth after crossing availability.
        policy = ControlPolicy("fixed", surface=base_case.surface, memory_available=mem is not None)
        trace = run_episode(base_case.goal, env, policy, memory=mem)
        new_expected = trace.answer if trace.status == "answered" else None
        cid = digest([base_case.case_id, "cov", cursor, mem_on, qry_on])
        cov_case = replace(
            base_case,
            case_id=cid,
            world_id=cid,
            goal=replace(base_case.goal, episode=cid),
            environment=env,
            memory=mem,
            truth=Truth(cid, new_expected, _admitted_ids(env, mem)),
            panel=f"coverage-{kind_tag}",
        )
        exs, _, fkeys = _case_supervision(cov_case)
        if (set(fkeys) & forbidden_families) or any(e.prompt in seen_prompts for e in exs):
            rejected += 1
            continue
        seen_worlds.add(cov_case.world_id)
        for ex in exs:
            if len(samples) >= target_examples:
                break
            seen_prompts.add(ex.prompt)
            frame = UTF8.frame(ex.prompt, ex.target, max_length=MAX_SEQUENCE)
            cases.append(cov_case)
            examples.append(ex)
            samples.append(
                TokenSample(
                    f"cov-{len(samples)}-{ex.task_id}",
                    frame.ids,
                    frame.loss_mask,
                    cov_case.world_id,
                    cov_case.kind,
                )
            )
    return tuple(cases), tuple(examples), tuple(samples), {"accepted": len(samples), "rejected": rejected}


def build_training_stream(
    arm: str,
    *,
    base_data: V6Dataset | None = None,
    total_updates: int,
    batch_size: int,
    data_seed: int = 606,
    order_seed: int = 7,
    relabel_seed: int = 1009,
    slot_assignment: str = "first-mention",
    heldout_suites: tuple[Any, ...] = (),
) -> dict[str, Any]:
    """Build deterministic training stream for B0, A, B, AB, C, AB+C, C+penalty, D-external."""
    t0 = time.perf_counter()
    if arm not in ("B0", "A", "B", "AB", "C", "AB+C", "C+penalty", "D-external", "S20-contrast"):
        raise ValueError(f"unsupported Hitchhiker training arm: {arm}")
    if type(total_updates) is not int or total_updates < 1:
        raise ValueError("total_updates must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    use_fresh = arm in ("A", "AB", "AB+C", "S20-contrast")
    use_relabel = arm in ("B", "AB", "AB+C", "S20-contrast")
    use_coverage = arm in ("C", "AB+C", "C+penalty")
    use_external_slots = arm == "D-external"
    coverage_recipe = "14-stratum-coverage" if use_coverage else "base-task-distribution"
    data_custody_verified = bool(heldout_suites)

    if base_data is None:
        base_data = make_dataset("train", target_bytes=40_000, seed=data_seed)

    # Collect heldout families/worlds/prompts to guarantee strict split separation.
    forbidden_families: set[str] = set()
    forbidden_worlds: set[str] = set()
    forbidden_prompts: set[str] = set()
    for suite in heldout_suites:
        if hasattr(suite, "cases"):
            cases = suite.cases
        elif isinstance(suite, dict) and "cohorts" in suite:
            cases = tuple(c for v in suite["cohorts"].values() for c in v)
        elif isinstance(suite, dict):
            cases = tuple(c for k, v in suite.items() if k != "split" and isinstance(v, (list, tuple)) for c in v)
        elif isinstance(suite, (list, tuple)):
            cases = tuple(suite)
        else:
            raise TypeError("unsupported heldout suite container")
        for c in cases:
            forbidden_worlds.add(c.world_id)
            exs, _, fkeys = _case_supervision(c)
            forbidden_families.update(fkeys)
            forbidden_prompts.update(e.prompt for e in exs)

    base_case_map = {c.case_id: c for c in base_data.cases}
    base_examples = base_data.examples
    base_samples = base_data.samples

    # Compute exact B0 update -> (example_index) plan using V7 ordered_indices.
    per_epoch = math.ceil(len(base_samples) / batch_size)
    epoch_orders: dict[int, tuple[int, ...]] = {}
    b0_slots: list[tuple[int, int]] = []
    for u in range(total_updates):
        epoch, batch_idx = divmod(u, per_epoch)
        if epoch not in epoch_orders:
            epoch_orders[epoch] = ordered_indices(base_samples, order_seed, epoch, "blocked")
        start = batch_idx * batch_size
        chosen_indices = epoch_orders[epoch][start : start + batch_size]
        while len(chosen_indices) < batch_size:
            chosen_indices = (*chosen_indices, *epoch_orders[epoch][: batch_size - len(chosen_indices)])
        for idx in chosen_indices:
            b0_slots.append((epoch, idx))

    contract_meta = {
        "protocol_sha256": PROTOCOL_SHA,
        "total_updates": total_updates,
        "batch_size": batch_size,
        "sampling_method": "blocked",
        "data_custody_verified": data_custody_verified,
        "coverage_recipe": coverage_recipe,
    }

    if arm == "B0":
        # Static baseline keeps base_data.samples and exact V7 blocked ordering.
        elapsed = time.perf_counter() - t0
        return {
            "arm": arm,
            **contract_meta,
            "generator_version": HITCHHIKER_STREAM_VERSION,
            "data_seed": data_seed,
            "order_seed": order_seed,
            "relabel_seed": relabel_seed,
            "samples": base_samples,
            "examples": base_examples,
            "cases": base_data.cases,
            "use_static_indexer": True,
            "stream_metrics": {
                "accepted_candidates": len(base_samples),
                "rejected_candidates": 0,
                "pool_size": len(base_samples),
                "target_pool_size": len(base_samples),
                "requested_examples": total_updates * batch_size,
                "total_updates": total_updates,
                "batch_size": batch_size,
                "generation_seconds": elapsed,
            },
        }

    if use_external_slots:
        canonicalizer = ExternalSlotCanonicalizer(assignment=slot_assignment, slot_seed=relabel_seed)
        canon_examples: list[V6Example] = []
        canon_samples: list[TokenSample] = []
        for case in base_data.cases:
            init_state = CognitiveState(case.goal).observe(case.environment.initial)
            mapping, _ = canonicalizer.build_mapping(init_state, case.surface)
            mapped_case = apply_case_permutation(case, mapping)
            exs, _, _ = _case_supervision(mapped_case)
            for ex in exs:
                frame = UTF8.frame(ex.prompt, ex.target, max_length=MAX_SEQUENCE)
                tid = f"dext-{ex.task_id}"
                canon_examples.append(replace(ex, task_id=tid))
                canon_samples.append(TokenSample(tid, frame.ids, frame.loss_mask, case.world_id, case.kind))
        elapsed = time.perf_counter() - t0
        return {
            "arm": arm,
            **contract_meta,
            "generator_version": HITCHHIKER_STREAM_VERSION,
            "data_seed": data_seed,
            "order_seed": order_seed,
            "relabel_seed": relabel_seed,
            "slot_assignment": slot_assignment,
            "samples": tuple(canon_samples),
            "examples": tuple(canon_examples),
            "cases": base_data.cases,
            "use_static_indexer": True,
            "stream_metrics": {
                "accepted_candidates": len(canon_samples),
                "rejected_candidates": 0,
                "pool_size": len(canon_samples),
                "target_pool_size": len(base_samples),
                "requested_examples": total_updates * batch_size,
                "total_updates": total_updates,
                "batch_size": batch_size,
                "generation_seconds": elapsed,
            },
        }

    # Build static coverage pool for C / C+penalty matching B0's pool size and blocked epoch repetition.
    stream_samples: list[TokenSample] = []
    stream_examples: list[V6Example] = []
    accepted = 0
    rejected = 0
    seen_worlds: set[str] = set(forbidden_worlds)
    seen_prompts: set[str] = set(forbidden_prompts)
    if use_coverage and not use_fresh:
        _, cov_exs, cov_samps, cov_stats = _coverage_dataset(
            seed=data_seed,
            target_examples=len(base_samples),
            forbidden_families=forbidden_families,
            forbidden_worlds=seen_worlds,
            forbidden_prompts=seen_prompts,
        )
        accepted += cov_stats["accepted"]
        rejected += cov_stats["rejected"]
        elapsed = time.perf_counter() - t0
        return {
            "arm": arm,
            **contract_meta,
            "generator_version": HITCHHIKER_STREAM_VERSION,
            "data_seed": data_seed,
            "order_seed": order_seed,
            "relabel_seed": relabel_seed,
            "samples": cov_samps,
            "examples": cov_exs,
            "cases": (),
            "use_static_indexer": True,
            "stream_metrics": {
                "accepted_candidates": accepted,
                "rejected_candidates": rejected,
                "pool_size": len(cov_samps),
                "target_pool_size": len(base_samples),
                "requested_examples": total_updates * batch_size,
                "total_updates": total_updates,
                "batch_size": batch_size,
                "generation_seconds": elapsed,
            },
        }

    for flat_idx in range(total_updates * batch_size):
        u, slot = divmod(flat_idx, batch_size)
        epoch, base_idx = b0_slots[flat_idx]
        ref_ex = base_examples[base_idx]
        ref_case = base_case_map[ref_ex.case_id]

        if use_fresh:
            target_kind = ref_ex.kind
            hops_ov = None
            mem_on = target_kind == "memory"
            qry_on = target_kind in ("known", "missing", "conflict", "null")
            if use_coverage:
                kind_tag, cov_hops, mem_on, qry_on = COVERAGE_PATTERNS[flat_idx % len(COVERAGE_PATTERNS)]
                target_kind = "route" if kind_tag == "route" else ("memory" if kind_tag == "memory" else kind_tag)
                hops_ov = cov_hops if target_kind in ("route", "memory") else None

            chosen_case = None
            exs = ()
            for retry in range(128):
                fresh_case, rej = _sample_fresh_case_of_kind(
                    generator_version=HITCHHIKER_STREAM_VERSION,
                    data_seed=data_seed,
                    update=u,
                    batch_slot=slot + retry * batch_size,
                    target_kind=target_kind,
                    forbidden_families=forbidden_families,
                    forbidden_worlds=seen_worlds,
                    forbidden_prompts=seen_prompts,
                    hops_override=hops_ov,
                )
                rejected += rej
                if use_coverage:
                    env = fresh_case.environment if qry_on else PublicEnvironment(fresh_case.environment.initial, {})
                    mem = fresh_case.memory if mem_on else None
                    if target_kind == "known" and mem_on:
                        unused = next(s for s in SYMBOLS if s not in fresh_case.goal.slots)
                        mem = MemoryVersion((ExperienceEvent(fresh_case.goal.reusable_scope, unused, "audit", 1, unused),))
                    policy = ControlPolicy("fixed", surface=fresh_case.surface, memory_available=mem is not None)
                    trace = run_episode(fresh_case.goal, env, policy, memory=mem)
                    exp_val = trace.answer if trace.status == "answered" else None
                    fresh_case = replace(
                        fresh_case,
                        environment=env,
                        memory=mem,
                        truth=Truth(fresh_case.case_id, exp_val, _admitted_ids(env, mem)),
                    )
                cand_case = fresh_case
                if use_relabel:
                    mapping, _ = entity_bijection((relabel_seed, u, slot, retry, cand_case.case_id))
                    cand_case = apply_case_permutation(cand_case, mapping)
                cand_exs, _, cand_fams = _case_supervision(cand_case)
                if (set(cand_fams) & forbidden_families) or any(e.prompt in forbidden_prompts for e in cand_exs):
                    rejected += 1
                    continue
                seen_worlds.add(fresh_case.world_id)
                chosen_case = cand_case
                exs = cand_exs
                break
            if chosen_case is None:
                raise RuntimeError("fresh stream could not sample a split-disjoint candidate after crossing/permutation")
        else:
            chosen_case = ref_case
            if use_relabel:
                mapping, _ = entity_bijection((relabel_seed, epoch, chosen_case.case_id))
                chosen_case = apply_case_permutation(chosen_case, mapping)
            exs, _, _ = _case_supervision(chosen_case)

        matching = [e for e in exs if e.stage == ref_ex.stage]
        chosen_ex = matching[0] if matching else exs[0]
        frame = UTF8.frame(chosen_ex.prompt, chosen_ex.target, max_length=MAX_SEQUENCE)
        tid = f"hh-{arm}-u{u}-s{slot}-{chosen_ex.task_id}"
        stream_examples.append(replace(chosen_ex, task_id=tid))
        stream_samples.append(
            TokenSample(
                tid,
                frame.ids,
                frame.loss_mask,
                chosen_case.world_id,
                chosen_case.kind,
            )
        )
        accepted += 1

    elapsed = time.perf_counter() - t0
    return {
        "arm": arm,
        **contract_meta,
        "generator_version": HITCHHIKER_STREAM_VERSION,
        "data_seed": data_seed,
        "order_seed": order_seed,
        "relabel_seed": relabel_seed,
        "samples": tuple(stream_samples),
        "examples": tuple(stream_examples),
        "cases": (),
        "use_static_indexer": False,
        "stream_metrics": {
            "accepted_candidates": accepted,
            "rejected_candidates": rejected,
            "pool_size": len(stream_samples),
            "requested_examples": total_updates * batch_size,
            "total_updates": total_updates,
            "batch_size": batch_size,
            "generation_seconds": elapsed,
        },
    }


def validate_stream_contract(
    stream: dict[str, Any],
    *,
    expected_arm: str,
    total_updates: int,
    batch_size: int,
) -> dict[str, Any]:
    """Validate stream contract keys, protocol binding, and static-vs-materialized sizing."""
    if not isinstance(stream, dict):
        raise ValueError("stream must be a dictionary")
    required = (
        "protocol_sha256",
        "arm",
        "total_updates",
        "batch_size",
        "sampling_method",
        "data_custody_verified",
        "use_static_indexer",
        "samples",
        "examples",
    )
    missing = [key for key in required if key not in stream]
    if missing:
        raise ValueError(f"stream missing required contract keys: {missing}")
    if stream["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("stream protocol_sha256 does not match current PROTOCOL_SHA")
    if stream["arm"] != expected_arm:
        raise ValueError(f"stream arm {stream['arm']!r} != expected {expected_arm!r}")
    if type(total_updates) is not int or total_updates < 1 or stream["total_updates"] != total_updates:
        raise ValueError("stream total_updates contract mismatch")
    if type(batch_size) is not int or batch_size < 1 or stream["batch_size"] != batch_size:
        raise ValueError("stream batch_size contract mismatch")
    if stream["sampling_method"] != "blocked":
        raise ValueError(f"stream sampling_method must be 'blocked', got {stream['sampling_method']!r}")
    if type(stream["data_custody_verified"]) is not bool:
        raise ValueError("stream data_custody_verified must be bool")
    if type(stream["use_static_indexer"]) is not bool:
        raise ValueError("stream use_static_indexer must be bool")
    samples = stream["samples"]
    examples = stream["examples"]
    if not samples or len(examples) != len(samples):
        raise ValueError("stream requires non-empty aligned samples and examples")
    if stream["use_static_indexer"]:
        if len(samples) <= 0:
            raise ValueError("static stream pool must be positive")
    else:
        expected_len = total_updates * batch_size
        if len(samples) != expected_len:
            raise ValueError(
                f"non-static materialized stream length {len(samples)} != {expected_len}"
            )
    return {
        "status": "PASS",
        "arm": expected_arm,
        "protocol_sha256": PROTOCOL_SHA,
        "total_updates": total_updates,
        "batch_size": batch_size,
        "sampling_method": stream["sampling_method"],
        "use_static_indexer": stream["use_static_indexer"],
        "pool_size": len(samples),
        "data_custody_verified": stream["data_custody_verified"],
    }


def _target_to_operation(target: str) -> str:
    if target in SYMBOLS:
        return "ANSWER"
    if target == "Q":
        return "QUERY"
    if target == "!":
        return "ABSTAIN"
    if target == "R":
        return "RETRIEVE"
    if target.startswith("~"):
        return "DELIBERATE"
    raise ValueError(f"unsupported Hitchhiker target action: {target!r}")


def arm_comparison_table(
    streams_or_entries: dict[str, dict[str, Any]] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-arm stream statistics, target marginals, and deltas against B0."""
    entries = (
        list(streams_or_entries.values())
        if isinstance(streams_or_entries, dict)
        else list(streams_or_entries)
    )
    rows: list[dict[str, Any]] = []
    for entry in entries:
        arm = entry["arm"]
        samples = entry["samples"]
        examples = entry["examples"]
        batch_size = int(entry["batch_size"])
        updates = int(entry["total_updates"])
        use_static = bool(entry["use_static_indexer"])
        sampling_method = entry.get("sampling_method", "blocked")
        order_seed = int(entry.get("order_seed", 7))

        if use_static:
            per_epoch = math.ceil(len(samples) / batch_size)
            epoch_orders: dict[int, tuple[int, ...]] = {}
            served_indices: list[int] = []
            for u in range(updates):
                epoch, batch_idx = divmod(u, per_epoch)
                if epoch not in epoch_orders:
                    epoch_orders[epoch] = ordered_indices(samples, order_seed, epoch, sampling_method)
                start = batch_idx * batch_size
                chosen = epoch_orders[epoch][start : start + batch_size]
                while len(chosen) < batch_size:
                    chosen = (*chosen, *epoch_orders[epoch][: batch_size - len(chosen)])
                served_indices.extend(chosen)
        else:
            served_indices = list(range(updates * batch_size))

        unique_worlds = len({samples[i].world_id for i in served_indices})
        unique_samples = len({(samples[i].tokens, samples[i].eligible) for i in served_indices})
        total_served = len(served_indices)
        mean_repetitions = (total_served / unique_samples) if unique_samples else 0.0
        supervised_tokens = sum(sum(1 for flag in samples[i].eligible if flag) for i in served_indices)
        input_positions = sum(len(samples[i].tokens) for i in served_indices)

        kind_counts = Counter(examples[i].kind for i in served_indices)
        op_counts = Counter(_target_to_operation(examples[i].target) for i in served_indices)
        ans_counts = Counter(examples[i].target for i in served_indices if examples[i].target in SYMBOLS)
        stage_counts = Counter(examples[i].stage for i in served_indices)

        data_mode = (
            ALL_ARMS[arm]["data_mode"]
            if arm in ALL_ARMS
            else ("static" if use_static else "fresh")
        )
        coverage_recipe = entry.get(
            "coverage_recipe",
            "14-stratum-coverage" if arm in ("C", "AB+C", "C+penalty") else "base-task-distribution",
        )
        rows.append(
            {
                "arm": arm,
                "data_mode": data_mode,
                "use_static_indexer": use_static,
                "sampling_method": sampling_method,
                "coverage_recipe": coverage_recipe,
                "unique_worlds": unique_worlds,
                "unique_samples": unique_samples,
                "total_examples_served": total_served,
                "mean_repetitions": mean_repetitions,
                "updates": updates,
                "batch_size": batch_size,
                "supervised_tokens": supervised_tokens,
                "input_positions": input_positions,
                "task_kind_marginals": dict(sorted(kind_counts.items())),
                "operation_marginals": dict(sorted(op_counts.items())),
                "answer_marginals": dict(sorted(ans_counts.items())),
                "stage_marginals": dict(sorted(stage_counts.items())),
                "deltas_vs_b0": None,
            }
        )

    b0_row = next((r for r in rows if r["arm"] == "B0"), None)
    if b0_row is not None:
        for row in rows:
            row["deltas_vs_b0"] = {
                "unique_worlds": row["unique_worlds"] - b0_row["unique_worlds"],
                "unique_samples": row["unique_samples"] - b0_row["unique_samples"],
                "total_examples_served": row["total_examples_served"] - b0_row["total_examples_served"],
                "mean_repetitions": row["mean_repetitions"] - b0_row["mean_repetitions"],
                "supervised_tokens": row["supervised_tokens"] - b0_row["supervised_tokens"],
                "input_positions": row["input_positions"] - b0_row["input_positions"],
                "task_kind_marginals": {
                    k: row["task_kind_marginals"].get(k, 0) - b0_row["task_kind_marginals"].get(k, 0)
                    for k in sorted(set(row["task_kind_marginals"]) | set(b0_row["task_kind_marginals"]))
                },
                "operation_marginals": {
                    k: row["operation_marginals"].get(k, 0) - b0_row["operation_marginals"].get(k, 0)
                    for k in sorted(set(row["operation_marginals"]) | set(b0_row["operation_marginals"]))
                },
            }
    return rows


class ExternalSlotCanonicalizer:
    """Public-only episode slot canonicalizer and output dereferencer (D-external).

    Assigns canonical slot labels over A-P using ONLY public state mentions
    (first-mention order or seeded random assignment), never accessing Truth,
    hidden channels, or unretrieved memory. Dereferences model outputs via pi^-1.
    """

    origin = "engineered_canonicalizer"

    def __init__(self, *, assignment: str = "first-mention", slot_seed: int = 17, strip_aliases: bool = True) -> None:
        if assignment not in ("first-mention", "random"):
            raise ValueError("slot assignment must be 'first-mention' or 'random'")
        self.assignment = assignment
        self.slot_seed = slot_seed
        self.strip_aliases = strip_aliases

    def build_mapping(self, state: CognitiveState, surface: Any) -> tuple[dict[str, str], dict[str, str]]:
        """Derive episode-consistent entity->slot bijection strictly from public state."""
        mentions: list[str] = [state.goal.subject]
        events = list(state.events)
        if surface.order == "reverse":
            events.reverse()
        for event in events:
            if event.subject not in mentions:
                mentions.append(event.subject)
            if event.value is not None and event.value not in mentions:
                mentions.append(event.value)
        remaining = [s for s in SYMBOLS if s not in mentions]
        ordered = [*mentions, *remaining]
        if self.assignment == "first-mention":
            slots = list(SYMBOLS)
        else:
            rng = random.Random(int(digest([HITCHHIKER_DATA_VERSION, "ext-slot", self.slot_seed, state.goal.episode]), 16))
            slots = list(SYMBOLS)
            rng.shuffle(slots)
        mapping = dict(zip(ordered, slots))
        inverse = {v: k for k, v in mapping.items()}
        return mapping, inverse

    def extend_mapping(
        self,
        mapping: dict[str, str],
        inverse: dict[str, str],
        state: CognitiveState,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Preserve existing public slot bindings across observations in one episode."""
        # Mapping already spans A-P bijectively; return copies to maintain immutability.
        return dict(mapping), dict(inverse)

    def canonicalize_public(
        self,
        state: CognitiveState,
        surface: Any,
        mapping: dict[str, str],
    ) -> tuple[CognitiveState, Any, dict[str, int]]:
        """Transform only public state and public surface into canonical slots."""
        mapped_events = tuple(
            ExperienceEvent(
                e.scope,
                mapping[e.subject],
                e.source,
                e.tick,
                _map_symbol(e.value, mapping),
                e.parents,
            )
            for e in state.events
        )
        mapped_goal = replace(
            state.goal,
            subject=mapping[state.goal.subject],
            slots=tuple(mapping[s] for s in state.goal.slots),
        )
        mapped_hyp = _map_symbol(state.hypothesis, mapping)
        mapped_state = replace(state, goal=mapped_goal, events=mapped_events, hypothesis=mapped_hyp)
        mapped_surface = (
            replace(surface, aliases="") if self.strip_aliases else surface
        )
        overhead = {
            "dictionary_entries": len({state.goal.subject, *(e.subject for e in state.events), *(e.value for e in state.events if e.value)}),
            "canonicalized_mentions": 1 + sum(1 + int(e.value is not None) for e in state.events),
        }
        return mapped_state, mapped_surface, overhead

    def dereference_output(self, operation: str, argument: str | None, inverse: dict[str, str]) -> tuple[str, str | None]:
        """Map canonical slot output back to the surface entity symbol."""
        if operation in ("ANSWER", "DELIBERATE"):
            if argument not in inverse:
                return "INVALID", None
            return operation, inverse[argument]
        return operation, argument
