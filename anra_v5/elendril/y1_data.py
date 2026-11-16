"""AN-RA V8-Y1 counterfactual world generator, evaluator-only truth, and epoch streams.

Public actor prompts never carry Truth, hidden channels, world IDs, or seeds.
All worlds satisfy the S3 512-token context bound (including target budget).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import math
import random
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
)
from .foundation_training import TokenSample
from .hitchhiker_data import (
    HitchhikerSurface,
    entity_bijection,
    family_split_key,
    render_hitchhiker_public,
)
from .representation import UTF8
from .v6_data import (
    CHANNEL,
    SYMBOLS,
    V6Case,
    V6Dataset,
    V6Example,
    _confirmation_case,
    _case as _v6_case,
    action_text,
    make_dataset as make_v6_dataset,
    public_action,
)
from .y1_protocol import (
    GENERATOR_SEEDS,
    HOLDOUT_SERIALIZATION_COMBINATION,
    NON_ROUTE_REHEARSAL_COUNTS,
    PROMPT_TOKEN_BOUND,
    RESERVED_TARGET_TOKENS,
    ROUTE_COVERAGE_COUNTS_PER_HOP,
    SPLIT_QUOTAS,
    T0_SPEC,
    TRAIN_HYPERPARAMS,
    derive_seed_int,
)


Y1_DATA_VERSION = "anra-v8-y1-data-v1"
PRIMARY_SOURCES = ("src0", "src1", "log0", "map0")  # Fixed 4-byte source labels.
RENAMED_SOURCES = ("alt0", "alt1", "db_0", "net0")  # Fixed 4-byte length-preserving rename.
SOURCE_RENAME_MAP = dict(zip(PRIMARY_SOURCES, RENAMED_SOURCES))
TIMESTAMP_MIN = 1000
TIMESTAMP_MAX = 8499
TIMESTAMP_SHIFT = 100  # Shifts 4-digit ticks within [1100..8599], preserving length and order.


@dataclass(frozen=True, slots=True)
class Y1Case:
    """Single Y1 evaluation or diagnostic case with public surface + evaluator-only metadata."""

    case_id: str
    world_id: str
    family_id: str
    split: str
    panel: str
    variant_index: int
    factor: str
    hops: int
    solvability: str  # "answerable", "query_needed", "unavailable"
    applicable: bool
    latent_terminal_pos: int | None
    visible_terminal_pos: int | None
    tracked_filler_pos: int | None
    queried_start_pos: int | None
    v6_case: V6Case
    public_prompt: str
    gold_operation: str
    gold_argument: str | None
    gold_target: str
    token_length: int
    reference_case_id: str | None = None

    @property
    def family(self) -> str:
        return self.v6_case.family

    @property
    def kind(self) -> str:
        return self.v6_case.kind

    @property
    def goal(self) -> Goal:
        return self.v6_case.goal

    @property
    def environment(self) -> PublicEnvironment:
        return self.v6_case.environment

    @property
    def truth(self) -> Truth:
        return self.v6_case.truth

    @property
    def surface(self) -> Any:
        return self.v6_case.surface

    @property
    def memory(self) -> MemoryVersion | None:
        return self.v6_case.memory

    @property
    def pair_id(self) -> str:
        return self.v6_case.pair_id


def _encode_prompt_len(prompt: str) -> int:
    return len(UTF8.encode(prompt)) + 2  # + BOS + SpecialToken.ANSWER


def check_prompt_budget(prompt: str, target: str = "~X") -> tuple[bool, int]:
    """Enforce S3 512-token bound including BOS, ANSWER, target, and EOS (no silent truncation)."""
    prompt_len = _encode_prompt_len(prompt)
    target_len = max(RESERVED_TARGET_TOKENS, len(UTF8.encode(target)) + 1)
    total = prompt_len + target_len
    return total <= PROMPT_TOKEN_BOUND, prompt_len


def independent_traversal_oracle(
    events: tuple[ExperienceEvent, ...],
    subject: str,
    hops: int,
    *,
    channels: dict[str, tuple[ExperienceEvent, ...]] | None = None,
    memory: MemoryVersion | None = None,
    goal_family: str = "route",
) -> tuple[str, str | None]:
    """Independent gold traversal & legality oracle (cross-checks generator truth)."""
    steps = hops if goal_family == "route" else 1
    working = list(events)
    avail_channels = dict(channels or {})
    avail_mem = memory

    for _pass in range(3):
        cur = subject
        blocked_status: str | None = None
        for _ in range(steps):
            b = belief(tuple(working), cur)
            if b.status == "null":
                return "ABSTAIN", None
            if b.status != "derived":
                blocked_status = b.status
                break
            cur = str(b.value)
        if blocked_status is None:
            return "ANSWER", cur
        if avail_mem is not None:
            working.extend(avail_mem.events)
            avail_mem = None
            continue
        if CHANNEL in avail_channels:
            working.extend(avail_channels.pop(CHANNEL))
            continue
        return "ABSTAIN", None
    return "ABSTAIN", None


def instantiated_state_key(
    events: tuple[ExperienceEvent, ...],
    subject: str,
    family: str,
    channels_present: bool,
    memory_present: bool,
) -> str:
    """Exact instantiated semantic fact/query/availability identity for split disjointness."""
    norm = sorted((e.subject, e.source, e.tick, e.value or "<NULL>") for e in events)
    return digest([Y1_DATA_VERSION, "instantiated-state", family, subject, norm, channels_present, memory_present])


def _is_holdout_combo(
    *,
    hops: int,
    terminal_pos_0based: int | None,
    distractor_count: int,
    queried_start_pos_0based: int | None,
) -> bool:
    return (
        hops == HOLDOUT_SERIALIZATION_COMBINATION["hop"]
        and distractor_count == HOLDOUT_SERIALIZATION_COMBINATION["distractor_count"]
        and terminal_pos_0based is not None
        and queried_start_pos_0based is not None
        and (terminal_pos_0based + 1) == HOLDOUT_SERIALIZATION_COMBINATION["terminal_edge_row_1based"]
        and (queried_start_pos_0based + 1) == HOLDOUT_SERIALIZATION_COMBINATION["queried_start_edge_row_1based"]
    )


def _positions_in_rows(
    rows: list[tuple[str, str, int, str | None]],
    *,
    queried_subject: str,
    terminal_subject: str | None,
    filler_subject: str | None = None,
) -> tuple[int | None, int | None, int | None]:
    q_pos = next((i for i, r in enumerate(rows) if r[0] == queried_subject), None)
    t_pos = None if terminal_subject is None else next((i for i, r in enumerate(rows) if r[0] == terminal_subject), None)
    f_pos = None if filler_subject is None else next((i for i, r in enumerate(rows) if r[0] == filler_subject), None)
    return q_pos, t_pos, f_pos


def _render_from_rows(
    *,
    case_id: str,
    world_id: str,
    split: str,
    panel: str,
    variant_index: int,
    factor: str,
    hops: int,
    solvability: str,
    applicable: bool,
    queried_subject: str,
    rows: list[tuple[str, str, int, str | None]],
    hidden_channel_rows: list[tuple[str, str, int, str | None]],
    terminal_subject: str | None,
    filler_subject: str | None,
    gold_val: str | None,
    pair_id: str = "",
    reference_case_id: str | None = None,
    latent_terminal_pos: int | None = None,
) -> Y1Case:
    scope = f"y1-{case_id}"
    reusable = f"y1mem-{case_id}"
    initial = tuple(ExperienceEvent(scope, s, src, t, v) for s, src, t, v in rows)
    channels = (
        {CHANNEL: tuple(ExperienceEvent(scope, s, src, t, v) for s, src, t, v in hidden_channel_rows)}
        if hidden_channel_rows
        else {}
    )
    slots = tuple(dict.fromkeys([queried_subject, *(s for s, _, _, _ in rows), *(s for s, _, _, _ in hidden_channel_rows)]))
    goal = Goal(case_id, "route", queried_subject, scope, reusable, slots)
    all_ev = (*initial, *(e for p in channels.values() for e in p))
    truth = Truth(case_id, gold_val, tuple(e.identity for e in all_ev))
    surface = HitchhikerSurface(template=0, aliases="", hops=hops, scratch=False, order="original")
    kind = "route" if solvability == "answerable" else ("missing" if solvability == "query_needed" else "unavailable")
    v6_case = V6Case(
        case_id=case_id,
        world_id=world_id,
        family="route",
        kind=kind,
        goal=goal,
        environment=PublicEnvironment(initial, channels),
        truth=truth,
        surface=surface,
        memory=None,
        pair_id=pair_id,
        panel=panel,
    )
    oracle_op, oracle_arg = independent_traversal_oracle(initial, queried_subject, hops, channels=channels)
    expected_final = ("ANSWER", gold_val) if gold_val is not None else ("ABSTAIN", None)
    if (oracle_op, oracle_arg) != expected_final:
        raise ValueError(f"oracle disagreement on {case_id}: {(oracle_op, oracle_arg)} != {expected_final}")

    state = CognitiveState(goal).observe(initial)
    ch_names = tuple(channels)
    prompt = render_hitchhiker_public(state, ch_names, surface, False)
    pub_op, pub_arg = public_action(state, ch_names, surface, False)
    pub_target = action_text(pub_op, pub_arg)
    ok_len, prompt_len = check_prompt_budget(prompt, pub_target)
    if not ok_len:
        raise ValueError(f"prompt exceeds {PROMPT_TOKEN_BOUND}-token bound ({prompt_len})")

    q_pos, vis_t_pos, f_pos = _positions_in_rows(
        rows,
        queried_subject=queried_subject,
        terminal_subject=terminal_subject if solvability == "answerable" else None,
        filler_subject=filler_subject,
    )
    return Y1Case(
        case_id=case_id,
        world_id=world_id,
        family_id=world_id,
        split=split,
        panel=panel,
        variant_index=variant_index,
        factor=factor,
        hops=hops,
        solvability=solvability,
        applicable=applicable,
        latent_terminal_pos=latent_terminal_pos if solvability != "answerable" else vis_t_pos,
        visible_terminal_pos=vis_t_pos,
        tracked_filler_pos=f_pos,
        queried_start_pos=q_pos,
        v6_case=v6_case,
        public_prompt=prompt,
        gold_operation=pub_op,
        gold_argument=pub_arg,
        gold_target=pub_target,
        token_length=prompt_len,
        reference_case_id=reference_case_id,
    )


def _construct_base_graph(
    rng: random.Random,
    *,
    hops: int,
    gold_letter: str,
    solvability: str,
    target_row_pos: int,
    forbid_holdout_combo: bool = False,
    force_holdout_combo: bool = False,
) -> dict[str, Any]:
    """Construct an h+2 row functional chain graph with 2 non-chain distractors and balanced positions."""
    if not 1 <= hops <= 4 or not 0 <= target_row_pos < hops + 2:
        raise ValueError("invalid hop or target row position")
    # Distinct chain nodes: h+1 symbols ending in fixed gold_letter (with unique terminal value).
    other_syms = [s for s in SYMBOLS if s != gold_letter]
    chain_nodes = rng.sample(other_syms, hops) + [gold_letter]
    remaining = [s for s in SYMBOLS if s not in chain_nodes]
    # Two distinct non-chain distractor source keys + objects disjoint from queried start/chain subjects.
    d1_sub, d2_sub = rng.sample(remaining, 2)
    pool_vals = [s for s in SYMBOLS if s not in (d1_sub, d2_sub, gold_letter)]
    d1_val, d2_val = rng.sample(pool_vals, 2)
    # Filler symbol (distinct from all subjects) for incomplete replacement of hidden terminal/bridge edge.
    filler_candidates = [s for s in remaining if s not in (d1_sub, d2_sub)]
    filler_sub = rng.choice(filler_candidates)
    filler_val = rng.choice([s for s in SYMBOLS if s != filler_sub and s != gold_letter])

    ticks = sorted(rng.sample(range(TIMESTAMP_MIN, TIMESTAMP_MAX + 1), hops + 3))
    srcs = [rng.choice(PRIMARY_SOURCES) for _ in range(hops + 3)]

    chain_edges = [
        (chain_nodes[i], srcs[i], ticks[i], chain_nodes[i + 1])
        for i in range(hops)
    ]
    distractor_edges = [
        (d1_sub, srcs[hops], ticks[hops], d1_val),
        (d2_sub, srcs[hops + 1], ticks[hops + 1], d2_val),
    ]
    filler_edge = (filler_sub, srcs[hops + 2], ticks[hops + 2], filler_val)

    hidden_edge: tuple[str, str, int, str | None] | None = None
    terminal_edge = chain_edges[-1]
    other_edges = list(chain_edges[:-1]) + list(distractor_edges)

    # Place target_edge (terminal edge if answerable, or filler edge if incomplete) at exact target_row_pos.
    rng.shuffle(other_edges)
    if solvability == "answerable":
        lead_edge = terminal_edge
        hidden_rows: list[tuple[str, str, int, str | None]] = []
    else:
        lead_edge = filler_edge
        hidden_rows = [terminal_edge] if solvability == "query_needed" else []
        hidden_edge = terminal_edge

    # Place non-lead edges into remaining slots; enforce holdout inclusion/exclusion at hop==2.
    rows: list[tuple[str, str, int, str | None]] = [None] * (hops + 2)  # type: ignore[list-item]
    rows[target_row_pos] = lead_edge
    open_slots = [i for i in range(hops + 2) if i != target_row_pos]
    for idx, edge in zip(open_slots, other_edges):
        rows[idx] = edge

    if hops == 2 and solvability == "answerable":
        start_sub = chain_nodes[0]
        term_sub = chain_nodes[-2]
        q_pos = next(i for i, r in enumerate(rows) if r[0] == start_sub)
        t_pos = next(i for i, r in enumerate(rows) if r[0] == term_sub)
        if force_holdout_combo:
            # Put terminal edge at row 0 (1-based 1) and start edge at row 2 (1-based 3).
            swap_rows = list(other_edges) + [lead_edge]
            term_e = next(e for e in swap_rows if e[0] == term_sub)
            start_e = next(e for e in swap_rows if e[0] == start_sub)
            rem_e = [e for e in swap_rows if e not in (term_e, start_e)]
            rows = [term_e, rem_e[0], start_e, rem_e[1]]
        elif forbid_holdout_combo and (t_pos + 1 == 1 and q_pos + 1 == 3):
            # Swap queried start edge (pos 2) with another non-terminal slot (pos 1 or 3).
            alt_pos = 1 if t_pos != 1 else 3
            rows[q_pos], rows[alt_pos] = rows[alt_pos], rows[q_pos]

    return {
        "hops": hops,
        "solvability": solvability,
        "queried_subject": chain_nodes[0],
        "terminal_subject": chain_nodes[-2],
        "gold_letter": gold_letter,
        "filler_subject": None if solvability == "answerable" else filler_sub,
        "rows": rows,
        "hidden_channel_rows": hidden_rows,
        "hidden_edge": hidden_edge,
        "distractors": distractor_edges,
        "target_row_pos": target_row_pos,
    }


def _find_moved_row_order(rows: list[tuple[str, str, int, str | None]], move_subject: str, rng: random.Random) -> list[tuple[str, str, int, str | None]]:
    orig_pos = next(i for i, r in enumerate(rows) if r[0] == move_subject)
    other_positions = [i for i in range(len(rows)) if i != orig_pos]
    new_pos = rng.choice(other_positions)
    others = [r for i, r in enumerate(rows) if i != orig_pos]
    rng.shuffle(others)
    moved = [None] * len(rows)
    moved[new_pos] = rows[orig_pos]
    for slot, item in zip([i for i in range(len(rows)) if i != new_pos], others):
        moved[slot] = item
    return moved  # type: ignore[return-value]


def _apply_entity_rename(
    graph: dict[str, Any],
    seed_material: Any,
) -> tuple[str, str | None, str | None, str, list[tuple[str, str, int, str | None]], list[tuple[str, str, int, str | None]]]:
    mapping, _ = entity_bijection(seed_material)
    map_row = lambda r: (mapping[r[0]], r[1], r[2], None if r[3] is None else mapping[r[3]])
    return (
        mapping[graph["queried_subject"]],
        mapping[graph["terminal_subject"]],
        None if graph["filler_subject"] is None else mapping[graph["filler_subject"]],
        mapping[graph["gold_letter"]],
        [map_row(r) for r in graph["rows"]],
        [map_row(r) for r in graph["hidden_channel_rows"]],
    )


def build_base_family_variants(
    graph: dict[str, Any],
    *,
    world_id: str,
    split: str,
    seed_base: int,
) -> list[Y1Case]:
    """Build exact Cases 0..7 for one instantiated Y1 base family."""
    hops = graph["hops"]
    solvability = graph["solvability"]
    q_sub = graph["queried_subject"]
    t_sub = graph["terminal_subject"]
    f_sub = graph["filler_subject"]
    gold = None if solvability == "unavailable" else graph["gold_letter"]
    rows = list(graph["rows"])
    hidden = list(graph["hidden_channel_rows"])
    latent_pos = graph["target_row_pos"] if solvability != "answerable" else None
    cases: list[Y1Case] = []

    # Case 0: Base
    cases.append(
        _render_from_rows(
            case_id=f"{world_id}:v0",
            world_id=world_id,
            split=split,
            panel="main-base",
            variant_index=0,
            factor="base",
            hops=hops,
            solvability=solvability,
            applicable=True,
            queried_subject=q_sub,
            rows=rows,
            hidden_channel_rows=hidden,
            terminal_subject=t_sub,
            filler_subject=f_sub,
            gold_val=gold,
            latent_terminal_pos=latent_pos,
        )
    )

    # Case 1: Row permutation moving visible target edge (answerable -> applicable=True; incomplete -> moves filler, applicable=False)
    move_sub = t_sub if solvability == "answerable" else str(f_sub)
    rng1 = random.Random(derive_seed_int(Y1_DATA_VERSION, "case1", seed_base, world_id))
    rows1 = _find_moved_row_order(rows, move_sub, rng1)
    cases.append(
        _render_from_rows(
            case_id=f"{world_id}:v1",
            world_id=world_id,
            split=split,
            panel="main-row-perm",
            variant_index=1,
            factor="row-permutation",
            hops=hops,
            solvability=solvability,
            applicable=(solvability == "answerable"),
            queried_subject=q_sub,
            rows=rows1,
            hidden_channel_rows=hidden,
            terminal_subject=t_sub,
            filler_subject=f_sub,
            gold_val=gold,
            latent_terminal_pos=latent_pos,
        )
    )

    # Case 2: Bijective entity rename
    rq, rt, rf, rg, rrows, rhidden = _apply_entity_rename(graph, ("case2", seed_base, world_id))
    cases.append(
        _render_from_rows(
            case_id=f"{world_id}:v2",
            world_id=world_id,
            split=split,
            panel="main-entity-rename",
            variant_index=2,
            factor="entity-rename",
            hops=hops,
            solvability=solvability,
            applicable=True,
            queried_subject=rq,
            rows=rrows,
            hidden_channel_rows=rhidden,
            terminal_subject=rt,
            filler_subject=rf,
            gold_val=None if solvability == "unavailable" else rg,
            latent_terminal_pos=latent_pos,
        )
    )

    # Case 3: Replace irrelevant distractor at same serialized length and position
    d0 = graph["distractors"][0]
    used_vals = {r[0] for r in rows} | {r[3] for r in rows if r[3] is not None} | {graph["gold_letter"]}
    alt_pool = [s for s in SYMBOLS if s != d0[3] and s not in used_vals] or [s for s in SYMBOLS if s != d0[3]]
    new_d0 = (d0[0], d0[1], d0[2], alt_pool[0])
    rows3 = [new_d0 if r == d0 else r for r in rows]
    cases.append(
        _render_from_rows(
            case_id=f"{world_id}:v3",
            world_id=world_id,
            split=split,
            panel="main-distractor-replace",
            variant_index=3,
            factor="distractor-replace",
            hops=hops,
            solvability=solvability,
            applicable=True,
            queried_subject=q_sub,
            rows=rows3,
            hidden_channel_rows=hidden,
            terminal_subject=t_sub,
            filler_subject=f_sub,
            gold_val=gold,
            latent_terminal_pos=latent_pos,
        )
    )

    # Case 4: Rename source labels preserving byte length
    rows4 = [(s, SOURCE_RENAME_MAP[src], t, v) for s, src, t, v in rows]
    hidden4 = [(s, SOURCE_RENAME_MAP[src], t, v) for s, src, t, v in hidden]
    cases.append(
        _render_from_rows(
            case_id=f"{world_id}:v4",
            world_id=world_id,
            split=split,
            panel="main-source-rename",
            variant_index=4,
            factor="source-rename",
            hops=hops,
            solvability=solvability,
            applicable=True,
            queried_subject=q_sub,
            rows=rows4,
            hidden_channel_rows=hidden4,
            terminal_subject=t_sub,
            filler_subject=f_sub,
            gold_val=gold,
            latent_terminal_pos=latent_pos,
        )
    )

    # Case 5: Shift timestamps by fixed length-preserving +100 keeping order
    rows5 = [(s, src, t + TIMESTAMP_SHIFT, v) for s, src, t, v in rows]
    hidden5 = [(s, src, t + TIMESTAMP_SHIFT, v) for s, src, t, v in hidden]
    cases.append(
        _render_from_rows(
            case_id=f"{world_id}:v5",
            world_id=world_id,
            split=split,
            panel="main-timestamp-shift",
            variant_index=5,
            factor="timestamp-shift",
            hops=hops,
            solvability=solvability,
            applicable=True,
            queried_subject=q_sub,
            rows=rows5,
            hidden_channel_rows=hidden5,
            terminal_subject=t_sub,
            filler_subject=f_sub,
            gold_val=gold,
            latent_terminal_pos=latent_pos,
        )
    )

    # Case 6: Change terminal edge's value on answerable; identity/no-public-change control (applicable=False) on incomplete
    if solvability == "answerable":
        term_row = next(r for r in rows if r[0] == t_sub)
        alt_gold = next(s for s in SYMBOLS if s != graph["gold_letter"] and s != t_sub)
        rows6 = [(r[0], r[1], r[2], alt_gold) if r[0] == t_sub else r for r in rows]
        cases.append(
            _render_from_rows(
                case_id=f"{world_id}:v6",
                world_id=world_id,
                split=split,
                panel="main-terminal-value-change",
                variant_index=6,
                factor="terminal-value-change",
                hops=hops,
                solvability=solvability,
                applicable=True,
                queried_subject=q_sub,
                rows=rows6,
                hidden_channel_rows=hidden,
                terminal_subject=t_sub,
                filler_subject=f_sub,
                gold_val=alt_gold,
                pair_id=f"{world_id}:c6pair",
            )
        )
    else:
        cases.append(
            _render_from_rows(
                case_id=f"{world_id}:v6",
                world_id=world_id,
                split=split,
                panel="main-terminal-value-change",
                variant_index=6,
                factor="terminal-value-change-identity-control",
                hops=hops,
                solvability=solvability,
                applicable=False,
                queried_subject=q_sub,
                rows=rows,
                hidden_channel_rows=hidden,
                terminal_subject=t_sub,
                filler_subject=f_sub,
                gold_val=gold,
                latent_terminal_pos=latent_pos,
            )
        )

    # Case 7: Query a distractor source at depth 1 (choosing different gold) on answerable; identity control on incomplete
    if solvability == "answerable":
        chosen_d = next((d for d in graph["distractors"] if d[3] != graph["gold_letter"]), graph["distractors"][0])
        cases.append(
            _render_from_rows(
                case_id=f"{world_id}:v7",
                world_id=world_id,
                split=split,
                panel="main-distractor-query",
                variant_index=7,
                factor="distractor-query",
                hops=1,
                solvability=solvability,
                applicable=True,
                queried_subject=chosen_d[0],
                rows=rows,
                hidden_channel_rows=[],
                terminal_subject=chosen_d[0],
                filler_subject=None,
                gold_val=chosen_d[3],
                pair_id=f"{world_id}:c7pair",
            )
        )
    else:
        cases.append(
            _render_from_rows(
                case_id=f"{world_id}:v7",
                world_id=world_id,
                split=split,
                panel="main-distractor-query",
                variant_index=7,
                factor="distractor-query-identity-control",
                hops=hops,
                solvability=solvability,
                applicable=False,
                queried_subject=q_sub,
                rows=rows,
                hidden_channel_rows=hidden,
                terminal_subject=t_sub,
                filler_subject=f_sub,
                gold_val=gold,
                latent_terminal_pos=latent_pos,
            )
        )
    return cases


def _balanced_allocations(count: int, n_slots: int, rng_seed: Any) -> list[int]:
    base, rem = divmod(count, n_slots)
    counts = [base] * n_slots
    offset = derive_seed_int(rng_seed) % n_slots
    for k in range(rem):
        counts[(offset + k) % n_slots] += 1
    seq = [slot for slot, c in enumerate(counts) for _ in range(c)]
    rng = random.Random(derive_seed_int("alloc-perm", rng_seed))
    rng.shuffle(seq)
    return seq


def _build_split_suite(split: str, seed: int) -> dict[str, Any]:
    """Build exact 11,840-case (+ 80 historical continuity) split suite."""
    all_cases: list[Y1Case] = []
    cohorts: dict[str, list[Y1Case]] = defaultdict(list)
    seen_inst_keys: set[str] = set()
    seen_prompts: set[str] = set()
    over_context_rejected = 0

    # 1. 1,024 main base families (per hop: 128 answerable, 64 query_needed, 64 unavailable) -> 8,192 cases
    main_idx = 0
    for hops in SPLIT_QUOTAS["hops"]:
        n_rows = hops + 2
        for solvability, count in (("answerable", 128), ("query_needed", 64), ("unavailable", 64)):
            pos_plan = _balanced_allocations(count, n_rows, (split, seed, hops, solvability, "pos"))
            gold_plan = _balanced_allocations(count, len(SYMBOLS), (split, seed, hops, solvability, "gold"))
            for item_idx in range(count):
                attempt = 0
                while True:
                    rng = random.Random(derive_seed_int(Y1_DATA_VERSION, split, seed, "main", hops, solvability, item_idx, attempt))
                    attempt += 1
                    try:
                        graph = _construct_base_graph(
                            rng,
                            hops=hops,
                            gold_letter=SYMBOLS[gold_plan[item_idx]],
                            solvability=solvability,
                            target_row_pos=pos_plan[item_idx],
                        )
                        world_id = f"{split}-m{main_idx:04d}"
                        variants = build_base_family_variants(graph, world_id=world_id, split=split, seed_base=seed)
                    except ValueError:
                        over_context_rejected += 1
                        continue
                    v0 = variants[0]
                    ikey = instantiated_state_key(v0.environment.initial, v0.goal.subject, v0.family, bool(v0.environment.channels), False)
                    prompts = [c.public_prompt for c in variants]
                    if ikey in seen_inst_keys or any(p in seen_prompts for p in prompts):
                        continue
                    seen_inst_keys.add(ikey)
                    seen_prompts.update(prompts)
                    main_idx += 1
                    all_cases.extend(variants)
                    cohorts["main"].extend(variants)
                    for vc in variants:
                        cohorts[vc.panel].append(vc)
                    break

    # 2. 256 edge-change-panel families (64/hop): orig_ref + fixed_pos + moved_pos = 768 cases
    for hops in SPLIT_QUOTAS["hops"]:
        n_rows = hops + 2
        pos_plan = _balanced_allocations(64, n_rows, (split, seed, hops, "edge_panel", "pos"))
        gold_plan = _balanced_allocations(64, len(SYMBOLS), (split, seed, hops, "edge_panel", "gold"))
        for item_idx in range(64):
            attempt = 0
            while True:
                rng = random.Random(derive_seed_int(Y1_DATA_VERSION, split, seed, "edge-panel", hops, item_idx, attempt))
                attempt += 1
                try:
                    graph = _construct_base_graph(
                        rng,
                        hops=hops,
                        gold_letter=SYMBOLS[gold_plan[item_idx]],
                        solvability="answerable",
                        target_row_pos=pos_plan[item_idx],
                    )
                    fid = f"{split}-ep{hops}-{item_idx:03d}"
                    rows = list(graph["rows"])
                    t_sub = graph["terminal_subject"]
                    q_sub = graph["queried_subject"]
                    alt_gold = next(s for s in SYMBOLS if s != graph["gold_letter"] and s != t_sub and s != q_sub)
                    fixed_rows = [(r[0], r[1], r[2], alt_gold) if r[0] == t_sub else r for r in rows]
                    moved_rows = _find_moved_row_order(fixed_rows, t_sub, random.Random(derive_seed_int("ep-move", seed, fid)))

                    ref_c = _render_from_rows(
                        case_id=f"{fid}:orig",
                        world_id=fid,
                        split=split,
                        panel="edge-change-panel",
                        variant_index=0,
                        factor="orig_ref",
                        hops=hops,
                        solvability="answerable",
                        applicable=True,
                        queried_subject=q_sub,
                        rows=rows,
                        hidden_channel_rows=[],
                        terminal_subject=t_sub,
                        filler_subject=None,
                        gold_val=graph["gold_letter"],
                        pair_id=f"{fid}:pair",
                    )
                    fix_c = _render_from_rows(
                        case_id=f"{fid}:fixed",
                        world_id=fid,
                        split=split,
                        panel="edge-change-panel",
                        variant_index=1,
                        factor="fixed_pos",
                        hops=hops,
                        solvability="answerable",
                        applicable=True,
                        queried_subject=q_sub,
                        rows=fixed_rows,
                        hidden_channel_rows=[],
                        terminal_subject=t_sub,
                        filler_subject=None,
                        gold_val=alt_gold,
                        pair_id=f"{fid}:pair",
                        reference_case_id=ref_c.case_id,
                    )
                    mov_c = _render_from_rows(
                        case_id=f"{fid}:moved",
                        world_id=fid,
                        split=split,
                        panel="edge-change-panel",
                        variant_index=2,
                        factor="moved_pos",
                        hops=hops,
                        solvability="answerable",
                        applicable=True,
                        queried_subject=q_sub,
                        rows=moved_rows,
                        hidden_channel_rows=[],
                        terminal_subject=t_sub,
                        filler_subject=None,
                        gold_val=alt_gold,
                        pair_id=f"{fid}:pair",
                        reference_case_id=ref_c.case_id,
                    )
                except ValueError:
                    over_context_rejected += 1
                    continue
                prompts = [ref_c.public_prompt, fix_c.public_prompt, mov_c.public_prompt]
                if any(p in seen_prompts for p in prompts):
                    continue
                seen_prompts.update(prompts)
                all_cases.extend((ref_c, fix_c, mov_c))
                cohorts["edge-change-panel"].extend((ref_c, fix_c, mov_c))
                break

    # 3. 256 missing-bridge-panel families (64/hop): query available vs unavailable = 512 cases
    for hops in SPLIT_QUOTAS["hops"]:
        n_rows = hops + 2
        pos_plan = _balanced_allocations(64, n_rows, (split, seed, hops, "bridge_panel", "pos"))
        gold_plan = _balanced_allocations(64, len(SYMBOLS), (split, seed, hops, "bridge_panel", "gold"))
        for item_idx in range(64):
            attempt = 0
            while True:
                rng = random.Random(derive_seed_int(Y1_DATA_VERSION, split, seed, "bridge-panel", hops, item_idx, attempt))
                attempt += 1
                try:
                    graph = _construct_base_graph(
                        rng,
                        hops=hops,
                        gold_letter=SYMBOLS[gold_plan[item_idx]],
                        solvability="query_needed",
                        target_row_pos=pos_plan[item_idx],
                    )
                    fid = f"{split}-bp{hops}-{item_idx:03d}"
                    avail_c = _render_from_rows(
                        case_id=f"{fid}:avail",
                        world_id=fid,
                        split=split,
                        panel="missing-bridge-panel",
                        variant_index=0,
                        factor="query_available",
                        hops=hops,
                        solvability="query_needed",
                        applicable=True,
                        queried_subject=graph["queried_subject"],
                        rows=graph["rows"],
                        hidden_channel_rows=graph["hidden_channel_rows"],
                        terminal_subject=graph["terminal_subject"],
                        filler_subject=graph["filler_subject"],
                        gold_val=graph["gold_letter"],
                        pair_id=f"{fid}:pair",
                        latent_terminal_pos=pos_plan[item_idx],
                    )
                    unavail_c = _render_from_rows(
                        case_id=f"{fid}:unavail",
                        world_id=fid,
                        split=split,
                        panel="missing-bridge-panel",
                        variant_index=1,
                        factor="query_unavailable",
                        hops=hops,
                        solvability="unavailable",
                        applicable=True,
                        queried_subject=graph["queried_subject"],
                        rows=graph["rows"],
                        hidden_channel_rows=[],
                        terminal_subject=graph["terminal_subject"],
                        filler_subject=graph["filler_subject"],
                        gold_val=None,
                        pair_id=f"{fid}:pair",
                        latent_terminal_pos=pos_plan[item_idx],
                    )
                except ValueError:
                    over_context_rejected += 1
                    continue
                if avail_c.public_prompt in seen_prompts or unavail_c.public_prompt in seen_prompts:
                    continue
                seen_prompts.update((avail_c.public_prompt, unavail_c.public_prompt))
                all_cases.extend((avail_c, unavail_c))
                cohorts["missing-bridge-panel"].extend((avail_c, unavail_c))
                break

    # 4. 256 dual-chain fixed-depth query families (64/hop): two disjoint chains + 2 distractors = 512 cases
    for hops in SPLIT_QUOTAS["hops"]:
        gold_plan = _balanced_allocations(64, len(SYMBOLS), (split, seed, hops, "dual_query", "gold"))
        for item_idx in range(64):
            attempt = 0
            while True:
                rng = random.Random(derive_seed_int(Y1_DATA_VERSION, split, seed, "dual-query", hops, item_idx, attempt))
                attempt += 1
                gold_a = SYMBOLS[gold_plan[item_idx]]
                pool = [s for s in SYMBOLS if s != gold_a]
                rng.shuffle(pool)
                gold_b = pool[0]
                chain_a = pool[1 : 1 + hops] + [gold_a]
                chain_b = pool[1 + hops : 1 + 2 * hops] + [gold_b]
                rem = pool[1 + 2 * hops :]
                # 2 distractor edges using distinct unused subjects (at h=4: 5 + 5 = 10 symbols + 2 d_sub + 2 vals = 14!)
                d1_sub, d2_sub = rem[0], rem[1]
                val_pool = [s for s in SYMBOLS if s not in (d1_sub, d2_sub, gold_a, gold_b)]
                d1_val = rem[2] if len(rem) > 2 else val_pool[0]
                d2_val = rem[3] if len(rem) > 3 else val_pool[1]
                ticks = sorted(rng.sample(range(TIMESTAMP_MIN, TIMESTAMP_MAX + 1), 2 * hops + 2))
                rows = [
                    *((chain_a[i], rng.choice(PRIMARY_SOURCES), ticks[i], chain_a[i + 1]) for i in range(hops)),
                    *((chain_b[i], rng.choice(PRIMARY_SOURCES), ticks[hops + i], chain_b[i + 1]) for i in range(hops)),
                    (d1_sub, rng.choice(PRIMARY_SOURCES), ticks[2 * hops], d1_val),
                    (d2_sub, rng.choice(PRIMARY_SOURCES), ticks[2 * hops + 1], d2_val),
                ]
                rng.shuffle(rows)
                fid = f"{split}-dq{hops}-{item_idx:03d}"
                try:
                    c_a = _render_from_rows(
                        case_id=f"{fid}:A",
                        world_id=fid,
                        split=split,
                        panel="dual-chain-query-panel",
                        variant_index=0,
                        factor="query_chain_a",
                        hops=hops,
                        solvability="answerable",
                        applicable=True,
                        queried_subject=chain_a[0],
                        rows=rows,
                        hidden_channel_rows=[],
                        terminal_subject=chain_a[-2],
                        filler_subject=None,
                        gold_val=gold_a,
                        pair_id=f"{fid}:pair",
                    )
                    c_b = _render_from_rows(
                        case_id=f"{fid}:B",
                        world_id=fid,
                        split=split,
                        panel="dual-chain-query-panel",
                        variant_index=1,
                        factor="query_chain_b",
                        hops=hops,
                        solvability="answerable",
                        applicable=True,
                        queried_subject=chain_b[0],
                        rows=rows,
                        hidden_channel_rows=[],
                        terminal_subject=chain_b[-2],
                        filler_subject=None,
                        gold_val=gold_b,
                        pair_id=f"{fid}:pair",
                    )
                except ValueError:
                    over_context_rejected += 1
                    continue
                if c_a.public_prompt in seen_prompts or c_b.public_prompt in seen_prompts:
                    continue
                seen_prompts.update((c_a.public_prompt, c_b.public_prompt))
                all_cases.extend((c_a, c_b))
                cohorts["dual-chain-query-panel"].extend((c_a, c_b))
                break

    # 5. 64 serialization-holdout families (hop=2, terminal_row_1based=1, distractor=2, start_row_1based=3)
    gold_plan_sh = _balanced_allocations(64, len(SYMBOLS), (split, seed, "serialization-holdout", "gold"))
    for item_idx in range(64):
        attempt = 0
        while True:
            rng = random.Random(derive_seed_int(Y1_DATA_VERSION, split, seed, "serialization-holdout", item_idx, attempt))
            attempt += 1
            try:
                graph = _construct_base_graph(
                    rng,
                    hops=2,
                    gold_letter=SYMBOLS[gold_plan_sh[item_idx]],
                    solvability="answerable",
                    target_row_pos=0,
                    force_holdout_combo=True,
                )
                fid = f"{split}-sh-{item_idx:03d}"
                sh_case = _render_from_rows(
                    case_id=fid,
                    world_id=fid,
                    split=split,
                    panel="serialization-holdout",
                    variant_index=0,
                    factor="serialization_holdout",
                    hops=2,
                    solvability="answerable",
                    applicable=True,
                    queried_subject=graph["queried_subject"],
                    rows=graph["rows"],
                    hidden_channel_rows=[],
                    terminal_subject=graph["terminal_subject"],
                    filler_subject=None,
                    gold_val=graph["gold_letter"],
                )
            except ValueError:
                over_context_rejected += 1
                continue
            if not _is_holdout_combo(
                hops=2,
                terminal_pos_0based=sh_case.visible_terminal_pos,
                distractor_count=2,
                queried_start_pos_0based=sh_case.queried_start_pos,
            ):
                raise RuntimeError("serialization holdout case violated holdout invariant")
            if sh_case.public_prompt in seen_prompts:
                continue
            seen_prompts.add(sh_case.public_prompt)
            all_cases.append(sh_case)
            cohorts["serialization-holdout"].append(sh_case)
            break

    # 6. 1,792 preservation suite families (256 per each of 7 original V6 kinds, distinct family IDs & visible prompts)
    orig_kinds = ("conflict", "known", "memory", "missing", "null", "route", "unavailable")
    pres_counts = {k: 0 for k in orig_kinds}
    pres_seed = derive_seed_int(Y1_DATA_VERSION, "preservation", split, seed)
    for idx in range(500_000):
        raw_case = _confirmation_case(idx, pres_seed) if split == "confirmation" else _v6_case(idx, pres_seed, split)
        if pres_counts[raw_case.kind] >= 256:
            if all(v == 256 for v in pres_counts.values()):
                break
            continue
        fid = f"{split}-pres-{raw_case.kind}-{pres_counts[raw_case.kind]:03d}"
        wid = f"{split}-pw-{raw_case.world_id[:16]}"
        state = CognitiveState(raw_case.goal).observe(raw_case.environment.initial)
        ch = tuple(raw_case.environment.channels)
        mem = raw_case.memory is not None
        prompt = render_hitchhiker_public(state, ch, raw_case.surface, mem)
        pub_op, pub_arg = public_action(state, ch, raw_case.surface, mem)
        target = action_text(pub_op, pub_arg)
        ok_len, plen = check_prompt_budget(prompt, target)
        if not ok_len or prompt in seen_prompts:
            if not ok_len:
                over_context_rejected += 1
            continue
        seen_prompts.add(prompt)
        pres_counts[raw_case.kind] += 1
        c = Y1Case(
            case_id=fid,
            world_id=wid,
            family_id=wid,
            split=split,
            panel="preservation-suite",
            variant_index=0,
            factor=f"preservation-{raw_case.kind}",
            hops=raw_case.surface.hops,
            solvability=raw_case.kind,
            applicable=True,
            latent_terminal_pos=None,
            visible_terminal_pos=None,
            tracked_filler_pos=None,
            queried_start_pos=None,
            v6_case=replace(raw_case, case_id=fid, world_id=wid, panel="preservation-suite"),
            public_prompt=prompt,
            gold_operation=pub_op,
            gold_argument=pub_arg,
            gold_target=target,
            token_length=plen,
        )
        all_cases.append(c)
        cohorts["preservation-suite"].append(c)
    if not all(v == 256 for v in pres_counts.values()):
        raise RuntimeError(f"failed to generate exact preservation quotas: {pres_counts}")

    # 7. Separate 80 historical continuity cases (excluded from 11,840 primary count)
    hist_ds = make_v6_dataset("validation" if split != "confirmation" else "confirmation", worlds=80, seed=707 if split != "confirmation" else 909)
    hist_cases: list[Y1Case] = []
    for i, hc in enumerate(hist_ds.cases):
        st = CognitiveState(hc.goal).observe(hc.environment.initial)
        ch = tuple(hc.environment.channels)
        mem = hc.memory is not None
        pr = render_hitchhiker_public(st, ch, hc.surface, mem)
        op, arg = public_action(st, ch, hc.surface, mem)
        tgt = action_text(op, arg)
        _, plen = check_prompt_budget(pr, tgt)
        hist_cases.append(
            Y1Case(
                case_id=f"{split}-hist-{i:02d}",
                world_id=f"{split}-hw-{hc.world_id[:16]}",
                family_id=hc.world_id,
                split=split,
                panel="historical-continuity",
                variant_index=0,
                factor=f"historical-{hc.kind}",
                hops=hc.surface.hops,
                solvability=hc.kind,
                applicable=True,
                latent_terminal_pos=None,
                visible_terminal_pos=None,
                tracked_filler_pos=None,
                queried_start_pos=None,
                v6_case=replace(hc, case_id=f"{split}-hist-{i:02d}", panel="historical-continuity"),
                public_prompt=pr,
                gold_operation=op,
                gold_argument=arg,
                gold_target=tgt,
                token_length=plen,
            )
        )

    assert len(all_cases) == SPLIT_QUOTAS["total_budgeted_cases_per_split_model"], f"unexpected split count {len(all_cases)}"
    return {
        "schema": "anra-v8-y1-suite-v1",
        "split": split,
        "seed": seed,
        "cases": tuple(all_cases),
        "cohorts": {k: tuple(v) for k, v in cohorts.items()},
        "historical_continuity": tuple(hist_cases),
        "over_context_rejected": over_context_rejected,
        "suite_sha256": digest([c.case_id for c in all_cases]),
    }


def build_development_suite() -> dict[str, Any]:
    return _build_split_suite("development", GENERATOR_SEEDS["development"])


def build_confirmation_suite() -> dict[str, Any]:
    return _build_split_suite("confirmation", GENERATOR_SEEDS["confirmation"])


def build_followup_suite() -> dict[str, Any]:
    return _build_split_suite("capacity_followup", GENERATOR_SEEDS["capacity_followup"])


def build_d2_probe_dataset() -> dict[str, Any]:
    """Distinct disjoint 1,024-answerable-family D2 probe dataset (256/hop: 128 train + 128 test)."""
    seed = GENERATOR_SEEDS["probe_d2"]
    train_cases: list[Y1Case] = []
    test_cases: list[Y1Case] = []
    seen_prompts: set[str] = set()
    for hops in (1, 2, 3, 4):
        pos_plan = _balanced_allocations(256, hops + 2, ("d2", seed, hops, "pos"))
        gold_plan = _balanced_allocations(256, len(SYMBOLS), ("d2", seed, hops, "gold"))
        for idx in range(256):
            attempt = 0
            while True:
                rng = random.Random(derive_seed_int(Y1_DATA_VERSION, "d2-probe", seed, hops, idx, attempt))
                attempt += 1
                try:
                    g = _construct_base_graph(
                        rng,
                        hops=hops,
                        gold_letter=SYMBOLS[gold_plan[idx]],
                        solvability="answerable",
                        target_row_pos=pos_plan[idx],
                    )
                    fid = f"d2-h{hops}-{idx:03d}"
                    c = _render_from_rows(
                        case_id=fid,
                        world_id=fid,
                        split="probe_d2",
                        panel="d2-probe",
                        variant_index=0,
                        factor="probe",
                        hops=hops,
                        solvability="answerable",
                        applicable=True,
                        queried_subject=g["queried_subject"],
                        rows=g["rows"],
                        hidden_channel_rows=[],
                        terminal_subject=g["terminal_subject"],
                        filler_subject=None,
                        gold_val=g["gold_letter"],
                    )
                except ValueError:
                    continue
                if c.public_prompt in seen_prompts:
                    continue
                seen_prompts.add(c.public_prompt)
                (train_cases if idx < 128 else test_cases).append(c)
                break
    return {
        "schema": "anra-v8-y1-d2-probe-v1",
        "seed": seed,
        "train_cases": tuple(train_cases),
        "test_cases": tuple(test_cases),
        "overlaps_confirmation": False,
    }


def _case_to_decision_samples(case: Y1Case) -> tuple[list[TokenSample], list[dict[str, Any]]]:
    """Expand one Y1Case into explicit public decision-state TokenSamples (BOS+prompt -> target+EOS)."""
    v6 = case.v6_case
    state = CognitiveState(v6.goal).observe(v6.environment.initial)
    ch = tuple(v6.environment.channels)
    avail = v6.memory is not None
    samples: list[TokenSample] = []
    meta: list[dict[str, Any]] = []
    for decision in range(4):
        op, arg = public_action(state, ch, v6.surface, avail)
        target = action_text(op, arg)
        prompt = render_hitchhiker_public(state, ch, v6.surface, avail)
        framed = UTF8.frame(prompt, target, max_length=PROMPT_TOKEN_BOUND)
        tid = f"{case.case_id}/d{decision}"
        samples.append(TokenSample(tid, framed.ids, framed.loss_mask, case.world_id, f"{case.solvability}_d{decision}"))
        meta.append(
            {
                "task_id": tid,
                "family_id": case.family_id,
                "world_id": case.world_id,
                "hops": case.hops,
                "solvability": case.solvability,
                "decision": decision,
                "state_kind": "post_query" if decision > 0 else ("query_initial" if op == "QUERY" else case.solvability),
                "target": target,
                "prompt_len": len(framed.ids),
            }
        )
        if op == "QUERY":
            state = state.observe(v6.environment.channels[str(arg)])
            ch = ()
        elif op == "RETRIEVE":
            assert v6.memory is not None
            state = state.observe(v6.memory.recall(v6.goal))
            avail = False
        else:
            break
    return samples, meta


def build_t0_dataset() -> dict[str, Any]:
    """Build exact 128-family (160 supervised decision-state) T0 learnability dataset."""
    seed = GENERATOR_SEEDS["positive_t0"]
    cases: list[Y1Case] = []
    samples: list[TokenSample] = []
    meta: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for hops in (1, 2, 3, 4):
        for solvability, count in (("answerable", 16), ("query_needed", 8), ("unavailable", 8)):
            pos_plan = _balanced_allocations(count, hops + 2, ("t0", seed, hops, solvability, "pos"))
            gold_plan = _balanced_allocations(count, len(SYMBOLS), ("t0", seed, hops, solvability, "gold"))
            for idx in range(count):
                attempt = 0
                while True:
                    rng = random.Random(derive_seed_int(Y1_DATA_VERSION, "t0", seed, hops, solvability, idx, attempt))
                    attempt += 1
                    try:
                        g = _construct_base_graph(
                            rng,
                            hops=hops,
                            gold_letter=SYMBOLS[gold_plan[idx]],
                            solvability=solvability,
                            target_row_pos=pos_plan[idx],
                            forbid_holdout_combo=True,
                        )
                        fid = f"t0-h{hops}-{solvability[:3]}-{idx:02d}"
                        c = _render_from_rows(
                            case_id=fid,
                            world_id=fid,
                            split="train",
                            panel="t0-learnability",
                            variant_index=0,
                            factor="t0",
                            hops=hops,
                            solvability=solvability,
                            applicable=True,
                            queried_subject=g["queried_subject"],
                            rows=g["rows"],
                            hidden_channel_rows=g["hidden_channel_rows"],
                            terminal_subject=g["terminal_subject"],
                            filler_subject=g["filler_subject"],
                            gold_val=None if solvability == "unavailable" else g["gold_letter"],
                        )
                        csamples, cmeta = _case_to_decision_samples(c)
                    except ValueError:
                        continue
                    if c.public_prompt in seen_prompts:
                        continue
                    seen_prompts.add(c.public_prompt)
                    cases.append(c)
                    samples.extend(csamples)
                    meta.extend(cmeta)
                    break
    if len(cases) != T0_SPEC["families_total"] or len(samples) != T0_SPEC["supervised_states_total"]:
        raise RuntimeError(f"T0 size mismatch: cases={len(cases)}, states={len(samples)}")
    # Cycle deterministically for 256 batch-16 updates = 4,096 exposures (25 full cycles + 96 partial).
    batches = []
    n = len(samples)
    order = list(range(n))
    random.Random(derive_seed_int("t0-cycle", seed)).shuffle(order)
    cursor = 0
    total_exposures = T0_SPEC["updates"] * T0_SPEC["batch_size"]
    cycled_samples: list[TokenSample] = []
    for ex_idx in range(total_exposures):
        base_s = samples[order[cursor % n]]
        cycled_samples.append(replace(base_s, task_id=f"{base_s.task_id}#ex{ex_idx}"))
        cursor += 1
    return {
        "schema": "anra-v8-y1-t0-v1",
        "cases": tuple(cases),
        "unique_samples": tuple(samples),
        "cycled_samples": tuple(cycled_samples),
        "meta": meta,
        "partial_final_cycle_exposures": total_exposures % n,
        "data_sha256": digest([asdict(s) for s in cycled_samples]),
    }


def build_non_route_rehearsal_pool(
    *,
    forbidden_prompts: set[str] | None = None,
    counts: dict[str, int] | None = None,
) -> tuple[TokenSample, ...]:
    """Materialize exact 2,368 historical decision-state rehearsal samples across 6 non-route kinds."""
    target_counts = dict(counts or NON_ROUTE_REHEARSAL_COUNTS)
    seed = GENERATOR_SEEDS["train"]
    collected: dict[str, list[TokenSample]] = {k: [] for k in sorted(target_counts)}
    blocked = set(forbidden_prompts or ())
    seen_prompts: set[str] = set()

    for idx in range(500_000):
        raw_case = _v6_case(idx, seed, "train")
        k = raw_case.kind
        if k not in collected or len(collected[k]) >= target_counts[k]:
            if all(len(v) >= target_counts[name] for name, v in collected.items()):
                break
            continue
        state = CognitiveState(raw_case.goal).observe(raw_case.environment.initial)
        ch = tuple(raw_case.environment.channels)
        avail = raw_case.memory is not None
        for d in range(4):
            if len(collected[k]) >= target_counts[k]:
                break
            op, arg = public_action(state, ch, raw_case.surface, avail)
            tgt = action_text(op, arg)
            prompt = render_hitchhiker_public(state, ch, raw_case.surface, avail)
            if prompt in blocked or prompt in seen_prompts:
                break
            try:
                framed = UTF8.frame(prompt, tgt, max_length=PROMPT_TOKEN_BOUND)
            except ValueError:
                break
            seen_prompts.add(prompt)
            tid = f"rehearsal-{k}-{len(collected[k]):04d}-{raw_case.case_id[:12]}-d{d}"
            collected[k].append(TokenSample(tid, framed.ids, framed.loss_mask, f"reh-{raw_case.world_id[:16]}", f"rehearsal-{k}"))
            if op == "QUERY":
                state = state.observe(raw_case.environment.channels[str(arg)])
                ch = ()
            elif op == "RETRIEVE":
                assert raw_case.memory is not None
                state = state.observe(raw_case.memory.recall(raw_case.goal))
                avail = False
            else:
                break
    for k, target_n in target_counts.items():
        if len(collected[k]) != target_n:
            raise RuntimeError(f"failed exact non-route quota for {k}: {len(collected[k])} != {target_n}")
    return tuple(s for k in sorted(collected) for s in collected[k])


def _build_route_epoch(
    *,
    coverage: str,
    epoch_seed_tag: Any,
    epoch: int,
    forbidden_prompts: set[str],
    route_counts_per_hop: dict[str, dict[str, int]] | None = None,
) -> tuple[list[TokenSample], dict[str, Any]]:
    counts_cfg = (route_counts_per_hop or ROUTE_COVERAGE_COUNTS_PER_HOP)[coverage]
    n_ans = counts_cfg["answerable"]
    n_qry_fam = counts_cfg["query_needed_families"]
    n_unavail = counts_cfg["unavailable"]

    samples: list[TokenSample] = []
    pos_audit: dict[str, dict[int, int]] = {}
    length_bins: Counter[int] = Counter()
    seen_prompts = set(forbidden_prompts)
    over_context_rejected = 0

    for hops in (1, 2, 3, 4):
        n_rows = hops + 2
        for solvability, fam_count in (("answerable", n_ans), ("query_needed", n_qry_fam), ("unavailable", n_unavail)):
            pos_plan = _balanced_allocations(fam_count, n_rows, ("y1-epoch-pos", epoch, hops, solvability))
            gold_plan = _balanced_allocations(fam_count, len(SYMBOLS), ("y1-epoch-gold", epoch_seed_tag, epoch, hops, solvability))
            pos_counter: Counter[int] = Counter()
            for idx in range(fam_count):
                attempt = 0
                while True:
                    rng = random.Random(
                        derive_seed_int(Y1_DATA_VERSION, "train-route", epoch_seed_tag, epoch, hops, solvability, idx, attempt)
                    )
                    attempt += 1
                    try:
                        g = _construct_base_graph(
                            rng,
                            hops=hops,
                            gold_letter=SYMBOLS[gold_plan[idx]],
                            solvability=solvability,
                            target_row_pos=pos_plan[idx],
                            forbid_holdout_combo=True,
                        )
                        fid = f"tr-e{epoch}-h{hops}-{solvability[:3]}-{idx:04d}"
                        c = _render_from_rows(
                            case_id=fid,
                            world_id=fid,
                            split="train",
                            panel="train-route",
                            variant_index=0,
                            factor="train",
                            hops=hops,
                            solvability=solvability,
                            applicable=True,
                            queried_subject=g["queried_subject"],
                            rows=g["rows"],
                            hidden_channel_rows=g["hidden_channel_rows"],
                            terminal_subject=g["terminal_subject"],
                            filler_subject=g["filler_subject"],
                            gold_val=None if solvability == "unavailable" else g["gold_letter"],
                        )
                        if _is_holdout_combo(
                            hops=hops,
                            terminal_pos_0based=c.visible_terminal_pos,
                            distractor_count=2,
                            queried_start_pos_0based=c.queried_start_pos,
                        ):
                            continue
                        csamples, _ = _case_to_decision_samples(c)
                    except ValueError:
                        over_context_rejected += 1
                        continue
                    # Guarantee split disjointness against held-out suites across ALL decision states.
                    prompts = [UTF8.decode([t for t in cs.tokens[1:] if t >= 4]) for cs in csamples]
                    if any(p in seen_prompts for p in prompts):
                        continue
                    seen_prompts.update(prompts)
                    samples.extend(csamples)
                    pos_counter[pos_plan[idx]] += 1
                    for cs in csamples:
                        length_bins[len(cs.tokens) // 32 * 32] += 1
                    break
            diff = max(pos_counter.values()) - min(pos_counter.values()) if pos_counter else 0
            if diff > 1:
                raise ValueError(f"stratum position imbalance > 1: {pos_counter}")
            pos_audit[f"h{hops}_{solvability}"] = dict(sorted(pos_counter.items()))
    return samples, {
        "position_audit": pos_audit,
        "length_bins": dict(sorted(length_bins.items())),
        "over_context_rejected": over_context_rejected,
    }


def build_arm_training_stream(
    *,
    arm: str,
    fresh: bool,
    coverage: str,
    seed: int,
    order_seed: int,
    epochs: int = 4,
    batch_size: int = 16,
    forbidden_prompts: set[str] | None = None,
    rehearsal_counts: dict[str, int] | None = None,
    route_counts_per_hop: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Materialize full deterministic multi-epoch Y1 training stream (static vs fresh, balanced vs imbalanced)."""
    blocked = set(forbidden_prompts or ())
    rehearsal = build_non_route_rehearsal_pool(forbidden_prompts=blocked, counts=rehearsal_counts)
    all_samples: list[TokenSample] = []
    epoch_audits: list[dict[str, Any]] = []
    static_route_epoch: tuple[list[TokenSample], dict[str, Any]] | None = None

    for ep in range(epochs):
        if not fresh and static_route_epoch is not None:
            r_samples, r_audit = static_route_epoch
        else:
            epoch_tag = f"fresh-{arm}-s{GENERATOR_SEEDS['train']}" if fresh else f"static-s{GENERATOR_SEEDS['train']}"
            r_samples, r_audit = _build_route_epoch(
                coverage=coverage,
                epoch_seed_tag=epoch_tag,
                epoch=ep if fresh else 0,
                forbidden_prompts=blocked,
                route_counts_per_hop=route_counts_per_hop,
            )
            if not fresh:
                static_route_epoch = (r_samples, r_audit)

        ep_pool = [
            replace(s, task_id=f"e{ep}/{s.task_id}")
            for s in (*rehearsal, *r_samples)
        ]
        # Deterministic stratum-matched permutation using SHA256 seed derivation.
        perm = list(range(len(ep_pool)))
        random.Random(derive_seed_int(Y1_DATA_VERSION, "epoch-order", order_seed, ep)).shuffle(perm)
        ordered = [ep_pool[i] for i in perm]
        all_samples.extend(ordered)
        epoch_audits.append(
            {
                "epoch": ep,
                "examples": len(ordered),
                "rehearsal_examples": len(rehearsal),
                "route_examples": len(r_samples),
                **r_audit,
            }
        )

    total_examples = len(all_samples)
    total_updates = total_examples // batch_size
    trimmed = tuple(all_samples[: total_updates * batch_size])
    return {
        "schema": "anra-v8-y1-stream-v1",
        "arm": arm,
        "fresh": fresh,
        "coverage": coverage,
        "seed": seed,
        "order_seed": order_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "total_updates": total_updates,
        "samples": trimmed,
        "epoch_audits": epoch_audits,
        "data_sha256": digest([asdict(s) for s in trimmed]),
        "provenance": {
            "schema": Y1_DATA_VERSION,
            "split": "train",
            "license": "repository LICENSE; newly authored AN-RA V8-Y1 procedural worlds",
            "arm": arm,
            "fresh": fresh,
            "coverage": coverage,
            "generator_seed": GENERATOR_SEEDS["train"],
            "order_seed": order_seed,
            "epochs": epochs,
            "rehearsal_counts": dict(rehearsal_counts or NON_ROUTE_REHEARSAL_COUNTS),
        },
    }


def collect_forbidden_prompts(*suites: dict[str, Any]) -> set[str]:
    """Collect all public prompts across held-out/diagnostic suites to guarantee zero train overlap."""
    out: set[str] = set()
    for s in suites:
        for key in ("cases", "historical_continuity", "train_cases", "test_cases"):
            for c in s.get(key, ()):
                out.add(c.public_prompt)
    return out


def audit_y1_split_custody(*suites: dict[str, Any]) -> dict[str, Any]:
    """Verify zero world/prompt/instantiated-state overlap across Y1 splits and check 512-token bound."""
    seen_worlds: dict[str, str] = {}
    seen_prompts: dict[str, str] = {}
    seen_states: dict[str, str] = {}
    topo_sets: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}

    for suite in suites:
        sname = suite["split"]
        cases = suite["cases"]
        split_counts[sname] = len(cases)
        topos: set[str] = set()
        for c in cases:
            if c.token_length > PROMPT_TOKEN_BOUND - RESERVED_TARGET_TOKENS:
                raise ValueError(f"case {c.case_id} violates S3 token budget ({c.token_length})")
            if c.public_prompt in seen_prompts and seen_prompts[c.public_prompt] != sname:
                raise ValueError(f"prompt leak across splits: {seen_prompts[c.public_prompt]} vs {sname}")
            seen_prompts[c.public_prompt] = sname
            if c.panel == "main-base":
                if c.world_id in seen_worlds and seen_worlds[c.world_id] != sname:
                    raise ValueError(f"world_id leak: {c.world_id}")
                seen_worlds[c.world_id] = sname
                ikey = instantiated_state_key(c.environment.initial, c.goal.subject, c.family, bool(c.environment.channels), False)
                if ikey in seen_states and seen_states[ikey] != sname:
                    raise ValueError(f"instantiated state leak across splits: {seen_states[ikey]} vs {sname}")
                seen_states[ikey] = sname
            topos.add(family_split_key(CognitiveState(c.goal).observe(c.environment.initial), tuple(c.environment.channels), c.memory is not None))
        topo_sets[sname] = topos

    topo_overlaps = {}
    names = list(topo_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            topo_overlaps[f"{a}__vs__{b}"] = len(topo_sets[a] & topo_sets[b])

    return {
        "schema": "anra-v8-y1-custody-audit-v1",
        "status": "PASS",
        "split_counts": split_counts,
        "unique_prompts_checked": len(seen_prompts),
        "unique_instantiated_base_states_checked": len(seen_states),
        "canonical_topology_overlaps_disclosed": topo_overlaps,
    }


def y1_case_to_record(case: Y1Case) -> dict[str, Any]:
    """Serialize one Y1Case losslessly without writing duplicated prompt strings into model predictions."""
    v6 = case.v6_case
    surf_cls = "HitchhikerSurface" if isinstance(v6.surface, HitchhikerSurface) else "Surface"
    return {
        "case_id": case.case_id,
        "world_id": case.world_id,
        "family_id": case.family_id,
        "split": case.split,
        "panel": case.panel,
        "variant_index": case.variant_index,
        "factor": case.factor,
        "hops": case.hops,
        "solvability": case.solvability,
        "applicable": case.applicable,
        "latent_terminal_pos": case.latent_terminal_pos,
        "visible_terminal_pos": case.visible_terminal_pos,
        "tracked_filler_pos": case.tracked_filler_pos,
        "queried_start_pos": case.queried_start_pos,
        "surface_type": surf_cls,
        "surface": asdict(v6.surface),
        "family": v6.family,
        "kind": v6.kind,
        "goal": asdict(v6.goal),
        "initial": [asdict(e) for e in v6.environment.initial],
        "channels": {k: [asdict(e) for e in v] for k, v in v6.environment.channels.items()},
        "truth": asdict(v6.truth),
        "memory": None if v6.memory is None else asdict(v6.memory),
        "pair_id": v6.pair_id,
        "public_prompt": case.public_prompt,
        "gold_operation": case.gold_operation,
        "gold_argument": case.gold_argument,
        "gold_target": case.gold_target,
        "token_length": case.token_length,
        "reference_case_id": case.reference_case_id,
    }


def y1_case_from_record(row: dict[str, Any]) -> Y1Case:
    """Deserialize a Y1Case record deterministically."""
    from .v6_data import Surface

    ev = lambda item: ExperienceEvent(**{**item, "parents": tuple(item["parents"])})
    goal = Goal(**{**row["goal"], "slots": tuple(row["goal"]["slots"])})
    mem = row["memory"]
    memory = None if mem is None else MemoryVersion(tuple(ev(e) for e in mem["events"]), mem["parent"], mem["capacity"])
    surf = HitchhikerSurface(**row["surface"]) if row["surface_type"] == "HitchhikerSurface" else Surface(**row["surface"])
    v6 = V6Case(
        row["case_id"],
        row["world_id"],
        row["family"],
        row["kind"],
        goal,
        PublicEnvironment(tuple(ev(e) for e in row["initial"]), {k: tuple(ev(e) for e in v) for k, v in row["channels"].items()}),
        Truth(**{**row["truth"], "admitted_public": tuple(row["truth"]["admitted_public"])}),
        surf,
        memory,
        row["pair_id"],
        row["panel"],
    )
    return Y1Case(
        case_id=row["case_id"],
        world_id=row["world_id"],
        family_id=row["family_id"],
        split=row["split"],
        panel=row["panel"],
        variant_index=row["variant_index"],
        factor=row["factor"],
        hops=row["hops"],
        solvability=row["solvability"],
        applicable=row["applicable"],
        latent_terminal_pos=row["latent_terminal_pos"],
        visible_terminal_pos=row["visible_terminal_pos"],
        tracked_filler_pos=row["tracked_filler_pos"],
        queried_start_pos=row["queried_start_pos"],
        v6_case=v6,
        public_prompt=row["public_prompt"],
        gold_operation=row["gold_operation"],
        gold_argument=row["gold_argument"],
        gold_target=row["gold_target"],
        token_length=row["token_length"],
        reference_case_id=row.get("reference_case_id"),
    )


def serialize_y1_suite(suite: dict[str, Any]) -> dict[str, Any]:
    """Export compact deduplicated world_table + variant_table with strict evaluator integrity hash."""
    world_table: dict[str, dict[str, Any]] = {}
    variant_table: list[dict[str, Any]] = []
    for c in (*suite["cases"], *suite.get("historical_continuity", ())):
        if c.world_id not in world_table:
            world_table[c.world_id] = {
                "world_id": c.world_id,
                "split": c.split,
                "hops": c.hops,
                "family_id": c.family_id,
                "solvability": c.solvability,
                "latent_terminal_pos": c.latent_terminal_pos,
            }
        rec = y1_case_to_record(c)
        rec["public_prompt_sha256"] = digest(c.public_prompt)
        variant_table.append(rec)
    payload = {
        "schema": suite["schema"],
        "split": suite["split"],
        "seed": suite["seed"],
        "over_context_rejected": suite["over_context_rejected"],
        "suite_sha256": suite["suite_sha256"],
        "world_table": world_table,
        "variant_table": variant_table,
    }
    payload["payload_sha256"] = digest([payload["world_table"], payload["variant_table"]])
    return payload


def load_y1_suite(doc: dict[str, Any]) -> dict[str, Any]:
    """Load and SHA-verify a serialized Y1 suite bundle."""
    if digest([doc["world_table"], doc["variant_table"]]) != doc.get("payload_sha256"):
        raise ValueError("Y1 serialized suite failed payload_sha256 verification")
    all_cases: list[Y1Case] = []
    hist_cases: list[Y1Case] = []
    cohorts: dict[str, list[Y1Case]] = defaultdict(list)
    for row in doc["variant_table"]:
        c = y1_case_from_record(row)
        if c.panel == "historical-continuity":
            hist_cases.append(c)
        else:
            all_cases.append(c)
            if c.panel.startswith("main-"):
                cohorts["main"].append(c)
            cohorts[c.panel].append(c)
    return {
        "schema": doc["schema"],
        "split": doc["split"],
        "seed": doc["seed"],
        "cases": tuple(all_cases),
        "cohorts": {k: tuple(v) for k, v in cohorts.items()},
        "historical_continuity": tuple(hist_cases),
        "over_context_rejected": doc["over_context_rejected"],
        "suite_sha256": doc["suite_sha256"],
    }

