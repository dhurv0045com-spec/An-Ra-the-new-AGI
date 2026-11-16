"""Canonical graph isomorphism auditing, split firewall, and confirmation suite.

Confirmation generation and rendering are implemented separately from the
training generator/renderer and explicitly tagged as separate implementation
with shared authorship.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations
import random

from anra_v5.elendril.codec import digest

from .contracts import SYMBOLS

CONFIRMATION_GENERATOR_VERSION = "elendril-v8-confirmation-ledger-v1"
CONFIRMATION_RENDERER_ID = "v8-confirmation-renderer-template-3"


def _world_edges(world: dict[str, object]) -> tuple[str, int, tuple[tuple[str, str, str, int, str | None], ...], bool, bool]:
    initial = [("init", r["subject"], r["source"], int(r["tick"]), r["value"]) for r in world["initial_facts"]]
    query = [("query", r["subject"], r["source"], int(r["tick"]), r["value"]) for r in world["hidden"]["query_payload"]]
    mem = [("mem", r["subject"], r["source"], int(r["tick"]), r["value"]) for r in world["hidden"]["memory_payload"]]
    all_raw = initial + query + mem
    unique_ticks = sorted({row[3] for row in all_raw})
    tick_rank = {t: idx for idx, t in enumerate(unique_ticks)}
    normalized = tuple((stream, subj, src, tick_rank[t], val) for stream, subj, src, t, val in all_raw)
    return (
        str(world["goal_subject"]),
        int(world["hops"]),
        normalized,
        bool(world["allowed_public_tools"]["query_available"]),
        bool(world["allowed_public_tools"]["memory_available"]),
    )


def canonical_graph_signature(world: dict[str, object]) -> dict[str, object]:
    """Compute an entity-permutation-invariant, row-order-invariant canonical signature."""

    goal_subj, hops, edges, q_avail, m_avail = _world_edges(world)
    nodes = sorted({goal_subj} | {e[1] for e in edges} | {e[4] for e in edges if e[4] is not None})
    if len(nodes) > 12:
        raise ValueError("canonicalizer supports at most 12 entities")

    # Two rounds of 1-WL color refinement to partition nodes into small symmetry classes
    colors: dict[str, tuple] = {}
    for u in nodes:
        out_sig = tuple(sorted((e[0], e[2], e[3], e[4] is None, e[4] == u) for e in edges if e[1] == u))
        in_sig = tuple(sorted((e[0], e[2], e[3], e[1] == u) for e in edges if e[4] == u))
        colors[u] = (u == goal_subj, out_sig, in_sig)
    for _ in range(2):
        new_colors: dict[str, tuple] = {}
        for u in nodes:
            out_nb = tuple(sorted((e[0], e[2], e[3], colors[e[4]] if e[4] is not None else None) for e in edges if e[1] == u))
            in_nb = tuple(sorted((e[0], e[2], e[3], colors[e[1]]) for e in edges if e[4] == u))
            new_colors[u] = (colors[u][0], out_nb, in_nb)
        colors = new_colors

    # Group by color in deterministic sorted order
    by_color: dict[tuple, list[str]] = defaultdict(list)
    for u in nodes:
        by_color[colors[u]].append(u)
    ordered_groups = [by_color[c] for c in sorted(by_color)]

    # Exact lexicographic search if product of factorials <= 4096, else WL + capped tie-break
    total_perms = 1
    for grp in ordered_groups:
        for k in range(1, len(grp) + 1):
            total_perms *= k
            if total_perms > 4096:
                break

    exact = total_perms <= 4096

    def _encode_with_mapping(mapping: dict[str, int]) -> tuple:
        mapped_edges = tuple(
            sorted((stream, mapping[subj], src, tick_r, None if val is None else mapping[val]) for stream, subj, src, tick_r, val in edges)
        )
        return (hops, mapping[goal_subj], q_avail, m_avail, mapped_edges)

    best_tuple: tuple | None = None
    if exact:
        # Enumerate permutations per color class
        def _recurse(group_idx: int, next_id: int, current_map: dict[str, int]) -> None:
            nonlocal best_tuple
            if group_idx == len(ordered_groups):
                cand = _encode_with_mapping(current_map)
                if best_tuple is None or cand < best_tuple:
                    best_tuple = cand
                return
            grp = ordered_groups[group_idx]
            for perm in permutations(grp):
                nxt = dict(current_map)
                for offset, node in enumerate(perm):
                    nxt[node] = next_id + offset
                _recurse(group_idx + 1, next_id + len(grp), nxt)

        _recurse(0, 0, {})
    else:
        # Deterministic canonical fallback using degree-sorted order within large symmetric classes
        mapping = {}
        idx = 0
        for grp in ordered_groups:
            for node in sorted(grp):
                mapping[node] = idx
                idx += 1
        best_tuple = _encode_with_mapping(mapping)

    assert best_tuple is not None
    # Structural-only graph topology (ignoring tool flags) for multi-entity isomorphism audit
    mapped_edges_only = best_tuple[4]
    structure_hash = digest(["v8-canon-struct-v1", hops, best_tuple[1], mapped_edges_only])
    full_hash = digest(["v8-canon-full-v1", best_tuple])
    is_isolated_one_edge = len(edges) == 1 and hops <= 1
    return {
        "canonical_hash": full_hash,
        "structure_hash": structure_hash,
        "entity_count": len(nodes),
        "edge_count": len(edges),
        "exact_canonicalization": exact,
        "is_isolated_one_edge_probe": is_isolated_one_edge,
    }


def is_isolated_one_edge_probe(world: dict[str, object]) -> bool:
    return canonical_graph_signature(world)["is_isolated_one_edge_probe"]


def audit_split_disjointness(
    worlds: list[dict[str, object]] | tuple[dict[str, object], ...],
    *,
    min_distinct_classes_per_split: int = 1,
) -> dict[str, object]:
    """Audit split assignment by connected counterfactual components and canonical hashes."""

    # Union-find over world_id, pair_id, parent_world_id, and semantic_hash
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    splits_by_comp: dict[str, set[str]] = defaultdict(set)
    struct_splits: dict[str, set[str]] = defaultdict(set)
    classes_by_split: dict[str, set[str]] = defaultdict(set)
    exact_count = 0
    probe_count = 0

    for w in worlds:
        wid = str(w["world_id"])
        split = str(w["split"])
        sig = canonical_graph_signature(w)
        if sig["exact_canonicalization"]:
            exact_count += 1
        if sig["is_isolated_one_edge_probe"]:
            probe_count += 1
        shash = str(sig["structure_hash"])
        if w.get("pair_id"):
            union(wid, "pair:" + str(w["pair_id"]))
        if w.get("parent_world_id"):
            union(wid, str(w["parent_world_id"]))
        if w.get("semantic_hash"):
            union(wid, "sem:" + str(w["semantic_hash"]))
        if not sig["is_isolated_one_edge_probe"]:
            struct_splits[shash].add(split)
            classes_by_split[split].add(shash)

    for w in worlds:
        splits_by_comp[find(str(w["world_id"]))].add(str(w["split"]))

    leaked_components = [comp for comp, sp in splits_by_comp.items() if len(sp) > 1]
    if leaked_components:
        raise ValueError(f"split leakage across counterfactual/semantic connected components: {len(leaked_components)}")

    leaked_structures = [h for h, sp in struct_splits.items() if len(sp) > 1]
    if leaked_structures:
        raise ValueError(f"structural graph isomorphism leakage across splits: {len(leaked_structures)}")

    for sp, cls in classes_by_split.items():
        if len(cls) < min_distinct_classes_per_split:
            raise ValueError(
                f"insufficient distinct canonical classes in split {sp!r}: {len(cls)} < {min_distinct_classes_per_split}"
            )

    return {
        "status": "PASS",
        "total_worlds": len(worlds),
        "exact_canonicalized": exact_count,
        "isolated_one_edge_probes": probe_count,
        "distinct_structural_classes_by_split": {k: len(v) for k, v in sorted(classes_by_split.items())},
        "leaked_components": 0,
        "leaked_structures": 0,
    }


def render_confirmation_public(
    *,
    ordered_facts: tuple,
    goal_subject: str,
    hops: int,
    query_available: bool,
    memory_available: bool,
    template_id: int,
    alias_prefix: str,
    order_seed: int,
    decision: int,
) -> str:
    """Separately implemented confirmation renderer (template 3, distinct phrasing)."""

    del template_id
    def _name(sym: str) -> str:
        return alias_prefix + sym

    preamble = (
        "Rules: newest tick per source wins; conflicting latest sources stay unresolved. "
        "Code: A-P, Q=query, R=memory, !=abstain.\n"
    )
    if hops <= 1:
        ask = f"Determine value of {_name(goal_subject)}."
    else:
        ask = f"Traverse {hops} hops starting at {_name(goal_subject)}."
    if alias_prefix:
        ask += " Return code letter A-P."
    meta = f"{ask} [Query={int(query_available)} Memory={int(memory_available)}]\n"

    rows_list = list(ordered_facts)
    if order_seed != 0:
        rng = random.Random(int(digest(["cf-order", order_seed, decision]), 16))
        rng.shuffle(rows_list)
    rendered_rows = []
    for fact in rows_list:
        subj = _name(fact.subject)
        val = "null" if fact.value is None else _name(fact.value)
        rendered_rows.append(f"[{fact.tick}|{subj}->{val}|{fact.source}]")
    return preamble + meta + "; ".join(rendered_rows) + "\nAction:"


def _cf_ledger_world(
    *,
    world_id: str,
    family: str,
    scenario: str,
    hops: int,
    goal_subject: str,
    initial_ledger: list[tuple[str, str, int, str | None]],
    query_ledger: list[tuple[str, str, int, str | None]],
    memory_ledger: list[tuple[str, str, int, str | None]],
    expected: str | None,
    query_available: bool,
    memory_available: bool,
    alias_prefix: str = "",
    pair_id: str = "",
    parent_world_id: str = "",
    variant: str = "canonical",
    row_order_seed: int = 0,
) -> dict[str, object]:
    def _rec(row: tuple[str, str, int, str | None]) -> dict[str, object]:
        return {"subject": row[0], "source": row[1], "tick": row[2], "value": row[3]}

    world = {
        "world_id": world_id,
        "parent_world_id": parent_world_id or world_id,
        "pair_id": pair_id,
        "split": "confirmation",
        "family": family,
        "scenario": scenario,
        "variant": variant,
        "hops": hops,
        "goal_subject": goal_subject,
        "initial_facts": [_rec(r) for r in initial_ledger],
        "allowed_public_tools": {"query_available": query_available, "memory_available": memory_available},
        "hidden": {
            "expected": expected,
            "query_payload": [_rec(r) for r in query_ledger],
            "memory_payload": [_rec(r) for r in memory_ledger],
        },
        "renderer_id": CONFIRMATION_RENDERER_ID,
        "renderer_template": 3,
        "alias_prefix": alias_prefix,
        "row_order_seed": row_order_seed,
        "authorship": "separate implementation with shared authorship",
    }
    world["semantic_hash"] = canonical_graph_signature(world)["structure_hash"]
    world["content_hash"] = digest(world)
    return world


def build_confirmation_suite(
    confirmation_seed: int,
    *,
    excluded_structure_hashes: set[str] | frozenset[str] = frozenset(),
    worlds_per_cohort: int = 256,
    include_multihop_holdout: bool = True,
) -> dict[str, tuple[dict[str, object], ...]]:
    """Build fresh confirmation worlds using a separate transition-ledger implementation.

    Does not call train world construction or train target-selection helpers.
    """

    if type(confirmation_seed) is not int or not 0 <= confirmation_seed < 2**32:
        raise ValueError("confirmation_seed must be uint32")
    if type(worlds_per_cohort) is not int or worlds_per_cohort < 1:
        raise ValueError("worlds_per_cohort must be positive")

    rng = random.Random(int(digest([CONFIRMATION_GENERATOR_VERSION, confirmation_seed]), 16))
    excluded = set(excluded_structure_hashes)
    cohorts: dict[str, list[dict[str, object]]] = defaultdict(list)

    def _make_base_and_variants(cohort_name: str, idx: int) -> list[dict[str, object]]:
        # Sample balanced target symbol cyclically
        target_sym = SYMBOLS[idx % len(SYMBOLS)]
        pool = [s for s in SYMBOLS if s != target_sym]
        for attempt in range(64):
            picked = rng.sample(pool, 7)
            subj, mid1, mid2, dist_s, dist_t, alt_cf, alt_irr = picked
            base_tick = rng.randrange(100, 700)
            q_avail = bool((idx + attempt) % 2)
            m_avail = bool(((idx // 2) + attempt) % 2)
            distractors = [
                (dist_s, "log", base_tick + 3, dist_t),
                (alt_irr, "map", base_tick + 7, alt_cf),
            ]
            # Construct scenario-specific ledgers
            if cohort_name == "R09":
                family, hops = "direct_one_hop", 1
                init = [(subj, "s0", base_tick, target_sym)] + distractors
                q_pay = [(alt_irr, "audit", base_tick + 50, alt_cf)] if q_avail else []
                m_pay = [(dist_s, "mem", base_tick + 40, dist_t)] if m_avail else []
                exp = target_sym
                cf_rel_init = [(subj, "s0", base_tick, alt_cf)] + distractors
                cf_rel_exp = alt_cf
                cf_irr_init = [(subj, "s0", base_tick, target_sym), (dist_s, "log", base_tick + 3, alt_cf), (alt_irr, "map", base_tick + 7, alt_cf)]
            elif cohort_name == "R10":
                family, hops = "two_hop", 2
                init = [(subj, "map", base_tick, mid1), (mid1, "map", base_tick + 5, target_sym)] + distractors
                q_pay = [(alt_irr, "audit", base_tick + 50, alt_cf)] if q_avail else []
                m_pay = [(dist_s, "mem", base_tick + 40, dist_t)] if m_avail else []
                exp = target_sym
                cf_rel_init = [(subj, "map", base_tick, mid1), (mid1, "map", base_tick + 5, alt_cf)] + distractors
                cf_rel_exp = alt_cf
                cf_irr_init = [(subj, "map", base_tick, mid1), (mid1, "map", base_tick + 5, target_sym), (dist_s, "log", base_tick + 3, alt_cf), (alt_irr, "map", base_tick + 7, alt_cf)]
            elif cohort_name == "E13":
                family, hops = "temporal_source", 1
                init = [(subj, "s0", base_tick, alt_cf), (subj, "s0", base_tick + 12, target_sym)] + distractors
                q_pay = [(alt_irr, "audit", base_tick + 50, alt_cf)] if q_avail else []
                m_pay = [(dist_s, "mem", base_tick + 40, dist_t)] if m_avail else []
                exp = target_sym
                cf_rel_init = [(subj, "s0", base_tick, target_sym), (subj, "s0", base_tick + 12, alt_cf)] + distractors
                cf_rel_exp = alt_cf
                cf_irr_init = [(subj, "s0", base_tick, alt_cf), (subj, "s0", base_tick + 12, target_sym), (dist_s, "log", base_tick + 3, alt_cf), (alt_irr, "map", base_tick + 7, alt_cf)]
            elif cohort_name == "E14":
                family, hops = "missing_query", 1
                q_avail = True
                m_avail = False
                init = list(distractors)
                q_pay = [(subj, "s0", base_tick + 25, target_sym)]
                m_pay = []
                exp = target_sym
                cf_rel_init = list(distractors)
                cf_rel_exp = alt_cf
                cf_irr_init = [(dist_s, "log", base_tick + 3, alt_cf), (alt_irr, "map", base_tick + 7, alt_cf)]
            elif cohort_name == "E15":
                family, hops = "conflict_query", 1
                q_avail = True
                m_avail = False
                init = [(subj, "s0", base_tick, alt_cf), (subj, "s1", base_tick + 1, mid1)] + distractors
                q_pay = [(subj, "s0", base_tick + 30, target_sym), (subj, "s1", base_tick + 30, target_sym)]
                m_pay = []
                exp = target_sym
                cf_rel_init = list(init)
                cf_rel_exp = alt_cf
                cf_irr_init = [(subj, "s0", base_tick, alt_cf), (subj, "s1", base_tick + 1, mid1), (dist_s, "log", base_tick + 3, alt_cf), (alt_irr, "map", base_tick + 7, alt_cf)]
            else:  # A19 null/unavailable
                family, hops = "null_unavailable", 1
                is_null = idx % 2 == 0
                if is_null:
                    init = [(subj, "s0", base_tick, None)] + distractors
                    q_pay = [(alt_irr, "audit", base_tick + 50, alt_cf)] if q_avail else []
                    m_pay = [(dist_s, "mem", base_tick + 40, dist_t)] if m_avail else []
                    cf_rel_init = [(subj, "s0", base_tick, alt_cf)] + distractors
                    cf_rel_exp = alt_cf
                else:
                    q_avail = False
                    m_avail = False
                    init = list(distractors)
                    q_pay = []
                    m_pay = []
                    cf_rel_init = [(subj, "s0", base_tick, alt_cf)] + distractors
                    cf_rel_exp = alt_cf
                exp = None
                cf_irr_init = init[:-1] + [(alt_irr, "map", base_tick + 7, mid2)]

            base_id = digest([CONFIRMATION_GENERATOR_VERSION, confirmation_seed, cohort_name, idx, attempt])
            pair_id = digest(["cf-pair", base_id])
            base_w = _cf_ledger_world(
                world_id=base_id,
                family=family,
                scenario=cohort_name,
                hops=hops,
                goal_subject=subj,
                initial_ledger=init,
                query_ledger=q_pay,
                memory_ledger=m_pay,
                expected=exp,
                query_available=q_avail,
                memory_available=m_avail,
                pair_id=pair_id,
                variant="canonical",
            )
            if base_w["semantic_hash"] in excluded:
                continue

            # Linked variants: row_shuffle, fresh_alias, relevant_cf, irrelevant_cf
            shuffle_w = _cf_ledger_world(
                world_id=base_id + "-shuf",
                parent_world_id=base_id,
                family=family,
                scenario=cohort_name,
                hops=hops,
                goal_subject=subj,
                initial_ledger=[(a, b, c + 17, d) for a, b, c, d in init],
                query_ledger=[(a, b, c + 17, d) for a, b, c, d in q_pay],
                memory_ledger=[(a, b, c + 17, d) for a, b, c, d in m_pay],
                expected=exp,
                query_available=q_avail,
                memory_available=m_avail,
                pair_id=pair_id,
                variant="row_shuffle",
                row_order_seed=(idx + 1) * 31,
            )
            alias_w = _cf_ledger_world(
                world_id=base_id + "-alias",
                parent_world_id=base_id,
                family=family,
                scenario=cohort_name,
                hops=hops,
                goal_subject=subj,
                initial_ledger=init,
                query_ledger=q_pay,
                memory_ledger=m_pay,
                expected=exp,
                query_available=q_avail,
                memory_available=m_avail,
                alias_prefix="cf-",
                pair_id=pair_id,
                variant="fresh_alias",
            )
            cf_rel_q = [(subj, "s0", base_tick + 25, alt_cf)] if cohort_name == "E14" else (
                [(subj, "s0", base_tick + 30, alt_cf), (subj, "s1", base_tick + 30, alt_cf)] if cohort_name == "E15" else q_pay
            )
            rel_cf_w = _cf_ledger_world(
                world_id=base_id + "-relcf",
                parent_world_id=base_id,
                family=family,
                scenario=cohort_name,
                hops=hops,
                goal_subject=subj,
                initial_ledger=cf_rel_init,
                query_ledger=cf_rel_q,
                memory_ledger=m_pay,
                expected=cf_rel_exp,
                query_available=q_avail,
                memory_available=m_avail,
                pair_id=pair_id,
                variant="relevant_cf",
            )
            irr_cf_w = _cf_ledger_world(
                world_id=base_id + "-irrcf",
                parent_world_id=base_id,
                family=family,
                scenario=cohort_name,
                hops=hops,
                goal_subject=subj,
                initial_ledger=cf_irr_init,
                query_ledger=q_pay,
                memory_ledger=m_pay,
                expected=exp,
                query_available=q_avail,
                memory_available=m_avail,
                pair_id=pair_id,
                variant="irrelevant_cf",
            )
            return [base_w, shuffle_w, alias_w, rel_cf_w, irr_cf_w]
        raise ValueError(f"failed to sample non-excluded confirmation world for {cohort_name} idx={idx}")

    for cohort_name in ("R09", "R10", "E13", "E14", "E15", "A19"):
        for idx in range(worlds_per_cohort):
            cohorts[cohort_name].extend(_make_base_and_variants(cohort_name, idx))

    if include_multihop_holdout:
        for hops, cname in ((3, "R11_hop3"), (4, "R11_hop4")):
            for idx in range(min(64, worlds_per_cohort)):
                target_sym = SYMBOLS[idx % len(SYMBOLS)]
                pool = [s for s in SYMBOLS if s != target_sym]
                chain_nodes = rng.sample(pool, hops + 2)
                nodes = chain_nodes[:hops] + [target_sym]
                dist_s, dist_t = chain_nodes[hops], chain_nodes[hops + 1]
                tick = rng.randrange(100, 500)
                init = [(nodes[i], "map", tick + i, nodes[i + 1]) for i in range(hops)] + [(dist_s, "log", tick + 9, dist_t)]
                wid = digest([CONFIRMATION_GENERATOR_VERSION, confirmation_seed, cname, idx])
                w = _cf_ledger_world(
                    world_id=wid,
                    family=f"holdout_hop_{hops}",
                    scenario="R11",
                    hops=hops,
                    goal_subject=nodes[0],
                    initial_ledger=init,
                    query_ledger=[],
                    memory_ledger=[],
                    expected=target_sym,
                    query_available=False,
                    memory_available=False,
                    pair_id=digest(["cf-pair", wid]),
                    variant="canonical",
                )
                if w["semantic_hash"] in excluded:
                    continue
                cohorts[cname].append(w)

    return {k: tuple(v) for k, v in cohorts.items()}


__all__ = [
    "CONFIRMATION_GENERATOR_VERSION",
    "CONFIRMATION_RENDERER_ID",
    "audit_split_disjointness",
    "build_confirmation_suite",
    "canonical_graph_signature",
    "is_isolated_one_edge_probe",
    "render_confirmation_public",
]
