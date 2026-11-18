"""Avendi V8-Y2 world/family generators, finite 1-hop pool allocator, custody checks, and P0 instrument."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import itertools
import random
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import canonical, digest
from .oracle import (
    evaluate_reference_policies_on_prompt,
    parse_tokenized_public_observation,
    run_pilot_reference_check,
    verify_target_tokens_have_eos,
    walk_parsed_graph,
)
from .protocol import (
    DESIGN_SHA,
    GENERATOR_SEEDS,
    ORDERED_UNEQUAL_PAIRS,
    SPLIT_FAMILY_COUNTS,
    SYMBOLS,
    derive_seed_int,
)
from .render import (
    RenderedRow,
    encode_prompt_and_target,
    erase_goal_token,
    extract_prompt_prefix_from_framed_tokens,
    mask_bridge_edges_in_prompt,
    render_avendi_prompt,
)


SOURCES: tuple[str, ...] = ("s0", "s1", "s2", "s3")
CELL_KEYS: tuple[str, ...] = ("00", "01", "10", "11")  # (query_idx, swapped_idx)


@dataclass(frozen=True, slots=True)
class AvendiCase:
    """Single presentation cell within an Avendi family or literal copy control."""

    case_id: str
    family_id: str
    split: str
    panel: str
    variant: str
    cell_key: str  # '00', '01', '10', '11', or 'literal'
    hops: int
    query_subject: str
    swapped: bool
    query_available: bool
    memory_available: bool
    template: int
    public_prompt: str
    prompt_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    full_tokens: tuple[int, ...]
    target_boundary_index: int
    prompt_token_sha256: str
    prompt_utf8_sha256: str
    instantiated_state_sha256: str
    canonical_family_key: str
    # Evaluator-only private truth (never exposed to actor):
    expected_answer: str
    rival_answer: str
    expected_operation: str
    proof_path: tuple[str, ...]
    endpoint_g: str
    endpoint_h: str
    bridge_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AvendiFamily:
    """Four-cell counterfactual crossover family (2 queries x 2 bridge rewires)."""

    family_id: str
    split: str
    panel: str
    variant: str
    hops: int
    canonical_family_key: str
    query_a: str
    query_b: str
    endpoint_g: str
    endpoint_h: str
    bridge_subjects: tuple[str, ...]
    cells: tuple[AvendiCase, ...]  # ordered in graph-major / cell_key order: 00, 10, 01, 11 or by cell_key

    def cell_map(self) -> dict[str, AvendiCase]:
        return {c.cell_key: c for c in self.cells}


def enumerate_canonical_one_hop_pool() -> tuple[tuple[tuple[str, str], tuple[str, str]], ...]:
    """Enumerate all 10,920 canonical 1-hop ({a, b}, {g, h}) unordered key/value family classes."""
    pool: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for a_idx in range(len(SYMBOLS)):
        for b_idx in range(a_idx + 1, len(SYMBOLS)):
            ka, kb = SYMBOLS[a_idx], SYMBOLS[b_idx]
            rem = [s for s in SYMBOLS if s not in (ka, kb)]
            for g_idx in range(len(rem)):
                for h_idx in range(g_idx + 1, len(rem)):
                    vg, vh = rem[g_idx], rem[h_idx]
                    pool.append(((ka, kb), (vg, vh)))
    return tuple(pool)


_CANONICAL_ONE_HOP_POOL = enumerate_canonical_one_hop_pool()


def _allocate_disjoint_one_hop_specs() -> dict[str, list[tuple[str, str, str, str]]]:
    """Deterministically allocate disjoint canonical 1-hop families across train/dev/confirm/transfer with balanced (g,h) pairs."""
    total_needed = (
        SPLIT_FAMILY_COUNTS["train_p3"]["one_hop_families"]
        + SPLIT_FAMILY_COUNTS["dev"]["one_hop_families"]
        + SPLIT_FAMILY_COUNTS["confirm"]["one_hop_families"]
        + SPLIT_FAMILY_COUNTS["transfer"]["one_hop_families"]
        + SPLIT_FAMILY_COUNTS["train_p4"]["one_hop_families"]
    )
    if len(_CANONICAL_ONE_HOP_POOL) < total_needed:
        raise ValueError(f"insufficient finite 1-hop support: {len(_CANONICAL_ONE_HOP_POOL)} < {total_needed}")

    # Group canonical classes by unordered endpoint set {g, h}
    by_val_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (ka, kb), (vg, vh) in _CANONICAL_ONE_HOP_POOL:
        by_val_pair.setdefault((vg, vh), []).append((ka, kb))

    rng = random.Random(derive_seed_int("avendi-one-hop-pool-alloc", DESIGN_SHA))
    for k_list in by_val_pair.values():
        rng.shuffle(k_list)

    used_canonical: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    allocations: dict[str, list[tuple[str, str, str, str]]] = {}

    split_requests = (
        ("train_p3", SPLIT_FAMILY_COUNTS["train_p3"]["one_hop_families"], GENERATOR_SEEDS["train"]),
        ("dev", SPLIT_FAMILY_COUNTS["dev"]["one_hop_families"], GENERATOR_SEEDS["dev"]),
        ("confirm", SPLIT_FAMILY_COUNTS["confirm"]["one_hop_families"], GENERATOR_SEEDS["confirm"]),
        ("transfer", SPLIT_FAMILY_COUNTS["transfer"]["one_hop_families"], GENERATOR_SEEDS["transfer"]),
        ("train_p4", SPLIT_FAMILY_COUNTS["train_p4"]["one_hop_families"], GENERATOR_SEEDS["p4_train"]),
    )

    for split_name, count, split_seed in split_requests:
        split_rng = random.Random(derive_seed_int("avendi-one-hop-order", split_name, split_seed))
        pair_cycle = list(ORDERED_UNEQUAL_PAIRS)
        split_rng.shuffle(pair_cycle)
        chosen: list[tuple[str, str, str, str]] = []
        for idx in range(count):
            g, h = pair_cycle[idx % len(pair_cycle)]
            val_key = (g, h) if g < h else (h, g)
            candidates = by_val_pair[val_key]
            picked_keys: tuple[str, str] | None = None
            while candidates:
                cand = candidates.pop()
                if (cand, val_key) not in used_canonical:
                    picked_keys = cand
                    used_canonical.add((cand, val_key))
                    break
            if picked_keys is None:
                raise RuntimeError(f"exhausted canonical 1-hop candidates for {val_key}")
            ka, kb = picked_keys
            if split_rng.random() < 0.5:
                ka, kb = kb, ka
            chosen.append((ka, kb, g, h))
        allocations[split_name] = chosen
    return allocations


_ONE_HOP_ALLOCATIONS = _allocate_disjoint_one_hop_specs()


def _instantiated_state_digest(
    *,
    rows: Sequence[RenderedRow],
    query_subject: str,
    hops: int,
    query_available: bool,
    memory_available: bool,
    is_literal_copy: bool,
) -> str:
    """Hash normalized (fact set, query, hops, visibility/availability) independent of surface row order or template."""
    norm_facts = sorted(
        (r.subject, "" if r.value is None else r.value)
        for r in rows
    )
    return digest([
        "avendi-instantiated-state-v1",
        norm_facts,
        query_subject,
        int(hops),
        bool(query_available),
        bool(memory_available),
        bool(is_literal_copy),
    ])


def _build_validated_case(
    *,
    case_id: str,
    family_id: str,
    split: str,
    panel: str,
    variant: str,
    cell_key: str,
    hops: int,
    query_subject: str,
    swapped: bool,
    rows: Sequence[RenderedRow],
    query_available: bool,
    memory_available: bool,
    template: int,
    alias_map: Mapping[str, str] | None,
    is_literal_copy: bool,
    expected_answer: str,
    rival_answer: str,
    endpoint_g: str,
    endpoint_h: str,
    bridge_subjects: tuple[str, ...],
    canonical_family_key: str,
) -> AvendiCase:
    prompt = render_avendi_prompt(
        query_subject=query_subject,
        hops=hops,
        rows=rows,
        query_available=query_available,
        memory_available=memory_available,
        template=template,
        alias_map=alias_map,
        is_literal_copy=is_literal_copy,
    )
    enc = encode_prompt_and_target(prompt, expected_answer)
    verify_target_tokens_have_eos(enc["target_tokens"])

    # Second independent parser must agree on exact tokenized public representation!
    parsed = parse_tokenized_public_observation(enc["prompt_tokens"])
    walked = walk_parsed_graph(parsed)
    if walked["action_text"] != expected_answer:
        raise ValueError(
            f"independent oracle mismatch on {case_id}: walked={walked['action_text']!r} != expected={expected_answer!r}"
        )

    inst_sha = _instantiated_state_digest(
        rows=rows,
        query_subject=query_subject,
        hops=hops,
        query_available=query_available,
        memory_available=memory_available,
        is_literal_copy=is_literal_copy,
    )
    return AvendiCase(
        case_id=case_id,
        family_id=family_id,
        split=split,
        panel=panel,
        variant=variant,
        cell_key=cell_key,
        hops=hops,
        query_subject=query_subject,
        swapped=swapped,
        query_available=query_available,
        memory_available=memory_available,
        template=template,
        public_prompt=prompt,
        prompt_tokens=enc["prompt_tokens"],
        target_tokens=enc["target_tokens"],
        full_tokens=enc["full_tokens"],
        target_boundary_index=enc["target_boundary_index"],
        prompt_token_sha256=enc["prompt_token_sha256"],
        prompt_utf8_sha256=enc["prompt_utf8_sha256"],
        instantiated_state_sha256=inst_sha,
        canonical_family_key=canonical_family_key,
        expected_answer=expected_answer,
        rival_answer=rival_answer,
        expected_operation=walked["operation"],
        proof_path=tuple(walked["proof_path"]),
        endpoint_g=endpoint_g,
        endpoint_h=endpoint_h,
        bridge_subjects=bridge_subjects,
    )


def _build_crossover_family(
    *,
    family_id: str,
    split: str,
    panel: str,
    variant: str,
    hops: int,
    labels: Sequence[str],
    order: Sequence[int],
    ticks: Sequence[int],
    sources: Sequence[str],
    template: int = 0,
    alias_map: Mapping[str, str] | None = None,
    distractor_edges: Sequence[tuple[str, str, int, str]] = (),
    distractor_insert_positions: Sequence[int] = (),
) -> AvendiFamily:
    """Construct a 4-cell crossover family (1, 2, 3, or 4 hops) with fixed nuisance parameters across cells."""
    if hops == 1:
        a, b, g, h = labels[:4]
        bridge_subjs = (a, b)
        can_key = f"1hop:{sorted((a, b))}:{sorted((g, h))}"

        def _edges(swapped: bool) -> list[tuple[str, str]]:
            return [(a, h if swapped else g), (b, g if swapped else h)]

    elif hops == 2:
        a, b, c, d, g, h = labels[:6]
        bridge_subjs = (c, d)
        can_key = f"2hop:{sorted(((a, c), (b, d)))}:{sorted((g, h))}"

        def _edges(swapped: bool) -> list[tuple[str, str]]:
            return [(a, c), (b, d), (c, h if swapped else g), (d, g if swapped else h)]

    elif hops == 3:
        a, b, c, d, e, f, g, h = labels[:8]
        bridge_subjs = (c, d)
        can_key = f"3hop:{sorted(((a, c), (b, d)))}:{sorted(((e, g), (f, h)))}"

        def _edges(swapped: bool) -> list[tuple[str, str]]:
            return [
                (a, c),
                (b, d),
                (c, f if swapped else e),
                (d, e if swapped else f),
                (e, g),
                (f, h),
            ]

    elif hops == 4:
        a, b, c, d, e, f, i_node, j_node, g, h = labels[:10]
        bridge_subjs = (c, d)
        can_key = f"4hop:{sorted(((a, c), (b, d)))}:{sorted(((e, i_node, g), (f, j_node, h)))}"

        def _edges(swapped: bool) -> list[tuple[str, str]]:
            return [
                (a, c),
                (b, d),
                (c, f if swapped else e),
                (d, e if swapped else f),
                (e, i_node),
                (f, j_node),
                (i_node, g),
                (j_node, h),
            ]
    else:
        raise ValueError(f"unsupported crossover hop count: {hops}")

    # Graph-major order: (swapped=False, query=a)->G, (swapped=False, query=b)->H, (swapped=True, query=a)->H, (swapped=True, query=b)->G
    cell_specs = (
        ("00", False, a, g, h),
        ("10", False, b, h, g),
        ("01", True, a, h, g),
        ("11", True, b, g, h),
    )

    cells: list[AvendiCase] = []
    for cell_key, swapped, q_sym, exp_ans, riv_ans in cell_specs:
        base_edges = _edges(swapped)
        ordered_rows = [
            RenderedRow(
                subject=base_edges[idx][0],
                value=base_edges[idx][1],
                tick=int(ticks[idx]),
                source=str(sources[idx]),
            )
            for idx in order
        ]
        if distractor_edges:
            # Deterministically interleave distractor rows at fixed positions
            full_rows = list(ordered_rows)
            for d_idx, (ds, dv, dt, dsrc) in enumerate(distractor_edges):
                pos = distractor_insert_positions[d_idx] if d_idx < len(distractor_insert_positions) else len(full_rows)
                full_rows.insert(min(pos, len(full_rows)), RenderedRow(subject=ds, value=dv, tick=dt, source=dsrc))
            ordered_rows = full_rows

        cid = f"{family_id}__c{cell_key}"
        cells.append(
            _build_validated_case(
                case_id=cid,
                family_id=family_id,
                split=split,
                panel=panel,
                variant=variant,
                cell_key=cell_key,
                hops=hops,
                query_subject=q_sym,
                swapped=swapped,
                rows=ordered_rows,
                query_available=False,
                memory_available=False,
                template=template,
                alias_map=alias_map,
                is_literal_copy=False,
                expected_answer=exp_ans,
                rival_answer=riv_ans,
                endpoint_g=g,
                endpoint_h=h,
                bridge_subjects=bridge_subjs,
                canonical_family_key=can_key,
            )
        )

    # Verify all 4 observations within the family are distinct
    if len({c.prompt_token_sha256 for c in cells}) != 4:
        raise ValueError(f"family {family_id} did not produce 4 distinct tokenized observations")

    return AvendiFamily(
        family_id=family_id,
        split=split,
        panel=panel,
        variant=variant,
        hops=hops,
        canonical_family_key=can_key,
        query_a=a,
        query_b=b,
        endpoint_g=g,
        endpoint_h=h,
        bridge_subjects=bridge_subjs,
        cells=tuple(cells),
    )


def build_literal_copy_cases(
    split: str,
    *,
    seed: int,
    count: int = 256,
    forbidden_state_hashes: set[str] | None = None,
) -> tuple[AvendiCase, ...]:
    """Generate 256 literal copy control cases (16 answers A-P x 16 balanced contexts)."""
    if count % 16 != 0:
        raise ValueError("literal copy count must be a multiple of 16")
    rng = random.Random(derive_seed_int("avendi-literal-copy", split, seed))
    seen_states = forbidden_state_hashes if forbidden_state_hashes is not None else set()
    cases: list[AvendiCase] = []
    reps = count // 16
    idx = 0
    for rep in range(reps):
        syms = list(SYMBOLS)
        rng.shuffle(syms)
        for ans in syms:
            other_pool = [s for s in SYMBOLS if s != ans]
            reg_pos = (rep + idx) % 4
            for _attempt in range(128):
                picked_keys = rng.sample(other_pool, 4)
                q_sym = picked_keys[0]
                dist_keys = picked_keys[1:]
                rival = dist_keys[0]
                ticks = rng.sample(range(1000, 9000), 4)
                srcs = [rng.choice(SOURCES) for _ in range(4)]

                rows: list[RenderedRow] = []
                d_ptr = 0
                for r_i in range(4):
                    if r_i == reg_pos:
                        rows.append(RenderedRow(subject=q_sym, value=ans, tick=ticks[r_i], source=srcs[r_i]))
                    else:
                        rows.append(RenderedRow(subject=dist_keys[d_ptr], value=None, tick=ticks[r_i], source=srcs[r_i]))
                        d_ptr += 1

                cand_state_sha = _instantiated_state_digest(
                    rows=rows,
                    query_subject=q_sym,
                    hops=1,
                    query_available=False,
                    memory_available=False,
                    is_literal_copy=True,
                )
                if cand_state_sha not in seen_states:
                    seen_states.add(cand_state_sha)
                    break
            else:
                raise RuntimeError("failed to sample disjoint literal copy state")

            fid = f"AV-{split}-literal-{idx:04d}"
            cid = f"{fid}__literal"
            cases.append(
                _build_validated_case(
                    case_id=cid,
                    family_id=fid,
                    split=split,
                    panel="literal_copy",
                    variant="native",
                    cell_key="literal",
                    hops=1,
                    query_subject=q_sym,
                    swapped=False,
                    rows=rows,
                    query_available=False,
                    memory_available=False,
                    template=0,
                    alias_map=None,
                    is_literal_copy=True,
                    expected_answer=ans,
                    rival_answer=rival,
                    endpoint_g=ans,
                    endpoint_h=rival,
                    bridge_subjects=(q_sym,),
                    canonical_family_key=f"literal:{q_sym}->{ans}:{','.join(sorted(dist_keys))}",
                )
            )
            idx += 1
    return tuple(cases)


def _sample_multi_hop_families(
    split: str,
    *,
    hops: int,
    count: int,
    seed: int,
    panel_name: str,
    forbidden_canonical_keys: set[str],
) -> tuple[AvendiFamily, ...]:
    """Sample `count` disjoint crossover families at `hops` (2, 3, or 4) with balanced (g, h) pairs."""
    rng = random.Random(derive_seed_int("avendi-multihop", split, hops, seed))
    pair_cycle = list(ORDERED_UNEQUAL_PAIRS)
    rng.shuffle(pair_cycle)

    n_nodes = {2: 6, 3: 8, 4: 10}[hops]
    n_edges = {2: 4, 3: 6, 4: 8}[hops]
    families: list[AvendiFamily] = []

    for idx in range(count):
        g, h = pair_cycle[idx % len(pair_cycle)]
        rem_pool = [s for s in SYMBOLS if s not in (g, h)]
        for _attempt in range(256):
            chosen_others = rng.sample(rem_pool, n_nodes - 2)
            labels = (*chosen_others, g, h)
            order = rng.sample(range(n_edges), n_edges)
            ticks = rng.sample(range(1000, 9000), n_edges)
            sources = [rng.choice(SOURCES) for _ in range(n_edges)]
            fam = _build_crossover_family(
                family_id=f"AV-{split}-{panel_name}-{idx:04d}",
                split=split,
                panel=panel_name,
                variant="native",
                hops=hops,
                labels=labels,
                order=order,
                ticks=ticks,
                sources=sources,
                template=0,
            )
            if fam.canonical_family_key not in forbidden_canonical_keys:
                forbidden_canonical_keys.add(fam.canonical_family_key)
                families.append(fam)
                break
        else:
            raise RuntimeError(f"failed to sample unique {hops}-hop family for {split} at index {idx}")
    return tuple(families)


def _build_one_hop_families_from_allocation(
    split_key: str,
    *,
    split_label: str,
    seed: int,
    limit: int | None = None,
) -> tuple[AvendiFamily, ...]:
    specs = _ONE_HOP_ALLOCATIONS[split_key]
    if limit is not None:
        specs = specs[:limit]
    rng = random.Random(derive_seed_int("avendi-1hop-render", split_key, seed))
    families: list[AvendiFamily] = []
    for idx, (a, b, g, h) in enumerate(specs):
        order = [0, 1] if (idx % 2 == 0) else [1, 0]
        if rng.random() < 0.5:
            order = list(reversed(order))
        ticks = rng.sample(range(1000, 9000), 2)
        sources = [rng.choice(SOURCES), rng.choice(SOURCES)]
        families.append(
            _build_crossover_family(
                family_id=f"AV-{split_label}-one_hop-{idx:04d}",
                split=split_label,
                panel="one_hop",
                variant="native",
                hops=1,
                labels=(a, b, g, h),
                order=order,
                ticks=ticks,
                sources=sources,
                template=0,
            )
        )
    return tuple(families)


def _build_transfer_variants_for_families(
    base_families: Sequence[AvendiFamily],
    *,
    seed: int,
) -> dict[str, tuple[AvendiFamily, ...]]:
    """Build format_only, topology_only, joint_format_topology, and alias_bijection variants on the SAME families."""
    rng = random.Random(derive_seed_int("avendi-transfer-variants", seed))
    format_only: list[AvendiFamily] = []
    topology_only: list[AvendiFamily] = []
    joint: list[AvendiFamily] = []
    alias_bij: list[AvendiFamily] = []

    alias_tokens = [f"N{i:02d}" for i in range(1, 17)]

    for fam in base_families:
        # Re-extract labels/order/ticks/sources from cell 00 & cell 01 so nuisance stays fixed across the 4 cells
        c00 = fam.cell_map()["00"]
        parsed00 = parse_tokenized_public_observation(c00.prompt_tokens)
        used_syms = {r.subject for r in parsed00.rows} | {r.value for r in parsed00.rows if r.value is not None}
        unused = [s for s in SYMBOLS if s not in used_syms]
        dist_syms = rng.sample(unused, 3)
        dticks = rng.sample(range(1000, 9000), 2)
        dsrcs = [rng.choice(SOURCES), rng.choice(SOURCES)]
        distractor_edges = (
            (dist_syms[0], dist_syms[1], dticks[0], dsrcs[0]),
            (dist_syms[1], dist_syms[2], dticks[1], dsrcs[1]),
        )
        insert_pos = (1, 3) if fam.hops >= 2 else (1, 2)

        shuffled_aliases = list(alias_tokens)
        rng.shuffle(shuffled_aliases)
        amap = {sym: shuffled_aliases[i] for i, sym in enumerate(SYMBOLS)}

        # Re-render each of the 4 cells under the 4 transfer conditions
        for vname, tmpl, dedges, am in (
            ("format_only", 1, (), None),
            ("topology_only", 0, distractor_edges, None),
            ("joint_format_topology", 1, distractor_edges, None),
            ("alias_bijection", 0, (), amap),
        ):
            new_cells: list[AvendiCase] = []
            for cell in fam.cells:
                p_cell = parse_tokenized_public_observation(cell.prompt_tokens)
                base_rows = [
                    RenderedRow(subject=r.subject, value=r.value, tick=r.tick, source=r.source)
                    for r in p_cell.rows
                ]
                if dedges:
                    for d_i, (ds, dv, dt, dsrc) in enumerate(dedges):
                        pos = insert_pos[d_i] if d_i < len(insert_pos) else len(base_rows)
                        base_rows.insert(min(pos, len(base_rows)), RenderedRow(subject=ds, value=dv, tick=dt, source=dsrc))
                new_cells.append(
                    _build_validated_case(
                        case_id=f"{fam.family_id}__{vname}__c{cell.cell_key}",
                        family_id=f"{fam.family_id}__{vname}",
                        split=fam.split,
                        panel=f"{fam.panel}_{vname}",
                        variant=vname,
                        cell_key=cell.cell_key,
                        hops=fam.hops,
                        query_subject=cell.query_subject,
                        swapped=cell.swapped,
                        rows=base_rows,
                        query_available=False,
                        memory_available=False,
                        template=tmpl,
                        alias_map=am,
                        is_literal_copy=False,
                        expected_answer=cell.expected_answer,
                        rival_answer=cell.rival_answer,
                        endpoint_g=fam.endpoint_g,
                        endpoint_h=fam.endpoint_h,
                        bridge_subjects=fam.bridge_subjects,
                        canonical_family_key=fam.canonical_family_key,
                    )
                )
            vfam = replace(
                fam,
                family_id=f"{fam.family_id}__{vname}",
                panel=f"{fam.panel}_{vname}",
                variant=vname,
                cells=tuple(new_cells),
            )
            if vname == "format_only":
                format_only.append(vfam)
            elif vname == "topology_only":
                topology_only.append(vfam)
            elif vname == "joint_format_topology":
                joint.append(vfam)
            else:
                alias_bij.append(vfam)

    return {
        "format_only": tuple(format_only),
        "topology_only": tuple(topology_only),
        "joint_format_topology": tuple(joint),
        "alias_bijection": tuple(alias_bij),
    }


def build_avendi_suite(*, scale_factor: int = 1) -> dict[str, Any]:
    """Build complete P0/P2/P3/P4 datasets across dev, confirm, transfer, and train with custody hashes.

    ``scale_factor=1`` builds the exact frozen production counts (256 families/panel in dev/confirm/transfer,
    128 2-hop/4-hop families, 1,024 P3 train families). Larger ``scale_factor`` values divide counts for
    fast unit tests while preserving all structural/balance invariants.
    """
    sf = max(1, int(scale_factor))
    n_eval_fam = max(4, 256 // sf)
    n_lit = max(16, (256 // sf // 16) * 16)
    n_len_fam = max(4, 128 // sf)
    n_train_p3 = max(8, 1024 // sf)
    n_train_p4 = max(4, 512 // sf)

    forbidden_multihop: set[str] = set()
    forbidden_literal_states: set[str] = set()
    splits: dict[str, dict[str, Any]] = {}

    for split_name in ("dev", "confirm", "transfer"):
        s_seed = GENERATOR_SEEDS[split_name]
        lit_cases = build_literal_copy_cases(
            split_name,
            seed=s_seed,
            count=n_lit,
            forbidden_state_hashes=forbidden_literal_states,
        )
        one_hop_fams = _build_one_hop_families_from_allocation(
            split_name,
            split_label=split_name,
            seed=s_seed,
            limit=n_eval_fam,
        )
        three_hop_fams = _sample_multi_hop_families(
            split_name,
            hops=3,
            count=n_eval_fam,
            seed=s_seed,
            panel_name="three_hop",
            forbidden_canonical_keys=forbidden_multihop,
        )
        split_bundle: dict[str, Any] = {
            "split": split_name,
            "seed": s_seed,
            "literal_cases": lit_cases,
            "one_hop_families": one_hop_fams,
            "three_hop_families": three_hop_fams,
        }
        if split_name == "transfer":
            two_hop_fams = _sample_multi_hop_families(
                "transfer",
                hops=2,
                count=n_len_fam,
                seed=s_seed + 10,
                panel_name="length_two_hop",
                forbidden_canonical_keys=forbidden_multihop,
            )
            four_hop_fams = _sample_multi_hop_families(
                "transfer",
                hops=4,
                count=n_len_fam,
                seed=s_seed + 20,
                panel_name="length_four_hop",
                forbidden_canonical_keys=forbidden_multihop,
            )
            split_bundle["two_hop_families"] = two_hop_fams
            split_bundle["four_hop_families"] = four_hop_fams
            split_bundle["one_hop_transfer_variants"] = _build_transfer_variants_for_families(one_hop_fams, seed=s_seed + 30)
            split_bundle["three_hop_transfer_variants"] = _build_transfer_variants_for_families(three_hop_fams, seed=s_seed + 40)
        splits[split_name] = split_bundle

    train_p3_fams = _build_one_hop_families_from_allocation(
        "train_p3",
        split_label="train_p3",
        seed=GENERATOR_SEEDS["train"],
        limit=n_train_p3,
    )
    train_p4_1hop = _build_one_hop_families_from_allocation(
        "train_p4",
        split_label="train_p4",
        seed=GENERATOR_SEEDS["p4_train"],
        limit=n_train_p4,
    )
    train_p4_2hop = _sample_multi_hop_families(
        "train_p4",
        hops=2,
        count=n_train_p4,
        seed=GENERATOR_SEEDS["p4_train"],
        panel_name="two_hop",
        forbidden_canonical_keys=forbidden_multihop,
    )

    splits["train_p3"] = {
        "split": "train_p3",
        "seed": GENERATOR_SEEDS["train"],
        "one_hop_families": train_p3_fams,
    }
    splits["train_p4"] = {
        "split": "train_p4",
        "seed": GENERATOR_SEEDS["p4_train"],
        "one_hop_families": train_p4_1hop,
        "two_hop_families": train_p4_2hop,
    }

    return {
        "schema": "anra-avendi-v8-y2-suite-v1",
        "design_sha256": DESIGN_SHA,
        "scale_factor": sf,
        "canonical_one_hop_pool_size": len(_CANONICAL_ONE_HOP_POOL),
        "splits": splits,
    }


def iter_split_cases(split_bundle: Mapping[str, Any], *, include_transfer_variants: bool = True) -> list[AvendiCase]:
    """Flatten all cases inside a split bundle."""
    cases: list[AvendiCase] = []
    if "literal_cases" in split_bundle:
        cases.extend(split_bundle["literal_cases"])
    for k in ("one_hop_families", "two_hop_families", "three_hop_families", "four_hop_families"):
        for fam in split_bundle.get(k, ()):
            cases.extend(fam.cells)
    if include_transfer_variants:
        for vk in ("one_hop_transfer_variants", "three_hop_transfer_variants"):
            vmap = split_bundle.get(vk, {})
            for fams in vmap.values():
                for fam in fams:
                    cases.extend(fam.cells)
    return cases


def audit_avendi_custody_and_overlap(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Verify prompt-only token hashes (at target boundary), instantiated state hashes, and family keys are disjoint across splits."""
    splits = suite["splits"]
    prompt_hashes_by_split: dict[str, set[str]] = {}
    state_hashes_by_split: dict[str, set[str]] = {}
    canonical_keys_by_split: dict[str, set[str]] = {}
    parent_split_by_family: dict[str, str] = {}
    defects: list[dict[str, Any]] = []

    for sname, sbundle in splits.items():
        p_set: set[str] = set()
        st_set: set[str] = set()
        ck_set: set[str] = set()
        for c in iter_split_cases(sbundle, include_transfer_variants=True):
            # Re-verify prompt-only prefix extraction at target_boundary_index
            prefix = extract_prompt_prefix_from_framed_tokens(c.full_tokens, c.target_boundary_index)
            re_sha = digest(list(prefix))
            if re_sha != c.prompt_token_sha256:
                defects.append({"type": "TARGET_BOUNDARY_HASH_MISMATCH", "case_id": c.case_id})
            p_set.add(re_sha)
            st_set.add(c.instantiated_state_sha256)
            ck_set.add(c.canonical_family_key)
            base_fid = c.family_id.split("__")[0]
            if base_fid in parent_split_by_family and parent_split_by_family[base_fid] != sname:
                defects.append({
                    "type": "FAMILY_VIEW_SPLIT_LEAK",
                    "family_id": base_fid,
                    "splits": [parent_split_by_family[base_fid], sname],
                })
            parent_split_by_family[base_fid] = sname
        prompt_hashes_by_split[sname] = p_set
        state_hashes_by_split[sname] = st_set
        canonical_keys_by_split[sname] = ck_set

    split_names = sorted(splits)
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            p_ov = prompt_hashes_by_split[s1] & prompt_hashes_by_split[s2]
            st_ov = state_hashes_by_split[s1] & state_hashes_by_split[s2]
            ck_ov = canonical_keys_by_split[s1] & canonical_keys_by_split[s2]
            if p_ov:
                defects.append({"type": "PROMPT_PREFIX_OVERLAP", "splits": [s1, s2], "count": len(p_ov)})
            if st_ov:
                defects.append({"type": "INSTANTIATED_STATE_OVERLAP", "splits": [s1, s2], "count": len(st_ov)})
            if ck_ov:
                defects.append({"type": "CANONICAL_FAMILY_OVERLAP", "splits": [s1, s2], "count": len(ck_ov)})

    return {
        "schema": "anra-avendi-v8-y2-custody-audit-v1",
        "status": "PASS" if not defects else "FAIL",
        "canonical_one_hop_pool_size": suite["canonical_one_hop_pool_size"],
        "prompt_counts_by_split": {k: len(v) for k, v in prompt_hashes_by_split.items()},
        "state_counts_by_split": {k: len(v) for k, v in state_hashes_by_split.items()},
        "canonical_family_counts_by_split": {k: len(v) for k, v in canonical_keys_by_split.items()},
        "defects": defects,
    }


def plant_custody_collision_negative_control(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Plant a duplicate prompt with changed target + changed ID and verify custody check catches both prompt and state collisions."""
    mutated = {**suite, "splits": dict(suite["splits"])}
    dev_one_hop = suite["splits"]["dev"]["one_hop_families"][0]
    victim_cell = dev_one_hop.cells[0]

    # Change target token (to rival_answer) and example/family ID while keeping identical public prompt!
    wrong_enc = encode_prompt_and_target(victim_cell.public_prompt, victim_cell.rival_answer)
    planted_case = replace(
        victim_cell,
        case_id="PLANTED-COLLISION-CASE-9999",
        family_id="PLANTED-COLLISION-FAM-9999",
        split="train_p3",
        target_tokens=wrong_enc["target_tokens"],
        full_tokens=wrong_enc["full_tokens"],
        expected_answer=victim_cell.rival_answer,
    )
    planted_fam = replace(
        dev_one_hop,
        family_id="PLANTED-COLLISION-FAM-9999",
        split="train_p3",
        cells=(planted_case, *dev_one_hop.cells[1:]),
    )
    orig_train = suite["splits"]["train_p3"]
    mutated["splits"]["train_p3"] = {
        **orig_train,
        "one_hop_families": (*orig_train["one_hop_families"], planted_fam),
    }
    audit_res = audit_avendi_custody_and_overlap(mutated)
    defect_types = {d["type"] for d in audit_res["defects"]}
    caught_both = (
        audit_res["status"] == "FAIL"
        and "PROMPT_PREFIX_OVERLAP" in defect_types
        and "INSTANTIATED_STATE_OVERLAP" in defect_types
    )
    return {
        "status": "PASS" if caught_both else "FAIL",
        "caught_prompt_prefix_overlap": "PROMPT_PREFIX_OVERLAP" in defect_types,
        "caught_instantiated_state_overlap": "INSTANTIATED_STATE_OVERLAP" in defect_types,
        "defects_detected": audit_res["defects"],
    }


def run_destructive_input_checks(families: Sequence[AvendiFamily]) -> dict[str, Any]:
    """Verify erasing goal token or masking bridge edges makes pairs render identically with conflicting targets."""
    goal_erasure_identical_pairs = 0
    goal_erasure_conflicting_targets = 0
    bridge_mask_identical_pairs = 0
    bridge_mask_conflicting_targets = 0
    total_pairs = 0

    for fam in families:
        cmap = fam.cell_map()
        # 1. Erase goal token across (00 vs 10) and (01 vs 11)
        for k_a, k_b in (("00", "10"), ("01", "11")):
            ca, cb = cmap[k_a], cmap[k_b]
            total_pairs += 1
            ea = erase_goal_token(ca.public_prompt)
            eb = erase_goal_token(cb.public_prompt)
            if ea == eb:
                goal_erasure_identical_pairs += 1
            if ca.expected_answer != cb.expected_answer:
                goal_erasure_conflicting_targets += 1

        # 2. Mask bridge edges across rewires (00 vs 01) and (10 vs 11)
        for k_orig, k_swap in (("00", "01"), ("10", "11")):
            co, cs = cmap[k_orig], cmap[k_swap]
            mo = mask_bridge_edges_in_prompt(co.public_prompt, fam.bridge_subjects)
            ms = mask_bridge_edges_in_prompt(cs.public_prompt, fam.bridge_subjects)
            if mo == ms:
                bridge_mask_identical_pairs += 1
            if co.expected_answer != cs.expected_answer:
                bridge_mask_conflicting_targets += 1

    passed = (
        total_pairs > 0
        and goal_erasure_identical_pairs == total_pairs
        and goal_erasure_conflicting_targets == total_pairs
        and bridge_mask_identical_pairs == total_pairs
        and bridge_mask_conflicting_targets == total_pairs
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "families_checked": len(families),
        "pairs_checked": total_pairs,
        "goal_erasure_identical_pairs": goal_erasure_identical_pairs,
        "goal_erasure_conflicting_targets": goal_erasure_conflicting_targets,
        "goal_erasure_deterministic_all_four_ceiling": 0,
        "bridge_mask_identical_pairs": bridge_mask_identical_pairs,
        "bridge_mask_conflicting_targets": bridge_mask_conflicting_targets,
        "bridge_mask_deterministic_all_four_ceiling": 0,
    }


def evaluate_policies_on_families(families: Sequence[AvendiFamily]) -> dict[str, Any]:
    """Evaluate all 8 reference/shortcut policies across a sequence of 4-cell families."""
    policy_names = (
        "public_traversal",
        "copied_terminal",
        "last_row",
        "rank_h_timestamp",
        "always_abstain",
        "empirical_majority",
        "query_blind",
        "bridge_blind",
    )
    case_hits: Counter[str] = Counter()
    family_hits: Counter[str] = Counter()

    for fam in families:
        f_counts: Counter[str] = Counter()
        for cell in fam.cells:
            preds = evaluate_reference_policies_on_prompt(cell.prompt_tokens)
            for pname in policy_names:
                hit = int(preds[pname] == cell.expected_answer)
                case_hits[pname] += hit
                f_counts[pname] += hit
        for pname in policy_names:
            family_hits[pname] += int(f_counts[pname] == 4)

    n_fams = max(1, len(families))
    n_cases = max(1, 4 * len(families))
    return {
        "families": len(families),
        "cases": 4 * len(families),
        "case_hits": {p: int(case_hits[p]) for p in policy_names},
        "case_accuracy": {p: case_hits[p] / n_cases for p in policy_names},
        "complete_families": {p: int(family_hits[p]) for p in policy_names},
        "complete_family_rate": {p: family_hits[p] / n_fams for p in policy_names},
    }


def verify_p0_instrument(*, scale_factor: int = 1) -> dict[str, Any]:
    """Run complete P0 CPU instrument qualification: custody, negative control, destructive checks, pilot, and policies."""
    suite = build_avendi_suite(scale_factor=scale_factor)
    custody = audit_avendi_custody_and_overlap(suite)
    planted_neg = plant_custody_collision_negative_control(suite)
    pilot_ref = run_pilot_reference_check(seed=GENERATOR_SEEDS["pilot_reference"], families=256)

    dev = suite["splits"]["dev"]
    dev_1hop_pol = evaluate_policies_on_families(dev["one_hop_families"])
    dev_3hop_pol = evaluate_policies_on_families(dev["three_hop_families"])
    dest_1hop = run_destructive_input_checks(dev["one_hop_families"])
    dest_3hop = run_destructive_input_checks(dev["three_hop_families"])

    # Verify balanced (g, h) endpoint pairs
    pair_counts_3hop = Counter((f.endpoint_g, f.endpoint_h) for f in dev["three_hop_families"])
    max_pair_count = max(pair_counts_3hop.values()) if pair_counts_3hop else 0
    min_pair_count = min(pair_counts_3hop.values()) if pair_counts_3hop else 0

    blind_policies = ("last_row", "rank_h_timestamp", "always_abstain", "empirical_majority", "query_blind", "bridge_blind")
    blind_zero_ok = all(
        dev_1hop_pol["complete_families"][p] == 0 and dev_3hop_pol["complete_families"][p] == 0
        for p in blind_policies
    )
    oracle_100_ok = (
        dev_1hop_pol["complete_families"]["public_traversal"] == len(dev["one_hop_families"])
        and dev_3hop_pol["complete_families"]["public_traversal"] == len(dev["three_hop_families"])
    )
    pilot_ok = (
        pilot_ref["all_four_correct"]["traversal"] == 256
        and pilot_ref["all_four_correct"]["last_object"] == 0
        and pilot_ref["all_four_correct"]["third_timestamp"] == 0
        and pilot_ref["all_four_correct"]["ignore_query_oracle_elsewhere"] == 0
        and pilot_ref["all_four_correct"]["ignore_bridge_oracle_elsewhere"] == 0
    )

    all_ok = (
        custody["status"] == "PASS"
        and planted_neg["status"] == "PASS"
        and dest_1hop["status"] == "PASS"
        and dest_3hop["status"] == "PASS"
        and oracle_100_ok
        and blind_zero_ok
        and pilot_ok
    )
    return {
        "schema": "anra-avendi-v8-y2-p0-receipt-v1",
        "status": "QUALIFIED" if all_ok else "ENGINEERING_FAILURE",
        "verification_row_status": "PASS" if all_ok else "FAIL",
        "design_sha256": DESIGN_SHA,
        "scale_factor": scale_factor,
        "custody_audit": custody,
        "planted_collision_negative_control": planted_neg,
        "pilot_reference_1801": pilot_ref,
        "dev_one_hop_policies": dev_1hop_pol,
        "dev_three_hop_policies": dev_3hop_pol,
        "destructive_checks": {
            "one_hop": dest_1hop,
            "three_hop": dest_3hop,
        },
        "endpoint_pair_balance_dev_3hop": {
            "distinct_ordered_pairs": len(pair_counts_3hop),
            "min_count": min_pair_count,
            "max_count": max_pair_count,
        },
    }
