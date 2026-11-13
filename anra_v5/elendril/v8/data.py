"""V8 dataset construction (D0 preservation, coverage-controlled D1, dev cohorts, and truth firewall)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_data import make_dataset as make_v6_dataset

from .contracts import OBSERVATION_VERSION, SCREEN_MAX_SEQUENCE, SYMBOLS, PublicInput
from .holdout import audit_split_disjointness, canonical_graph_signature
from .teacher import (
    PublicFact,
    TeacherStep,
    audit_teacher_samples,
    build_teacher_steps_for_world,
    compute_shortcut_baselines,
)
from .verifier import independent_expected_outcome, verify_episode_trace

D0_EXPECTED_IDENTITY = "7786244187963f75e6cda5b10e4d4e0d81927b7000154126fff538dea15539e5"
D1_FAMILIES = (
    "direct_one_hop",
    "two_hop",
    "temporal_source",
    "missing_query",
    "conflict_query",
    "memory_retrieval",
    "null_unavailable",
)
D1_WORLD_QUOTA_PER_FAMILY = 512
D1_TRAIN_SEED = 80801
D1_DEV_SEED = 80802


def load_d0_dataset():
    """Load exact inherited V7.17 D0 training dataset (9472 samples, seed 606)."""

    dataset = make_v6_dataset("train", seed=606, target_bytes=2200000)
    if dataset.identity != D0_EXPECTED_IDENTITY or len(dataset.samples) != 9472:
        raise ValueError(
            f"D0 identity mismatch: got {dataset.identity} ({len(dataset.samples)} samples), "
            f"expected {D0_EXPECTED_IDENTITY} (9472)"
        )
    return dataset


def render_public_v8(
    *,
    ordered_facts: tuple[PublicFact, ...],
    goal_subject: str,
    hops: int,
    query_available: bool,
    memory_available: bool,
    template_id: int,
    alias_prefix: str,
    order_seed: int,
    decision: int,
) -> str:
    """Render public observation only; never includes world_id, family, split, or truth."""

    if template_id not in (0, 1, 2):
        raise ValueError(f"unsupported train/dev template_id: {template_id!r}")
    if alias_prefix not in ("", "tr-", "dev-"):
        raise ValueError(f"unsupported train/dev alias_prefix: {alias_prefix!r}")
    entity = lambda sym: alias_prefix + sym
    instructions = (
        "Latest time per source counts; disagreement needs evidence. "
        "Output A-P, Q=query, !=abstain, R=memory.\n"
    )
    if hops <= 1:
        question = f"Current value of {entity(goal_subject)}?"
    else:
        question = f"Follow {hops} links from {entity(goal_subject)}."
    if alias_prefix:
        question += " Output the letter after the alias."
    header = f"{question} Query={int(query_available)} Memory={int(memory_available)}\n"

    facts_list = list(ordered_facts)
    if order_seed != 0:
        rng = random.Random(int(digest(["v8-row-order", order_seed, decision]), 16))
        rng.shuffle(facts_list)

    rows = []
    for fact in facts_list:
        subj = entity(fact.subject)
        val = "null" if fact.value is None else entity(fact.value)
        if template_id == 0:
            rows.append(f"{subj}={val}@{fact.tick}/{fact.source}")
        elif template_id == 1:
            rows.append(f"At {fact.tick} {fact.source} says {subj}:{val}")
        else:
            rows.append(f"({fact.source},{fact.tick},{subj},{val})")
    prompt = instructions + header + "; ".join(rows) + "\nAction:"
    if len(prompt.encode("utf-8")) + 5 > SCREEN_MAX_SEQUENCE:
        raise ValueError("V8 public prompt exceeds 512-byte screen limit")
    return prompt


def render_world_initial_public_input(world: dict[str, object], *, render_fn=render_public_v8) -> PublicInput:
    """Redact a world record to its actor-facing initial PublicInput."""

    facts = tuple(PublicFact(**row) for row in world["initial_facts"])
    text = render_fn(
        ordered_facts=facts,
        goal_subject=str(world["goal_subject"]),
        hops=int(world["hops"]),
        query_available=bool(world["allowed_public_tools"]["query_available"]),
        memory_available=bool(world["allowed_public_tools"]["memory_available"]),
        template_id=int(world["renderer_template"]),
        alias_prefix=str(world["alias_prefix"]),
        order_seed=int(world["row_order_seed"]),
        decision=0,
    )
    return PublicInput(
        text_utf8=text,
        query_available=bool(world["allowed_public_tools"]["query_available"]),
        memory_available=bool(world["allowed_public_tools"]["memory_available"]),
        action_budget=4,
        observation_version=OBSERVATION_VERSION,
    )


class ActorWorldView:
    """Truth-firewalled actor view exposing only PublicInput and public tool replies."""

    def __init__(self, world: dict[str, object], *, render_fn=render_public_v8) -> None:
        self._initial = render_world_initial_public_input(world, render_fn=render_fn)

    @property
    def initial_public_input(self) -> PublicInput:
        return self._initial


def _fact_dict(subject: str, source: str, tick: int, value: str | None) -> dict[str, object]:
    return {"subject": subject, "source": source, "tick": int(tick), "value": value}


def _make_distractors(
    rng: random.Random,
    *,
    forbidden_subjects: set[str],
    count: int,
    tick_pool: list[int],
    palette: list[str],
) -> list[dict[str, object]]:
    if count == 0:
        return []
    avail = [s for s in palette if s not in forbidden_subjects]
    rows = []
    for i in range(count):
        s = avail[i % len(avail)]
        v = rng.choice(palette)
        src = rng.choice(("s0", "s1", "map", "log"))
        t = tick_pool[i % len(tick_pool)]
        rows.append(_fact_dict(s, src, t, v))
    return rows


def _construct_world(
    *,
    rng: random.Random,
    world_id: str,
    split: str,
    family: str,
    scenario: str,
    target_symbol: str,
    template_id: int,
    alias_prefix: str,
    row_order_seed: int,
    tool_combo_idx: int,
    null_subkind: str | None = None,
    distractor_count: int | None = None,
    pair_id: str = "",
    parent_world_id: str = "",
    variant: str = "canonical",
) -> dict[str, object]:
    other_syms = [s for s in SYMBOLS if s != target_symbol]
    picked = rng.sample(other_syms, 9)
    subj, mid1, irr1, irr2, irr3, alt_v, alt_v2, spare, extra = picked
    palette = [target_symbol, *picked]  # 10 entities, leaving room within 4..12 even after CFs
    # Sample independent ticks across a wide range so highest tick never leaks answer
    ticks = rng.sample(range(50, 950), 12)
    n_dist = rng.choice((0, 2, 4)) if distractor_count is None else distractor_count
    forbidden = {subj, mid1}
    distractors = _make_distractors(
        rng, forbidden_subjects=forbidden, count=n_dist, tick_pool=ticks[4:], palette=palette
    )

    # Independent availability flags (tools vary across answerable/unanswerable worlds)
    q_avail = bool(tool_combo_idx & 1)
    m_avail = bool((tool_combo_idx >> 1) & 1)
    query_payload: list[dict[str, object]] = []
    memory_payload: list[dict[str, object]] = []
    initial_facts: list[dict[str, object]] = []
    hops = 1
    expected: str | None = target_symbol

    if family == "direct_one_hop":
        src = rng.choice(("s0", "s1", "map"))
        initial_facts = [_fact_dict(subj, src, ticks[0], target_symbol)] + distractors
        if q_avail:
            query_payload = [_fact_dict(irr1, "audit", ticks[1], irr2)]
        if m_avail:
            memory_payload = [_fact_dict(irr3, "mem", ticks[2], spare)]
    elif family == "two_hop":
        hops = 2
        src1, src2 = rng.choice(("map", "s0")), rng.choice(("map", "s1"))
        initial_facts = [
            _fact_dict(subj, src1, ticks[0], mid1),
            _fact_dict(mid1, src2, ticks[1], target_symbol),
        ] + distractors
        if q_avail:
            query_payload = [_fact_dict(irr1, "audit", ticks[2], irr2)]
        if m_avail:
            memory_payload = [_fact_dict(irr3, "mem", ticks[3], spare)]
    elif family == "temporal_source":
        src = rng.choice(("s0", "s1", "log"))
        t_early, t_late = sorted((ticks[0], ticks[1]))
        initial_facts = [
            _fact_dict(subj, src, t_early, alt_v),
            _fact_dict(subj, src, t_late, target_symbol),
        ] + distractors
        if q_avail:
            query_payload = [_fact_dict(irr1, "audit", ticks[2], irr2)]
        if m_avail:
            memory_payload = [_fact_dict(irr3, "mem", ticks[3], spare)]
    elif family == "missing_query":
        q_avail = True
        initial_facts = list(distractors)
        if not initial_facts:
            initial_facts = [_fact_dict(irr1, "map", ticks[0], irr2)]
        query_payload = [_fact_dict(subj, "s0", ticks[1], target_symbol)]
        if m_avail:
            # Irrelevant memory when memory_available=True so actor checks memory then queries
            memory_payload = [_fact_dict(irr3, "mem", ticks[2], spare)]
    elif family == "conflict_query":
        q_avail = True
        initial_facts = [
            _fact_dict(subj, "s0", ticks[0], alt_v),
            _fact_dict(subj, "s1", ticks[1], alt_v2),
        ] + distractors
        t_res = max(ticks[0], ticks[1]) + 10
        query_payload = [
            _fact_dict(subj, "s0", t_res, target_symbol),
            _fact_dict(subj, "s1", t_res, target_symbol),
        ]
        if m_avail:
            memory_payload = [_fact_dict(irr3, "mem", ticks[2], spare)]
    elif family == "memory_retrieval":
        hops = 2
        m_avail = True
        initial_facts = [_fact_dict(subj, "map", ticks[0], mid1)] + distractors
        memory_payload = [_fact_dict(mid1, "map", ticks[1], target_symbol)]
        if q_avail:
            query_payload = [_fact_dict(irr1, "audit", ticks[2], irr2)]
    elif family == "null_unavailable":
        expected = None
        if null_subkind == "null":
            src = rng.choice(("s0", "s1"))
            initial_facts = [_fact_dict(subj, src, ticks[0], None)] + distractors
            if q_avail:
                query_payload = [_fact_dict(irr1, "audit", ticks[1], irr2)]
            if m_avail:
                memory_payload = [_fact_dict(irr3, "mem", ticks[2], spare)]
        else:  # unavailable
            initial_facts = list(distractors)
            if not initial_facts:
                initial_facts = [_fact_dict(irr1, "map", ticks[0], irr2)]
            # Half have no tools; half have irrelevant tools (testing safe abstention after futile query/mem)
            if q_avail:
                query_payload = [_fact_dict(irr1, "audit", ticks[1], irr2)]
            if m_avail:
                memory_payload = [_fact_dict(irr3, "mem", ticks[2], spare)]
    else:
        raise ValueError(f"unknown D1 family: {family!r}")

    # Independent initial row shuffle
    rng.shuffle(initial_facts)
    world = {
        "world_id": world_id,
        "parent_world_id": parent_world_id or world_id,
        "pair_id": pair_id,
        "split": split,
        "family": family,
        "scenario": scenario,
        "variant": variant,
        "hops": hops,
        "goal_subject": subj,
        "initial_facts": initial_facts,
        "allowed_public_tools": {"query_available": q_avail, "memory_available": m_avail},
        "hidden": {
            "expected": expected,
            "query_payload": query_payload,
            "memory_payload": memory_payload,
        },
        "renderer_id": f"v8-renderer-template-{template_id}",
        "renderer_template": template_id,
        "alias_prefix": alias_prefix,
        "row_order_seed": row_order_seed,
        "target_provenance": "independent_expected_outcome_and_public_teacher",
    }
    sig = canonical_graph_signature(world)
    world["semantic_hash"] = sig["structure_hash"]
    world["canonical_hash"] = sig["canonical_hash"]
    world["is_isolated_one_edge_probe"] = sig["is_isolated_one_edge_probe"]
    world["content_hash"] = digest(world)
    # Cross-check independent verifier outcome matches intended expected
    if independent_expected_outcome(world) != expected:
        raise AssertionError(
            f"world {world_id} ({family}/{null_subkind}) failed independent expected check: "
            f"got {independent_expected_outcome(world)!r}, wanted {expected!r}"
        )
    return world


def _crosstabs(worlds: list[dict[str, object]], steps: list[TeacherStep]) -> dict[str, object]:
    by_wid = defaultdict(list)
    for st in steps:
        by_wid[st.world_id].append(st)
    table = Counter()
    for w in worlds:
        wid = str(w["world_id"])
        st0 = by_wid[wid][0]
        avail = f"Q{int(st0.public_input.query_available)}M{int(st0.public_input.memory_available)}"
        length_bin = f"b{(st0.prompt_bytes // 64) * 64}"
        table[f"{w['family']}|{st0.gold_op}|{avail}|{length_bin}"] += 1
    return dict(sorted(table.items()))


@dataclass(frozen=True)
class V8Dataset:
    split: str
    identity: str
    worlds: tuple[dict[str, object], ...]
    steps: tuple[TeacherStep, ...]
    audit: dict[str, object]
    shortcuts: dict[str, object]
    crosstabs: dict[str, object]
    structural_hashes: frozenset[str]


def build_d1_training_dataset(
    *,
    seed: int = D1_TRAIN_SEED,
    quota_per_family: int = D1_WORLD_QUOTA_PER_FAMILY,
) -> V8Dataset:
    """Build the D1 coverage-controlled training corpus (default 3584 worlds = 512 x 7)."""

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    if type(quota_per_family) is not int or quota_per_family < 16:
        raise ValueError("quota_per_family must be >= 16")

    rng = random.Random(int(digest(["v8-d1-train-v1", seed]), 16))
    worlds: list[dict[str, object]] = []
    steps: list[TeacherStep] = []

    for fam_idx, family in enumerate(D1_FAMILIES):
        # Permute target symbols deterministically per family for balanced <=1.1 ratio
        sym_order = list(SYMBOLS)
        rng.shuffle(sym_order)
        for i in range(quota_per_family):
            target_sym = sym_order[i % len(sym_order)]
            null_sub = None
            if family == "null_unavailable":
                null_sub = "null" if i < quota_per_family // 2 else "unavailable"
            template_id = (i + fam_idx) % 2  # Train uses templates 0 and 1
            alias_prefix = "tr-" if (i % 2 == 1) else ""
            row_order_seed = (i + 1) * 17 + fam_idx
            tool_combo_idx = (i + fam_idx) % 4
            # Keep some isolated 1-edge probes in direct_one_hop, majority multi-entity
            dist_count = None
            if family == "direct_one_hop" and i % 3 == 0:
                dist_count = 0
            world_id = digest(["v8-d1-train-world", seed, family, i])
            world = _construct_world(
                rng=rng,
                world_id=world_id,
                split="train",
                family=family,
                scenario="D1_TRAIN",
                target_symbol=target_sym,
                template_id=template_id,
                alias_prefix=alias_prefix,
                row_order_seed=row_order_seed,
                tool_combo_idx=tool_combo_idx,
                null_subkind=null_sub,
                distractor_count=dist_count,
            )
            world_steps = build_teacher_steps_for_world(world, render_fn=render_public_v8)
            world["token_length"] = world_steps[0].prompt_bytes
            worlds.append(world)
            steps.extend(world_steps)

    teacher_audit = audit_teacher_samples(steps)
    ratio = teacher_audit["answer_max_min_ratio"]
    if ratio is not None and ratio > 1.1:
        raise ValueError(f"D1 training target balance ratio {ratio:.3f} exceeds 1.1 limit")
    split_audit = audit_split_disjointness(worlds, min_distinct_classes_per_split=min(32, quota_per_family))
    shortcuts = compute_shortcut_baselines(worlds)
    crosstabs = _crosstabs(worlds, steps)
    struct_hashes = frozenset(
        str(w["semantic_hash"]) for w in worlds if not w["is_isolated_one_edge_probe"]
    )
    dataset_identity = digest(
        {
            "schema": "elendril-v8-d1-dataset-v1",
            "seed": seed,
            "quota_per_family": quota_per_family,
            "worlds": [w["content_hash"] for w in worlds],
            "steps": [(s.task_id, s.public_input.text_utf8, s.target_text) for s in steps],
        }
    )
    return V8Dataset(
        split="train",
        identity=dataset_identity,
        worlds=tuple(worlds),
        steps=tuple(steps),
        audit={"teacher": teacher_audit, "split": split_audit},
        shortcuts=shortcuts,
        crosstabs=crosstabs,
        structural_hashes=struct_hashes,
    )


def make_positive_canary_cohort(d1_dataset: V8Dataset, *, worlds_target: int = 32) -> V8Dataset:
    """Select a fixed 32-world training-only cohort covering all 7 families and availability states."""

    if d1_dataset.split != "train":
        raise ValueError("positive canary cohort must come from training split")
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for w in d1_dataset.worlds:
        by_family[str(w["family"])].append(w)
    selected: list[dict[str, object]] = []
    # Select 4 or 5 per family (total 32 across 7 families: 5*4 + 4*3 = 32)
    quotas = {fam: (5 if idx < 4 else 4) for idx, fam in enumerate(D1_FAMILIES)}
    for fam in D1_FAMILIES:
        pool = by_family[fam]
        # Ensure null_unavailable includes both null and unavailable, and vary tool availability
        chosen = []
        seen_avail = set()
        if fam == "null_unavailable":
            nulls = [w for w in pool if any(f["value"] is None for f in w["initial_facts"])]
            unavails = [w for w in pool if not any(f["value"] is None for f in w["initial_facts"])]
            chosen.extend(nulls[:2])
            chosen.extend(unavails[:2])
        else:
            for w in pool:
                key = (bool(w["allowed_public_tools"]["query_available"]), bool(w["allowed_public_tools"]["memory_available"]))
                if key not in seen_avail:
                    chosen.append(w)
                    seen_avail.add(key)
                if len(chosen) >= quotas[fam]:
                    break
            for w in pool:
                if len(chosen) >= quotas[fam]:
                    break
                if w not in chosen:
                    chosen.append(w)
        selected.extend(chosen[: quotas[fam]])
    selected = selected[:worlds_target]
    selected_ids = {str(w["world_id"]) for w in selected}
    canary_steps = tuple(s for s in d1_dataset.steps if s.world_id in selected_ids)
    canary_id = digest(["v8-positive-canary-32", d1_dataset.identity, sorted(selected_ids)])
    return V8Dataset(
        split="train",
        identity=canary_id,
        worlds=tuple(selected),
        steps=canary_steps,
        audit={"teacher": audit_teacher_samples(canary_steps), "parent_identity": d1_dataset.identity},
        shortcuts=compute_shortcut_baselines(selected),
        crosstabs=_crosstabs(selected, list(canary_steps)),
        structural_hashes=d1_dataset.structural_hashes,
    )


def _linked_variants_for_world(
    base_world: dict[str, object],
    *,
    alt_symbol: str,
    irr_symbol: str,
    alias_prefix: str = "dev-",
) -> list[dict[str, object]]:
    base_id = str(base_world["world_id"])
    pair_id = str(base_world["pair_id"]) or digest(["dev-pair", base_id])
    base_clean = {**base_world, "pair_id": pair_id, "parent_world_id": base_id, "variant": "canonical"}
    init = [dict(r) for r in base_world["initial_facts"]]
    q_pay = [dict(r) for r in base_world["hidden"]["query_payload"]]
    m_pay = [dict(r) for r in base_world["hidden"]["memory_payload"]]
    subj = str(base_world["goal_subject"])

    def _clone(suffix: str, variant: str, new_init, new_q, new_m, new_exp, *, alias="", order_seed=0):
        w = {
            **base_clean,
            "world_id": f"{base_id}-{suffix}",
            "variant": variant,
            "initial_facts": new_init,
            "hidden": {"expected": new_exp, "query_payload": new_q, "memory_payload": new_m},
            "alias_prefix": alias,
            "row_order_seed": order_seed,
        }
        sig = canonical_graph_signature(w)
        w["semantic_hash"] = sig["structure_hash"]
        w["canonical_hash"] = sig["canonical_hash"]
        w["is_isolated_one_edge_probe"] = sig["is_isolated_one_edge_probe"]
        w["content_hash"] = digest(w)
        return w

    shuffle_w = _clone(
        "shuf",
        "row_shuffle",
        [{**r, "tick": r["tick"] + 13} for r in init],
        [{**r, "tick": r["tick"] + 13} for r in q_pay],
        [{**r, "tick": r["tick"] + 13} for r in m_pay],
        base_world["hidden"]["expected"],
        order_seed=97,
    )
    alias_w = _clone(
        "alias",
        "fresh_alias",
        init,
        q_pay,
        m_pay,
        base_world["hidden"]["expected"],
        alias=alias_prefix,
    )

    # Relevant counterfactual: flip the terminal target fact so answer flips to alt_symbol
    rel_init = [dict(r) for r in init]
    rel_q = [dict(r) for r in q_pay]
    rel_m = [dict(r) for r in m_pay]
    family = str(base_world["family"])
    old_exp = base_world["hidden"]["expected"]
    if family in ("direct_one_hop", "temporal_source"):
        max_t = max((r["tick"] for r in rel_init if r["subject"] == subj), default=0)
        for r in rel_init:
            if r["subject"] == subj and r["tick"] == max_t:
                r["value"] = alt_symbol
    elif family == "two_hop":
        hop1_val = next(r["value"] for r in rel_init if r["subject"] == subj)
        for r in rel_init:
            if r["subject"] == hop1_val:
                r["value"] = alt_symbol
    elif family == "missing_query":
        for r in rel_q:
            if r["subject"] == subj:
                r["value"] = alt_symbol
    elif family == "conflict_query":
        for r in rel_q:
            if r["subject"] == subj:
                r["value"] = alt_symbol
    elif family == "memory_retrieval":
        hop1_val = next(r["value"] for r in rel_init if r["subject"] == subj)
        for r in rel_m:
            if r["subject"] == hop1_val:
                r["value"] = alt_symbol
    else:  # null_unavailable -> make relevant observation directly answerable as alt_symbol
        replaced = False
        for r in rel_init:
            if r["subject"] == subj:
                r["value"] = alt_symbol
                replaced = True
        if not replaced:
            rel_init = [_fact_dict(subj, "s0", 500, alt_symbol)] + rel_init[:-1]
    rel_w = _clone("relcf", "relevant_cf", rel_init, rel_q, rel_m, alt_symbol)
    if independent_expected_outcome(rel_w) != alt_symbol:
        raise AssertionError(f"relevant_cf failed independent check for {base_id} ({family})")

    # Irrelevant counterfactual: flip a distractor edge (or add/replace a non-goal edge)
    irr_init = [dict(r) for r in init]
    changed_irr = False
    for r in irr_init:
        if r["subject"] != subj and r["value"] != old_exp:
            # Ensure we don't accidentally hit hop1 node in two_hop/memory
            if family in ("two_hop", "memory_retrieval"):
                hop1 = next((x["value"] for x in irr_init if x["subject"] == subj), None)
                if r["subject"] == hop1:
                    continue
            r["value"] = irr_symbol if r["value"] != irr_symbol else alt_symbol
            changed_irr = True
            break
    if not changed_irr:
        # Pick an unused symbol as irrelevant distractor replacement
        used = {r["subject"] for r in irr_init} | {subj}
        unused = next(s for s in SYMBOLS if s not in used)
        irr_init.append(_fact_dict(unused, "log", 777, irr_symbol))
    irr_w = _clone("irrcf", "irrelevant_cf", irr_init, q_pay, m_pay, old_exp)
    if independent_expected_outcome(irr_w) != old_exp:
        raise AssertionError(f"irrelevant_cf changed outcome for {base_id} ({family})")

    return [base_clean, shuffle_w, alias_w, rel_w, irr_w]


def build_d1_dev_suite(
    *,
    seed: int = D1_DEV_SEED,
    worlds_per_cohort: int = 128,
    excluded_structure_hashes: set[str] | frozenset[str] = frozenset(),
) -> dict[str, tuple[dict[str, object], ...]]:
    """Build primary development cohorts R09, R10, E13, E14, E15, A19 with 5 linked forms each."""

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    if type(worlds_per_cohort) is not int or worlds_per_cohort < 1:
        raise ValueError("worlds_per_cohort must be positive")

    cohort_spec = {
        "R09": "direct_one_hop",
        "R10": "two_hop",
        "E13": "temporal_source",
        "E14": "missing_query",
        "E15": "conflict_query",
        "A19": "null_unavailable",
    }
    rng = random.Random(int(digest(["v8-d1-dev-v1", seed]), 16))
    excluded = set(excluded_structure_hashes)
    cohorts: dict[str, list[dict[str, object]]] = {}

    for cohort_name, family in cohort_spec.items():
        rows: list[dict[str, object]] = []
        sym_order = list(SYMBOLS)
        rng.shuffle(sym_order)
        for idx in range(worlds_per_cohort):
            target_sym = sym_order[idx % len(sym_order)]
            null_sub = ("null" if idx % 2 == 0 else "unavailable") if family == "null_unavailable" else None
            for attempt in range(64):
                world_id = digest(["v8-d1-dev-world", seed, cohort_name, idx, attempt])
                pair_id = digest(["v8-d1-dev-pair", world_id])
                dist_count = rng.choice((2, 4))  # Always >=2 distractors for structural distinctness in dev
                w = _construct_world(
                    rng=rng,
                    world_id=world_id,
                    split="validation",
                    family=family,
                    scenario=cohort_name,
                    target_symbol=target_sym,
                    template_id=2,  # Held-out dev template
                    alias_prefix="",
                    row_order_seed=(idx + 1) * 23,
                    tool_combo_idx=(idx + attempt) % 4,
                    null_subkind=null_sub,
                    distractor_count=dist_count,
                    pair_id=pair_id,
                )
                if w["semantic_hash"] in excluded:
                    continue
                alt_sym = next(s for s in SYMBOLS if s != target_sym and s != w["goal_subject"])
                irr_sym = next(s for s in SYMBOLS if s not in (target_sym, alt_sym, w["goal_subject"]))
                linked = _linked_variants_for_world(w, alt_symbol=alt_sym, irr_symbol=irr_sym, alias_prefix="dev-")
                if any(v["semantic_hash"] in excluded for v in linked):
                    continue
                rows.extend(linked)
                break
            else:
                raise ValueError(f"could not sample non-excluded dev world for {cohort_name} idx={idx}")
        cohorts[cohort_name] = rows
    return {k: tuple(v) for k, v in cohorts.items()}


def write_v8_dataset_json(dataset: V8Dataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "elendril-v8-dataset-file-v1",
        "split": dataset.split,
        "identity": dataset.identity,
        "worlds": list(dataset.worlds),
        "audit": dataset.audit,
        "shortcuts": dataset.shortcuts,
        "crosstabs": dataset.crosstabs,
        "structural_hashes": sorted(dataset.structural_hashes),
    }
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


__all__ = [
    "ActorWorldView",
    "D0_EXPECTED_IDENTITY",
    "D1_DEV_SEED",
    "D1_FAMILIES",
    "D1_TRAIN_SEED",
    "D1_WORLD_QUOTA_PER_FAMILY",
    "V8Dataset",
    "build_d1_dev_suite",
    "build_d1_training_dataset",
    "load_d0_dataset",
    "make_positive_canary_cohort",
    "render_public_v8",
    "render_world_initial_public_input",
    "write_v8_dataset_json",
]
