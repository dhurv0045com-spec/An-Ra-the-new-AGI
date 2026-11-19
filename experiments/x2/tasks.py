"""Elendril V8-X2 tasks, Generator B, D1-H selective edit, T13 FST, T14 costly info, stateful kernel & policy."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
import math
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.v8.contracts import ACTION_19_LABELS, SYMBOLS
from anra_v5.elendril.v8.policies import _apply_mode_to_distribution
from anra_v5.planthir.contracts import SubgoalProbeSpec, UniversalAction, UniversalObservation

from .models import OP_TO_INDEX, X2ModelSpec, X2TypedModel

X2_OBSERVATION_VERSION = "elendril-x2-public-obs-v1"
X2_KERNEL_ID = "elendril-x2-stateful-kernel-v1"

T14_CHEAP_COST = 0.02
T14_EXPENSIVE_COST = 0.35
T14_STRATA = (
    "immediately_decidable",
    "one_useful_query",
    "one_useful_retrieve",
    "complementary_tools",
    "unresolved_after_exhaustion",
    "stale_conflict_revision",
)
T13_STATE_LABELS = ("A", "B", "C", "D")
T13_EVENTS = ("E0", "E1", "E2", "E3")
T13_DOSES = (0, 4, 16, 64)
T13_EPISODIC_CACHE_BUDGET_BYTES = 16 * 1024


# =============================================================================
# 1. Independent Generator B + Renderer B + Reference Interpreter (§5)
# =============================================================================

def interpret_facts_reference(
    facts: Sequence[Mapping[str, Any]],
    *,
    goal_subject: str,
    hops: int,
    min_epoch: int = 1,
) -> dict[str, Any]:
    """Pure independent reference interpreter over fact multisets with authority, epoch & ambiguity rules."""
    active: dict[str, list[Mapping[str, Any]]] = {}
    for row in facts:
        if int(row.get("epoch", 1)) < int(min_epoch):
            continue
        subj = str(row["subj"])
        active.setdefault(subj, []).append(row)

    resolved: dict[str, str | None] = {}
    ambiguous: set[str] = set()
    for subj, rows in active.items():
        # Highest authority wins; within highest authority, highest epoch wins; distinct ties -> ambiguous
        best_rank = max(int(r.get("authority", 1)) for r in rows)
        top_auth = [r for r in rows if int(r.get("authority", 1)) == best_rank]
        best_ep = max(int(r.get("epoch", 1)) for r in top_auth)
        top = [r for r in top_auth if int(r.get("epoch", 1)) == best_ep]
        vals = sorted({str(r["val"]) for r in top})
        if len(vals) == 1:
            resolved[subj] = vals[0]
        else:
            resolved[subj] = None
            ambiguous.add(subj)

    cur = str(goal_subject)
    path: list[str] = [cur]
    for _ in range(int(hops)):
        if cur in ambiguous:
            return {"status": "AMBIGUOUS", "answer": None, "path": path, "missing_node": cur}
        nxt = resolved.get(cur)
        if nxt is None:
            return {"status": "MISSING", "answer": None, "path": path, "missing_node": cur}
        cur = nxt
        path.append(cur)
    return {"status": "SUPPORTED", "answer": cur, "path": path, "missing_node": None}


def render_x2_factorial(
    *,
    facts: Sequence[Mapping[str, Any]],
    goal_subject: str,
    hops: int,
    min_epoch: int = 1,
    fact_style: str = "canonical",
    header_style: str = "canonical",
    query_available: bool = False,
    retrieve_available: bool = False,
    query_cost: float = 0.0,
    retrieve_cost: float = 0.0,
    budget_remaining: int = 2,
    pad_to_bytes: int | None = None,
    extra_lines: Sequence[str] = (),
) -> str:
    """2x2 canonical/narrative fact x header renderer with optional balanced harmless padding."""
    if header_style == "canonical":
        hdr = (
            f"TASK=X2 GOAL={goal_subject} HOPS={int(hops)} MIN_EP={int(min_epoch)} "
            f"Q={'1' if query_available else '0'}:{query_cost:.2f} "
            f"R={'1' if retrieve_available else '0'}:{retrieve_cost:.2f} BUD={int(budget_remaining)}"
        )
    elif header_style == "narrative":
        q_txt = f"query open ({query_cost:.2f})" if query_available else "query closed"
        r_txt = f"retrieve open ({retrieve_cost:.2f})" if retrieve_available else "retrieve closed"
        hdr = (
            f"Trace {int(hops)} steps from start {goal_subject} using records from epoch {int(min_epoch)} onward; "
            f"{q_txt}; {r_txt}; turns left {int(budget_remaining)}."
        )
    else:
        raise ValueError(f"unsupported header_style {header_style!r}")

    lines = [hdr]
    for row in facts:
        s, v = str(row["subj"]), str(row["val"])
        ep = int(row.get("epoch", 1))
        auth = int(row.get("authority", 1))
        if fact_style == "canonical":
            lines.append(f"E {s}->{v} ep={ep} auth={auth}")
        elif fact_style == "narrative":
            lines.append(f"Record at epoch {ep} (trust {auth}) links node {s} to {v}.")
        else:
            raise ValueError(f"unsupported fact_style {fact_style!r}")
    for line in extra_lines:
        lines.append(str(line))
    text = "\n".join(lines)
    if pad_to_bytes is not None:
        raw_len = len(text.encode("utf-8"))
        if raw_len < int(pad_to_bytes):
            pad_needed = int(pad_to_bytes) - raw_len
            if pad_needed >= 4:
                text = text + "\n# " + ("." * (pad_needed - 3))
    return text


def build_generator_b_multihop_panel(
    *,
    split: str = "dev",
    num_skeletons: int = 128,
    seed: int = 7109,
    depths: Sequence[int] = (3, 4, 5),
    include_exhaustive_perm: bool = True,
) -> dict[str, Any]:
    """Independent Generator B (§5): 128 skeletons x depths (3,4,5) x 4 variants x 2 presentations."""
    split_salt = 0 if split == "dev" else 90_001
    rng = random.Random(int(seed) + split_salt)
    episodes: list[dict[str, Any]] = []
    exhaustive_panel: list[dict[str, Any]] = []
    sampled_perm_panel: list[dict[str, Any]] = []

    for sk_idx in range(int(num_skeletons)):
        # Degree-preserving 10-node skeleton with two disjoint 5-hop paths + 2 alt-branch nodes
        nodes = list(SYMBOLS)
        rng.shuffle(nodes)
        chain_a = nodes[:6]  # 5 edges: a0->a1->a2->a3->a4->a5
        chain_b = nodes[6:12]  # 5 edges: b0->b1->b2->b3->b4->b5
        base_edges = [
            {"subj": chain_a[i], "val": chain_a[i + 1], "epoch": 2, "authority": 2}
            for i in range(5)
        ] + [
            {"subj": chain_b[i], "val": chain_b[i + 1], "epoch": 2, "authority": 2}
            for i in range(5)
        ]
        # Balance terminal output bin rotation across skeletons
        for depth in depths:
            start_node = chain_a[5 - int(depth)]
            alt_start = chain_b[5 - int(depth)]
            clean_facts = deepcopy(base_edges)
            clean_res = interpret_facts_reference(clean_facts, goal_subject=start_node, hops=int(depth))
            assert clean_res["status"] == "SUPPORTED"

            # Relevant interior-edge swap: redirect the middle edge of chain_a into chain_b
            mid_idx = max(0, (5 - int(depth)) + (int(depth) // 2))
            rel_facts = deepcopy(base_edges)
            rel_facts[mid_idx]["val"] = chain_b[mid_idx + 1]
            rel_res = interpret_facts_reference(rel_facts, goal_subject=start_node, hops=int(depth))
            assert rel_res["status"] == "SUPPORTED" and rel_res["answer"] != clean_res["answer"]

            # Equal-size irrelevant swap: modify an untouched edge in chain_b
            irrel_facts = deepcopy(base_edges)
            irrel_facts[5]["val"] = nodes[12]
            irrel_res = interpret_facts_reference(irrel_facts, goal_subject=start_node, hops=int(depth))
            assert irrel_res["status"] == "SUPPORTED" and irrel_res["answer"] == clean_res["answer"]

            # Decoy variant: place a plausible wrong endpoint from chain_b on the LAST row
            decoy_facts = deepcopy(base_edges)
            decoy_facts.append({"subj": nodes[13], "val": chain_b[-1], "epoch": 2, "authority": 2})
            decoy_res = interpret_facts_reference(decoy_facts, goal_subject=start_node, hops=int(depth))
            assert decoy_res["status"] == "SUPPORTED" and decoy_res["answer"] == clean_res["answer"]

            variants = (
                ("clean", clean_facts, clean_res["answer"], start_node),
                ("relevant_interior_edit", rel_facts, rel_res["answer"], start_node),
                ("irrelevant_edit", irrel_facts, irrel_res["answer"], start_node),
                ("last_row_decoy", decoy_facts, decoy_res["answer"], start_node),
            )
            for vname, vfacts, expected_ans, g_subj in variants:
                for pres_idx, (f_style, h_style) in enumerate((("canonical", "canonical"), ("narrative", "canonical"))):
                    wid = f"genB_{split}_sk{sk_idx:03d}_d{depth}_{vname}_p{pres_idx}"
                    episodes.append(
                        {
                            "world_id": wid,
                            "task_type": "generator_b_multihop",
                            "split": split,
                            "generator": "generator_b_v1",
                            "skeleton_id": f"sk{sk_idx:03d}",
                            "sig": f"genB_{split}_sk{sk_idx:03d}_d{depth}_{vname}",
                            "hops": int(depth),
                            "variant": vname,
                            "presentation": pres_idx,
                            "fact_style": f_style,
                            "header_style": h_style,
                            "goal_subject": g_subj,
                            "alt_start_subject": alt_start,
                            "min_epoch": 1,
                            "facts": vfacts,
                            "hidden": {
                                "expected_operation": "ANSWER",
                                "expected_answer": expected_ans,
                                "expected": expected_ans,
                                "supported": True,
                            },
                        }
                    )

        # Exhaustive <=6-row permutation panel on first 2 skeletons
        if include_exhaustive_perm and sk_idx < 2:
            small_facts = deepcopy(base_edges[:4])  # 4 active rows -> 24 exact permutations
            small_ans = interpret_facts_reference(small_facts, goal_subject=chain_a[0], hops=3)["answer"]
            for p_idx, perm_rows in enumerate(permutations(small_facts)):
                exhaustive_panel.append(
                    {
                        "world_id": f"genB_{split}_exh_sk{sk_idx:03d}_perm{p_idx:02d}",
                        "task_type": "generator_b_multihop",
                        "split": split,
                        "generator": "generator_b_exhaustive_perm_v1",
                        "skeleton_id": f"sk{sk_idx:03d}",
                        "sig": f"genB_{split}_exh_sk{sk_idx:03d}",
                        "hops": 3,
                        "variant": "exhaustive_perm",
                        "fact_style": "canonical",
                        "header_style": "canonical",
                        "goal_subject": chain_a[0],
                        "min_epoch": 1,
                        "facts": list(perm_rows),
                        "hidden": {"expected_operation": "ANSWER", "expected_answer": small_ans, "supported": True},
                    }
                )
        # 16 seeded permutations for first 4 skeletons
        if sk_idx < 4:
            for p_idx in range(16):
                perm_rows = deepcopy(base_edges)
                random.Random(seed + sk_idx * 100 + p_idx).shuffle(perm_rows)
                p_ans = interpret_facts_reference(perm_rows, goal_subject=chain_a[0], hops=5)["answer"]
                sampled_perm_panel.append(
                    {
                        "world_id": f"genB_{split}_sperm_sk{sk_idx:03d}_p{p_idx:02d}",
                        "task_type": "generator_b_multihop",
                        "split": split,
                        "generator": "generator_b_sampled_perm_v1",
                        "skeleton_id": f"sk{sk_idx:03d}",
                        "sig": f"genB_{split}_sperm_sk{sk_idx:03d}",
                        "hops": 5,
                        "variant": "sampled_perm",
                        "fact_style": "canonical",
                        "header_style": "canonical",
                        "goal_subject": chain_a[0],
                        "min_epoch": 1,
                        "facts": perm_rows,
                        "hidden": {"expected_operation": "ANSWER", "expected_answer": p_ans, "supported": True},
                    }
                )

    return {
        "schema": "elendril-x2-generator-b-v1",
        "split": split,
        "seed": int(seed) + split_salt,
        "num_skeletons": int(num_skeletons),
        "episodes": episodes,
        "exhaustive_perm_panel": exhaustive_panel,
        "sampled_perm_panel": sampled_perm_panel,
        "content_sha256": digest([e["world_id"] for e in episodes]),
    }


def eval_shortcut_baselines(worlds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Run cheap heuristic baselines + deterministic reference interpreter on Generator B / D1-H worlds (§5)."""
    labels = [str(w["hidden"]["expected_answer"]) for w in worlds]
    maj_label = Counter(labels).most_common(1)[0][0] if labels else "A"
    counts = {k: 0 for k in ("majority", "first_row", "last_row", "terminal_only", "bag_of_codes", "reference_interpreter")}
    for w in worlds:
        gold = str(w["hidden"]["expected_answer"])
        facts = w.get("facts", ())
        if maj_label == gold:
            counts["majority"] += 1
        if facts and str(facts[0]["val"]) == gold:
            counts["first_row"] += 1
        if facts and str(facts[-1]["val"]) == gold:
            counts["last_row"] += 1
        # Terminal-only: pick the most common leaf value that never appears as a subject
        subjs = {str(r["subj"]) for r in facts}
        leaves = [str(r["val"]) for r in facts if str(r["val"]) not in subjs]
        term_guess = leaves[-1] if leaves else (str(facts[-1]["val"]) if facts else "A")
        if term_guess == gold:
            counts["terminal_only"] += 1
        # Bag-of-codes: most frequent value token
        vals = [str(r["val"]) for r in facts]
        bag_guess = Counter(vals).most_common(1)[0][0] if vals else "A"
        if bag_guess == gold:
            counts["bag_of_codes"] += 1
        ref = interpret_facts_reference(facts, goal_subject=str(w["goal_subject"]), hops=int(w.get("hops", 1)))
        if ref["status"] == "SUPPORTED" and ref["answer"] == gold:
            counts["reference_interpreter"] += 1
    n = max(1, len(worlds))
    return {
        "n_worlds": len(worlds),
        "accuracies": {k: float(v / n) for k, v in counts.items()},
        "reference_interpreter_exact": counts["reference_interpreter"] == len(worlds),
    }


# =============================================================================
# 2. D1-H Public-Update Selective-Edit Panel (§8, §11)
# =============================================================================

def build_selective_edit_panel(
    *,
    num_histories: int = 128,
    seed: int = 8201,
    split: str = "dev",
) -> list[dict[str, Any]]:
    """Build D1-H public-update selective-edit histories: edited key + dependent + untouched sibling + stale late lure."""
    rng = random.Random(int(seed) + (0 if split == "dev" else 50_000))
    worlds: list[dict[str, Any]] = []
    for idx in range(int(num_histories)):
        syms = list(SYMBOLS)
        rng.shuffle(syms)
        k_edit, v_old, v_new, v_dep_old, v_dep_new = syms[0], syms[1], syms[2], syms[3], syms[4]
        k_sib, v_sib, v_stale = syms[5], syms[6], syms[7]

        # Initial reliable bindings (epoch=1, auth=2), reliable correction (epoch=2, auth=3),
        # plus optional low-authority stale late lure (epoch=3, auth=1) on half the panel
        include_stale_lure = (idx % 2 == 1)
        events = [
            {"subj": k_edit, "val": v_old, "epoch": 1, "authority": 2},
            {"subj": v_old, "val": v_dep_old, "epoch": 1, "authority": 2},
            {"subj": v_new, "val": v_dep_new, "epoch": 1, "authority": 2},
            {"subj": k_sib, "val": v_sib, "epoch": 1, "authority": 2},
            {"subj": k_edit, "val": v_new, "epoch": 2, "authority": 3},  # authoritative correction
        ]
        if include_stale_lure:
            events.append({"subj": k_edit, "val": v_stale, "epoch": 3, "authority": 1})  # low-authority late lure

        dep_res = interpret_facts_reference(events, goal_subject=k_edit, hops=2)
        sib_res = interpret_facts_reference(events, goal_subject=k_sib, hops=1)
        assert dep_res["answer"] == v_dep_new and sib_res["answer"] == v_sib

        base_id = f"sel_edit_{split}_{idx:03d}"
        # Paired queries from the identical post-correction public history:
        worlds.append(
            {
                "world_id": f"{base_id}::edited_dep",
                "pair_id": base_id,
                "role": "edited_dependent",
                "task_type": "selective_edit",
                "split": split,
                "sig": f"{base_id}::edited_dep",
                "hops": 2,
                "goal_subject": k_edit,
                "min_epoch": 1,
                "facts": events,
                "correction_event_index": 4,
                "has_stale_late_lure": include_stale_lure,
                "hidden": {
                    "expected_operation": "ANSWER",
                    "expected_answer": v_dep_new,
                    "expected": v_dep_new,
                    "old_answer": v_dep_old,
                    "supported": True,
                },
            }
        )
        worlds.append(
            {
                "world_id": f"{base_id}::untouched_sibling",
                "pair_id": base_id,
                "role": "untouched_sibling",
                "task_type": "selective_edit",
                "split": split,
                "sig": f"{base_id}::untouched_sibling",
                "hops": 1,
                "goal_subject": k_sib,
                "min_epoch": 1,
                "facts": events,
                "correction_event_index": 4,
                "has_stale_late_lure": include_stale_lure,
                "hidden": {
                    "expected_operation": "ANSWER",
                    "expected_answer": v_sib,
                    "expected": v_sib,
                    "supported": True,
                },
            }
        )
    return worlds


# =============================================================================
# 3. T13 Finite-State Transducer Rule Learning + Bounded Symbolic Solver (§6, §11)
# =============================================================================

@dataclass(frozen=True)
class T13Law:
    """Finite-state transducer law over 4 states (0..3) and 4 events (0..3)."""

    law_id: str
    family: str
    schema_kind: str
    unary_table: tuple[int, int, int, int]
    binary_table: tuple[tuple[int, int, int, int], ...]  # [state][event] -> next_state

    def eval_unary(self, x: int) -> int:
        return int(self.unary_table[int(x) % 4])

    def eval_binary(self, x: int, ev: int) -> int:
        return int(self.binary_table[int(x) % 4][int(ev) % 4])

    def full_truth_table(self) -> tuple[int, ...]:
        flat_bin = tuple(v for row in self.binary_table for v in row)
        return self.unary_table + flat_bin


def _rot_plus_one(x: int) -> int:
    return (int(x) + 1) % 4


def _eval_t13_expr(law: T13Law, expr: Mapping[str, Any]) -> int:
    kind = str(expr["kind"])
    if kind == "f":
        return law.eval_unary(int(expr["x"]))
    if kind == "r_f":
        return _rot_plus_one(law.eval_unary(int(expr["x"])))
    if kind == "f_r":
        return law.eval_unary(_rot_plus_one(int(expr["x"])))
    if kind == "g":
        return law.eval_binary(int(expr["x"]), int(expr["e"]))
    if kind == "r_g":
        return _rot_plus_one(law.eval_binary(int(expr["x"]), int(expr["e"])))
    if kind == "g_f":
        return law.eval_binary(law.eval_unary(int(expr["x"])), int(expr["e"]))
    raise ValueError(f"unsupported T13 expr kind {kind!r}")


def format_t13_expr(expr: Mapping[str, Any]) -> str:
    kind = str(expr["kind"])
    x = T13_STATE_LABELS[int(expr["x"]) % 4]
    if kind == "f":
        return f"f({x})"
    if kind == "r_f":
        return f"r(f({x}))"
    if kind == "f_r":
        return f"f(r({x}))"
    e = T13_EVENTS[int(expr["e"]) % 4]
    if kind == "g":
        return f"g({x},{e})"
    if kind == "r_g":
        return f"r(g({x},{e}))"
    if kind == "g_f":
        return f"g(f({x}),{e})"
    raise ValueError(f"unsupported expr kind {kind!r}")


def enumerate_t13_candidate_prior(max_laws: int = 512) -> list[T13Law]:
    """Enumerate 256-1,024 distinct T13 candidate laws deduplicated by full semantic truth table (§11)."""
    seen_tables: set[tuple[int, ...]] = set()
    laws: list[T13Law] = []

    def _add(law: T13Law) -> None:
        if len(laws) >= max_laws:
            return
        tbl = law.full_truth_table()
        if tbl not in seen_tables:
            seen_tables.add(tbl)
            laws.append(law)

    # 1. Include exact §11 identity and swap(0,1) unary laws with standard shift binary table first
    std_bin = tuple(tuple((s + e) % 4 for e in range(4)) for s in range(4))
    _add(T13Law("law_id_std", "perm_sub", "identity_perm", (0, 1, 2, 3), std_bin))
    _add(T13Law("law_swap01_std", "perm_sub", "swap_perm", (1, 0, 2, 3), std_bin))

    # 2. All 24 unary permutations x structured binary laws (shift, xor, guarded reset, context reversal)
    binary_templates: list[tuple[str, str, tuple[tuple[int, int, int, int], ...]]] = [
        ("perm_sub", "mod_shift", std_bin),
        ("perm_sub", "xor_mix", tuple(tuple((s ^ e) % 4 for e in range(4)) for s in range(4))),
        ("perm_sub", "rev_shift", tuple(tuple((s - e) % 4 for e in range(4)) for s in range(4))),
        ("reset_absorb", "reset_e0", tuple(tuple(0 if e == 0 else (s + e) % 4 for e in range(4)) for s in range(4))),
        ("reset_absorb", "absorb_s3", tuple(tuple(3 if s == 3 else (s + e) % 4 for e in range(4)) for s in range(4))),
        ("guarded_bin", "guard_even", tuple(tuple((s + 2 * e) % 4 if s % 2 == 0 else (s + 1) % 4 for e in range(4)) for s in range(4))),
        ("guarded_bin", "guard_gt1", tuple(tuple((s ^ 1) if (s >= 2 and e >= 2) else (s + e) % 4 for e in range(4)) for s in range(4))),
        ("noncommute", "affine_3s", tuple(tuple((3 * s + e) % 4 for e in range(4)) for s in range(4))),
        ("exception", "exc_s1e2", tuple(tuple(0 if (s == 1 and e == 2) else (s + e) % 4 for e in range(4)) for s in range(4))),
        # Held-out confirmation schemata (§6)
        ("heldout_gated_reset", "gated_reset_conf", tuple(tuple(e if (s == e) else (s + 1) % 4 for e in range(4)) for s in range(4))),
        ("heldout_ctx_reversal", "ctx_reversal_conf", tuple(tuple((3 - s) if e >= 2 else (s + e) % 4 for e in range(4)) for s in range(4))),
    ]
    for p_idx, perm in enumerate(permutations(range(4))):
        u_tbl = (int(perm[0]), int(perm[1]), int(perm[2]), int(perm[3]))
        for fam, skind, b_tbl in binary_templates:
            _add(T13Law(f"law_p{p_idx:02d}_{skind}", fam, skind, u_tbl, b_tbl))
    # 3. Reset/constant unary maps (e.g. absorb to c) to reach 512 distinct laws
    for c in range(4):
        for s_keep in range(4):
            u_tbl = tuple(s if s == s_keep else c for s in range(4))
            for fam, skind, b_tbl in binary_templates:
                _add(T13Law(f"law_rst{c}k{s_keep}_{skind}", "reset_absorb", f"u_rst_{skind}", u_tbl, b_tbl))
    return laws


class T13VersionSpaceSolver:
    """Bounded symbolic version-space solver certifying identifiability and Bayes ceiling (§6, §11)."""

    def __init__(self, candidates: Sequence[T13Law] | None = None, *, max_laws: int = 512) -> None:
        self.candidates = tuple(candidates) if candidates is not None else tuple(enumerate_t13_candidate_prior(max_laws))
        if not (256 <= len(self.candidates) <= 1024):
            raise ValueError(f"T13 prior size must be in [256, 1024], got {len(self.candidates)}")

    def posterior(self, support: Sequence[Mapping[str, Any]], query_expr: Mapping[str, Any]) -> dict[str, Any]:
        compatible = [
            law
            for law in self.candidates
            if all(_eval_t13_expr(law, ex["expr"]) == int(ex["output"]) for ex in support)
        ]
        counts = [0, 0, 0, 0]
        for law in compatible:
            counts[_eval_t13_expr(law, query_expr)] += 1
        total = len(compatible)
        if total == 0:
            probs = [0.25, 0.25, 0.25, 0.25]
            return {
                "compatible_laws": 0,
                "identifiable": False,
                "output_probs": probs,
                "bayes_optimal_label": 0,
                "bayes_ceiling": 0.25,
                "in_declared_prior": False,
            }
        probs = [float(c / total) for c in counts]
        best_idx = max(range(4), key=lambda i: probs[i])
        identifiable = bool(max(probs) == 1.0)
        return {
            "compatible_laws": total,
            "identifiable": identifiable,
            "output_probs": probs,
            "bayes_optimal_label": int(best_idx),
            "bayes_ceiling": float(max(probs)),
            "in_declared_prior": True,
        }


def build_t13_decisive_fixture(*, witness_position: str = "late") -> dict[str, Any]:
    """Exact §11 T13 pair: r(x)=(x+1)%4, f=id vs swap(0,1), 15 common + 1 witness f(0), held-out r(f(1)) & r(f(3))."""
    std_bin = tuple(tuple((s + e) % 4 for e in range(4)) for s in range(4))
    law_id = T13Law("law_id_std", "perm_sub", "identity_perm", (0, 1, 2, 3), std_bin)
    law_swap = T13Law("law_swap01_std", "perm_sub", "swap_perm", (1, 0, 2, 3), std_bin)

    # 15 distinct semantically diverse common examples where law_id and law_swap agree
    common_exprs: list[dict[str, Any]] = [
        {"kind": "f", "x": 2},
        {"kind": "f", "x": 3},
        {"kind": "r_f", "x": 2},
        {"kind": "r_f", "x": 3},
        {"kind": "f_r", "x": 1},
        {"kind": "f_r", "x": 2},
    ]
    for x in range(4):
        for e in range(4):
            if len(common_exprs) >= 15:
                break
            ex = {"kind": "g", "x": x, "e": e}
            if _eval_t13_expr(law_id, ex) == _eval_t13_expr(law_swap, ex):
                common_exprs.append(ex)
        if len(common_exprs) >= 15:
            break
    assert len(common_exprs) == 15

    common_support = [
        {"expr": ex, "output": _eval_t13_expr(law_id, ex)}
        for ex in common_exprs
    ]
    for item in common_support:
        assert _eval_t13_expr(law_swap, item["expr"]) == item["output"]

    witness_expr = {"kind": "f", "x": 0}
    w_id = {"expr": witness_expr, "output": _eval_t13_expr(law_id, witness_expr)}      # 0
    w_swap = {"expr": witness_expr, "output": _eval_t13_expr(law_swap, witness_expr)}  # 1
    assert w_id["output"] == 0 and w_swap["output"] == 1

    if witness_position == "early":
        sup_id = [w_id, *common_support]
        sup_swap = [w_swap, *common_support]
    else:
        sup_id = [*common_support, w_id]
        sup_swap = [*common_support, w_swap]

    disc_query = {"kind": "r_f", "x": 1}  # r(f(1)) -> 2 (id) vs 1 (swap)
    inv_query = {"kind": "r_f", "x": 3}   # r(f(3)) -> 0 in both
    assert _eval_t13_expr(law_id, disc_query) == 2
    assert _eval_t13_expr(law_swap, disc_query) == 1
    assert _eval_t13_expr(law_id, inv_query) == 0 and _eval_t13_expr(law_swap, inv_query) == 0

    def _make_world(wid: str, sup: list[dict[str, Any]], q: dict[str, Any], expected_idx: int, role: str) -> dict[str, Any]:
        return {
            "world_id": wid,
            "task_type": "t13_fst",
            "split": "fixture",
            "sig": wid,
            "role": role,
            "witness_position": witness_position,
            "support": sup,
            "query": q,
            "hops": 2,
            "hidden": {
                "expected_operation": "ANSWER",
                "expected_answer": T13_STATE_LABELS[expected_idx],
                "expected": T13_STATE_LABELS[expected_idx],
                "expected_state_idx": expected_idx,
                "supported": True,
            },
        }

    return {
        "schema": "elendril-x2-t13-decisive-fixture-v1",
        "witness_position": witness_position,
        "law_id": law_id,
        "law_swap": law_swap,
        "common_support": common_support,
        "disc_query": disc_query,
        "inv_query": inv_query,
        "worlds": {
            "id_disc": _make_world(f"t13_fix_{witness_position}_id_disc", sup_id, disc_query, 2, "discriminating_id"),
            "swap_disc": _make_world(f"t13_fix_{witness_position}_swap_disc", sup_swap, disc_query, 1, "discriminating_swap"),
            "id_inv": _make_world(f"t13_fix_{witness_position}_id_inv", sup_id, inv_query, 0, "invariant_id"),
            "swap_inv": _make_world(f"t13_fix_{witness_position}_swap_inv", sup_swap, inv_query, 0, "invariant_swap"),
            "ambiguous_disc_a": _make_world(f"t13_fix_{witness_position}_ambig_a", common_support, disc_query, 2, "ambiguous_a"),
            "ambiguous_disc_b": _make_world(f"t13_fix_{witness_position}_ambig_b", common_support, disc_query, 1, "ambiguous_b"),
        },
    }


def build_t13_family_bank(*, seed: int = 9301) -> dict[str, Any]:
    """Build 32 train / 8 dev / 12 locked confirmation T13 families with truth-table & AST dedup (§6)."""
    all_laws = enumerate_t13_candidate_prior(512)
    # Partition by held-out operator schemata so confirmation holds out entire schemata
    conf_pool = [l for l in all_laws if l.family.startswith("heldout_")]
    non_conf = [l for l in all_laws if not l.family.startswith("heldout_")]
    rng = random.Random(int(seed))
    rng.shuffle(non_conf)
    rng.shuffle(conf_pool)

    # 32 train families, 8 dev families, 12 confirmation families (4 instances each)
    train_laws = non_conf[: 32 * 4]
    dev_laws = non_conf[32 * 4 : (32 + 8) * 4]
    conf_laws = conf_pool[: 12 * 4]
    if len(conf_laws) < 48:
        conf_laws.extend(non_conf[(32 + 8) * 4 : (32 + 8) * 4 + (48 - len(conf_laws))])

    def _pack(name: str, items: list[T13Law], n_fams: int) -> list[dict[str, Any]]:
        out = []
        for f_idx in range(n_fams):
            insts = items[f_idx * 4 : (f_idx + 1) * 4]
            out.append(
                {
                    "family_id": f"T13_{name.upper()}_F{f_idx:02d}",
                    "split": name,
                    "instances": [
                        {
                            "law_id": l.law_id,
                            "schema_kind": l.schema_kind,
                            "unary_table": list(l.unary_table),
                            "binary_table": [list(r) for r in l.binary_table],
                            "truth_table_sha256": digest(list(l.full_truth_table())),
                        }
                        for l in insts
                    ],
                }
            )
        return out

    train_fams = _pack("train", train_laws, 32)
    dev_fams = _pack("dev", dev_laws, 8)
    conf_fams = _pack("confirmation", conf_laws, 12)

    # Assert strict semantic truth-table disjointness across splits
    t_hashes = {ins["truth_table_sha256"] for f in train_fams for ins in f["instances"]}
    d_hashes = {ins["truth_table_sha256"] for f in dev_fams for ins in f["instances"]}
    c_hashes = {ins["truth_table_sha256"] for f in conf_fams for ins in f["instances"]}
    assert not (t_hashes & d_hashes) and not (t_hashes & c_hashes) and not (d_hashes & c_hashes)

    return {
        "schema": "elendril-x2-t13-family-bank-v1",
        "counts": {"train": len(train_fams), "dev": len(dev_fams), "confirmation": len(conf_fams)},
        "train": train_fams,
        "dev": dev_fams,
        "confirmation": conf_fams,
        "split_hashes": {
            "train": digest(sorted(t_hashes)),
            "dev": digest(sorted(d_hashes)),
            "confirmation": digest(sorted(c_hashes)),
        },
    }


def eval_t13_retrieval_buffer_baseline(
    support: Sequence[Mapping[str, Any]],
    query_expr: Mapping[str, Any],
) -> dict[str, Any]:
    """Explicit T13 retrieval-buffer competitor + cache budget check (§6)."""
    rendered = "\n".join(f"{format_t13_expr(ex['expr'])}={T13_STATE_LABELS[int(ex['output'])]}" for ex in support)
    byte_len = len(rendered.encode("utf-8"))
    overflow = byte_len > T13_EPISODIC_CACHE_BUDGET_BYTES
    q_str = format_t13_expr(query_expr)
    exact = [int(ex["output"]) for ex in support if format_t13_expr(ex["expr"]) == q_str]
    if exact:
        return {"predicted_idx": exact[-1], "match_type": "exact_lookup", "bytes": byte_len, "overflow": overflow}
    # Nearest neighbor by same input x
    same_x = [int(ex["output"]) for ex in support if int(ex["expr"].get("x", -1)) == int(query_expr.get("x", -2))]
    pred = Counter(same_x).most_common(1)[0][0] if same_x else 0
    return {"predicted_idx": pred, "match_type": "nearest_x", "bytes": byte_len, "overflow": overflow}


# =============================================================================
# 4. T14 Costly Information / Price-Crossover Environment + Exact DP (§7, §11)
# =============================================================================

def solve_t14_public_dp(world: Mapping[str, Any], *, cost_override: float | None = None) -> dict[str, Any]:
    """Exact public-belief dynamic programming solver for T14 (§7, §11). NEVER reads hidden truth."""
    pub = world["public_belief"]
    prior = [float(p) for p in pub["prior_probs"]]
    cert_already = bool(pub.get("certified", False))
    q_avail = bool(pub.get("query_available", False))
    r_avail = bool(pub.get("retrieve_available", False))
    p_q_cert = float(pub.get("query_cert_prob", 0.0))
    p_r_cert = float(pub.get("retrieve_cert_prob", 0.0))
    p_both_cert = float(pub.get("complementary_cert_prob", max(p_q_cert, p_r_cert)))
    q_cost = float(cost_override if cost_override is not None else pub.get("query_cost", T14_CHEAP_COST))
    r_cost = float(cost_override if cost_override is not None else pub.get("retrieve_cost", T14_CHEAP_COST))
    budget = int(pub.get("budget", 1))

    # Supported means public posterior assigns all mass (1.0) to one answer (or certified=True)
    def _terminal_value(is_cert: bool) -> tuple[str, float]:
        if is_cert:
            return ("ANSWER", 1.0)
        # Unsupported answer always earns -1.0 under §7/§11 registered utility; ABSTAIN earns 0.0
        return ("ABSTAIN", 0.0)

    def _dp(is_cert: bool, qa: bool, ra: bool, bud: int, used_q: bool, used_r: bool) -> tuple[str, float, dict[str, float]]:
        term_op, term_ev = _terminal_value(is_cert)
        action_evs: dict[str, float] = {
            "ANSWER": 1.0 if is_cert else -1.0,
            "ABSTAIN": 0.0,
        }
        best_op, best_ev = term_op, term_ev
        if is_cert or bud <= 0:
            return best_op, best_ev, action_evs

        if qa:
            eff_p = p_both_cert if used_r else p_q_cert
            _, v_yes, _ = _dp(True, False, ra, bud - 1, True, used_r)
            _, v_no, _ = _dp(False, False, ra, bud - 1, True, used_r)
            ev_q = eff_p * v_yes + (1.0 - eff_p) * v_no - q_cost
            action_evs["QUERY"] = float(ev_q)
            if ev_q > best_ev + 1e-12:
                best_op, best_ev = "QUERY", float(ev_q)
        if ra:
            eff_p = p_both_cert if used_q else p_r_cert
            _, v_yes, _ = _dp(True, qa, False, bud - 1, used_q, True)
            _, v_no, _ = _dp(False, qa, False, bud - 1, used_q, True)
            ev_r = eff_p * v_yes + (1.0 - eff_p) * v_no - r_cost
            action_evs["RETRIEVE"] = float(ev_r)
            if ev_r > best_ev + 1e-12:
                best_op, best_ev = "RETRIEVE", float(ev_r)
        return best_op, float(best_ev), action_evs

    opt_op, opt_ev, action_evs = _dp(cert_already, q_avail, r_avail, budget, False, False)
    return {
        "optimal_operation": opt_op,
        "optimal_ev": round(float(opt_ev), 8),
        "action_evs": {k: round(float(v), 8) for k, v in action_evs.items()},
        "prior_probs": prior,
        "certified": cert_already,
        "clairvoyant_ceiling": 1.0,
    }


def build_t14_price_crossover_fixture() -> dict[str, Any]:
    """Exact §7/§11 T14 price-crossover fixture: p_cert=0.20, cheap=0.02 (+0.18 QUERY) vs expensive=0.35 (-0.15 ABSTAIN)."""
    def _make(regime: str, cost: float, *, cert_realized: bool, hidden_ans: str = "A") -> dict[str, Any]:
        return {
            "world_id": f"t14_crossover_{regime}_{'cert' if cert_realized else 'nocert'}_{hidden_ans}",
            "task_type": "t14_costly",
            "stratum": "one_useful_query",
            "cost_regime": regime,
            "split": "fixture",
            "sig": f"t14_crossover_{regime}_{'cert' if cert_realized else 'nocert'}_{hidden_ans}",
            "hops": 1,
            "goal_subject": "X",
            "public_belief": {
                "active_labels": ["A", "B"],
                "prior_probs": [0.5, 0.5],
                "certified": False,
                "query_available": True,
                "retrieve_available": True,
                "query_cert_prob": 0.20,
                "retrieve_cert_prob": 0.00,  # dominated 2nd tool (§11)
                "complementary_cert_prob": 0.20,
                "query_cost": float(cost),
                "retrieve_cost": float(cost),
                "budget": 1,
            },
            "initial_facts": [{"subj": "X", "val": "A", "epoch": 1, "authority": 1}],
            "forced_revision": None,
            "hidden": {
                "true_answer": hidden_ans,
                "expected": hidden_ans,
                "query_certifies": bool(cert_realized),
                "retrieve_certifies": False,
                "query_payload": {"subj": "X", "val": hidden_ans, "epoch": 2, "authority": 3} if cert_realized else None,
                "retrieve_payload": None,
            },
        }

    cheap_cert = _make("cheap", T14_CHEAP_COST, cert_realized=True, hidden_ans="A")
    cheap_nocert = _make("cheap", T14_CHEAP_COST, cert_realized=False, hidden_ans="B")
    exp_cert = _make("expensive", T14_EXPENSIVE_COST, cert_realized=True, hidden_ans="A")
    exp_nocert = _make("expensive", T14_EXPENSIVE_COST, cert_realized=False, hidden_ans="B")
    dp_cheap = solve_t14_public_dp(cheap_cert)
    dp_exp = solve_t14_public_dp(exp_cert)
    assert dp_cheap["optimal_operation"] == "QUERY" and abs(dp_cheap["action_evs"]["QUERY"] - 0.18) < 1e-6
    assert dp_exp["optimal_operation"] == "ABSTAIN" and abs(dp_exp["action_evs"]["QUERY"] - (-0.15)) < 1e-6
    return {
        "schema": "elendril-x2-t14-price-crossover-fixture-v1",
        "cheap_cert": cheap_cert,
        "cheap_nocert": cheap_nocert,
        "expensive_cert": exp_cert,
        "expensive_nocert": exp_nocert,
        "dp_cheap": dp_cheap,
        "dp_expensive": dp_exp,
    }


def build_t14_panel(
    *,
    split: str = "dev",
    histories_per_stratum: int = 8,
    seed: int = 9401,
) -> list[dict[str, Any]]:
    """Build 6-stratum T14 paired cheap/expensive panel + forced-exposure revision cases (§7)."""
    rng = random.Random(int(seed) + (0 if split == "dev" else 70_000))
    worlds: list[dict[str, Any]] = []
    stratum_cfg = {
        "immediately_decidable": (True, True, True, 0.0, 0.0, 0.0, 1),
        "one_useful_query": (False, True, True, 0.25, 0.0, 0.25, 1),
        "one_useful_retrieve": (False, True, True, 0.0, 0.25, 0.25, 1),
        "complementary_tools": (False, True, True, 0.05, 0.05, 0.65, 2),
        "unresolved_after_exhaustion": (False, True, True, 0.0, 0.0, 0.0, 1),
        "stale_conflict_revision": (False, True, True, 0.30, 0.20, 0.50, 2),
    }
    for s_idx, stratum in enumerate(T14_STRATA):
        cert0, qa, ra, pq, pr, pb, bud = stratum_cfg[stratum]
        for h_idx in range(int(histories_per_stratum)):
            ans = SYMBOLS[(s_idx + h_idx) % 4]
            alt = SYMBOLS[(s_idx + h_idx + 1) % 4]
            q_cert = (rng.random() < pq) if pq > 0 else False
            r_cert = (rng.random() < pr) if pr > 0 else False
            forced_rev = (
                {"subj": "X", "val": ans, "epoch": 3, "authority": 3}
                if stratum == "stale_conflict_revision"
                else None
            )
            for regime, cost in (("cheap", T14_CHEAP_COST), ("expensive", T14_EXPENSIVE_COST)):
                wid = f"t14_{split}_{stratum}_h{h_idx:03d}_{regime}"
                worlds.append(
                    {
                        "world_id": wid,
                        "pair_id": f"t14_{split}_{stratum}_h{h_idx:03d}",
                        "task_type": "t14_costly",
                        "stratum": stratum,
                        "cost_regime": regime,
                        "split": split,
                        "sig": f"t14_{split}_{stratum}_h{h_idx:03d}_{regime}",
                        "hops": 1,
                        "goal_subject": "X",
                        "public_belief": {
                            "active_labels": [ans, alt],
                            "prior_probs": [1.0, 0.0] if cert0 else [0.5, 0.5],
                            "certified": cert0,
                            "query_available": qa,
                            "retrieve_available": ra,
                            "query_cert_prob": pq,
                            "retrieve_cert_prob": pr,
                            "complementary_cert_prob": pb,
                            "query_cost": float(cost),
                            "retrieve_cost": float(cost),
                            "budget": int(bud),
                        },
                        "initial_facts": [{"subj": "X", "val": ans if cert0 else alt, "epoch": 1, "authority": 3 if cert0 else 1}],
                        "forced_revision": forced_rev,
                        "hidden": {
                            "true_answer": ans,
                            "expected": ans,
                            "query_certifies": bool(q_cert),
                            "retrieve_certifies": bool(r_cert),
                            "query_payload": {"subj": "X", "val": ans, "epoch": 2, "authority": 3} if q_cert else None,
                            "retrieve_payload": {"subj": "X", "val": ans, "epoch": 2, "authority": 3} if r_cert else None,
                        },
                    }
                )
    return worlds


def eval_t14_constant_baselines(cheap_world: Mapping[str, Any], exp_world: Mapping[str, Any]) -> dict[str, Any]:
    """Verify constant tool policies lose on >=1 side of every non-tied price crossover (§7, §9)."""
    dp_c = solve_t14_public_dp(cheap_world)
    dp_e = solve_t14_public_dp(exp_world)
    policies = ("QUERY", "RETRIEVE", "ANSWER", "ABSTAIN")
    regrets: dict[str, dict[str, float]] = {}
    for pol in policies:
        rc = dp_c["optimal_ev"] - dp_c["action_evs"].get(pol, -1.0)
        re = dp_e["optimal_ev"] - dp_e["action_evs"].get(pol, -1.0)
        regrets[pol] = {
            "cheap_regret": round(float(rc), 8),
            "expensive_regret": round(float(re), 8),
            "loses_on_at_least_one_side": bool(rc > 1e-9 or re > 1e-9),
        }
    return {
        "dp_cheap_opt": dp_c["optimal_operation"],
        "dp_expensive_opt": dp_e["optimal_operation"],
        "constant_policies": regrets,
        "all_constant_policies_lose_somewhere": all(v["loses_on_at_least_one_side"] for v in regrets.values()),
    }


# =============================================================================
# 5. Stateful X2EnvironmentKernel & X2ObservablePolicy (§0.2)
# =============================================================================

@dataclass(frozen=True)
class X2KernelState:
    world: dict[str, Any]
    task_type: str
    public_lines: tuple[str, ...]
    query_available: bool
    retrieve_available: bool
    consumed_tools: frozenset[str]
    query_cost: float
    retrieve_cost: float
    certified: bool
    certified_answer: str | None
    budget_remaining: int
    charged_cost: float
    step_count: int
    executed_actions: tuple[tuple[str, str | None], ...]
    terminal: bool = False


class X2EnvironmentKernel:
    """Stateful truth-firewalled EnvironmentKernel for X2 (selective-edit, Generator B, T13, T14)."""

    kernel_id = X2_KERNEL_ID

    def __init__(self, *, max_actions: int = 4) -> None:
        self.max_actions = int(max_actions)

    def canonical_config(self) -> dict[str, Any]:
        return {"kernel_id": self.kernel_id, "max_actions": self.max_actions}

    def structural_signature(self, world: Mapping[str, Any]) -> str:
        return str(world.get("sig") or world.get("world_id") or "x2_w0")

    def inject_distractor(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = deepcopy(dict(world))
        w["world_id"] = f"{w.get('world_id', 'w')}::dist"
        w["sig"] = f"{w.get('sig', w['world_id'])}::dist"
        if "facts" in w and isinstance(w["facts"], list):
            w["facts"] = list(w["facts"]) + [{"subj": "P", "val": "O", "epoch": 1, "authority": 1}]
        elif "initial_facts" in w and isinstance(w["initial_facts"], list):
            w["initial_facts"] = list(w["initial_facts"]) + [{"subj": "P", "val": "O", "epoch": 1, "authority": 1}]
        return w

    def permute_world(self, world: Mapping[str, Any], shift: int = 1) -> dict[str, Any]:
        w = deepcopy(dict(world))
        w["world_id"] = f"{w.get('world_id', 'w')}::perm{shift}"
        w["sig"] = f"{w.get('sig', w['world_id'])}::perm{shift}"
        if "facts" in w and isinstance(w["facts"], list) and len(w["facts"]) > 1:
            s = shift % len(w["facts"])
            w["facts"] = w["facts"][s:] + w["facts"][:s]
        return w

    def reverse_world(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = deepcopy(dict(world))
        w["world_id"] = f"{w.get('world_id', 'w')}::rev"
        w["sig"] = f"{w.get('sig', w['world_id'])}::rev"
        if "facts" in w and isinstance(w["facts"], list):
            w["facts"] = list(reversed(w["facts"]))
        return w

    def initial_state(self, world: Mapping[str, Any]) -> X2KernelState:
        w = deepcopy(dict(world))
        tt = str(w.get("task_type", "selective_edit"))
        if tt == "t14_costly":
            pb = w["public_belief"]
            cert = bool(pb.get("certified", False))
            cert_ans = str(w["initial_facts"][0]["val"]) if (cert and w.get("initial_facts")) else None
            base_lines = [
                f"TASK=T14 GOAL=X Q_P={pb['query_cert_prob']:.2f} R_P={pb['retrieve_cert_prob']:.2f} "
                f"Q_C={pb['query_cost']:.2f} R_C={pb['retrieve_cost']:.2f} CERT={'1' if cert else '0'}",
                f"CANDS={','.join(pb['active_labels'])}",
            ]
            for f in w.get("initial_facts", ()):
                base_lines.append(f"OBS {f['subj']}->{f['val']} ep={f['epoch']} auth={f['authority']}")
            return X2KernelState(
                world=w,
                task_type=tt,
                public_lines=tuple(base_lines),
                query_available=bool(pb.get("query_available", True)),
                retrieve_available=bool(pb.get("retrieve_available", True)),
                consumed_tools=frozenset(),
                query_cost=float(pb.get("query_cost", T14_CHEAP_COST)),
                retrieve_cost=float(pb.get("retrieve_cost", T14_CHEAP_COST)),
                certified=cert,
                certified_answer=cert_ans,
                budget_remaining=int(pb.get("budget", 1)) + 1,  # tools + final terminal action
                charged_cost=0.0,
                step_count=0,
                executed_actions=(),
                terminal=False,
            )
        if tt == "t13_fst":
            lines = ["TASK=T13 FST_RULE_INDUCTION"]
            for idx, ex in enumerate(w.get("support", ())):
                lines.append(f"SUP{idx:02d} {format_t13_expr(ex['expr'])}={T13_STATE_LABELS[int(ex['output'])]}")
            lines.append(f"QUERY {format_t13_expr(w['query'])}")
            return X2KernelState(
                world=w,
                task_type=tt,
                public_lines=tuple(lines),
                query_available=False,
                retrieve_available=False,
                consumed_tools=frozenset(),
                query_cost=0.0,
                retrieve_cost=0.0,
                certified=True,
                certified_answer=None,
                budget_remaining=1,
                charged_cost=0.0,
                step_count=0,
                executed_actions=(),
                terminal=False,
            )
        # Generator B / D1-H selective edit
        rendered = render_x2_factorial(
            facts=w.get("facts", ()),
            goal_subject=str(w.get("goal_subject", "A")),
            hops=int(w.get("hops", 1)),
            min_epoch=int(w.get("min_epoch", 1)),
            fact_style=str(w.get("fact_style", "canonical")),
            header_style=str(w.get("header_style", "canonical")),
            query_available=False,
            retrieve_available=False,
            budget_remaining=1,
            pad_to_bytes=w.get("pad_to_bytes"),
        )
        return X2KernelState(
            world=w,
            task_type=tt,
            public_lines=tuple(rendered.splitlines()),
            query_available=False,
            retrieve_available=False,
            consumed_tools=frozenset(),
            query_cost=0.0,
            retrieve_cost=0.0,
            certified=True,
            certified_answer=None,
            budget_remaining=1,
            charged_cost=0.0,
            step_count=0,
            executed_actions=(),
            terminal=False,
        )

    def observe(self, state: X2KernelState) -> UniversalObservation:
        # Strict truth firewall: ONLY public_lines + public tool flags/costs/budget
        text = "\n".join(state.public_lines)
        if state.task_type == "t14_costly":
            text = (
                f"{text}\nSTATE step={state.step_count} Q={'1' if state.query_available else '0'} "
                f"R={'1' if state.retrieve_available else '0'} CERT={'1' if state.certified else '0'} "
                f"BUD={max(1, state.budget_remaining)}"
            )
        obs_id = digest([X2_OBSERVATION_VERSION, state.step_count, text])[:16]
        return UniversalObservation(
            obs_id=obs_id,
            rendered_text=text,
            available_tools={"query": bool(state.query_available), "memory": bool(state.retrieve_available)},
            consumed_tools=state.consumed_tools,
            decision_index=int(state.step_count),
            budget_remaining=max(1, int(state.budget_remaining)),
            metadata={
                "world_id": str(state.world.get("world_id", "x2_w0")),
                "task_type": state.task_type,
                "query_cost": state.query_cost,
                "retrieve_cost": state.retrieve_cost,
            },
        )

    def step(self, state: X2KernelState, operation: str, argument: str | None = None) -> X2KernelState:
        if state.terminal:
            return state
        op = str(operation).upper()
        actions = state.executed_actions + ((op, argument),)
        next_step = state.step_count + 1
        next_bud = max(0, state.budget_remaining - 1)
        lines = list(state.public_lines)
        qa, ra = state.query_available, state.retrieve_available
        consumed = set(state.consumed_tools)
        cert, cert_ans = state.certified, state.certified_answer
        cost = state.charged_cost

        if op in ("ANSWER", "ABSTAIN") or next_bud == 0 or next_step >= self.max_actions:
            return X2KernelState(
                world=state.world,
                task_type=state.task_type,
                public_lines=tuple(lines),
                query_available=qa,
                retrieve_available=ra,
                consumed_tools=frozenset(consumed),
                query_cost=state.query_cost,
                retrieve_cost=state.retrieve_cost,
                certified=cert,
                certified_answer=cert_ans,
                budget_remaining=next_bud,
                charged_cost=cost,
                step_count=next_step,
                executed_actions=actions,
                terminal=True,
            )

        hidden = state.world.get("hidden", {})
        if op == "QUERY":
            cost += state.query_cost
            if qa:
                qa = False
                consumed.add("query")
                payload = hidden.get("query_payload")
                if payload is not None:
                    cert = True
                    cert_ans = str(payload["val"])
                    lines.append(f"TOOL_Q CERT {payload['subj']}->{payload['val']} ep={payload['epoch']} auth={payload['authority']}")
                else:
                    lines.append("TOOL_Q NO_CERTIFICATION")
            else:
                lines.append("TOOL_Q UNAVAILABLE")
        elif op == "RETRIEVE":
            cost += state.retrieve_cost
            if ra:
                ra = False
                consumed.add("memory")
                payload = hidden.get("retrieve_payload")
                if payload is not None:
                    cert = True
                    cert_ans = str(payload["val"])
                    lines.append(f"TOOL_R CERT {payload['subj']}->{payload['val']} ep={payload['epoch']} auth={payload['authority']}")
                else:
                    lines.append("TOOL_R NO_CERTIFICATION")
            else:
                lines.append("TOOL_R UNAVAILABLE")

        # Optional public forced-revision event arriving at step boundary independent of first action (§7)
        frev = state.world.get("forced_revision")
        if frev is not None and next_step == 1:
            cert = True
            cert_ans = str(frev["val"])
            lines.append(f"FORCED_REV {frev['subj']}->{frev['val']} ep={frev['epoch']} auth={frev['authority']}")

        return X2KernelState(
            world=state.world,
            task_type=state.task_type,
            public_lines=tuple(lines),
            query_available=qa,
            retrieve_available=ra,
            consumed_tools=frozenset(consumed),
            query_cost=state.query_cost,
            retrieve_cost=state.retrieve_cost,
            certified=cert,
            certified_answer=cert_ans,
            budget_remaining=next_bud,
            charged_cost=cost,
            step_count=next_step,
            executed_actions=actions,
            terminal=bool(next_bud <= 0),
        )

    def ideal_action(self, state: X2KernelState) -> tuple[str, str | None]:
        """Public-belief optimal action (never peeks at hidden futures on T14!)."""
        if state.task_type == "t14_costly":
            if state.certified and state.certified_answer is not None:
                return ("ANSWER", state.certified_answer)
            # Re-solve public DP from remaining public availability & budget
            pb = dict(state.world["public_belief"])
            pb["certified"] = bool(state.certified)
            pb["query_available"] = bool(state.query_available)
            pb["retrieve_available"] = bool(state.retrieve_available)
            pb["budget"] = max(0, int(state.budget_remaining) - 1)
            dp = solve_t14_public_dp({"public_belief": pb})
            op = dp["optimal_operation"]
            if op == "ANSWER":
                return ("ANSWER", state.certified_answer or str(pb["active_labels"][0]))
            return (op, None)
        hidden = state.world.get("hidden", {})
        exp_op = str(hidden.get("expected_operation", "ANSWER"))
        exp_ans = str(hidden["expected_answer"]) if hidden.get("expected_answer") is not None else None
        return (exp_op, exp_ans if exp_op == "ANSWER" else None)

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        """Return intermediate hop subgoals for multi-hop Generator B / selective-edit worlds."""
        hops = int(world.get("hops", 1))
        if hops <= 1 or "facts" not in world or "goal_subject" not in world:
            return ()
        probes: list[SubgoalProbeSpec] = []
        for h in range(1, hops):
            res = interpret_facts_reference(
                world["facts"],
                goal_subject=str(world["goal_subject"]),
                hops=h,
                min_epoch=int(world.get("min_epoch", 1)),
            )
            if res["status"] != "SUPPORTED" or res["answer"] is None:
                continue
            mw = deepcopy(dict(world))
            mw["world_id"] = f"{world.get('world_id', 'w')}::hop{h}"
            mw["sig"] = f"{world.get('sig', mw['world_id'])}::hop{h}"
            mw["hops"] = h
            mw["hidden"] = {"expected_operation": "ANSWER", "expected_answer": res["answer"], "supported": True}
            probes.append(
                SubgoalProbeSpec(
                    subgoal_id=f"{world.get('world_id', 'w')}_h{h}",
                    stage_name=f"hop_{h}",
                    hop_index=h,
                    modified_world=mw,
                    expected_operation="ANSWER",
                    expected_argument=str(res["answer"]),
                    description=f"Intermediate {h}-hop symbol resolution",
                )
            )
        return tuple(probes)

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        st = self.initial_state(world)
        for op, arg in actions:
            st = self.step(st, op, arg)
            if st.terminal:
                break
        last_op, last_arg = actions[-1] if actions else ("INVALID", None)
        hidden = world.get("hidden", {})

        if st.task_type == "t14_costly":
            true_ans = str(hidden.get("true_answer", "A"))
            if last_op == "ANSWER":
                if st.certified and str(last_arg) == true_ans:
                    reward = 1.0 - st.charged_cost
                    correct = True
                    supported = True
                else:
                    reward = -1.0 - st.charged_cost
                    correct = False
                    supported = False
            elif last_op == "ABSTAIN":
                reward = 0.0 - st.charged_cost
                correct = not st.certified
                supported = True
            else:
                reward = -1.0 - st.charged_cost
                correct = False
                supported = False
            dp = solve_t14_public_dp(world)
            first_op = actions[0][0] if actions else "INVALID"
            return {
                "correct": bool(correct),
                "supported": bool(supported),
                "utility": float(reward),
                "charged_cost": float(st.charged_cost),
                "first_action_optimal_dp": bool(first_op == dp["optimal_operation"]),
                "optimal_dp_op": dp["optimal_operation"],
                "optimal_dp_ev": dp["optimal_ev"],
            }

        exp_op = str(hidden.get("expected_operation", "ANSWER"))
        exp_ans = hidden.get("expected_answer")
        ok = bool(last_op == exp_op and (exp_op != "ANSWER" or str(last_arg) == str(exp_ans)))
        return {
            "correct": ok,
            "supported": bool(hidden.get("supported", True)),
            "expected": [exp_op, exp_ans],
            "observed": [last_op, last_arg],
        }


def encode_x2_text(text: str, *, max_length: int = 256) -> tuple[list[int], int]:
    """Encode public observation text into UTF-8 framed token IDs with explicit action position."""
    raw = UTF8.encode(text)
    if len(raw) > max_length - 2:
        raw = raw[-(max_length - 2) :]
    toks = [UTF8.special(SpecialToken.BOS), *raw, UTF8.special(SpecialToken.ANSWER)]
    return toks, len(toks) - 1


def _decode_action_19(idx: int) -> tuple[str, str | None]:
    i = int(idx)
    if 0 <= i < 16:
        return ("ANSWER", SYMBOLS[i])
    if i == 16:
        return ("QUERY", None)
    if i == 17:
        return ("RETRIEVE", None)
    return ("ABSTAIN", None)


class X2ObservablePolicy:
    """Public-history-only ObservablePolicy with episodic memory reset/clone/lesions for Planthir."""

    def __init__(
        self,
        model: X2TypedModel,
        model_spec: Any,
        *,
        model_identity: str = "elendril-x2-policy",
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
        lesion_mode: str | None = None,
        block_write_index: int | None = None,
    ) -> None:
        self.model = model
        self.model_spec = model_spec
        self.model_identity = str(model_identity)
        self.policy_id = str(model_identity)
        self.architecture_family = str(model.x2_spec.architecture)
        self.parameter_count = sum(int(p.numel()) for p in model.parameters())
        self.mode = str(mode)
        self.copy_enabled_override = copy_enabled_override
        self.lesion_mode = lesion_mode
        self.block_write_index = block_write_index
        self._episode_memory: torch.Tensor | None = None

    def reset_episode(self) -> None:
        self._episode_memory = None

    def clone_episode_state(self) -> torch.Tensor | None:
        return None if self._episode_memory is None else self._episode_memory.detach().clone()

    def restore_episode_state(self, state: torch.Tensor | None) -> None:
        self._episode_memory = None if state is None else state.detach().clone()

    def _batch_encode(self, obs_batch: Sequence[UniversalObservation]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = next(self.model.parameters()).device
        max_ctx = int(self.model.x2_spec.base_spec.context_length)
        encoded = [encode_x2_text(o.rendered_text, max_length=max_ctx) for o in obs_batch]
        max_len = max(len(t) for t, _ in encoded)
        toks = torch.zeros((len(encoded), max_len), dtype=torch.long, device=dev)
        segs = torch.full((len(encoded), max_len), -1, dtype=torch.long, device=dev)
        apos = torch.zeros((len(encoded),), dtype=torch.long, device=dev)
        for i, (seq, ap) in enumerate(encoded):
            toks[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=dev)
            segs[i, : len(seq)] = 0
            apos[i] = ap
        return toks, segs, apos

    @torch.no_grad()
    def propose_batch(
        self,
        obs_batch: Sequence[UniversalObservation],
        *,
        carry_episode_memory: bool = False,
        forecast_override: torch.Tensor | None = None,
    ) -> list[UniversalAction]:
        self.model.eval()
        toks, segs, apos = self._batch_encode(obs_batch)
        init_mem = self._episode_memory if (carry_episode_memory and len(obs_batch) == 1) else None
        out = self.model.forward_action(
            toks,
            segs,
            apos,
            copy_enabled_override=self.copy_enabled_override,
            lesion_mode=self.lesion_mode,
            init_memory=init_mem,
            block_write_index=self.block_write_index,
            forecast_override=forecast_override,
        )
        if carry_episode_memory and out.memory_state is not None and len(obs_batch) == 1:
            self._episode_memory = out.memory_state.detach().clone()
        probs_19 = out.log_probs_19.exp().cpu()
        results: list[UniversalAction] = []
        for idx, obs in enumerate(obs_batch):
            raw_probs = probs_19[idx]
            raw_i = int(torch.argmax(raw_probs).item())
            raw_op, raw_arg = _decode_action_19(raw_i)
            chosen_i, lp = _apply_mode_to_distribution(
                raw_probs,
                mode=self.mode,
                query_available=bool(obs.available_tools.get("query", False)),
                memory_available=bool(obs.available_tools.get("memory", False)),
            )
            op, arg = _decode_action_19(chosen_i)
            dist = {ACTION_19_LABELS[j]: float(raw_probs[j].item()) for j in range(19)}
            results.append(
                UniversalAction(
                    operation=op,
                    argument=arg,
                    log_probability=lp,
                    action_distribution=dist,
                    mode=self.mode,
                    raw_operation=raw_op,
                    raw_argument=raw_arg,
                )
            )
        return results

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        return self.propose_batch([obs], carry_episode_memory=True)[0]

    @torch.no_grad()
    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        self.model.eval()
        toks, segs, apos = self._batch_encode(obs_batch)
        out = self.model.forward_action(
            toks,
            segs,
            apos,
            copy_enabled_override=self.copy_enabled_override,
            lesion_mode=self.lesion_mode,
        )
        return {
            "decision_hidden": out.decision_hidden.detach().cpu(),
            "provisional_val_logits": out.provisional_val_logits.detach().cpu(),
            "certifiable_logit": out.certifiable_logit.detach().cpu(),
        }


__all__ = [
    "T13Law",
    "T13VersionSpaceSolver",
    "T13_DOSES",
    "T13_EPISODIC_CACHE_BUDGET_BYTES",
    "T13_EVENTS",
    "T13_STATE_LABELS",
    "T14_CHEAP_COST",
    "T14_EXPENSIVE_COST",
    "T14_STRATA",
    "X2EnvironmentKernel",
    "X2KernelState",
    "X2ObservablePolicy",
    "X2_KERNEL_ID",
    "build_generator_b_multihop_panel",
    "build_selective_edit_panel",
    "build_t13_decisive_fixture",
    "build_t13_family_bank",
    "build_t14_panel",
    "build_t14_price_crossover_fixture",
    "encode_x2_text",
    "enumerate_t13_candidate_prior",
    "eval_shortcut_baselines",
    "eval_t13_retrieval_buffer_baseline",
    "eval_t14_constant_baselines",
    "format_t13_expr",
    "interpret_facts_reference",
    "render_x2_factorial",
    "solve_t14_public_dp",
]
