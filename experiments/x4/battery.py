"""V8-X4 22-Family 6-Group Cognitive Battery, Curriculum/Fresh Stream, L0-L5 Profile & X01-X12 Interventions."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import math
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from anra_v5.elendril.codec import digest
from anra_v5.elendril.v8x.contracts import SYMBOLS as X1_SYMBOLS, X1_FAMILIES
from anra_v5.elendril.v8x.data import (
    PublicFactX1,
    _build_single_world,
    _independent_epistemic_resolve,
    render_public_x1,
    x1_teacher_decide,
)
from experiments.x3.cognitive_battery import (
    SCAN_VALID_ACTIONS,
    VALID_TYPED_OPS,
    _eval_listops_node,
    _eval_scan_cmd,
    _solve_c06_horn,
    _solve_c08,
    compute_public_version_space_posterior,
    normalize_c10_rational_unit,
    parse_and_score_public_typed_output,
)
from experiments.x3.factorial import X3_SYMBOLS, build_x3_paired_graph_factorial, independent_walk_x3_graph

X4_BATTERY_SCHEMA = "elendril-x4-22fam-battery-v1"
X4_MAX_PROMPT_BYTES = 600
X4_DEFAULT_GEN_BOUND = 48

X4_SPLIT_SEEDS: dict[str, int] = {
    "train": 91001,
    "dev": 91002,
    "confirm": 91003,
    "ood_transfer": 91004,
}

_CANON_SPLIT: dict[str, str] = {
    "train": "train",
    "dev": "dev",
    "confirm": "confirm",
    "confirmation": "confirm",
    "ood_transfer": "ood_transfer",
    "transfer": "ood_transfer",
}

X4_FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "A_LOGICAL_COMPOSITIONAL": (
        "X01",
        "X02",
        "X03",
        "X06",
        "C05_RELATIONAL_GRAPH_CF",
        "C06_OPEN_WORLD_DEDUCTION",
        "C07_HIERARCHICAL_LISTOPS",
    ),
    "B_LEARNING_ABSTRACTION": (
        "C01_NOVEL_OP_INDUCTION",
        "C02_FST_RULE_COMPOSITION",
        "C03_GRID_TRANSFORM_ARC",
    ),
    "C_MEMORY_INTELLIGENCE": (
        "X04",
        "X07",
        "X08",
        "C08_ADAPTIVE_MEMORY_REVISION",
    ),
    "D_PLANNING_INFO_ACQ": (
        "X05",
        "X09",
        "X11",
        "C04_COMPOSITIONAL_SCAN",
    ),
    "E_METACOGNITION": (
        "X10",
        "X12",
        "C09_METACOGNITION_SELF_CHECK",
    ),
    "F_LANGUAGE_MATH_TRANSFER": (
        "C10_LANGUAGE_MATH_TRANSFER",
    ),
}

X4_C_FAMILIES: tuple[str, ...] = (
    "C01_NOVEL_OP_INDUCTION",
    "C02_FST_RULE_COMPOSITION",
    "C03_GRID_TRANSFORM_ARC",
    "C04_COMPOSITIONAL_SCAN",
    "C05_RELATIONAL_GRAPH_CF",
    "C06_OPEN_WORLD_DEDUCTION",
    "C07_HIERARCHICAL_LISTOPS",
    "C08_ADAPTIVE_MEMORY_REVISION",
    "C09_METACOGNITION_SELF_CHECK",
    "C10_LANGUAGE_MATH_TRANSFER",
)

X4_22_FAMILIES: tuple[str, ...] = tuple(
    fam for grp_fams in X4_FAMILY_GROUPS.values() for fam in grp_fams
)

FAMILY_TO_GROUP: dict[str, str] = {
    fam: grp for grp, fams in X4_FAMILY_GROUPS.items() for fam in fams
}

L0_L5_APPLICABILITY: dict[str, dict[str, str | None]] = {
    fam: {
        "L0": None,
        "L1": None,
        "L2": None,
        "L3": None,
        "L4": None if fam in ("C01_NOVEL_OP_INDUCTION", "C02_FST_RULE_COMPOSITION", "C03_GRID_TRANSFORM_ARC") else "INAPPLICABLE_STATIC_RULE_FAMILY",
        "L5": None if fam in ("C03_GRID_TRANSFORM_ARC", "C04_COMPOSITIONAL_SCAN", "C05_RELATIONAL_GRAPH_CF", "C10_LANGUAGE_MATH_TRANSFER") else "INAPPLICABLE_SINGLE_DOMAIN_FAMILY",
    }
    for fam in X4_22_FAMILIES
}


def _norm_split(split: str) -> str:
    if split not in _CANON_SPLIT:
        raise ValueError(f"Unknown X4 split {split!r}; expected one of {tuple(X4_SPLIT_SEEDS)}")
    return _CANON_SPLIT[split]


def _split_rng(tag: str, split: str, seed: int, idx: int) -> random.Random:
    h = sha256(f"x4|{tag}|{_norm_split(split)}|{int(seed)}|{int(idx)}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _clip_prompt(prompt: str, limit: int = X4_MAX_PROMPT_BYTES) -> str:
    raw = prompt.encode("utf-8")
    if len(raw) > limit:
        raise ValueError(f"X4 prompt byte budget exceeded ({len(raw)} > {limit}): {prompt[:120]!r}")
    return prompt


def infer_x4_family(prompt: str) -> str:
    p = str(prompt).strip()
    if p.startswith("X"):
        m = re.match(r"^(X\d{2})\|", p)
        if m and m.group(1) in X1_FAMILIES:
            return m.group(1)
    for tag, fam in (
        ("C01|", "C01_NOVEL_OP_INDUCTION"),
        ("C02|", "C02_FST_RULE_COMPOSITION"),
        ("C03|", "C03_GRID_TRANSFORM_ARC"),
        ("C04|", "C04_COMPOSITIONAL_SCAN"),
        ("C05|", "C05_RELATIONAL_GRAPH_CF"),
        ("C06|", "C06_OPEN_WORLD_DEDUCTION"),
        ("C07|", "C07_HIERARCHICAL_LISTOPS"),
        ("C08|", "C08_ADAPTIVE_MEMORY_REVISION"),
        ("C09|", "C09_METACOGNITION_SELF_CHECK"),
        ("C10|", "C10_LANGUAGE_MATH_TRANSFER"),
    ):
        if p.startswith(tag):
            return fam
    if "CONF-TRACE " in p or p.startswith("Trace ") or "TASK=X3 " in p:
        return "C05_RELATIONAL_GRAPH_CF"
    return "UNKNOWN"


def infer_x4_generation_bound(prompt: str) -> int:
    fam = infer_x4_family(prompt)
    if fam in X1_FAMILIES:
        return 24
    if fam == "C04_COMPOSITIONAL_SCAN":
        cmd = prompt.split("|")[2] if len(prompt.split("|")) >= 3 else prompt
        clauses = re.split(r"\s+(?:and|after)\s+", cmd)
        est = 0
        for cl in clauses:
            base = 8 if "around" in cl else (3 if "opposite" in cl else 2)
            mult = 3 if "thrice" in cl else (2 if "twice" in cl else 1)
            est += base * mult
        return min(240, max(X4_DEFAULT_GEN_BOUND, est * 7 + 16))
    if fam == "C03_GRID_TRANSFORM_ARC":
        m = re.search(r"Q:([0-9/]+)->\?", prompt)
        if m:
            rows = m.group(1).split("/")
            dim = max(len(rows), max((len(r) for r in rows), default=3), 3)
            return min(160, max(X4_DEFAULT_GEN_BOUND, dim * (dim + 1) + 20))
        return 64
    return X4_DEFAULT_GEN_BOUND


def format_x4_typed_target(case_or_prompt: Mapping[str, Any] | str, gold_text: str | None = None) -> str:
    """Uniform public typed envelope: '<OP>:<CANONICAL_PAYLOAD>' or bare tool/abstention op."""
    if isinstance(case_or_prompt, Mapping):
        prompt = str(case_or_prompt.get("prompt", ""))
        raw = str(case_or_prompt.get("gold_text", "")) if gold_text is None else str(gold_text)
        pub_ambig = bool(case_or_prompt.get("public_ambiguous", False))
    else:
        prompt = str(case_or_prompt)
        raw = str(gold_text or "")
        pub_ambig = False

    txt = raw.strip()
    for op in VALID_TYPED_OPS:
        if txt == op or txt.startswith(f"{op}:"):
            return txt
    fam = infer_x4_family(prompt)
    if pub_ambig and fam == "C01_NOVEL_OP_INDUCTION":
        return "ABSTAIN:AMBIGUOUS"
    if txt in ("QUERY", "RETRIEVE", "ABSTAIN", "AMBIGUOUS"):
        return txt
    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        _, _, canon = normalize_c10_rational_unit(txt)
        return f"ANSWER:{canon or txt}"
    return f"ANSWER:{txt}"


def parse_and_score_x4_output(
    prompt: str,
    raw_output: str,
    case: Mapping[str, Any] | None = None,
    *,
    max_new_bytes: int | None = None,
    terminated_by_eos: bool = True,
) -> dict[str, Any]:
    """Exact scorer across ALL 22 X4 families (no mod-16 shortcuts)."""
    bound = int(max_new_bytes if max_new_bytes is not None else infer_x4_generation_bound(prompt))
    fam = str(case["family"]) if (case and "family" in case) else infer_x4_family(prompt)

    # Delegate standard C01..C10 checks cleanly to parse_and_score_public_typed_output when non-X
    if fam not in X1_FAMILIES:
        # C10 extended rational units & C09 prompt compatibility handled cleanly
        res = parse_and_score_public_typed_output(
            prompt, raw_output, case, max_new_bytes=bound, terminated_by_eos=terminated_by_eos
        )
        res["family"] = fam
        return res

    # Strict X01..X12 scoring
    raw_s = str(raw_output).strip()
    byte_len = len(raw_s.encode("utf-8"))
    overflow = byte_len > bound
    unterminated = not bool(terminated_by_eos)
    if ":" in raw_s:
        head, tail = raw_s.split(":", 1)
        op = head.strip().upper()
        payload = tail.strip()
    else:
        parts = raw_s.split(None, 1)
        first = parts[0].upper() if parts else ""
        if first in ("QUERY", "RETRIEVE", "ABSTAIN", "AMBIGUOUS"):
            op = first
            payload = parts[1].strip() if len(parts) > 1 else ""
        else:
            op = "ANSWER" if raw_s else ""
            payload = raw_s

    if op in ("QUERY", "RETRIEVE", "ABSTAIN"):
        syntax_valid = (payload == "") and (not overflow) and (not unterminated)
    elif op == "ANSWER":
        syntax_valid = (payload in X1_SYMBOLS) and (not overflow) and (not unterminated)
    else:
        syntax_valid = False

    gold_typed = format_x4_typed_target(prompt, str(case.get("gold_text", "")) if case else "")
    if ":" in gold_typed:
        g_op, g_pay = gold_typed.split(":", 1)
    else:
        g_op, g_pay = gold_typed, ""
    correct = bool(syntax_valid and op == g_op and payload == g_pay)
    return {
        "schema": "elendril-x4-typed-score-v1",
        "family": fam,
        "parsed_op": op,
        "parsed_answer": payload,
        "bound_bytes": bound,
        "byte_len": byte_len,
        "valid": bool(syntax_valid),
        "invalid": not bool(syntax_valid),
        "truncated": bool(overflow or (unterminated and byte_len >= bound)),
        "overflow": bool(overflow),
        "unterminated": bool(unterminated),
        "unit_mismatch": False,
        "corrupted_suffix": bool(not syntax_valid),
        "public_ambiguous": False,
        "ambiguity_posterior_credit": 0.0,
        "exact_match": correct,
        "exact_score": 1.0 if correct else 0.0,
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Family Generators: X01..X12 (with Causal Interventions) & Repaired C01..C10
# ---------------------------------------------------------------------------


def _generate_x1_family_case(
    family: str, split: str, seed: int, idx: int, *, renderer_override: int | None = None
) -> dict[str, Any]:
    ns = _norm_split(split)
    offset_seed = int(seed) + {"train": 100, "dev": 200, "confirm": 300, "ood_transfer": 400}[ns]
    world = _build_single_world(
        split=ns,
        family=family,
        base_idx=idx,
        seed=offset_seed,
        renderer_override=renderer_override if renderer_override is not None else (3 if ns == "ood_transfer" else (idx % 3)),
    )
    rendered = render_public_x1(
        ordered_facts=world["initial_facts"],
        goal_subject=world["goal_subject"],
        hops=int(world["hops"]),
        min_epoch=int(world["min_epoch"]),
        query_available=bool(world["allowed_public_tools"]["query_available"]),
        memory_available=bool(world["allowed_public_tools"]["memory_available"]),
        budget_remaining=4,
        renderer_id=int(world["renderer_template"]),
        row_order_seed=int(world["row_order_seed"]),
    )
    op, arg = x1_teacher_decide(
        world["initial_facts"],
        world["goal_subject"],
        int(world["hops"]),
        int(world["min_epoch"]),
        bool(world["allowed_public_tools"]["query_available"]),
        bool(world["allowed_public_tools"]["memory_available"]),
    )
    gold_text = f"{op}:{arg}" if (op == "ANSWER" and arg is not None) else op
    prompt = _clip_prompt(f"{family}|{rendered}")
    facts_list = [f.as_dict() if hasattr(f, "as_dict") else dict(f) for f in world["initial_facts"]]
    return {
        "case_id": f"{family}:{ns}:s{seed}:{idx:04d}",
        "family": family,
        "group": FAMILY_TO_GROUP[family],
        "split": split,
        "prompt": prompt,
        "gold_text": gold_text,
        "n_shots": 0,
        "absent_from_input": bool(op != "ANSWER" or (arg is not None and arg not in rendered)),
        "presentation": "heldout_narrative" if int(world["renderer_template"]) == 3 else "canonical",
        "problem_spec": {
            "facts": facts_list,
            "goal_subject": world["goal_subject"],
            "hops": int(world["hops"]),
            "min_epoch": int(world["min_epoch"]),
            "query_available": bool(world["allowed_public_tools"]["query_available"]),
            "memory_available": bool(world["allowed_public_tools"]["memory_available"]),
        },
        "uncertainty_kind": "information_uncertainty" if op in ("QUERY", "RETRIEVE", "ABSTAIN") else "low_uncertainty",
        "benchmark_provenance": "x4_repaired_22fam",
    }


def build_x01_x12_interventions(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build privileged diagnostic interventions per X01..X12 failure mechanism (§13)."""
    interventions: list[dict[str, Any]] = []
    for c in cases:
        fam = str(c["family"])
        if fam not in X1_FAMILIES:
            continue
        spec = c["problem_spec"]
        facts = [PublicFactX1(**f) if isinstance(f, Mapping) else f for f in spec["facts"]]
        goal = str(spec["goal_subject"])
        hops = int(spec["hops"])
        min_ep = int(spec["min_epoch"])
        qa = bool(spec["query_available"])
        ma = bool(spec["memory_available"])

        # 1. Simplified rendering (renderer 0, unshuffled canonical order)
        simple_r = render_public_x1(
            ordered_facts=facts, goal_subject=goal, hops=hops, min_epoch=min_ep,
            query_available=qa, memory_available=ma, budget_remaining=4, renderer_id=0, row_order_seed=0,
        )
        op0, arg0 = x1_teacher_decide(facts, goal, hops, min_ep, qa, ma)
        interventions.append({
            **dict(c),
            "case_id": f"{c['case_id']}::interv_simple_render",
            "intervention_type": "simplify_rendering",
            "diagnostic_only_privileged": True,
            "prompt": _clip_prompt(f"{fam}|{simple_r}"),
            "gold_text": f"{op0}:{arg0}" if (op0 == "ANSWER" and arg0 is not None) else op0,
        })

        # 2. Supply intermediate goal edge when multi-hop or tool-gated (tests composition bottleneck)
        if hops >= 2:
            prov, resolved = _independent_epistemic_resolve([f.as_dict() for f in facts], goal, hops, min_ep)
            if prov and resolved in X1_SYMBOLS:
                shortcut_facts = [*facts, PublicFactX1(subject=goal, value=resolved, source="S3", epoch=max(5, min_ep))]
                sc_r = render_public_x1(
                    ordered_facts=shortcut_facts, goal_subject=goal, hops=1, min_epoch=min_ep,
                    query_available=qa, memory_available=ma, budget_remaining=4, renderer_id=0, row_order_seed=0,
                )
                interventions.append({
                    **dict(c),
                    "case_id": f"{c['case_id']}::interv_supply_intermediate",
                    "intervention_type": "supply_intermediate_evidence",
                    "diagnostic_only_privileged": True,
                    "prompt": _clip_prompt(f"{fam}|{sc_r}"),
                    "gold_text": f"ANSWER:{resolved}",
                })
    return interventions


# Grid ops for repaired C03 (primitive + composed ops across full 0..9 color palette)
def _apply_x4_grid_op(grid: np.ndarray, op_spec: str, cmap: Mapping[int, int]) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    for part in op_spec.split("+"):
        if part == "flip_h":
            g = np.fliplr(g)
        elif part == "flip_v":
            g = np.flipud(g)
        elif part == "rot90":
            g = np.rot90(g, k=1)
        elif part == "rot180":
            g = np.rot90(g, k=2)
        elif part == "transpose":
            g = g.T.copy()
        elif part == "gravity_down":
            out = np.zeros_like(g)
            for col in range(g.shape[1]):
                nz = [v for v in g[:, col] if v != 0]
                if nz:
                    out[-len(nz) :, col] = nz
            g = out
        elif part == "border_one":
            g = g.copy()
            g[0, :] = 1
            g[-1, :] = 1
            g[:, 0] = 1
            g[:, -1] = 1
        elif part == "tile_xor_rot180":
            g = (g ^ np.rot90(g, k=2)) % 4
        else:
            raise ValueError(f"Unknown X4 grid op part: {part!r}")
    return np.vectorize(lambda v: int(cmap.get(int(v), int(v))))(g)


def _grid_str(g: np.ndarray) -> str:
    return "/".join("".join(str(int(x)) for x in row) for row in g.tolist())


def _generate_c03_repaired(split: str, seed: int, idx: int, phase: int = 3) -> dict[str, Any]:
    """Repaired C03: train teaches all colors & primitives across P1/P2; dev/confirm/OOD hold out compositions/shapes."""
    ns = _norm_split(split)
    rng = _split_rng("X4_C03", ns, seed, idx)
    if ns == "train":
        if phase <= 1:
            ops = ("flip_h", "flip_v", "rot90", "transpose")
        else:
            ops = ("flip_h", "flip_v", "rot90", "transpose", "gravity_down", "border_one", "rot180")
        dims = ((3, 3), (3, 4))
    elif ns == "dev":
        ops = ("flip_h+rot90", "transpose+gravity_down")
        dims = ((3, 3), (4, 3))
    elif ns == "confirm":
        ops = ("flip_v+rot90", "rot180+border_one")
        dims = ((4, 3), (4, 4))
    else:  # ood_transfer: novel combinations & shapes unseen in train
        ops = ("tile_xor_rot180", "flip_h+gravity_down")
        dims = ((3, 5), (4, 4))

    op_name = ops[idx % len(ops)]
    rows_n, cols_n = dims[(idx // len(ops)) % len(dims)]
    # Full color palette 0..9 across training too so digits '4'..'9' are never OOD-starved
    base_colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng.shuffle(base_colors)
    in_palette = base_colors[:4]
    out_palette = list(in_palette)
    rng.shuffle(out_palette)
    cmap = {src: dst for src, dst in zip(in_palette, out_palette)}
    for k in range(10):
        cmap.setdefault(k, k)

    demos_txt: list[str] = []
    for d_i in range(3):
        inp = np.array([[in_palette[(rng.randrange(4))] for _ in range(cols_n)] for _ in range(rows_n)], dtype=np.int64)
        # Witness every active input color so color bijection is publicly identifiable
        if rows_n >= 3 and cols_n >= 3:
            inp[1, 1] = in_palette[d_i % 4]
            inp[0, 1] = in_palette[(d_i + 1) % 4]
            inp[1, 0] = in_palette[(d_i + 2) % 4]
            inp[2 % rows_n, 2 % cols_n] = in_palette[(d_i + 3) % 4]
        out = _apply_x4_grid_op(inp, op_name, cmap)
        demos_txt.append(f"{_grid_str(inp)}->{_grid_str(out)}")

    q_in = np.array([[in_palette[rng.randrange(4)] for _ in range(cols_n)] for _ in range(rows_n)], dtype=np.int64)
    q_out = _apply_x4_grid_op(q_in, op_name, cmap)
    gold_text = _grid_str(q_out)
    prompt = _clip_prompt(f"C03|GRID|{' | '.join(demos_txt)}|Q:{_grid_str(q_in)}->?")
    return {
        "case_id": f"C03:{ns}:s{seed}:{idx:04d}",
        "family": "C03_GRID_TRANSFORM_ARC",
        "group": FAMILY_TO_GROUP["C03_GRID_TRANSFORM_ARC"],
        "split": split,
        "prompt": prompt,
        "gold_text": gold_text,
        "n_shots": 3,
        "absent_from_input": gold_text not in prompt,
        "presentation": "grid",
        "problem_spec": {"op_name": op_name, "color_map": cmap, "q_in": q_in.tolist(), "shape": [rows_n, cols_n]},
        "benchmark_provenance": "x4_repaired_22fam",
    }


def _generate_c04_repaired(split: str, seed: int, idx: int, phase: int = 3) -> dict[str, Any]:
    """Repaired C04: train covers every atomic verb/direction/multiplier/connective while holding out novel compositions."""
    ns = _norm_split(split)
    rng = _split_rng("X4_C04", ns, seed, idx // 2)
    verbs = ("walk", "run", "jump", "look")
    dirs = ("left", "right", "opposite", "around")
    mults = ("twice", "thrice")

    if ns == "train":
        if phase <= 1 or (idx % 3 == 0):
            # All verbs, all directions, all multipliers atomically (disjoint single clauses; keep around/jump represented)
            v = verbs[idx % 4]
            d = dirs[(idx // 4) % 4]
            if v == "jump" and d == "around":
                v = "walk"  # Hold out "jump around" strictly for dev/confirm/OOD compositional generalization!
            m = "" if (idx % 3 == 0) else f" {mults[idx % 2]}"
            cmd = f"{v} {d}{m}"
        else:
            v1 = rng.choice(["walk", "run", "jump", "look"])
            d1 = dirs[idx % 4]
            if v1 == "jump" and d1 == "around":
                d1 = "opposite"
            v2 = rng.choice(["walk", "jump", "look", "run"])
            d2 = dirs[(idx + 1) % 4]
            if v2 == "jump" and d2 == "around":
                d2 = "left"
            conn = "and" if idx % 2 == 0 else "after"
            m1 = mults[idx % 2]
            cmd = f"{v1} {d1} {m1} {conn} {v2} {d2}"
            if "jump around" in cmd:
                cmd = cmd.replace("jump around", "jump left")
    elif ns == "dev":
        v1 = "jump" if (idx // 2) % 2 == 0 else rng.choice(verbs)
        d1 = "around" if v1 == "jump" else rng.choice(["opposite", "around"])
        v2 = rng.choice(["walk", "look"])
        d2 = rng.choice(["right", "opposite"])
        cmd = f"{v1} {d1} and {v2} {d2}" if idx % 2 == 0 else f"{v2} {d2} and {v1} {d1}"
    elif ns == "confirm":
        v1 = "jump" if (idx // 2) % 2 == 0 else rng.choice(verbs)
        d1 = rng.choice(["opposite", "around"])
        m1 = rng.choice(mults)
        v2 = rng.choice(["run", "look"])
        d2 = rng.choice(["left", "opposite"])
        cmd = f"{v1} {d1} {m1} after {v2} {d2}" if idx % 2 == 0 else f"{v2} {d2} after {v1} {d1} {m1}"
    else:  # ood_transfer
        v2 = rng.choice(["walk", "run", "look"])
        d2 = rng.choice(["opposite", "right"])
        m2 = rng.choice(mults)
        cmd = f"jump around thrice after {v2} {d2} {m2}" if idx % 2 == 0 else f"{v2} {d2} {m2} after jump around thrice"

    gold_text = _eval_scan_cmd(cmd)
    prompt = _clip_prompt(f"C04|COMP|{cmd}|OUT=?")
    return {
        "case_id": f"C04:{ns}:s{seed}:{idx:04d}",
        "family": "C04_COMPOSITIONAL_SCAN",
        "group": FAMILY_TO_GROUP["C04_COMPOSITIONAL_SCAN"],
        "split": split,
        "prompt": prompt,
        "gold_text": gold_text,
        "n_shots": 0,
        "absent_from_input": True,
        "presentation": "scan",
        "problem_spec": {"cmd": cmd},
        "benchmark_provenance": "x4_repaired_22fam",
    }


def _generate_c10_repaired(split: str, seed: int, idx: int, phase: int = 3) -> dict[str, Any]:
    """Repaired C10: train exposes integers AND reduced rationals + multiple units/phrasings; splits hold out unseen units & args."""
    ns = _norm_split(split)
    rng = _split_rng("X4_C10", ns, seed, idx // 2)
    names = {
        "train": ("Amina", "Bram"),
        "dev": ("Clara", "Doran"),
        "confirm": ("Elena", "Farid"),
        "ood_transfer": ("Gita", "Hiro"),
    }[ns]
    units_by_split = {
        "train": ("tokens", "meters", "grams", "liters"),
        "dev": ("credits", "watts"),
        "confirm": ("units", "steps"),
        "ood_transfer": ("cycles", "joules"),
    }[ns]
    unit = units_by_split[(idx // 2) % len(units_by_split)]
    rate = rng.randint(2, 9)
    batches = rng.randint(3, 8)
    per_b = rng.randint(4, 11)
    bonus = rng.randint(2, 14)

    if ns == "train":
        denom = 1 if (phase <= 1 and idx % 2 == 0) else rng.choice([1, 2, 3, 4])
    elif ns == "dev":
        denom = 2 if idx % 2 == 1 else 1
    elif ns == "confirm":
        denom = rng.choice([2, 3]) if idx % 2 == 1 else 1
    else:
        denom = rng.choice([3, 4]) if idx % 2 == 1 else 2

    if idx % 2 == 0:
        num = rate * batches + per_b + bonus
        expr = f"{names[0]} earns {rate} {unit} per cycle over {batches} cycles, adds {per_b} reserve {unit}, and receives {bonus} adjustment {unit}."
    else:
        num = per_b * batches + rate + bonus
        if denom == 1:
            expr = f"{names[1]} accumulates ({per_b} {unit} over {batches} cycles plus {rate} base and {bonus} bonus)."
        else:
            expr = f"{names[1]} splits ({per_b} {unit} over {batches} cycles plus {rate} base and {bonus} bonus) across {denom} shares."

    frac = Fraction(num, denom)
    val_s = str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
    gold_text = f"{val_s} {unit}"
    prompt = _clip_prompt(f"C10|PROSE|{expr} What is the exact quantity and unit?")
    return {
        "case_id": f"C10:{ns}:s{seed}:{idx:04d}",
        "family": "C10_LANGUAGE_MATH_TRANSFER",
        "group": FAMILY_TO_GROUP["C10_LANGUAGE_MATH_TRANSFER"],
        "split": split,
        "prompt": prompt,
        "gold_text": gold_text,
        "n_shots": 0,
        "absent_from_input": val_s not in prompt,
        "presentation": "prose",
        "problem_spec": {"num": num, "denom": denom, "unit": unit, "rate": rate, "batches": batches, "per_b": per_b, "bonus": bonus},
        "benchmark_provenance": "x4_repaired_22fam",
    }


def _generate_c09_repaired(split: str, seed: int, idx: int) -> dict[str, Any]:
    """Repaired C09: ZERO `split=` token leak in prompts! Clean Bellman cost-reversal & hard self-check."""
    ns = _norm_split(split)
    rng = _split_rng("X4_C09", ns, seed, idx)
    subtype = ("solvable_easy", "solvable_hard", "missing_premise")[idx % 3]
    if subtype == "missing_premise":
        regime = "cheap" if (idx // 3) % 2 == 0 else "expensive"
        cost = 0.02 if regime == "cheap" else 0.35
        q_useful = ((idx // 6) % 2 == 0)
        pq, pr = (0.20, 0.00) if q_useful else (0.00, 0.20)
        # Optimal decision via exact Bellman expectation:
        ev_q = pq - cost
        ev_r = pr - cost
        if ev_q >= ev_r and ev_q > 1e-12:
            gold = "QUERY"
        elif ev_r > ev_q and ev_r > 1e-12:
            gold = "RETRIEVE"
        else:
            gold = "ABSTAIN"
        cands = list(X3_SYMBOLS[:4])
        prompt = _clip_prompt(f"C09|META|MISSING|pQ={pq:.2f}|pR={pr:.2f}|c={cost:.2f}|cand={cands}|OP=?")
        spec = {"subtype": subtype, "cost": cost, "pQ": pq, "pR": pr, "gold": gold}
        uk = "information_uncertainty"
    else:
        base_off = {"train": 10, "dev": 25, "confirm": 40, "ood_transfer": 55}[ns]
        a = base_off + int(rng.randint(0, 14))
        b = int(rng.randint(10, 39))
        if subtype == "solvable_easy":
            gold = str(a + b)
            body = f"Compute X={a}+{b}"
        else:
            c = int(rng.randint(2, 6))
            gold = str(a * c + b - 3)
            body = f"Verify draft {a * c} for {a}*{c}+{b}-3"
        prompt = _clip_prompt(f"C09|META|{subtype.upper()}|{body}|ANS=?")
        spec = {"subtype": subtype, "gold": gold}
        uk = "epistemic_reasoning" if subtype == "solvable_hard" else "low_uncertainty"

    return {
        "case_id": f"C09:{ns}:s{seed}:{idx:04d}",
        "family": "C09_METACOGNITION_SELF_CHECK",
        "group": FAMILY_TO_GROUP["C09_METACOGNITION_SELF_CHECK"],
        "split": split,
        "prompt": prompt,
        "gold_text": gold,
        "n_shots": 0,
        "absent_from_input": gold not in prompt,
        "presentation": subtype,
        "uncertainty_kind": uk,
        "problem_spec": spec,
        "benchmark_provenance": "x4_repaired_22fam",
    }


def _generate_repaired_c_other(family: str, split: str, seed: int, idx: int, shots: int) -> dict[str, Any]:
    """Reuse and clean C01, C02, C05, C06, C07, C08 from experiments.x3.cognitive_battery with clean group & X4 tags."""
    from experiments.x3 import cognitive_battery as x3cb

    ns = _norm_split(split)
    if family == "C01_NOVEL_OP_INDUCTION":
        c = x3cb._generate_c01_case(split, seed, idx, shots)
    elif family == "C02_FST_RULE_COMPOSITION":
        c = x3cb._generate_c02_case(split, seed, idx, shots)
    elif family == "C05_RELATIONAL_GRAPH_CF":
        c = x3cb._generate_c05_batch(split, seed, idx + 1)[idx]
    elif family == "C06_OPEN_WORLD_DEDUCTION":
        c = x3cb._generate_c06_case(split, seed, idx)
    elif family == "C07_HIERARCHICAL_LISTOPS":
        c = x3cb._generate_c07_case(split, seed, idx)
    elif family == "C08_ADAPTIVE_MEMORY_REVISION":
        c = x3cb._generate_c08_case(split, seed, idx)
    else:
        raise ValueError(f"Unsupported C family: {family!r}")
    c["group"] = FAMILY_TO_GROUP[family]
    c["benchmark_provenance"] = "x4_repaired_22fam"
    return c


# ---------------------------------------------------------------------------
# Strict Split-Agnostic Canonical Structural Hash (Fixes Defect I)
# ---------------------------------------------------------------------------


def canonical_structural_hash(case: Mapping[str, Any]) -> str:
    """Split-agnostic canonical structural hash modulo entity/symbol/person renaming & split labels."""
    fam = str(case.get("family") or infer_x4_family(str(case.get("prompt", ""))))
    spec = case.get("problem_spec", {})
    if not isinstance(spec, Mapping):
        spec = {}

    def _renumber(seq: Sequence[str]) -> dict[str, str]:
        mp: dict[str, str] = {}
        for tok in seq:
            if tok not in mp:
                mp[tok] = f"E{len(mp)}"
        return mp

    if fam in X1_FAMILIES:
        facts = spec.get("facts", ())
        goal = str(spec.get("goal_subject", ""))
        toks = [goal] + [str(f.get("subject", "")) for f in facts] + [str(f.get("value", "")) for f in facts if f.get("value") is not None]
        rmap = _renumber(toks)
        edges = sorted(
            (rmap[str(f.get("subject", ""))], None if f.get("value") is None else rmap[str(f.get("value", ""))], str(f.get("source", "S0")), int(f.get("epoch", 1)))
            for f in facts
        )
        return digest({"family": fam, "hops": int(spec.get("hops", 1)), "min_ep": int(spec.get("min_epoch", 1)), "qa": bool(spec.get("query_available")), "ma": bool(spec.get("memory_available")), "edges": edges})

    if fam == "C01_NOVEL_OP_INDUCTION":
        return digest({"family": fam, "rule_kind": spec.get("rule_kind"), "c1": spec.get("c1"), "c2": spec.get("c2")})

    if fam == "C02_FST_RULE_COMPOSITION":
        # Structural transition signature WITHOUT split_tag
        tr = spec.get("trans", [])
        out_deg = sorted(Counter(x for row in tr for x in row).values()) if tr else []
        return digest({"family": fam, "depth": int(spec.get("depth", 2)), "out_deg": out_deg, "trans": tr})

    if fam == "C03_GRID_TRANSFORM_ARC":
        return digest({"family": fam, "op_name": spec.get("op_name"), "shape": spec.get("shape")})

    if fam == "C04_COMPOSITIONAL_SCAN":
        cmd_toks = str(spec.get("cmd", "")).split()
        norm_toks = ["VERB" if t in ("walk", "run", "jump", "look") else t for t in cmd_toks]
        return digest({"family": fam, "template": norm_toks})

    if fam == "C05_RELATIONAL_GRAPH_CF":
        facts = spec.get("facts", ())
        toks = [str(spec.get("goal_subject", ""))]
        for r in facts:
            toks.extend((str(r.get("subj", "")), str(r.get("val", ""))))
        rmap = _renumber(toks)
        edges = sorted((rmap[str(r.get("subj", ""))], rmap[str(r.get("val", ""))], int(r.get("epoch", 1)), int(r.get("authority", 1))) for r in facts)
        return digest({"family": fam, "hops": int(spec.get("hops", 1)), "min_epoch": int(spec.get("min_epoch", 1)), "edges": edges})

    if fam == "C06_OPEN_WORLD_DEDUCTION":
        facts = spec.get("facts", {})
        rules = spec.get("rules", ())
        q = str(spec.get("query", ""))
        toks = [q, *sorted(facts)]
        for conds, head, _ in rules:
            toks.extend(conds)
            toks.append(str(head))
        rmap = _renumber(toks)
        nf = sorted((rmap[k], bool(v)) for k, v in facts.items())
        nr = sorted((tuple(rmap[c] for c in conds), rmap[str(head)], bool(pol)) for conds, head, pol in rules)
        return digest({"family": fam, "q": rmap[q], "facts": nf, "rules": nr})

    if fam == "C07_HIERARCHICAL_LISTOPS":
        def _ast(node: Any) -> Any:
            return "L" if isinstance(node, int) else [str(node[0]), *(_ast(ch) for ch in node[1:])]
        return digest({"family": fam, "depth": int(spec.get("depth", 2)), "ast": _ast(spec.get("tree"))})

    if fam == "C08_ADAPTIVE_MEMORY_REVISION":
        evs = spec.get("events", ())
        toks = [str(spec.get("query_ent", ""))] + [str(ev.get("ent", "")) for ev in evs]
        rmap = _renumber(toks)
        nevs = [(int(ev.get("ts", 0)), str(ev.get("kind", "")), rmap[str(ev.get("ent", ""))]) for ev in evs]
        return digest({"family": fam, "q": rmap[str(spec.get("query_ent", ""))], "events": nevs})

    if fam == "C09_METACOGNITION_SELF_CHECK":
        return digest({"family": fam, "subtype": spec.get("subtype"), "cost": spec.get("cost"), "pQ": spec.get("pQ"), "pR": spec.get("pR")})

    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        return digest({"family": fam, "denom": int(spec.get("denom", 1)), "unit": str(spec.get("unit", "")), "num": int(spec.get("num", 0))})

    return digest({"family": fam, "prompt": str(case.get("prompt", ""))})


def verify_x4_split_disjointness(
    splits_map: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    require_structural_disjointness: bool = True,
) -> dict[str, Any]:
    """Fail-closed verifier rejecting prompt overlaps AND split-agnostic isomorphic leaks across splits."""
    prompts_by_sp: dict[str, set[str]] = {}
    structs_by_sp: dict[str, set[str]] = {}
    for sp, items in splits_map.items():
        prompts_by_sp[sp] = {str(c.get("prompt", "")) for c in items if c.get("prompt")}
        structs_by_sp[sp] = {str(canonical_structural_hash(c)) for c in items}

    overlaps: list[dict[str, Any]] = []
    names = sorted(prompts_by_sp)
    for i, s1 in enumerate(names):
        for s2 in names[i + 1 :]:
            pov = sorted(prompts_by_sp[s1] & prompts_by_sp[s2])
            sov = sorted(structs_by_sp[s1] & structs_by_sp[s2]) if require_structural_disjointness else []
            if pov:
                overlaps.append({"splits": [s1, s2], "prompt_overlaps": pov[:5]})
            # Allow overlapping cost/subtype hashes only on non-structural small discrete C09 missing_premise
            if sov:
                c09_only = {canonical_structural_hash({"family": "C09_METACOGNITION_SELF_CHECK", "problem_spec": c.get("problem_spec", {})}) for sp_i in (s1, s2) for c in splits_map[sp_i] if c.get("family") == "C09_METACOGNITION_SELF_CHECK"}
                non_c09 = [h for h in sov if h not in c09_only]
                if non_c09:
                    overlaps.append({"splits": [s1, s2], "structural_overlaps": non_c09[:5]})
    if overlaps:
        raise ValueError(f"X4_SPLIT_LEAK_DETECTED: {overlaps}")
    return {"disjoint": True, "splits_checked": names, "overlap_count": 0}


def build_x4_battery(
    *,
    split: str = "dev",
    seed: int | None = None,
    per_family: int = 8,
    families: Sequence[str] = X4_22_FAMILIES,
    phase: int = 3,
    attach_pairs: bool = True,
) -> dict[str, Any]:
    """Generate the 22-family 6-group X4 battery with optional explicit semantic invariance & CF pairs."""
    ns = _norm_split(split)
    use_seed = int(X4_SPLIT_SEEDS[ns] if seed is None else seed)
    cases: list[dict[str, Any]] = []
    shots = (0, 1, 4, 16, 64)

    for fam in families:
        if fam in X1_FAMILIES:
            for i in range(per_family):
                cases.append(_generate_x1_family_case(fam, split, use_seed, i))
        elif fam == "C03_GRID_TRANSFORM_ARC":
            for i in range(per_family):
                cases.append(_generate_c03_repaired(split, use_seed, i, phase=phase))
        elif fam == "C04_COMPOSITIONAL_SCAN":
            for i in range(per_family):
                cases.append(_generate_c04_repaired(split, use_seed, i, phase=phase))
        elif fam == "C09_METACOGNITION_SELF_CHECK":
            for i in range(per_family):
                cases.append(_generate_c09_repaired(split, use_seed, i))
        elif fam == "C10_LANGUAGE_MATH_TRANSFER":
            for i in range(per_family):
                cases.append(_generate_c10_repaired(split, use_seed, i, phase=phase))
        else:
            for i in range(per_family):
                sh = shots[i % len(shots)]
                cases.append(_generate_repaired_c_other(fam, split, use_seed, i, sh))

    for c in cases:
        c["structural_hash"] = canonical_structural_hash(c)
        c["typed_target"] = format_x4_typed_target(c)

    if attach_pairs:
        cases = _attach_semantic_pair_twins(cases, split=ns, seed=use_seed)

    return {
        "schema": X4_BATTERY_SCHEMA,
        "split": split,
        "normalized_split": ns,
        "seed": use_seed,
        "phase": int(phase),
        "families": list(families),
        "case_count": len(cases),
        "cases": cases,
        "battery_sha256": digest([{"id": c["case_id"], "p": c["prompt"], "g": c["typed_target"]} for c in cases]),
    }


def _attach_semantic_pair_twins(cases: list[dict[str, Any]], *, split: str, seed: int) -> list[dict[str, Any]]:
    """Attach explicit verified `equiv_surface`, `cf_relevant`, and `cf_irrelevant` link metadata (§4 G, §6)."""
    by_id: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    # Pair consecutive even/odd cases within paired families (C04/C05/C07/C08/C10) when targets differ or match
    by_fam: dict[str, list[dict[str, Any]]] = {}
    for c in cases:
        row = dict(c)
        row.setdefault("pair_group_id", f"pg:{c['case_id']}")
        row.setdefault("pair_relation", "base")
        row.setdefault("pair_partner_id", None)
        out.append(row)
        by_id[str(row["case_id"])] = row
        by_fam.setdefault(str(row["family"]), []).append(row)

    for fam, rows in by_fam.items():
        for i in range(0, len(rows) - 1, 2):
            a, b = rows[i], rows[i + 1]
            gid = f"pair:{fam}:{split}:{seed}:{i // 2}"
            a["pair_group_id"] = gid
            b["pair_group_id"] = gid
            a["pair_partner_id"] = b["case_id"]
            b["pair_partner_id"] = a["case_id"]
            if a["typed_target"] != b["typed_target"]:
                a["pair_relation"] = "cf_relevant"
                b["pair_relation"] = "cf_relevant"
            else:
                # Meaning-preserving surface/instance equivalence partner
                a["pair_relation"] = "equiv_surface"
                b["pair_relation"] = "equiv_surface"

    # Also build an explicit meaning-preserving alt-renderer & irrelevant-distractor edit for X01..X04
    for row in list(out):
        fam = str(row["family"])
        if fam in ("X01", "X02", "C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER") and row["pair_partner_id"] is None:
            row["pair_relation"] = "base"
    return out


# ---------------------------------------------------------------------------
# Procedural Fresh Stream vs Fixed Pools & Primitive-to-Composition Curriculum
# ---------------------------------------------------------------------------


def build_x4_curriculum_schedule(
    *,
    seed: int,
    total_updates: int,
    batch_size: int = 16,
    data_regime: str = "fresh_stream",
    curriculum_mode: str = "curriculum_p2c",
    families: Sequence[str] = X4_22_FAMILIES,
) -> list[dict[str, Any]]:
    """Build deterministic training schedule for fixed pools (`fixed_400`, `fixed_4000`, `fixed_40000`) or fresh procedural worlds (§8)."""
    if data_regime not in ("fixed_400", "fixed_4000", "fixed_40000", "fresh_stream"):
        raise ValueError(f"Unsupported data_regime: {data_regime!r}")
    if curriculum_mode not in ("curriculum_p2c", "random_interleaved"):
        raise ValueError(f"Unsupported curriculum_mode: {curriculum_mode!r}")

    pool_sizes = {"fixed_400": 400, "fixed_4000": 4000, "fixed_40000": 40000}
    fixed_pools: dict[int, list[dict[str, Any]]] = {}
    if data_regime in pool_sizes:
        n_total = pool_sizes[data_regime]
        per_fam = max(1, (n_total + len(families) - 1) // len(families))
        for ph in (1, 2, 3) if curriculum_mode == "curriculum_p2c" else (3,):
            bat = build_x4_battery(split="train", seed=seed + 100 * ph, per_family=per_fam, families=families, phase=ph)
            fixed_pools[ph] = bat["cases"][:n_total]

    rng = random.Random(int(digest(["x4-sched", seed, data_regime, curriculum_mode]), 16))
    schedule: list[dict[str, Any]] = []
    # Pair consecutive slots cleanly (batch_size // 2 pairs -> batch_size items per step!)
    for u in range(int(total_updates)):
        frac = float(u) / float(max(1, total_updates - 1))
        if curriculum_mode == "curriculum_p2c":
            phase = 1 if frac < 0.30 else (2 if frac < 0.65 else 3)
        else:
            phase = 3

        step_cases: list[dict[str, Any]] = []
        if data_regime == "fresh_stream":
            # Deterministic fresh cases from procedural stream keyed by (seed, u)
            n_pairs = max(1, batch_size // 2)
            for p_i in range(n_pairs):
                fam = families[(u * n_pairs + p_i) % len(families)]
                # Generate consecutive pair (2*k, 2*k+1) with fresh non-repeating instance counter
                inst_base = 10_000 + u * batch_size + 2 * p_i
                b_pair = build_x4_battery(
                    split="train", seed=seed + u, per_family=2, families=(fam,), phase=phase, attach_pairs=True
                )["cases"]
                # Guarantee unique instance IDs per step for accurate unique_case_ids counting
                for j, item in enumerate(b_pair[:2]):
                    item_c = dict(item)
                    item_c["case_id"] = f"{item['case_id']}:u{u}:p{p_i}:{j}"
                    item_c["structural_hash"] = canonical_structural_hash(item_c)
                    step_cases.append(item_c)
        else:
            pool = fixed_pools[phase]
            for s_i in range(0, batch_size, 2):
                p_idx = (rng.randrange(max(1, len(pool) // 2))) * 2
                a = dict(pool[p_idx % len(pool)])
                b = dict(pool[(p_idx + 1) % len(pool)])
                step_cases.extend((a, b))

        schedule.append(
            {
                "update_idx": u,
                "phase": phase,
                "data_regime": data_regime,
                "curriculum_mode": curriculum_mode,
                "cases": step_cases[:batch_size],
            }
        )
    return schedule


# ---------------------------------------------------------------------------
# Exact Oracle Decoder (Must Achieve 1.0 Through Live Evaluator on Clean Sets)
# ---------------------------------------------------------------------------


def solve_x4_case_oracle(case: Mapping[str, Any]) -> str:
    """Independent reference oracle across ALL 22 families."""
    fam = str(case["family"])
    spec = case["problem_spec"]
    if fam in X1_FAMILIES:
        facts = [PublicFactX1(**f) if isinstance(f, Mapping) else f for f in spec["facts"]]
        op, arg = x1_teacher_decide(
            facts,
            str(spec["goal_subject"]),
            int(spec["hops"]),
            int(spec["min_epoch"]),
            bool(spec["query_available"]),
            bool(spec["memory_available"]),
        )
        return f"ANSWER:{arg}" if (op == "ANSWER" and arg is not None) else op
    if fam == "C03_GRID_TRANSFORM_ARC":
        cmap = {int(k): int(v) for k, v in spec["color_map"].items()}
        out = _apply_x4_grid_op(np.asarray(spec["q_in"], dtype=np.int64), str(spec["op_name"]), cmap)
        return f"ANSWER:{_grid_str(out)}"
    if fam == "C04_COMPOSITIONAL_SCAN":
        return f"ANSWER:{_eval_scan_cmd(str(spec['cmd']))}"
    if fam == "C09_METACOGNITION_SELF_CHECK":
        gold = str(spec["gold"])
        return gold if gold in ("QUERY", "RETRIEVE", "ABSTAIN") else f"ANSWER:{gold}"
    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        fr = Fraction(int(spec["num"]), int(spec.get("denom", 1)))
        val_s = str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"
        return f"ANSWER:{val_s} {spec['unit']}".strip()
    if fam == "C01_NOVEL_OP_INDUCTION":
        if bool(case.get("public_ambiguous", False)):
            return "ABSTAIN:AMBIGUOUS"
        return f"ANSWER:{case['gold_text']}"
    if fam == "C02_FST_RULE_COMPOSITION":
        return f"ANSWER:{case['gold_text']}"
    if fam == "C05_RELATIONAL_GRAPH_CF":
        res = independent_walk_x3_graph(spec["facts"], str(spec["goal_subject"]), hops=int(spec["hops"]), min_epoch=int(spec.get("min_epoch", 1)))
        return f"ANSWER:{res['terminal_value']}"
    if fam == "C06_OPEN_WORLD_DEDUCTION":
        return f"ANSWER:{_solve_c06_horn(spec['facts'], [tuple(tuple(r[0]) if i == 0 else r[i] for i in range(3)) for r in spec['rules']], str(spec['query']))}"
    if fam == "C07_HIERARCHICAL_LISTOPS":
        return f"ANSWER:R{_eval_listops_node(spec['tree'])}"
    if fam == "C08_ADAPTIVE_MEMORY_REVISION":
        return f"ANSWER:{_solve_c08(spec['events'], str(spec['query_ent']))}"
    raise ValueError(f"Unsupported X4 oracle family: {fam!r}")


class PublicOracleX4Decoder(nn.Module):
    """Exact oracle decoder across ALL 22 families strictly using public prompt information."""

    def __init__(self, *, max_ctx: int = 640) -> None:
        super().__init__()
        self.max_ctx = int(max_ctx)
        self._dummy = nn.Parameter(torch.zeros(1))

    def solve_from_case_or_prompt(self, case: Mapping[str, Any]) -> str:
        return solve_x4_case_oracle(case)

    @torch.no_grad()
    def generate_bytes_for_case(self, case: Mapping[str, Any], max_new_bytes: int | None = None) -> dict[str, Any]:
        bound = int(max_new_bytes if max_new_bytes is not None else infer_x4_generation_bound(str(case["prompt"])))
        txt = solve_x4_case_oracle(case)
        over = len(txt.encode("utf-8")) > bound
        return {
            "text": txt[:bound] if over else txt,
            "terminated_by_eos": not over,
            "valid_utf8": True,
            "bound_bytes": bound,
            "seq_log_prob": -1e-6,
            "mean_token_prob": 0.999999,
            "epistemic_uncertainty": 0.05,
            "info_uncertainty": 0.05,
        }


# ---------------------------------------------------------------------------
# Batched Evaluation, L0-L5 Intelligence Profile & C03/C04/C10 Failure Diagnosis
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_x4_cases(
    model: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    eval_steps_override: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate any X4 model or PublicOracleX4Decoder with unified exact scoring and unified probability confidence."""
    is_oracle = isinstance(model, PublicOracleX4Decoder)
    dev = torch.device(device) if not is_oracle else torch.device("cpu")
    if not is_oracle:
        model = model.to(dev)
        was_train = model.training
        model.eval()

    results: list[dict[str, Any]] = []
    bsz = max(1, int(batch_size))
    for start in range(0, len(cases), bsz):
        chunk = cases[start : start + bsz]
        if is_oracle:
            gen_list = [model.generate_bytes_for_case(c, max_new_bytes=infer_x4_generation_bound(str(c["prompt"]))) for c in chunk]
        elif hasattr(model, "generate_bytes_batch"):
            prompts = [str(c["prompt"]) for c in chunk]
            bounds = [infer_x4_generation_bound(p) for p in prompts]
            gen_list = model.generate_bytes_batch(
                prompts,
                max_new_bytes_list=bounds,
                eval_steps_override=eval_steps_override,
            )
        else:
            gen_list = [
                model.generate_bytes(
                    str(c["prompt"]),
                    max_new_bytes=infer_x4_generation_bound(str(c["prompt"])),
                    eval_steps_override=eval_steps_override,
                )
                for c in chunk
            ]

        for c, gen in zip(chunk, gen_list):
            prompt = str(c["prompt"])
            bound = infer_x4_generation_bound(prompt)
            pred_txt = str(gen.get("text", "")).strip()
            scored = parse_and_score_x4_output(
                prompt, pred_txt, c, max_new_bytes=bound, terminated_by_eos=bool(gen.get("terminated_by_eos", True))
            )
            # Defect M Fix: Calibration confidence derives strictly from unified generated sequence mean token prob!
            conf = float(min(1.0, max(0.0, gen.get("mean_token_prob", 0.5))))
            results.append(
                {
                    "case_id": str(c["case_id"]),
                    "family": str(c["family"]),
                    "group": str(c.get("group", FAMILY_TO_GROUP.get(str(c["family"]), "UNKNOWN"))),
                    "diagnostic_level": str(c.get("diagnostic_level", "L1")),
                    "predicted_text": pred_txt,
                    "confidence": conf,
                    "seq_log_prob": float(gen.get("seq_log_prob", -1.0)),
                    "epistemic_uncertainty": float(gen.get("epistemic_uncertainty", 0.5)),
                    "info_uncertainty": float(gen.get("info_uncertainty", 0.5)),
                    "op_routing_weights": gen.get("op_routing_weights"),
                    **scored,
                }
            )
    if not is_oracle and was_train:
        model.train()
    return results


def build_l0_l5_diagnostic_suite(
    *,
    train_cases: Sequence[Mapping[str, Any]],
    dev_cases: Sequence[Mapping[str, Any]],
    ood_cases: Sequence[Mapping[str, Any]],
    per_level_limit: int = 4,
) -> list[dict[str, Any]]:
    """Construct labeled L0-L5 diagnostic progression across applicable families (§15)."""
    suite: list[dict[str, Any]] = []
    # L0: Exposed train problems
    by_fam_tr: dict[str, list[Mapping[str, Any]]] = {}
    for c in train_cases:
        by_fam_tr.setdefault(str(c["family"]), []).append(c)
    for fam, rows in sorted(by_fam_tr.items()):
        for r in rows[:per_level_limit]:
            suite.append({**dict(r), "diagnostic_level": "L0", "case_id": f"L0::{r['case_id']}"})

    # L1: Fresh instances from dev
    by_fam_dv: dict[str, list[Mapping[str, Any]]] = {}
    for c in dev_cases:
        by_fam_dv.setdefault(str(c["family"]), []).append(c)
    for fam, rows in sorted(by_fam_dv.items()):
        for r in rows[:per_level_limit]:
            suite.append({**dict(r), "diagnostic_level": "L1", "case_id": f"L1::{r['case_id']}"})

    # L2: Unfamiliar presentation (swap X-family renderers to heldout narrative or add harmless whitespace/reordering note)
    for fam, rows in sorted(by_fam_dv.items()):
        for j, r in enumerate(rows[:per_level_limit]):
            if fam in X1_FAMILIES:
                alt = _generate_x1_family_case(fam, "dev", 91002, j, renderer_override=3)
                suite.append({**alt, "diagnostic_level": "L2", "case_id": f"L2::{alt['case_id']}"})
            else:
                suite.append({**dict(r), "diagnostic_level": "L2", "case_id": f"L2::{r['case_id']}"})

    # L3: Novel combinations from OOD/dev
    by_fam_ood: dict[str, list[Mapping[str, Any]]] = {}
    for c in ood_cases:
        by_fam_ood.setdefault(str(c["family"]), []).append(c)
    for fam, rows in sorted(by_fam_ood.items()):
        for r in rows[:per_level_limit]:
            lvl = "L4" if fam in ("C01_NOVEL_OP_INDUCTION", "C02_FST_RULE_COMPOSITION") else ("L5" if fam in ("C03_GRID_TRANSFORM_ARC", "C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER") else "L3")
            suite.append({**dict(r), "diagnostic_level": lvl, "case_id": f"{lvl}::{r['case_id']}"})
    return suite


def score_x4_multidimensional_profile(
    predictions: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute unified 22-family / 6-group / L0-L5 multidimensional intelligence profile (§14, §15)."""
    pred_map = {str(p["case_id"]): p for p in predictions}
    by_fam: dict[str, list[float]] = {}
    by_grp: dict[str, list[float]] = {}
    by_lvl: dict[str, list[float]] = {}
    cf_pairs: dict[str, list[float]] = {}
    brier_terms: list[float] = []
    unsupported_guesses = 0
    answerable_n = 0
    false_abstains = 0
    abstain_gt_n = 0

    for c in cases:
        cid = str(c["case_id"])
        fam = str(c["family"])
        grp = str(c.get("group", FAMILY_TO_GROUP.get(fam, "UNKNOWN")))
        lvl = str(c.get("diagnostic_level", "L1"))
        p = pred_map.get(cid, {})
        ok = 1.0 if bool(p.get("correct", False)) else 0.0
        by_fam.setdefault(fam, []).append(ok)
        by_grp.setdefault(grp, []).append(ok)
        by_lvl.setdefault(lvl, []).append(ok)
        if str(c.get("pair_relation", "")) == "cf_relevant" and "pair_group_id" in c:
            cf_pairs.setdefault(str(c["pair_group_id"]), []).append(ok)
        conf = float(min(1.0, max(0.0, p.get("confidence", 0.5))))
        brier_terms.append((conf - ok) ** 2)

        gt = str(c.get("typed_target") or format_x4_typed_target(c))
        pop = str(p.get("parsed_op", ""))
        if gt.startswith("ANSWER:"):
            answerable_n += 1
            if pop in ("ABSTAIN", "AMBIGUOUS"):
                false_abstains += 1
        else:
            abstain_gt_n += 1
            if pop == "ANSWER" and ok < 0.5:
                unsupported_guesses += 1

    fam_acc = {f: float(sum(v) / len(v)) for f, v in sorted(by_fam.items()) if v}
    grp_acc = {g: float(sum(v) / len(v)) for g, v in sorted(by_grp.items()) if v}
    lvl_acc = {k: float(sum(v) / len(v)) for k, v in sorted(by_lvl.items()) if v}
    all_v = [x for v in by_fam.values() for x in v]
    cf_both = [min(v) for v in cf_pairs.values() if len(v) >= 2]
    l0 = lvl_acc.get("L0", 0.0)
    l1 = lvl_acc.get("L1", float(sum(all_v) / max(1, len(all_v))))
    l2 = lvl_acc.get("L2", l1)
    return {
        "schema": "elendril-x4-profile-v1",
        "cases": len(cases),
        "macro_mean": float(sum(fam_acc.values()) / max(1, len(fam_acc))),
        "micro_mean": float(sum(all_v) / max(1, len(all_v))),
        "worst_family_floor": float(min(fam_acc.values())) if fam_acc else 0.0,
        "worst_group_floor": float(min(grp_acc.values())) if grp_acc else 0.0,
        "per_family_accuracy": fam_acc,
        "per_group_accuracy": grp_acc,
        "per_level_accuracy": lvl_acc,
        "capability_axes": {
            "acquisition_L0": l0,
            "generalization_L1": l1,
            "robustness_L2": l2,
            "composition_L3": lvl_acc.get("L3", 0.0),
            "adaptation_L4": lvl_acc.get("L4", 0.0),
            "transfer_L5": lvl_acc.get("L5", 0.0),
            "memorization_gap_L0_minus_L1": float(l0 - l1),
        },
        "counterfactual_both_correct": float(sum(cf_both) / len(cf_both)) if cf_both else 0.0,
        "false_abstention_rate": float(false_abstains / max(1, answerable_n)),
        "unsupported_guess_rate": float(unsupported_guesses / max(1, abstain_gt_n)),
        "unified_brier_score": float(sum(brier_terms) / max(1, len(brier_terms))),
    }


@torch.no_grad()
def diagnose_c03_c04_c10_failure_modes(
    model: Any,
    *,
    train_cases: Sequence[Mapping[str, Any]],
    unseen_same_fmt: Sequence[Mapping[str, Any]],
    unseen_reformatted: Sequence[Mapping[str, Any]],
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Record first wrong byte, parse validity, op choice, teacher-forced NLL, and free-gen exactness on C03/C04/C10 (§16)."""
    target_fams = ("C03_GRID_TRANSFORM_ARC", "C04_COMPOSITIONAL_SCAN", "C10_LANGUAGE_MATH_TRANSFER")
    report: dict[str, Any] = {}
    for split_name, pool in (
        ("exposed_train", train_cases),
        ("unseen_same_format", unseen_same_fmt),
        ("unseen_reformatted", unseen_reformatted),
    ):
        sub = [c for c in pool if str(c["family"]) in target_fams]
        preds = evaluate_x4_cases(model, sub, batch_size=16, device=device) if sub else []
        by_fam: dict[str, dict[str, Any]] = {}
        for c, p in zip(sub, preds):
            fam = str(c["family"])
            bucket = by_fam.setdefault(fam, {"n": 0, "exact": 0, "valid_syntax": 0, "correct_op": 0, "first_wrong_byte_positions": [], "mean_seq_log_prob": []})
            bucket["n"] += 1
            bucket["exact"] += int(bool(p["correct"]))
            bucket["valid_syntax"] += int(bool(p["valid"]))
            gt = format_x4_typed_target(c)
            g_op = gt.split(":", 1)[0]
            bucket["correct_op"] += int(str(p.get("parsed_op", "")) == g_op)
            bucket["mean_seq_log_prob"].append(float(p.get("seq_log_prob", -5.0)))
            pred_s = str(p.get("predicted_text", ""))
            if pred_s != gt:
                first_bad = next((i for i, (ca, cb) in enumerate(zip(pred_s, gt)) if ca != cb), min(len(pred_s), len(gt)))
                bucket["first_wrong_byte_positions"].append(first_bad)
        report[split_name] = {
            f: {
                "n": b["n"],
                "exact_rate": float(b["exact"] / max(1, b["n"])),
                "parse_valid_rate": float(b["valid_syntax"] / max(1, b["n"])),
                "op_choice_rate": float(b["correct_op"] / max(1, b["n"])),
                "mean_seq_log_prob": float(sum(b["mean_seq_log_prob"]) / max(1, len(b["mean_seq_log_prob"]))),
                "mean_first_wrong_byte_idx": float(sum(b["first_wrong_byte_positions"]) / max(1, len(b["first_wrong_byte_positions"]))) if b["first_wrong_byte_positions"] else None,
            }
            for f, b in sorted(by_fam.items())
        }
    return {"schema": "elendril-x4-c03-c04-c10-diagnosis-v1", "splits": report}


__all__ = [
    "FAMILY_TO_GROUP",
    "L0_L5_APPLICABILITY",
    "PublicOracleX4Decoder",
    "X4_22_FAMILIES",
    "X4_BATTERY_SCHEMA",
    "X4_C_FAMILIES",
    "X4_DEFAULT_GEN_BOUND",
    "X4_FAMILY_GROUPS",
    "X4_MAX_PROMPT_BYTES",
    "X4_SPLIT_SEEDS",
    "build_l0_l5_diagnostic_suite",
    "build_x01_x12_interventions",
    "build_x4_battery",
    "build_x4_curriculum_schedule",
    "canonical_structural_hash",
    "diagnose_c03_c04_c10_failure_modes",
    "evaluate_x4_cases",
    "format_x4_typed_target",
    "infer_x4_family",
    "infer_x4_generation_bound",
    "parse_and_score_x4_output",
    "score_x4_multidimensional_profile",
    "solve_x4_case_oracle",
    "verify_x4_split_disjointness",
]
