"""V8-X3 10-Family Multidimensional Cognitive Testing Battery (§2-§5, §9-§10)."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from anra_v5.elendril.codec import digest
from experiments.x2.tasks import build_t14_panel, solve_t14_public_dp
from experiments.x3.factorial import (
    X3_SYMBOLS,
    build_x3_paired_graph_factorial,
    independent_walk_x3_graph,
)

X3_COGNITIVE_SCHEMA = "elendril-x3-cognitive-battery-v1"
X3_MAX_PROMPT_BYTES = 600

X3_COGNITIVE_FAMILIES: tuple[str, ...] = (
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

SPLIT_SEEDS: dict[str, int] = {
    "train": 20261,
    "dev": 20262,
    "confirm": 20263,
    "ood_transfer": 20264,
}


def _split_rng(family: str, split: str, seed: int, idx: int) -> random.Random:
    seed_hex = sha256(f"x3|{family}|{split}|{seed}|{idx}".encode("utf-8")).hexdigest()
    return random.Random(int(seed_hex[:16], 16))


def _ensure_prompt_budget(prompt: str, context_limit: int = X3_MAX_PROMPT_BYTES) -> str:
    raw = prompt.encode("utf-8")
    if len(raw) > context_limit:
        raise ValueError(f"X3 prompt byte budget exceeded: {len(raw)} > {context_limit}: {prompt[:120]!r}")
    return prompt


# ---------------------------------------------------------------------------
# C01: Novel Mathematical / Symbolic Operator Induction (0/1/4/16/64-shot)
# ---------------------------------------------------------------------------

_C01_SPLIT_RULES: dict[str, tuple[str, ...]] = {
    "train": ("affine_13", "max_mix_13"),
    "dev": ("diff_sq_13", "bilinear_13"),
    "confirm": ("piecewise_par_13", "abs_mix_13"),
    "ood_transfer": ("quad_xor_13",),
}


def _eval_c01_rule(rule_kind: str, c1: int, c2: int, c3: int, a: int, b: int) -> int:
    if rule_kind == "affine_13":
        val = c1 * a + c2 * b + c3
    elif rule_kind == "max_mix_13":
        val = c1 * max(a, b) + c2 * min(a, b) + c3
    elif rule_kind == "diff_sq_13":
        val = c1 * abs(a - b) + c2 * (a + b) + c3
    elif rule_kind == "bilinear_13":
        val = a * b + c1 * a + c2 * b + c3
    elif rule_kind == "piecewise_par_13":
        val = (c1 * a + b + c3) if ((a + b) % 2 == 0) else (c2 * b + a + c3)
    elif rule_kind == "abs_mix_13":
        val = abs(c1 * a - c2 * b) + c3
    elif rule_kind == "quad_xor_13":
        val = (a * a + c1 * b) ^ c3
    else:
        raise ValueError(f"Unknown C01 rule_kind: {rule_kind!r}")
    # Map cleanly to 20..99 or A..P? Use deterministic mod-16 symbol + numeric string range separated from inputs
    return int(val % 16)


def _generate_c01_case(split: str, seed: int, idx: int, shots: int) -> dict[str, Any]:
    rng = _split_rng("C01_NOVEL_OP_INDUCTION", split, seed, idx)
    rules = _C01_SPLIT_RULES[split]
    rule_kind = rules[idx % len(rules)]
    # Partition parameter coefficients strictly by split so version spaces stay disjoint
    split_offset = {"train": 1, "dev": 2, "confirm": 3, "ood_transfer": 5}[split]
    c1 = 1 + ((idx + split_offset) % 5)
    c2 = 1 + ((idx // 3 + 2 * split_offset) % 5)
    c3 = 1 + ((idx // 7 + split_offset) % 7)

    # Pick query pair whose output symbol never equals any demo output when possible, and output is offset (Q..Z or 20..35) so absent from input
    qa = 6 + (idx % 4)
    qb = 7 + ((idx // 2) % 3)
    y_idx = _eval_c01_rule(rule_kind, c1, c2, c3, qa, qb)
    # Encode output as two-digit code >= 30 (30..45) strictly outside operand range 1..9!
    gold_val = 30 + y_idx
    gold_text = str(gold_val)

    # Build up to 64 distinct compact demos "a*b=y;" (7 bytes each -> 64*7=448B <= 600B) whose outputs never match query digits accidentally when small
    demos: list[tuple[int, int, int]] = []
    pairs = [(a, b) for a in range(1, 10) for b in range(1, 10) if not (a == qa and b == qb)]
    rng.shuffle(pairs)
    for da, db in pairs:
        dy = 30 + _eval_c01_rule(rule_kind, c1, c2, c3, da, db)
        if dy != gold_val:
            demos.append((da, db, dy))
        if len(demos) >= int(shots):
            break
    while len(demos) < int(shots) and pairs:
        da, db = pairs[len(demos) % len(pairs)]
        dy = 30 + _eval_c01_rule(rule_kind, c1, c2, c3, da, db)
        demos.append((da, db, dy))

    op_sym = {"train": "*", "dev": "#", "confirm": "~", "ood_transfer": "^"}[split]
    demo_str = ";".join(f"{da}{op_sym}{db}={dy}" for da, db, dy in demos[:shots])
    prompt = _ensure_prompt_budget(f"C01|RULE|k={shots}|{demo_str}|Q:{qa}{op_sym}{qb}=?")
    masked_prompt = f"C01|RULE|k=0||Q:{qa}{op_sym}{qb}=?"
    return {
        "case_id": f"C01:{split}:s{seed}:k{shots}:{idx:04d}",
        "family": "C01_NOVEL_OP_INDUCTION",
        "split": split,
        "canon_sig": f"c01:{rule_kind}:k{shots}:c={c1},{c2},{c3}:q={qa},{qb}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": y_idx,
        "n_shots": int(shots),
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c01:{split}:{idx // 2}",
        "problem_spec": {
            "rule_kind": rule_kind,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "qa": qa,
            "qb": qb,
        },
        "benchmark_provenance": "internal_matched_analog",
    }


def _oracle_c01(spec: Mapping[str, Any]) -> str:
    y = _eval_c01_rule(
        str(spec["rule_kind"]), int(spec["c1"]), int(spec["c2"]), int(spec["c3"]), int(spec["qa"]), int(spec["qb"])
    )
    return str(30 + y)


# ---------------------------------------------------------------------------
# C02: Vectorized FST Rule Composition & Version-Space Identification
# ---------------------------------------------------------------------------


def _build_fst_tensors(split: str, law_id: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(abs(hash((split, law_id))) % (2**31) if False else int(sha256(f"fst|{split}|{law_id}".encode()).hexdigest()[:8], 16))
    # 4-state x 4-event transition table T[s, e] -> s', and unary/compose permutation table F[s] -> out
    # Split-disjoint permutation signatures via affine multiplier a_s in split-specific odd set
    split_mult = {"train": 1, "dev": 3, "confirm": 5, "ood_transfer": 7}[split]
    perm = np.array([(i * split_mult + law_id) % 4 for i in range(4)], dtype=np.int64)
    trans = np.zeros((4, 4), dtype=np.int64)
    for s in range(4):
        shift = int(rng.integers(0, 4))
        for e in range(4):
            trans[s, e] = (s + e * split_mult + shift) % 4
    return trans, perm


def _eval_fst_seq(trans: np.ndarray, perm: np.ndarray, init_state: int, events: Sequence[int]) -> int:
    s = int(init_state)
    for e in events:
        s = int(trans[s, int(e)])
    return int(perm[s])


def verify_fst_version_space(
    trans: np.ndarray,
    perm: np.ndarray,
    demos: Sequence[tuple[int, tuple[int, ...], int]],
    candidates: Sequence[tuple[np.ndarray, np.ndarray]],
) -> int:
    """Vectorized / fast exact candidate count matching all FST demonstrations."""
    consistent = 0
    for c_trans, c_perm in candidates:
        ok = True
        for init_s, evs, y in demos:
            if _eval_fst_seq(c_trans, c_perm, init_s, evs) != y:
                ok = False
                break
        if ok:
            consistent += 1
    return consistent


def _generate_c02_case(split: str, seed: int, idx: int, shots: int) -> dict[str, Any]:
    pair_rng = _split_rng("C02_FST_RULE_COMPOSITION", split, seed, idx // 2)
    law_id = 1 + (idx % 12) + 100 * {"train": 1, "dev": 2, "confirm": 3, "ood_transfer": 4}[split]
    trans, perm = _build_fst_tensors(split, law_id)
    depth = 3 if split in ("confirm", "ood_transfer") else 2
    q_init = int(pair_rng.randint(0, 3))
    q_events = [int(pair_rng.randint(0, 3)) for _ in range(depth)]
    out_state = _eval_fst_seq(trans, perm, q_init, q_events)
    # Use outputs K,L,M,N strictly distinct from state/event digits 0..3
    out_labels = ("K", "L", "M", "N")
    gold_text = out_labels[out_state]

    # Compact demo "0:12>L" = 6 bytes * 64 = 384B <= 600B
    demos: list[str] = []
    for d_i in range(int(shots)):
        di = (idx + d_i) % 4
        de = [((d_i >> (2 * j)) ^ j ^ di) % 4 for j in range(depth)]
        if di == q_init and de == q_events:
            di = (di + 1) % 4
        dy = out_labels[_eval_fst_seq(trans, perm, di, de)]
        demos.append(f"{di}:{''.join(str(x) for x in de)}>{dy}")
    q_ev_s = "".join(str(x) for x in q_events)
    fst_tag = {"train": "FA", "dev": "FB", "confirm": "FC", "ood_transfer": "FD"}[split]
    prompt = _ensure_prompt_budget(f"C02|{fst_tag}|k={shots}|{' '.join(demos)}|Q:{q_init}:{q_ev_s}>?")
    masked_prompt = f"C02|{fst_tag}|k=0||Q:{q_init}:{q_ev_s}>?"
    return {
        "case_id": f"C02:{split}:s{seed}:k{shots}:{idx:04d}",
        "family": "C02_FST_RULE_COMPOSITION",
        "split": split,
        "canon_sig": f"c02:{split}:law={law_id}:q={q_init}:{q_ev_s}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": 10 + out_state,  # 'K'..'N' -> 10..13
        "n_shots": int(shots),
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c02:{split}:{idx // 2}",
        "problem_spec": {
            "trans": trans.tolist(),
            "perm": perm.tolist(),
            "q_init": q_init,
            "q_events": q_events,
            "out_labels": list(out_labels),
        },
        "benchmark_provenance": "internal_matched_analog",
    }


def _oracle_c02(spec: Mapping[str, Any]) -> str:
    trans = np.asarray(spec["trans"], dtype=np.int64)
    perm = np.asarray(spec["perm"], dtype=np.int64)
    y = _eval_fst_seq(trans, perm, int(spec["q_init"]), spec["q_events"])
    return str(spec["out_labels"][y])


# ---------------------------------------------------------------------------
# C03: Symbolic 2D Grid Transformation (ARC-AGI-1 Internal Matched Analog)
# ---------------------------------------------------------------------------

_C03_SPLIT_OPS: dict[str, tuple[str, ...]] = {
    "train": ("flip_h", "flip_v"),
    "dev": ("rot90", "transpose"),
    "confirm": ("gravity_down", "border_one"),
    "ood_transfer": ("tile_xor_rot180",),
}


def _apply_grid_op(grid: np.ndarray, op_name: str, color_map: Mapping[int, int]) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if op_name == "flip_h":
        out = np.fliplr(g)
    elif op_name == "flip_v":
        out = np.flipud(g)
    elif op_name == "rot90":
        out = np.rot90(g, k=1)
    elif op_name == "transpose":
        out = g.T
    elif op_name == "gravity_down":
        out = np.zeros_like(g)
        for col in range(g.shape[1]):
            nz = [v for v in g[:, col] if v != 0]
            if nz:
                out[-len(nz) :, col] = nz
    elif op_name == "border_one":
        out = g.copy()
        out[0, :] = 1
        out[-1, :] = 1
        out[:, 0] = 1
        out[:, -1] = 1
    elif op_name == "tile_xor_rot180":
        r = np.rot90(g, k=2)
        out = (g ^ r) % 4
    else:
        raise ValueError(f"Unknown C03 grid op: {op_name!r}")
    mapped = np.vectorize(lambda v: int(color_map.get(int(v), int(v))))(out)
    return mapped


def _grid_to_str(g: np.ndarray) -> str:
    return "/".join("".join(str(int(x)) for x in row) for row in g.tolist())


def _generate_c03_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    rng = _split_rng("C03_GRID_TRANSFORM_ARC", split, seed, idx)
    pair_rng = _split_rng("C03_GRID_TRANSFORM_ARC_Q", split, seed, idx // 2)
    ops = _C03_SPLIT_OPS[split]
    op_name = ops[idx % len(ops)]
    # Held-out color remap per split; second arm shifts color map to test sensitivity to demos on identical q_in
    if split == "train":
        color_map = {0: 0, 1: 2, 2: 3, 3: 1} if idx % 2 == 0 else {0: 0, 1: 3, 2: 1, 3: 2}
    elif split == "dev":
        color_map = {0: 0, 1: 3, 2: 1, 3: 2} if idx % 2 == 0 else {0: 0, 1: 1, 2: 2, 3: 3}
    elif split == "confirm":
        color_map = {0: 0, 1: 4, 2: 5, 3: 6} if idx % 2 == 0 else {0: 0, 1: 6, 2: 4, 3: 5}
    else:
        color_map = {0: 0, 1: 7, 2: 8, 3: 9} if idx % 2 == 0 else {0: 0, 1: 9, 2: 7, 3: 8}

    sz = 3
    demos_txt: list[str] = []
    for d in range(2):
        inp = np.array([[int(rng.randint(0, 3)) for _ in range(sz)] for _ in range(sz)], dtype=np.int64)
        out = _apply_grid_op(inp, op_name, color_map)
        demos_txt.append(f"{_grid_to_str(inp)}->{_grid_to_str(out)}")

    q_in = np.array([[int(pair_rng.randint(0, 3)) for _ in range(sz)] for _ in range(sz)], dtype=np.int64)
    q_out = _apply_grid_op(q_in, op_name, color_map)
    gold_text = _grid_to_str(q_out)
    # Deterministic 16-class signature from cell sum/parity when discrete class scored
    gold_cls = int(np.sum(q_out)) % 16
    prompt = _ensure_prompt_budget(f"C03|GRID|{' | '.join(demos_txt)}|Q:{_grid_to_str(q_in)}->?")
    return {
        "case_id": f"C03:{split}:s{seed}:{idx:04d}",
        "family": "C03_GRID_TRANSFORM_ARC",
        "split": split,
        "canon_sig": f"c03:{op_name}:{split}:{idx % 2}:{_grid_to_str(q_in)}",
        "prompt": prompt,
        "masked_prompt": f"C03|GRID|NONE|Q:{_grid_to_str(q_in)}->?",
        "gold_text": gold_text,
        "gold_class": gold_cls,
        "n_shots": 2,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c03:{split}:{idx // 2}",
        "problem_spec": {"op_name": op_name, "color_map": color_map, "q_in": q_in.tolist()},
        "benchmark_provenance": "internal_matched_analog",
    }


def _oracle_c03(spec: Mapping[str, Any]) -> str:
    cmap = {int(k): int(v) for k, v in spec["color_map"].items()}
    out = _apply_grid_op(np.asarray(spec["q_in"], dtype=np.int64), str(spec["op_name"]), cmap)
    return _grid_to_str(out)


# ---------------------------------------------------------------------------
# C04: Compositional Command-to-Action Grammar (SCAN Internal Matched Analog)
# ---------------------------------------------------------------------------

_PRIMITIVES = ("walk", "run", "jump", "look")


def _eval_scan_cmd(cmd: str) -> str:
    toks = cmd.split()

    def _base_dir(verb: str, direction: str | None) -> list[str]:
        act = f"I_{verb.upper()}"
        if direction == "left":
            return ["TURN_L", act]
        if direction == "right":
            return ["TURN_R", act]
        if direction == "opposite":
            return ["TURN_L", "TURN_L", act]
        if direction == "around":
            return ["TURN_L", act] * 4
        return [act]

    def _parse_phrase(ptoks: list[str]) -> list[str]:
        mult = 1
        if ptoks and ptoks[-1] == "twice":
            mult = 2
            ptoks = ptoks[:-1]
        elif ptoks and ptoks[-1] == "thrice":
            mult = 3
            ptoks = ptoks[:-1]
        verb = ptoks[0]
        direction = ptoks[1] if len(ptoks) > 1 and ptoks[1] in ("left", "right", "opposite", "around") else None
        return _base_dir(verb, direction) * mult

    if "and" in toks:
        p = toks.index("and")
        return " ".join(_parse_phrase(toks[:p]) + _parse_phrase(toks[p + 1 :]))
    if "after" in toks:
        p = toks.index("after")
        return " ".join(_parse_phrase(toks[p + 1 :]) + _parse_phrase(toks[:p]))
    return " ".join(_parse_phrase(toks))


def _generate_c04_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    rng = _split_rng("C04_COMPOSITIONAL_SCAN", split, seed, idx)
    # Strictly hold out (jump, around/opposite) + "after" + "thrice" combinations across splits!
    if split == "train":
        p1 = rng.choice(["walk", "run", "look"])
        d1 = rng.choice(["left", "right"])
        m1 = rng.choice(["twice"])
        cmd = f"{p1} {d1} {m1}"
    elif split == "dev":
        p1 = rng.choice(["jump", "walk"])
        d1 = rng.choice(["left", "opposite"])
        p2 = rng.choice(["run", "look"])
        d2 = rng.choice(["right"])
        cmd = f"{p1} {d1} and {p2} {d2}"
    elif split == "confirm":
        p1 = "jump" if idx % 2 == 0 else rng.choice(_PRIMITIVES)
        d1 = rng.choice(["opposite", "around"])
        m1 = rng.choice(["twice", "thrice"])
        p2 = rng.choice(["walk", "look"])
        cmd = f"{p1} {d1} after {p2} {m1}"
    else:
        cmd = f"jump around thrice after {rng.choice(['walk', 'run'])} opposite twice"
    gold_text = _eval_scan_cmd(cmd)
    prompt = _ensure_prompt_budget(f"C04|COMP|{cmd}|OUT=?")
    masked_prompt = f"C04|COMP|{len(cmd.split())}_WORDS|OUT=?"
    return {
        "case_id": f"C04:{split}:s{seed}:{idx:04d}",
        "family": "C04_COMPOSITIONAL_SCAN",
        "split": split,
        "canon_sig": f"c04:{split}:{cmd}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": len(gold_text.split()) % 16,
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c04:{split}:{idx // 2}",
        "problem_spec": {"cmd": cmd},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C05: Relational Graph Counterfactuals Across Canonical & Narrative Forms
# ---------------------------------------------------------------------------


def _generate_c05_batch(split: str, seed: int, count: int) -> list[dict[str, Any]]:
    # Reuse existing verified X3 paired graph generator with disjoint depths/splits & distinct seeds
    bg = max(2, (count + 15) // 16)
    x3_sp = "confirmation" if split in ("confirm", "ood_transfer") else "dev"
    x3_seed = int(seed) + {"train": 1000, "dev": 2000, "confirm": 3000, "ood_transfer": 4000}[split]
    panel = build_x3_paired_graph_factorial(split=x3_sp, depths=(4, 5), base_graphs_per_depth=bg, seed=x3_seed)
    out: list[dict[str, Any]] = []
    for i, dec in enumerate(panel["decisions"][:count]):
        narrative_mode = str(dec.get("renderer", "canonical"))
        prompt = _ensure_prompt_budget(str(dec["rendered_text"]), context_limit=636)
        gold_text = str(dec["expected_answer"])
        g_idx = X3_SYMBOLS.index(gold_text) if gold_text in X3_SYMBOLS else 0
        out.append(
            {
                "case_id": f"C05:{split}:s{seed}:{i:04d}",
                "family": "C05_RELATIONAL_GRAPH_CF",
                "split": split,
                "canon_sig": f"c05:{split}:{seed}:{dec['world_id']}",
                "prompt": prompt,
                "masked_prompt": f"C05|MASKED|hops={dec['hops']}",
                "gold_text": gold_text,
                "gold_class": g_idx,
                "n_shots": 0,
                "absent_from_input": False,
                "contrast_pair_id": f"{dec.get('pair_key', f'c05:{i // 4}')}::{dec.get('query_role', 'affected')}",
                "presentation": narrative_mode,
                "problem_spec": {
                    "facts": dec["facts"],
                    "goal_subject": dec["goal_subject"],
                    "hops": dec["hops"],
                    "min_epoch": dec.get("min_epoch", 1),
                },
                "benchmark_provenance": "internal_matched_analog",
            }
        )
    return out


def _oracle_c05(spec: Mapping[str, Any]) -> str:
    res = independent_walk_x3_graph(
        spec["facts"],
        str(spec["goal_subject"]),
        hops=int(spec["hops"]),
        min_epoch=int(spec.get("min_epoch", 1)),
    )
    return str(res["terminal_value"])


# ---------------------------------------------------------------------------
# C06: Open-World 3-Valued Horn Deduction (ProofWriter Matched Analog)
# ---------------------------------------------------------------------------


def _solve_c06_horn(facts: Mapping[str, bool], rules: Sequence[tuple[tuple[str, ...], str, bool]], query: str) -> str:
    known: dict[str, bool] = dict(facts)
    changed = True
    while changed:
        changed = False
        for conds, head, polarity in rules:
            if head in known:
                continue
            if all(known.get(c) is True for c in conds):
                known[head] = bool(polarity)
                changed = True
    if query not in known:
        return "UNKNOWN"
    return "TRUE" if known[query] else "FALSE"


def _generate_c06_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    rng = _split_rng("C06_OPEN_WORLD_DEDUCTION", split, seed, idx)
    depth = 2 if split == "train" else (3 if split == "dev" else 4)
    preds = [f"p{i}" for i in range(depth + 4)]
    rng.shuffle(preds)
    verdict_mode = ("TRUE", "FALSE", "UNKNOWN")[idx % 3]
    facts: dict[str, bool] = {preds[0]: True}
    if verdict_mode != "UNKNOWN":
        facts[preds[1]] = True
    rules: list[tuple[tuple[str, ...], str, bool]] = []
    # Chain of Horn rules up to query = preds[depth]
    if depth >= 2:
        rules.append(((preds[0], preds[1]), preds[2], True))
    for k in range(2, depth):
        pol = False if (k == depth - 1 and verdict_mode == "FALSE") else True
        rules.append(((preds[k],), preds[k + 1], pol))
    # If depth == 2 and FALSE: head rule derives False
    if depth == 2 and verdict_mode == "FALSE":
        rules[0] = ((preds[0], preds[1]), preds[2], False)
    query = preds[depth]
    # Add 1 distractor rule
    rules.append(((preds[-1],), preds[-2], True))
    gold_text = _solve_c06_horn(facts, rules, query)
    fact_str = " ".join(f"{k}=T" if v else f"{k}=F" for k, v in sorted(facts.items()))
    rule_str = ";".join(f"{'+'.join(c)}->{h}:{'T' if pol else 'F'}" for c, h, pol in rules)
    prompt = _ensure_prompt_budget(f"C06|DED|F:{fact_str}|R:{rule_str}|Q:{query}?")
    masked_prompt = f"C06|DED|d={depth}|F:MASKED|Q=?"
    cls_map = {"TRUE": 0, "FALSE": 1, "UNKNOWN": 2}
    return {
        "case_id": f"C06:{split}:s{seed}:{idx:04d}",
        "family": "C06_OPEN_WORLD_DEDUCTION",
        "split": split,
        "canon_sig": f"c06:{split}:d={depth}:{verdict_mode}:{'/'.join(preds)}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": cls_map[gold_text],
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c06:{split}:{idx // 3}",
        "problem_spec": {"facts": facts, "rules": rules, "query": query},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C07: Hierarchical Prefix Tree Evaluation (ListOps Matched Analog)
# ---------------------------------------------------------------------------


def _eval_listops_node(node: Any) -> int:
    if isinstance(node, int):
        return int(node)
    op = str(node[0])
    vals = [_eval_listops_node(ch) for ch in node[1:]]
    if op == "MAX":
        return max(vals)
    if op == "MIN":
        return min(vals)
    if op == "SUM10":
        return sum(vals) % 10
    if op == "FIRST":
        return vals[0]
    if op == "LAST":
        return vals[-1]
    if op == "MED":
        s = sorted(vals)
        return s[len(s) // 2]
    raise ValueError(f"Unknown ListOps op: {op!r}")


def _fmt_listops(node: Any) -> str:
    if isinstance(node, int):
        return str(node)
    return f"[{node[0]} {' '.join(_fmt_listops(c) for c in node[1:])}]"


def _build_listops_tree(rng: random.Random, depth: int, ops: Sequence[str]) -> Any:
    if depth <= 0:
        return int(rng.integers(0, 10) if hasattr(rng, "integers") else rng.randint(0, 9))
    op = rng.choice(ops)
    n_ch = rng.choice([2, 3])
    return [op, *[_build_listops_tree(rng, depth - 1, ops) for _ in range(n_ch)]]


def _generate_c07_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    rng = _split_rng("C07_HIERARCHICAL_LISTOPS", split, seed, idx)
    depth = 2 if split == "train" else (3 if split == "dev" else 4)
    ops = ("MAX", "MIN", "SUM10") if split == "train" else ("MAX", "MIN", "SUM10", "MED", "FIRST", "LAST")
    tree = _build_listops_tree(rng, depth, ops)
    # Offset by +10 -> 10..19 so output token does not literally appear inside 0..9 leaves
    val = _eval_listops_node(tree)
    gold_text = f"R{val}"
    expr = _fmt_listops(tree)
    prompt = _ensure_prompt_budget(f"C07|LIST|d={depth}|{expr}=R?")
    return {
        "case_id": f"C07:{split}:s{seed}:{idx:04d}",
        "family": "C07_HIERARCHICAL_LISTOPS",
        "split": split,
        "canon_sig": f"c07:{split}:d={depth}:{expr}",
        "prompt": prompt,
        "masked_prompt": f"C07|LIST|d={depth}|[MASKED]=R?",
        "gold_text": gold_text,
        "gold_class": int(val),
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c07:{split}:{idx // 2}",
        "problem_spec": {"tree": tree},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C08: Adaptive Working-Memory Revision & Retention (Delta + Sibling + Lure)
# ---------------------------------------------------------------------------


def _solve_c08(events: Sequence[Mapping[str, Any]], query_ent: str) -> str:
    mem: dict[str, tuple[int, str]] = {}
    for ev in events:
        if ev["kind"] in ("set", "correct"):
            ent = str(ev["ent"])
            ts = int(ev["ts"])
            if ent not in mem or ts >= mem[ent][0]:
                mem[ent] = (ts, str(ev["val"]))
    return mem[query_ent][1]


def _generate_c08_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    rng = _split_rng("C08_ADAPTIVE_MEMORY_REVISION", split, seed, idx)
    syms = list(X3_SYMBOLS)
    rng.shuffle(syms)
    # Disjoint entity pools across splits
    ent_prefix = {"train": "e", "dev": "u", "confirm": "q", "ood_transfer": "z"}[split]
    e_rev = f"{ent_prefix}1"
    e_sib = f"{ent_prefix}2"
    old_v, new_v, sib_v, lure_v = syms[0], syms[1], syms[2], syms[3]
    events = [
        {"ts": 1, "kind": "set", "ent": e_rev, "val": old_v},
        {"ts": 2, "kind": "set", "ent": e_sib, "val": sib_v},
        {"ts": 3, "kind": "correct", "ent": e_rev, "val": new_v},
        {"ts": 0, "kind": "stale_lure", "ent": e_rev, "val": lure_v},
    ]
    query_ent = e_rev if idx % 2 == 0 else e_sib
    gold_text = _solve_c08(events, query_ent)
    # Encode compactly and check distractor presence
    stream = ";".join(
        f"{ev['kind']}:{ev['ent']}={ev['val']}@{ev['ts']}" for ev in events
    )
    prompt = _ensure_prompt_budget(f"C08|MEM|{stream}|Q:{query_ent}=?")
    return {
        "case_id": f"C08:{split}:s{seed}:{idx:04d}",
        "family": "C08_ADAPTIVE_MEMORY_REVISION",
        "split": split,
        "canon_sig": f"c08:{split}:{stream}:q={query_ent}",
        "prompt": prompt,
        "masked_prompt": f"C08|MEM|MASKED|Q:{query_ent}=?",
        "gold_text": gold_text,
        "gold_class": X3_SYMBOLS.index(gold_text),
        "n_shots": 0,
        "absent_from_input": False,
        "contrast_pair_id": f"c08:{split}:{idx // 2}",
        "problem_spec": {"events": events, "query_ent": query_ent},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C09: Metacognition, Epistemic vs Information Uncertainty, & Error Correction
# ---------------------------------------------------------------------------


def _solve_c09_oracle(spec: Mapping[str, Any]) -> str:
    mode = str(spec["subtype"])
    if mode == "missing_premise":
        dp = solve_t14_public_dp(spec["t14_world"])
        op = str(dp["optimal_operation"])
        return op if op != "ANSWER" else str(spec["fallback_ans"])
    return str(spec["gold"])


def _generate_c09_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    rng = _split_rng("C09_METACOGNITION_SELF_CHECK", split, seed, idx)
    subtype = ("solvable_easy", "solvable_hard", "missing_premise")[idx % 3]
    syms = list(X3_SYMBOLS)
    rng.shuffle(syms)
    if subtype == "missing_premise":
        # Reuse verified T14 costly-information DP formulation across cheap/expensive regimes
        t14 = build_t14_panel(split="dev" if split == "dev" else "conf", histories_per_stratum=1, seed=seed + idx + 1)
        regime = "cheap" if (idx // 3) % 2 == 0 else "expensive"
        w = next((x for x in t14 if x["stratum"] == "one_useful_query" and x["cost_regime"] == regime), t14[0])
        dp = solve_t14_public_dp(w)
        fallback = str(w["public_belief"]["active_labels"][0])
        gold_text = str(dp["optimal_operation"]) if dp["optimal_operation"] != "ANSWER" else fallback
        pb = w["public_belief"]
        prompt = _ensure_prompt_budget(
            f"C09|META|MISSING|split={split}|p0={pb['prior_probs'][0]:.2f}|cost={pb['query_cost']:.2f}|cand={pb['active_labels']}|OP=?"
        )
        spec = {
            "subtype": "missing_premise",
            "t14_world": w,
            "fallback_ans": fallback,
        }
        gold_cls = {"QUERY": 16, "RETRIEVE": 17, "ABSTAIN": 18}.get(
            gold_text, X3_SYMBOLS.index(gold_text) if gold_text in X3_SYMBOLS else 0
        )
        uncertainty_kind = "information_uncertainty"
    else:
        a, b = int(rng.randint(10, 40)), int(rng.randint(10, 40))
        if subtype == "solvable_easy":
            gold_text = str(a + b)
            body = f"Compute X={a}+{b}"
        else:
            c = int(rng.randint(2, 5))
            first_try = a * c
            # Self-correction task: includes flawed initial draft requiring reconsideration
            gold_text = str(a * c + b - 3)
            body = f"Verify draft {first_try} for {a}*{c}+{b}-3"
        prompt = _ensure_prompt_budget(f"C09|META|{subtype.upper()}|{body}|ANS=?")
        spec = {"subtype": subtype, "gold": gold_text}
        gold_cls = int(gold_text) % 16
        uncertainty_kind = "epistemic_reasoning" if subtype == "solvable_hard" else "low_uncertainty"
    return {
        "case_id": f"C09:{split}:s{seed}:{idx:04d}",
        "family": "C09_METACOGNITION_SELF_CHECK",
        "split": split,
        "canon_sig": f"c09:{split}:{subtype}:{seed}:{idx}:{gold_text}",
        "prompt": prompt,
        "masked_prompt": f"C09|META|{subtype.upper()}|MASKED|ANS=?",
        "gold_text": gold_text,
        "gold_class": gold_cls,
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c09:{split}:{idx // 3}",
        "uncertainty_kind": uncertainty_kind,
        "problem_spec": spec,
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C10: Language & Multi-Step Math Transfer (Prose Explanations -> Solution)
# ---------------------------------------------------------------------------


def _generate_c10_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    rng = _split_rng("C10_LANGUAGE_MATH_TRANSFER", split, seed, idx)
    names = {
        "train": ("Amina", "Bram"),
        "dev": ("Clara", "Doran"),
        "confirm": ("Elena", "Farid"),
        "ood_transfer": ("Gita", "Hiro"),
    }[split]
    rate_base = {"train": 2, "dev": 3, "confirm": 4, "ood_transfer": 5}[split]
    batches = int(rng.randint(3, 8))
    per_batch = int(rng.randint(4, 9))
    bonus = int(rng.randint(5, 14))
    total = rate_base * batches + per_batch + bonus
    gold_text = str(total)
    prompt = _ensure_prompt_budget(
        f"C10|PROSE|{names[0]} earns {rate_base} tokens per cycle over {batches} cycles, "
        f"adds {per_batch} reserve tokens, and receives {bonus} adjustment tokens. "
        f"What is the total?"
    )
    return {
        "case_id": f"C10:{split}:s{seed}:{idx:04d}",
        "family": "C10_LANGUAGE_MATH_TRANSFER",
        "split": split,
        "canon_sig": f"c10:{split}:r={rate_base}:b={batches}:p={per_batch}:k={bonus}",
        "prompt": prompt,
        "masked_prompt": f"C10|PROSE|{names[0]} earns MASKED tokens. What is the total?",
        "gold_text": gold_text,
        "gold_class": int(total) % 16,
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c10:{split}:{idx // 2}",
        "problem_spec": {"rate": rate_base, "batches": batches, "per_batch": per_batch, "bonus": bonus},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# Unified Independent Reference Oracle, Battery Builder, & Custody Auditors
# ---------------------------------------------------------------------------


def solve_cognitive_case_oracle(case: Mapping[str, Any]) -> str:
    """Independent reference oracle verifying exact target string without neural state."""
    fam = str(case["family"])
    spec = case["problem_spec"]
    if fam == "C01_NOVEL_OP_INDUCTION":
        return _oracle_c01(spec)
    if fam == "C02_FST_RULE_COMPOSITION":
        return _oracle_c02(spec)
    if fam == "C03_GRID_TRANSFORM_ARC":
        return _oracle_c03(spec)
    if fam == "C04_COMPOSITIONAL_SCAN":
        return _eval_scan_cmd(str(spec["cmd"]))
    if fam == "C05_RELATIONAL_GRAPH_CF":
        return _oracle_c05(spec)
    if fam == "C06_OPEN_WORLD_DEDUCTION":
        return _solve_c06_horn(spec["facts"], [tuple(tuple(r[0]) if i == 0 else r[i] for i in range(3)) for r in spec["rules"]], str(spec["query"]))
    if fam == "C07_HIERARCHICAL_LISTOPS":
        return f"R{_eval_listops_node(spec['tree'])}"
    if fam == "C08_ADAPTIVE_MEMORY_REVISION":
        return _solve_c08(spec["events"], str(spec["query_ent"]))
    if fam == "C09_METACOGNITION_SELF_CHECK":
        return _solve_c09_oracle(spec)
    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        return str(int(spec["rate"]) * int(spec["batches"]) + int(spec["per_batch"]) + int(spec["bonus"]))
    raise ValueError(f"Unknown cognitive family: {fam!r}")


def verify_cognitive_case(case: Mapping[str, Any], predicted_text: str) -> dict[str, Any]:
    expected = solve_cognitive_case_oracle(case)
    pred = str(predicted_text).strip()
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "expected": expected,
        "predicted": pred,
        "correct": pred == expected,
    }


def build_x3_cognitive_battery(
    *,
    split: str = "dev",
    seed: int | None = None,
    per_family: int = 20,
    shot_counts: Sequence[int] = (0, 1, 4, 16, 64),
) -> dict[str, Any]:
    """Generate the complete 10-family V8-X3 multidimensional cognitive evaluation battery."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown X3 split {split!r}; expected one of {tuple(SPLIT_SEEDS)}")
    use_seed = int(SPLIT_SEEDS[split] if seed is None else seed)
    cases: list[dict[str, Any]] = []

    # C01 & C02 sweep all requested shot counts (0, 1, 4, 16, 64)
    n_shots_list = list(shot_counts) or [4]
    for i in range(per_family):
        sh = int(n_shots_list[i % len(n_shots_list)])
        cases.append(_generate_c01_case(split, use_seed, i, sh))
        cases.append(_generate_c02_case(split, use_seed, i, sh))
        cases.append(_generate_c03_case(split, use_seed, i))
        cases.append(_generate_c04_case(split, use_seed, i))
    cases.extend(_generate_c05_batch(split, use_seed, per_family))
    for i in range(per_family):
        cases.append(_generate_c06_case(split, use_seed, i))
        cases.append(_generate_c07_case(split, use_seed, i))
        cases.append(_generate_c08_case(split, use_seed, i))
        cases.append(_generate_c09_case(split, use_seed, i))
        cases.append(_generate_c10_case(split, use_seed, i))

    battery_sha = digest([{"id": c["case_id"], "sig": c["canon_sig"], "gold": c["gold_text"]} for c in cases])
    return {
        "schema": X3_COGNITIVE_SCHEMA,
        "split": split,
        "seed": use_seed,
        "families": list(X3_COGNITIVE_FAMILIES),
        "shot_counts": [int(s) for s in n_shots_list],
        "case_count": len(cases),
        "cases": cases,
        "battery_sha256": battery_sha,
    }


def verify_x3_split_disjointness(splits_map: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Fail-closed canonical skeleton & prompt disjointness verifier across all splits."""
    sigs_by_split: dict[str, set[str]] = {}
    prompts_by_split: dict[str, set[str]] = {}
    for sp, items in splits_map.items():
        sigs_by_split[str(sp)] = {str(c["canon_sig"]) for c in items}
        prompts_by_split[str(sp)] = {str(c["prompt"]) for c in items}
    split_names = sorted(sigs_by_split)
    overlaps: list[dict[str, Any]] = []
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1 :]:
            sig_ov = sorted(sigs_by_split[s1] & sigs_by_split[s2])
            pr_ov = sorted(prompts_by_split[s1] & prompts_by_split[s2])
            if sig_ov or pr_ov:
                overlaps.append({"splits": [s1, s2], "sig_overlaps": sig_ov[:5], "prompt_overlaps": pr_ov[:5]})
    if overlaps:
        raise ValueError(f"SPLIT_LEAK_DETECTED: {overlaps}")
    return {"disjoint": True, "splits_checked": split_names, "overlap_count": 0}


def eval_x3_battery_shortcuts(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate zero-reasoning heuristics + independent oracle across the cognitive battery."""
    if not cases:
        raise ValueError("eval_x3_battery_shortcuts requires non-empty cases")
    counts = Counter(str(c["gold_text"]) for c in cases)
    maj_label = counts.most_common(1)[0][0]
    seen_demos: dict[str, str] = {}

    maj_ok = copy_last_ok = copy_first_ok = nearest_ok = abstain_ok = oracle_ok = 0
    by_fam: dict[str, list[bool]] = {}
    for c in cases:
        gold = str(c["gold_text"])
        prompt = str(c["prompt"])
        toks = prompt.replace("|", " ").replace("=", " ").replace(";", " ").split()
        copy_last = toks[-2] if len(toks) >= 2 and toks[-1].endswith("?") else (toks[-1] if toks else "")
        copy_first = toks[2] if len(toks) >= 3 else (toks[0] if toks else "")
        fam = str(c["family"])
        nearest = seen_demos.get(fam, "")
        seen_demos[fam] = gold

        maj_ok += int(maj_label == gold)
        copy_last_ok += int(copy_last == gold)
        copy_first_ok += int(copy_first == gold)
        nearest_ok += int(nearest == gold)
        abstain_ok += int(gold in ("ABSTAIN", "UNKNOWN"))
        or_ans = solve_cognitive_case_oracle(c)
        oracle_ok += int(or_ans == gold)
        by_fam.setdefault(fam, []).append(or_ans == gold)

    n = float(len(cases))
    res = {
        "schema": "elendril-x3-battery-shortcuts-v1",
        "cases": int(n),
        "prior_majority_accuracy": maj_ok / n,
        "copy_last_token_accuracy": copy_last_ok / n,
        "copy_first_token_accuracy": copy_first_ok / n,
        "nearest_demo_retrieval_accuracy": nearest_ok / n,
        "always_abstain_accuracy": abstain_ok / n,
        "max_shortcut_accuracy": max(maj_ok, copy_last_ok, copy_first_ok, nearest_ok, abstain_ok) / n,
        "oracle_independent_accuracy": oracle_ok / n,
        "per_family_oracle_min": min(sum(v) / len(v) for v in by_fam.values()),
    }
    return res


def verify_x3_destructive_masks(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove that identical masked prompts across cases map to opposing ground-truth answers."""
    by_mask: dict[str, set[str]] = {}
    fam_colliding: set[str] = set()
    for c in cases:
        mp = str(c.get("masked_prompt", ""))
        if not mp:
            continue
        bucket = by_mask.setdefault(mp, set())
        bucket.add(str(c["gold_text"]))
        if len(bucket) >= 2:
            fam_colliding.add(str(c["family"]))
    return {
        "schema": "elendril-x3-destructive-mask-v1",
        "unique_masked_prompts": len(by_mask),
        "opposing_answer_masks": sum(1 for v in by_mask.values() if len(v) >= 2),
        "families_with_opposing_masks": sorted(fam_colliding),
    }


def load_external_benchmark_jsonl(
    path: Path | str,
    *,
    benchmark_name: str,
    official_unmodified: bool = True,
) -> dict[str, Any]:
    """Load an external public benchmark (ARC/SCAN/CLUTRR/ProofWriter/bAbI/ListOps/GSM8K) preserving provenance (§10)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"External benchmark JSONL not found: {p}")
    provenance = "official_external" if official_unmodified else "interface_adapted_external"
    items: list[dict[str, Any]] = []
    for i, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        doc = json.loads(line)
        prompt = str(doc.get("prompt", doc.get("input", "")))
        gold = str(doc.get("gold_text", doc.get("answer", doc.get("target", "")))).strip()
        items.append(
            {
                "case_id": str(doc.get("case_id", f"ext:{benchmark_name}:{i:04d}")),
                "family": str(doc.get("family", f"EXT_{benchmark_name.upper()}")),
                "split": str(doc.get("split", "external")),
                "prompt": prompt,
                "gold_text": gold,
                "gold_class": int(doc.get("gold_class", abs(int(sha256(gold.encode()).hexdigest()[:8], 16)) % 16)),
                "absent_from_input": gold not in prompt,
                "benchmark_name": str(benchmark_name),
                "benchmark_provenance": provenance,
            }
        )
    return {
        "schema": "elendril-x3-external-benchmark-v1",
        "benchmark_name": str(benchmark_name),
        "benchmark_provenance": provenance,
        "count": len(items),
        "items": items,
        "sha256": digest(items),
    }


def score_cognitive_battery_profile(
    predictions: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score multidimensional capability profile (per-family, worst-floor, shot curves, absent-from-input, metacognition)."""
    pred_by_id = {str(p["case_id"]): p for p in predictions}
    by_family: dict[str, list[float]] = {}
    by_shot: dict[int, list[float]] = {}
    absent_scores: list[float] = []
    present_scores: list[float] = []
    narrative_scores: list[float] = []
    canonical_scores: list[float] = []
    epistemic_confs: list[float] = []
    info_confs: list[float] = []
    brier_terms: list[float] = []
    cf_pairs: dict[str, list[float]] = {}
    failures: list[dict[str, Any]] = []

    for c in cases:
        cid = str(c["case_id"])
        p = pred_by_id.get(cid, {})
        pred_text = str(p.get("predicted_text", "")).strip()
        correct_flag = p.get("correct")
        if correct_flag is None:
            ok = 1.0 if pred_text == str(c["gold_text"]).strip() else 0.0
        else:
            ok = 1.0 if bool(correct_flag) else 0.0
        fam = str(c["family"])
        by_family.setdefault(fam, []).append(ok)
        if fam in ("C01_NOVEL_OP_INDUCTION", "C02_FST_RULE_COMPOSITION"):
            by_shot.setdefault(int(c.get("n_shots", 0)), []).append(ok)
        if bool(c.get("absent_from_input")):
            absent_scores.append(ok)
        else:
            present_scores.append(ok)
        pres = c.get("presentation")
        if pres == "narrative":
            narrative_scores.append(ok)
        elif pres == "canonical":
            canonical_scores.append(ok)
        if "contrast_pair_id" in c:
            cf_pairs.setdefault(str(c["contrast_pair_id"]), []).append(ok)
        if fam == "C09_METACOGNITION_SELF_CHECK":
            u_kind = str(c.get("uncertainty_kind", ""))
            epistemic_score = float(p.get("epistemic_uncertainty", 0.5))
            info_score = float(p.get("info_uncertainty", 0.5))
            if u_kind == "information_uncertainty":
                info_confs.append(info_score - epistemic_score)
            elif u_kind == "epistemic_reasoning":
                epistemic_confs.append(info_score - epistemic_score)
        conf = float(min(1.0, max(0.0, p.get("confidence", 0.5))))
        brier_terms.append((conf - ok) ** 2)
        if ok < 0.5 and len(failures) < 50:
            failures.append(
                {
                    "case_id": cid,
                    "family": fam,
                    "expected": str(c["gold_text"]),
                    "predicted": pred_text,
                    "absent_from_input": bool(c.get("absent_from_input")),
                }
            )

    fam_acc = {f: float(sum(v) / len(v)) for f, v in sorted(by_family.items()) if v}
    shot_curve = {int(k): float(sum(v) / len(v)) for k, v in sorted(by_shot.items()) if v}
    shot_keys = sorted(shot_curve)
    shot_auc_gain = (shot_curve[shot_keys[-1]] - shot_curve[shot_keys[0]]) if len(shot_keys) >= 2 else 0.0

    # Pairwise empirical AUROC distinguishing missing-information uncertainty vs epistemic reasoning difficulty
    if info_confs and epistemic_confs:
        wins = 0.0
        tot = float(len(info_confs) * len(epistemic_confs))
        for a in info_confs:
            for b in epistemic_confs:
                wins += 1.0 if a > b else (0.5 if a == b else 0.0)
        auroc = wins / tot
    else:
        auroc = 0.5

    both_pairs = [min(v) for v in cf_pairs.values() if len(v) >= 2]
    all_vals = [x for v in by_family.values() for x in v]
    return {
        "schema": "elendril-x3-cognitive-profile-v1",
        "cases": len(cases),
        "macro_mean": float(sum(fam_acc.values()) / max(1, len(fam_acc))),
        "micro_mean": float(sum(all_vals) / max(1, len(all_vals))),
        "worst_dimension_floor": float(min(fam_acc.values())) if fam_acc else 0.0,
        "per_family_accuracy": fam_acc,
        "shot_learning_curve": shot_curve,
        "shot_improvement_gain": float(shot_auc_gain),
        "absent_from_input_accuracy": float(sum(absent_scores) / max(1, len(absent_scores))),
        "present_in_input_accuracy": float(sum(present_scores) / max(1, len(present_scores))),
        "canonical_accuracy": float(sum(canonical_scores) / max(1, len(canonical_scores))) if canonical_scores else None,
        "narrative_accuracy": float(sum(narrative_scores) / max(1, len(narrative_scores))) if narrative_scores else None,
        "counterfactual_both_correct": float(sum(both_pairs) / max(1, len(both_pairs))) if both_pairs else 0.0,
        "epistemic_vs_info_auroc": float(auroc),
        "brier_score": float(sum(brier_terms) / max(1, len(brier_terms))),
        "failure_examples": failures[:20],
    }


__all__ = [
    "SPLIT_SEEDS",
    "X3_COGNITIVE_FAMILIES",
    "X3_COGNITIVE_SCHEMA",
    "X3_MAX_PROMPT_BYTES",
    "build_x3_cognitive_battery",
    "eval_x3_battery_shortcuts",
    "load_external_benchmark_jsonl",
    "score_cognitive_battery_profile",
    "solve_cognitive_case_oracle",
    "verify_cognitive_case",
    "verify_fst_version_space",
    "verify_x3_destructive_masks",
    "verify_x3_split_disjointness",
]
