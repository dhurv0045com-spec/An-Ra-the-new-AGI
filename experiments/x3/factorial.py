"""X3 4/5-hop paired graph factorial, independent walker, shortcut baselines & X2 crosswalk (§4)."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import random
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import digest
from experiments.x2.tasks import X2EnvironmentKernel, check_x2_encoding_eligibility

X3_SYMBOLS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOP")


def independent_walk_x3_graph(
    edges: Sequence[Mapping[str, Any]],
    start_node: str,
    *,
    hops: int,
    min_epoch: int = 1,
) -> dict[str, Any]:
    """Standalone graph walker completely independent of the X3 generator and X2 interpreter."""
    by_src: dict[str, list[Mapping[str, Any]]] = {}
    for row in edges:
        ep = int(row.get("epoch", 1))
        if ep < int(min_epoch):
            continue
        by_src.setdefault(str(row["subj"]), []).append(row)

    next_hop: dict[str, str] = {}
    ambiguous: set[str] = set()
    for src, candidates in by_src.items():
        top_auth = max(int(r.get("authority", 1)) for r in candidates)
        auth_filtered = [r for r in candidates if int(r.get("authority", 1)) == top_auth]
        top_ep = max(int(r.get("epoch", 1)) for r in auth_filtered)
        ep_filtered = [r for r in auth_filtered if int(r.get("epoch", 1)) == top_ep]
        targets = {str(r["val"]) for r in ep_filtered}
        if len(targets) != 1:
            ambiguous.add(src)
        else:
            next_hop[src] = next(iter(targets))

    cur = str(start_node)
    path: list[str] = [cur]
    for step in range(int(hops)):
        if cur in ambiguous:
            return {"reachable": False, "terminal_value": None, "path": tuple(path), "hops": step, "ambiguous": True}
        if cur not in next_hop:
            return {"reachable": False, "terminal_value": None, "path": tuple(path), "hops": step, "ambiguous": False}
        cur = next_hop[cur]
        path.append(cur)
    return {"reachable": True, "terminal_value": cur, "path": tuple(path), "hops": int(hops), "ambiguous": False}


def render_x3_prompt(
    facts: Sequence[Mapping[str, Any]],
    *,
    goal_subject: str,
    hops: int,
    renderer: str = "canonical",
    min_epoch: int = 1,
    pad_to_bytes: int | None = None,
) -> str:
    """Render X3 graph facts in canonical or narrative style, avoiding A..P uppercase prose tokens."""
    if renderer == "canonical":
        hdr = f"TASK=X3 GOAL={goal_subject} HOPS={int(hops)} MIN_EP={int(min_epoch)} Q=0:0.00 R=0:0.00 BUD=1"
        body = [f"E {r['subj']}->{r['val']} ep={int(r.get('epoch', 1))} auth={int(r.get('authority', 1))}" for r in facts]
    elif renderer == "narrative":
        hdr = f"Trace {int(hops)} steps from {goal_subject} (ep>={int(min_epoch)}, tools off)."
        body = [f"Record ep {int(r.get('epoch', 1))}: node {r['subj']} links to {r['val']}." for r in facts]
    else:
        raise ValueError(f"unsupported X3 renderer {renderer!r}")
    text = "\n".join([hdr, *body])
    if pad_to_bytes is not None:
        raw_len = len(text.encode("utf-8"))
        if raw_len < int(pad_to_bytes):
            pad_needed = int(pad_to_bytes) - raw_len
            if pad_needed >= 4:
                text = text + "\n# " + ("." * (pad_needed - 3))
    return text


def _facts_signature(facts: Sequence[Mapping[str, Any]], *, min_epoch: int = 1) -> dict[str, Any]:
    tok_multiset = Counter(str(r["subj"]) for r in facts) + Counter(str(r["val"]) for r in facts)
    out_deg = Counter(str(r["subj"]) for r in facts if int(r.get("epoch", 1)) >= min_epoch)
    in_deg = Counter(str(r["val"]) for r in facts if int(r.get("epoch", 1)) >= min_epoch)
    return {
        "symbol_multiset": dict(sorted(tok_multiset.items())),
        "out_degree_hist": dict(sorted(Counter(out_deg.values()).items())),
        "in_degree_hist": dict(sorted(Counter(in_deg.values()).items())),
        "num_rows": len(facts),
    }


def _build_single_base_graph(
    *,
    graph_id: str,
    depth: int,
    rng: random.Random,
    flip_roles: bool,
    terminal_share_a: bool,
    wrong_last_mode: int,
    split: str,
) -> dict[str, Any]:
    if depth not in (4, 5):
        raise ValueError(f"X3 depth must be 4 or 5, got {depth}")

    syms = list(X3_SYMBOLS)
    rng.shuffle(syms)
    # Allocate distinct symbols so every active path is strictly cycle-free and depth-matched
    a_chain = syms[0 : depth]          # length d (sources a_0 .. a_{d-1})
    k_leaf = syms[depth]               # K
    b_chain = syms[depth + 1 : 2 * depth + 1]  # length d (sources b_0 .. b_{d-1})
    m_leaf = syms[2 * depth + 1]       # M (!= K)

    if flip_roles:
        a_chain, b_chain = b_chain, a_chain
        k_leaf, m_leaf = m_leaf, k_leaf

    path_a = [(a_chain[i], a_chain[i + 1] if i + 1 < depth else k_leaf) for i in range(depth)]
    path_b = [(b_chain[i], b_chain[i + 1] if i + 1 < depth else m_leaf) for i in range(depth)]

    if depth == 4:
        # 4 edges for sibling: 15 distinct symbols (unused slice syms[10:15])
        s_chain = syms[10:14]
        s_leaf = syms[14]
        path_s = [(s_chain[i], s_chain[i + 1] if i + 1 < 4 else s_leaf) for i in range(4)]
        # Symmetric inactive ep=0 irrelevant pair reusing terminal symbols safely
        irrel_edges = [
            {"subj": a_chain[0], "val": k_leaf, "epoch": 0, "authority": 1, "role": "irrel_0"},
            {"subj": b_chain[0], "val": m_leaf, "epoch": 0, "authority": 1, "role": "irrel_1"},
        ]
        sib_leaf = s_leaf
    else:
        # Depth 5: 16 distinct symbols total; sibling uses 4 fresh nodes -> untouched terminal edge of A or B
        s_chain = syms[12:16]
        tail_src, tail_leaf = (a_chain[-1], k_leaf) if terminal_share_a else (b_chain[-1], m_leaf)
        path_s = [
            (s_chain[0], s_chain[1]),
            (s_chain[1], s_chain[2]),
            (s_chain[2], s_chain[3]),
            (s_chain[3], tail_src),
        ]
        irrel_edges = [
            {"subj": a_chain[0], "val": k_leaf, "epoch": 0, "authority": 1, "role": "irrel_0"},
            {"subj": b_chain[0], "val": m_leaf, "epoch": 0, "authority": 1, "role": "irrel_1"},
        ]
        sib_leaf = tail_leaf

    swap_idx = 1  # Interior edge: a1->a2 <-> b1->b2 (strictly before tail edge a_{d-1}->leaf)
    raw_rows: list[dict[str, Any]] = []
    for idx, (s, v) in enumerate(path_a):
        raw_rows.append({"subj": s, "val": v, "epoch": 1, "authority": 1, "role": f"A_{idx}"})
    for idx, (s, v) in enumerate(path_b):
        raw_rows.append({"subj": s, "val": v, "epoch": 1, "authority": 1, "role": f"B_{idx}"})
    for idx, (s, v) in enumerate(path_s):
        raw_rows.append({"subj": s, "val": v, "epoch": 1, "authority": 1, "role": f"S_{idx}"})
    raw_rows.extend(irrel_edges)

    # Identify distractor row placed LAST so last-row heuristic picks a plausible WRONG terminal symbol != gold
    terminal_a_row = next(r for r in raw_rows if r["role"] == f"A_{depth - 1}")
    terminal_b_row = next(r for r in raw_rows if r["role"] == f"B_{depth - 1}")
    # Also ensure non-terminal distractor when S already ends in K or M
    non_k_m_row = next(r for r in raw_rows if r["role"] == "B_0")
    last_row = terminal_b_row if wrong_last_mode % 2 == 0 else non_k_m_row

    prefix_rows = [r for r in raw_rows if r is not last_row]
    rng.shuffle(prefix_rows)
    facts_orig = [
        {"subj": r["subj"], "val": r["val"], "epoch": int(r["epoch"]), "authority": int(r["authority"]), "role": r["role"]}
        for r in [*prefix_rows, last_row]
    ]

    # Edited: exchange destinations of interior edges A_1 and B_1 IN PLACE (preserving row order & multiset)
    facts_edit = deepcopy(facts_orig)
    idx_a1 = next(i for i, r in enumerate(facts_edit) if r["role"] == f"A_{swap_idx}")
    idx_b1 = next(i for i, r in enumerate(facts_edit) if r["role"] == f"B_{swap_idx}")
    facts_edit[idx_a1]["val"], facts_edit[idx_b1]["val"] = facts_edit[idx_b1]["val"], facts_edit[idx_a1]["val"]

    # Irrelevant edit: exchange destinations of irrel_0 and irrel_1 IN PLACE (exact same 2-row edit distance)
    facts_irrel = deepcopy(facts_orig)
    idx_i0 = next(i for i, r in enumerate(facts_irrel) if r["role"] == "irrel_0")
    idx_i1 = next(i for i, r in enumerate(facts_irrel) if r["role"] == "irrel_1")
    facts_irrel[idx_i0]["val"], facts_irrel[idx_i1]["val"] = facts_irrel[idx_i1]["val"], facts_irrel[idx_i0]["val"]

    # Independent graph-walker certification across all conditions
    aff_subj = a_chain[0]
    sib_subj = s_chain[0]
    w_aff_orig = independent_walk_x3_graph(facts_orig, aff_subj, hops=depth, min_epoch=1)
    w_aff_edit = independent_walk_x3_graph(facts_edit, aff_subj, hops=depth, min_epoch=1)
    w_aff_irrel = independent_walk_x3_graph(facts_irrel, aff_subj, hops=depth, min_epoch=1)
    w_sib_orig = independent_walk_x3_graph(facts_orig, sib_subj, hops=depth, min_epoch=1)
    w_sib_edit = independent_walk_x3_graph(facts_edit, sib_subj, hops=depth, min_epoch=1)
    w_sib_irrel = independent_walk_x3_graph(facts_irrel, sib_subj, hops=depth, min_epoch=1)

    if not (
        w_aff_orig["reachable"]
        and w_aff_edit["reachable"]
        and w_aff_irrel["reachable"]
        and w_sib_orig["reachable"]
        and w_sib_edit["reachable"]
        and w_sib_irrel["reachable"]
    ):
        raise AssertionError(f"X3 graph walk unreachable in {graph_id}")
    if w_aff_orig["terminal_value"] != k_leaf or w_aff_edit["terminal_value"] != m_leaf:
        raise AssertionError(f"X3 affected edit failed to flip K->M in {graph_id}")
    if w_aff_irrel["terminal_value"] != k_leaf:
        raise AssertionError(f"X3 irrelevant edit altered affected answer in {graph_id}")
    if w_sib_orig["terminal_value"] != w_sib_edit["terminal_value"] or w_sib_orig["terminal_value"] != w_sib_irrel["terminal_value"]:
        raise AssertionError(f"X3 sibling answer changed in {graph_id}")
    if str(facts_orig[-1]["val"]) == k_leaf:
        # Swap last_row distractor to avoid matching k_leaf when wrong_last_mode collides cleanly
        alt_idx = next(i for i, r in enumerate(facts_orig[:-1]) if str(r["val"]) not in (k_leaf, sib_leaf))
        for arr in (facts_orig, facts_edit, facts_irrel):
            arr[alt_idx], arr[-1] = arr[-1], arr[alt_idx]

    sig_orig = _facts_signature(facts_orig)
    sig_edit = _facts_signature(facts_edit)
    sig_irrel = _facts_signature(facts_irrel)
    if sig_orig != sig_edit or sig_orig != sig_irrel:
        raise AssertionError(f"X3 multiset/degree invariance violated in {graph_id}")

    return {
        "graph_id": graph_id,
        "split": split,
        "depth": int(depth),
        "affected_query": aff_subj,
        "sibling_query": sib_subj,
        "expected_affected_before": str(w_aff_orig["terminal_value"]),
        "expected_affected_after": str(w_aff_edit["terminal_value"]),
        "expected_sibling": str(w_sib_orig["terminal_value"]),
        "distractor_value": str(facts_orig[-1]["val"]),
        "facts_original": facts_orig,
        "facts_edited": facts_edit,
        "facts_irrelevant": facts_irrel,
        "swap_row_indices": (int(idx_a1), int(idx_b1)),
        "irrelevant_row_indices": (int(idx_i0), int(idx_i1)),
        "graph_signature": sig_orig,
    }


def build_x3_paired_graph_factorial(
    *,
    split: str = "dev",
    depths: Sequence[int] = (4, 5),
    base_graphs_per_depth: int | None = None,
    seed: int | None = None,
    max_context_length: int = 640,
) -> dict[str, Any]:
    """Build X3 paired 4/5-hop graph factorial (1,024 dev / 2,048 confirmation decisions, byte-aligned & certified)."""
    if split not in ("dev", "confirmation"):
        raise ValueError(f"unsupported X3 split {split!r}; expected 'dev' or 'confirmation'")
    n_per_depth = int(base_graphs_per_depth if base_graphs_per_depth is not None else (64 if split == "dev" else 128))
    rng_seed = int(seed if seed is not None else (31001 if split == "dev" else 97003))
    rng = random.Random(rng_seed)

    base_graphs: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    irrelevant_controls: list[dict[str, Any]] = []
    aligned_pairs: list[dict[str, Any]] = []

    for dp in depths:
        for idx in range(n_per_depth):
            gid = f"x3_{split}_d{int(dp)}_{idx:04d}"
            # Counterbalance roles, tail sharing, and last-row distractor across base graphs
            bg = _build_single_base_graph(
                graph_id=gid,
                depth=int(dp),
                rng=rng,
                flip_roles=bool(idx % 2 == 1),
                terminal_share_a=bool((idx // 2) % 2 == 0),
                wrong_last_mode=(idx // 4) % 2,
                split=split,
            )
            base_graphs.append(bg)

            for renderer in ("canonical", "narrative"):
                for query_role, goal_subj, ans_orig, ans_edit in (
                    ("affected", bg["affected_query"], bg["expected_affected_before"], bg["expected_affected_after"]),
                    ("sibling", bg["sibling_query"], bg["expected_sibling"], bg["expected_sibling"]),
                ):
                    raw_o = render_x3_prompt(bg["facts_original"], goal_subject=goal_subj, hops=int(dp), renderer=renderer)
                    raw_e = render_x3_prompt(bg["facts_edited"], goal_subject=goal_subj, hops=int(dp), renderer=renderer)
                    raw_i = render_x3_prompt(bg["facts_irrelevant"], goal_subject=goal_subj, hops=int(dp), renderer=renderer)
                    target_len = max(len(raw_o.encode("utf-8")), len(raw_e.encode("utf-8")), len(raw_i.encode("utf-8")))
                    p_orig = render_x3_prompt(bg["facts_original"], goal_subject=goal_subj, hops=int(dp), renderer=renderer, pad_to_bytes=target_len)
                    p_edit = render_x3_prompt(bg["facts_edited"], goal_subject=goal_subj, hops=int(dp), renderer=renderer, pad_to_bytes=target_len)
                    p_irrel = render_x3_prompt(bg["facts_irrelevant"], goal_subject=goal_subj, hops=int(dp), renderer=renderer, pad_to_bytes=target_len)

                    for txt in (p_orig, p_edit, p_irrel):
                        elig = check_x2_encoding_eligibility(txt, max_length=max_context_length)
                        if not elig["eligible"]:
                            raise ValueError(f"X3_CONTEXT_OVERFLOW:{gid}:{renderer}:{elig['byte_len']}>{elig['payload_limit']}")

                    pair_key = f"{gid}::{renderer}"
                    base_meta = {
                        "graph_id": gid,
                        "pair_key": pair_key,
                        "split": split,
                        "depth": int(dp),
                        "hops": int(dp),
                        "renderer": renderer,
                        "query_role": query_role,
                        "goal_subject": goal_subj,
                        "distractor_value": bg["distractor_value"],
                        "expected_before": ans_orig,
                        "expected_after": ans_edit,
                        "swap_row_indices": bg["swap_row_indices"],
                        "irrelevant_row_indices": bg["irrelevant_row_indices"],
                    }
                    decisions.append(
                        {
                            **base_meta,
                            "world_id": f"{pair_key}::orig::{query_role}",
                            "world_variant": "original",
                            "rendered_text": p_orig,
                            "facts": bg["facts_original"],
                            "expected_answer": ans_orig,
                            "byte_len": len(p_orig.encode("utf-8")),
                        }
                    )
                    decisions.append(
                        {
                            **base_meta,
                            "world_id": f"{pair_key}::edit::{query_role}",
                            "world_variant": "edited",
                            "rendered_text": p_edit,
                            "facts": bg["facts_edited"],
                            "expected_answer": ans_edit,
                            "byte_len": len(p_edit.encode("utf-8")),
                        }
                    )
                    irrelevant_controls.append(
                        {
                            **base_meta,
                            "world_id": f"{pair_key}::irrel::{query_role}",
                            "world_variant": "irrelevant_edit",
                            "rendered_text": p_irrel,
                            "facts": bg["facts_irrelevant"],
                            "expected_answer": ans_orig,
                            "byte_len": len(p_irrel.encode("utf-8")),
                        }
                    )
                    aligned_pairs.append(
                        {
                            **base_meta,
                            "original_text": p_orig,
                            "edited_text": p_edit,
                            "irrelevant_text": p_irrel,
                            "byte_len_original": len(p_orig.encode("utf-8")),
                            "byte_len_edited": len(p_edit.encode("utf-8")),
                            "byte_len_irrelevant": len(p_irrel.encode("utf-8")),
                        }
                    )

    panel_sha = digest(
        [
            {"world_id": d["world_id"], "text": d["rendered_text"], "ans": d["expected_answer"]}
            for d in decisions
        ]
    )
    return {
        "schema": "elendril-x3-paired-factorial-v1",
        "split": split,
        "seed": rng_seed,
        "depths": [int(x) for x in depths],
        "base_graphs_per_depth": n_per_depth,
        "base_graph_count": len(base_graphs),
        "decision_count": len(decisions),
        "irrelevant_control_count": len(irrelevant_controls),
        "aligned_pair_count": len(aligned_pairs),
        "panel_sha256": panel_sha,
        "base_graphs": base_graphs,
        "decisions": decisions,
        "irrelevant_controls": irrelevant_controls,
        "aligned_pairs": aligned_pairs,
    }


def eval_x3_shortcut_baselines(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Run first-row, last-row, modal-frequency, and independent-walker oracle baselines before unblinding."""
    decisions = list(panel.get("decisions", ()))
    if not decisions:
        raise ValueError("panel contains zero decisions")

    first_ok = 0
    last_ok = 0
    freq_ok = 0
    oracle_ok = 0
    last_row_distractor_rate_num = 0

    for d in decisions:
        facts = list(d["facts"])
        exp = str(d["expected_answer"])
        first_pred = str(facts[0]["val"])
        last_pred = str(facts[-1]["val"])
        counts = Counter(str(r["val"]) for r in facts)
        freq_pred = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        walk = independent_walk_x3_graph(facts, str(d["goal_subject"]), hops=int(d["hops"]), min_epoch=1)
        oracle_pred = str(walk["terminal_value"])

        if first_pred == exp:
            first_ok += 1
        if last_pred == exp:
            last_ok += 1
        if last_pred == str(d["distractor_value"]):
            last_row_distractor_rate_num += 1
        if freq_pred == exp:
            freq_ok += 1
        if oracle_pred == exp:
            oracle_ok += 1

    n = len(decisions)
    return {
        "schema": "elendril-x3-shortcut-baselines-v1",
        "decision_count": n,
        "first_row_accuracy": round(first_ok / n, 6),
        "last_row_accuracy": round(last_ok / n, 6),
        "frequency_accuracy": round(freq_ok / n, 6),
        "oracle_independent_walker_accuracy": round(oracle_ok / n, 6),
        "wrong_last_row_trap_rate": round(last_row_distractor_rate_num / n, 6),
    }


def build_x3_x2_crosswalk(
    x3_panel: Mapping[str, Any],
    x2_worlds: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Check prompt & world signature overlap between X3 decisions and X2 cohorts so cases never double-count."""
    kernel = X2EnvironmentKernel()
    x2_text_hashes: set[str] = set()
    for w in x2_worlds:
        if w.get("rendered_text"):
            x2_text_hashes.add(digest(str(w["rendered_text"])))
        else:
            try:
                x2_text_hashes.add(digest(kernel.observe(kernel.initial_state(w)).rendered_text))
            except Exception:
                continue
    x2_world_ids = {str(w.get("world_id", "")) for w in x2_worlds if w.get("world_id")}

    x3_hashes = [digest(str(d["rendered_text"])) for d in x3_panel.get("decisions", ())]
    overlap_texts = [h for h in x3_hashes if h in x2_text_hashes]
    overlap_ids = [str(d["world_id"]) for d in x3_panel.get("decisions", ()) if str(d["world_id"]) in x2_world_ids]
    unique_x3 = len(set(x3_hashes))
    return {
        "schema": "elendril-x3-x2-crosswalk-v1",
        "x3_split": str(x3_panel.get("split", "")),
        "x3_decision_count": len(x3_hashes),
        "x3_unique_prompt_count": unique_x3,
        "x2_comparison_count": len(x2_worlds),
        "prompt_overlap_count": len(overlap_texts),
        "world_id_overlap_count": len(overlap_ids),
        "strictly_novel_for_confirmation": (len(overlap_texts) == 0 and len(overlap_ids) == 0),
    }


__all__ = [
    "X3_SYMBOLS",
    "build_x3_paired_graph_factorial",
    "build_x3_x2_crosswalk",
    "eval_x3_shortcut_baselines",
    "independent_walk_x3_graph",
    "render_x3_prompt",
]
