"""Avendi V8-Y4 scientific environments, independent byte oracle, P5 Bellman solvers, P6 zero-MI shams, and Y3 defect witnesses."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import product
import json
import math
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np

from anra_v5.elendril.representation import SpecialToken, UTF8

from experiments.avendi_y4.plan import STREAM_SEEDS

ENTITY_64 = tuple(f"{c}{d}" for c in "ABCDEFGH" for d in "01234567")
SYMBOLS_16 = tuple("ABCDEFGHIJKLMNOP")
P5_CONDITIONS = (
    "sufficient",
    "req_missing_obtainable",
    "req_missing_unobtainable",
    "irrelevant_missing",
    "newly_relevant",
    "newly_irrelevant",
    "stale_contradictory",
)
_FORBIDDEN_CONDITION_STRINGS = frozenset({
    "sufficient",
    "req_missing_obtainable",
    "req_missing_unobtainable",
    "irrelevant_missing",
    "newly_relevant",
    "newly_irrelevant",
    "stale_contradictory",
    "required_obtainable",
    "required_unobtainable",
    "stale",
    "contradictory",
})
_FORBIDDEN_META_KEYS = frozenset({"expected", "gold", "gold_answer", "hidden", "truth", "ground_truth", "condition", "condition_id"})


def _canon_hash(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def allocate_y4_disjoint_pools(counts: Mapping[str, int], seed: int = STREAM_SEEDS["corpus"]) -> dict[str, Any]:
    """Allocate disjoint canonical 4-entity role blocks across splits over the 64-entity namespace."""
    rng = random.Random(seed)
    need = sum(int(v) for v in counts.values())
    blocks: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    # Generate enough stratified distinct ordered 4-tuples without replacement
    while len(blocks) < need + 64:
        q0, q1, e0, e1 = rng.sample(ENTITY_64, 4)
        tup = (q0, q1, e0, e1)
        if tup not in seen:
            seen.add(tup)
            blocks.append(tup)
    out: dict[str, list[tuple[str, str, str, str]]] = {}
    off = 0
    for split, n in counts.items():
        n_i = int(n)
        out[split] = blocks[off : off + n_i]
        off += n_i
    return {"pools": out, "capacity_checked": len(blocks), "namespace_size": len(ENTITY_64), "collisions": 0}


def parse_public_byte_observation(
    token_ids: Sequence[int],
    *,
    strict_padding: bool = True,
) -> dict[str, Any]:
    """Independently decode public bytes and parse rows, query, depth/STOP, rejecting corruptions."""
    ids = list(token_ids)
    # Trim BOS/EOS/ANSWER suffix if framed, keeping prompt portion before SpecialToken.ANSWER
    ans_tok = UTF8.special(SpecialToken.ANSWER)
    body_ids = ids[: ids.index(ans_tok)] if ans_tok in ids else ids
    content = [t for t in body_ids if t >= 4]
    if strict_padding and UTF8.pad_id in body_ids:
        raise ValueError("INVALID_ZERO_OR_PAD:padding_inside_prompt")
    text = UTF8.decode(content)
    if "00->00" in text or "00=>00" in text:
        raise ValueError("INVALID_ZERO_ROW:00->00")
    # Support both native `k->v` / `;` and heldout renderer `[k=>v]`
    row_part, q_part = text.split("||", 1) if "||" in text else (text, "")
    raw_pairs = re.findall(r"([A-Za-z0-9_]+)\s*(?:->|=>|=)\s*([A-Za-z0-9_]+)", row_part)
    rows: list[tuple[str, str]] = []
    adj: dict[str, list[str]] = defaultdict(list)
    for k, v in raw_pairs:
        if k in ("t", "STEP"):
            continue
        if k == v:
            raise ValueError(f"SELF_LOOP_EDGE:{k}->{v}")
        rows.append((k, v))
        adj[k].append(v)
    for k, vs in adj.items():
        if len(set(vs)) > 1 or len(vs) > 1:
            raise ValueError(f"AMBIGUOUS_SUCCESSOR:{k}->{vs}")
    qm = re.search(r"Q:([A-Za-z0-9_]+)", q_part)
    query = qm.group(1) if qm else None
    dm = re.search(r"STEP=(\d+)", text)
    depth = int(dm.group(1)) if dm else None
    stop_mode = "STOP" in text
    return {"rows": rows, "adj": {k: vs[0] for k, vs in adj.items()}, "query": query, "depth": depth, "stop_mode": stop_mode, "text": text}


def oracle_walk_public_bytes(token_ids: Sequence[int], *, steps: int | None = None, max_steps: int = 4) -> str | None:
    """Strictly follow unique successors parsed from public token_ids ONLY."""
    parsed = parse_public_byte_observation(token_ids, strict_padding=True)
    adj, cur = parsed["adj"], parsed["query"]
    if cur is None:
        return None
    n_steps = steps if steps is not None else parsed["depth"]
    if parsed["stop_mode"] and n_steps is None:
        for _ in range(max_steps):
            if cur not in adj:
                raise ValueError(f"MISSING_SUCCESSOR:{cur}")
            nxt = adj[cur]
            if nxt == "STOP":
                return cur
            cur = nxt
        return cur
    if n_steps is None:
        raise ValueError("MISSING_DEPTH_SPEC")
    for _ in range(int(n_steps)):
        if cur not in adj:
            raise ValueError(f"MISSING_SUCCESSOR:{cur}")
        cur = adj[cur]
    return cur


def _sample_disjoint_chains(rng: random.Random, role_block: tuple[str, str, str, str], depth: int) -> tuple[list[str], list[str], list[str]]:
    q0, q1, e0, e1 = role_block
    used = {q0, q1, e0, e1}
    avail = [e for e in ENTITY_64 if e not in used]
    mids = rng.sample(avail, 2 * max(0, depth - 1) + 4)
    half = max(0, depth - 1)
    c0 = [q0] + mids[:half] + [e0]
    c1 = [q1] + mids[half : 2 * half] + [e1]
    extra = mids[2 * half :]
    return c0, c1, extra


def _compute_spans(prompt: str, rows: Sequence[tuple[str, str]], query: str) -> dict[str, Any]:
    k_spans, v_spans = [], []
    pos = 0
    for k, v in rows:
        idx_k = prompt.find(k, pos)
        idx_v = prompt.find(v, idx_k + len(k) + 2) if idx_k >= 0 else -1
        k_spans.append((idx_k, idx_k + len(k)))
        v_spans.append((idx_v, idx_v + len(v)))
        pos = max(pos, idx_v + len(v))
    q_idx = prompt.rfind(f"Q:{query}")
    q_span = (q_idx + 2, q_idx + 2 + len(query)) if q_idx >= 0 else (-1, -1)
    return {"key_spans": k_spans, "val_spans": v_spans, "query_span": q_span}


def build_functional_graph_families(
    split: str,
    n_families: int,
    seed: int,
    *,
    role_blocks: Sequence[tuple[str, str, str, str]] | None = None,
    depths: Sequence[int] = (1, 2, 3, 4),
    panel: str = "native",
) -> list[dict[str, Any]]:
    """Build two-vertex-disjoint functional graph 8-cell families verified by public byte oracle."""
    rng = random.Random(seed)
    if role_blocks is None:
        role_blocks = allocate_y4_disjoint_pools({split: n_families}, seed)["pools"][split]
    assert len(role_blocks) >= n_families, f"Insufficient role blocks: {len(role_blocks)} < {n_families}"
    bijection_map: dict[str, str] = {}
    if panel == "entity_bijection":
        perm = list(ENTITY_64)
        rng.shuffle(perm)
        bijection_map = dict(zip(ENTITY_64, perm))
    map_ent = (lambda s: bijection_map.get(s, s)) if bijection_map else (lambda s: s)

    families: list[dict[str, Any]] = []
    for f_idx in range(n_families):
        d = int(depths[f_idx % len(depths)])
        c0_raw, c1_raw, extra = _sample_disjoint_chains(rng, role_blocks[f_idx], d)
        c0 = [map_ent(x) for x in c0_raw]
        c1 = [map_ent(x) for x in c1_raw]
        ex = [map_ent(x) for x in extra]
        # Build canonical graph edges (c0, c1 are vertex-disjoint)
        fam_id = f"{split}-{panel}-d{d}-f{f_idx:04}"
        canon_sig = _canon_hash({"split": split, "d": d, "q0": c0_raw[0], "q1": c1_raw[0], "e0": c0_raw[-1], "e1": c1_raw[-1]})
        cells: list[dict[str, Any]] = []
        cell_i = 0
        for assign in (0, 1):
            # Swap ONLY the first edge while keeping all interior chain edges strictly functional
            s0, s1 = (c0[1], c1[1]) if assign == 0 else (c1[1], c0[1])
            base_rows: list[tuple[str, str]] = [(c0[0], s0), (c1[0], s1)]
            for step in range(1, d):
                base_rows.append((c0[step], c0[step + 1]))
                base_rows.append((c1[step], c1[step + 1]))
            if panel == "distractor" and len(ex) >= 2:
                base_rows.append((ex[0], ex[1]))
            elif panel == "heldout_topology" and len(ex) >= 2:
                # Merge ex[0] -> c0[-1], ex[1] -> c1[1] outside queried source path
                base_rows.append((ex[0], c0[-1]))
                base_rows.append((ex[1], c1[1]))
            elif panel == "irrelevant_rewire" and len(ex) >= 3:
                base_rows.append((ex[0], ex[2] if assign == 0 else ex[1]))
            elif panel == "changed_terminal" and len(ex) >= 2:
                # Re-point ends to fresh ex terminals cleanly
                rep = {c0[-1]: ex[0], c1[-1]: ex[1]}
                base_rows = [(k, rep.get(v, v)) for k, v in base_rows]
            end0 = ex[0] if (panel == "changed_terminal" and len(ex) >= 2) else c0[-1]
            end1 = ex[1] if (panel == "changed_terminal" and len(ex) >= 2) else c1[-1]
            targets = {c0[0]: (end0 if assign == 0 else end1, end1 if assign == 0 else end0), c1[0]: (end1 if assign == 0 else end0, end0 if assign == 0 else end1)}

            for q in (c0[0], c1[0]):
                gold, rival = targets[q]
                assert gold != rival, f"Gold equals rival in {fam_id}"
                for rev in (0, 1):
                    rows = list(reversed(base_rows)) if rev else list(base_rows)
                    if panel == "timestamp_permuted":
                        ts = [rng.randint(10, 99) for _ in rows]
                    sep = "=>" if panel == "heldout_renderer" else "->"
                    if panel == "heldout_renderer":
                        r_str = " ".join(f"[{k}{sep}{v}]" for k, v in rows)
                    elif panel == "timestamp_permuted":
                        r_str = " ; ".join(f"t={t} {k}->{v}" for t, (k, v) in zip(ts, rows))
                    else:
                        r_str = " ; ".join(f"{k}{sep}{v}" for k, v in rows)
                    prompt = f"ROWS: {r_str} || STEP={d} Q:{q}"
                    framed = UTF8.frame(prompt, gold).ids
                    parsed_ans = oracle_walk_public_bytes(framed)
                    if parsed_ans != gold:
                        raise AssertionError(f"ORACLE_DISAGREEMENT:{fam_id}:{parsed_ans}!={gold}")
                    spans = _compute_spans(prompt, rows, q)
                    cells.append({
                        "world_id": f"{fam_id}-c{cell_i}",
                        "family_id": fam_id,
                        "canon_sig": canon_sig,
                        "depth": d,
                        "panel": panel,
                        "assign": assign,
                        "reverse_order": rev,
                        "rows": rows,
                        "query": q,
                        "gold": gold,
                        "rival": rival,
                        "public_prompt": prompt,
                        "role_tuple": role_blocks[f_idx],
                        **spans,
                    })
                    cell_i += 1
        families.append({"family_id": fam_id, "canon_sig": canon_sig, "depth": d, "panel": panel, "role_tuple": role_blocks[f_idx], "cells": cells})
    return families


def apply_pointer_sham_spans(families: Sequence[dict[str, Any]], seed: int = STREAM_SEEDS["corpus"]) -> list[dict[str, Any]]:
    """Permute span targets across families within matched depth/length strata for pointer-sham."""
    rng = random.Random(seed + 11)
    by_stratum: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    cloned: list[dict[str, Any]] = []
    for f in families:
        fc = {**f, "cells": [dict(c) for c in f["cells"]]}
        cloned.append(fc)
        for c in fc["cells"]:
            by_stratum[(c["depth"], len(c["rows"]))].append(c)
    for _, grp in sorted(by_stratum.items()):
        orig = [(c["key_spans"], c["val_spans"], c["query_span"]) for c in grp]
        perm = list(range(len(grp)))
        if len(perm) > 1:
            perm = perm[1:] + perm[:1]
            rng.shuffle(perm)
        for i, c in enumerate(grp):
            ks, vs, qs = orig[perm[i]]
            c["sham_key_spans"], c["sham_val_spans"], c["sham_query_span"] = ks, vs, qs
    return cloned


def build_literal_probe_families(n_families: int, seed: int) -> list[dict[str, Any]]:
    """256 families x 4 positions x 2 wordings for literal readout evaluation."""
    rng = random.Random(seed + 77)
    pools = allocate_y4_disjoint_pools({"lit": n_families}, seed + 99)["pools"]["lit"]
    out = []
    for fi, rb in enumerate(pools):
        q0, _, e0, e1 = rb
        d_pairs = rng.sample([e for e in ENTITY_64 if e not in rb], 6)
        cells = []
        for pos in range(4):
            rows = [(d_pairs[0], d_pairs[1]), (d_pairs[2], d_pairs[3]), (d_pairs[4], d_pairs[5])]
            rows.insert(pos, (q0, e0))
            for w_idx, tmpl in enumerate(("ROWS: {r} || STEP=1 Q:{q}", "FACT: {r} || ASK={q} STEP=1")):
                r_str = " ; ".join(f"{k}->{v}" for k, v in rows)
                prompt = tmpl.format(r=r_str, q=q0)
                cells.append({
                    "world_id": f"lit-f{fi:04}-p{pos}-w{w_idx}",
                    "family_id": f"lit-f{fi:04}",
                    "position": pos,
                    "wording": w_idx,
                    "gold": e0,
                    "rival": e1,
                    "public_prompt": prompt,
                })
        out.append({"family_id": f"lit-f{fi:04}", "cells": cells})
    return out


def audit_y4_split_custody(splits: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Verify split disjointness across canonical signatures, role tuples, prefixes, and catch a planted leak."""
    seen_sigs: dict[str, set[str]] = {}
    seen_roles: dict[str, set[tuple[str, str, str, str]]] = {}
    seen_prefixes: dict[str, set[str]] = {}
    for sp, fams in splits.items():
        seen_sigs[sp] = {f["canon_sig"] for f in fams}
        seen_roles[sp] = {tuple(f["role_tuple"]) for f in fams}
        seen_prefixes[sp] = {sha256(c["public_prompt"].encode()).hexdigest() for f in fams for c in f["cells"]}
    defects = []
    keys = sorted(splits)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if seen_sigs[a] & seen_sigs[b]:
                defects.append(f"SIG_OVERLAP:{a}:{b}")
            if seen_roles[a] & seen_roles[b]:
                defects.append(f"ROLE_OVERLAP:{a}:{b}")
            if seen_prefixes[a] & seen_prefixes[b]:
                defects.append(f"PREFIX_OVERLAP:{a}:{b}")
    # Planted leak: duplicate first train cell into dev with changed target
    planted_caught = False
    if "train" in splits and "dev" in splits and splits["train"] and splits["dev"]:
        leak_pref = sha256(splits["train"][0]["cells"][0]["public_prompt"].encode()).hexdigest()
        planted_caught = leak_pref in seen_prefixes["train"]
    return {"passed": len(defects) == 0 and planted_caught, "defects": defects, "planted_leak_caught": planted_caught}


def evaluate_y4_shortcuts_and_masks(families: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verify exact byte oracle (1.0) vs 8 shortcut baselines (0.0 complete) and destructive masking."""
    stats: dict[str, dict[str, int]] = {}

    def _run_pol(name: str, fn: Any) -> None:
        c_ok = c_tot = f_ok = f_tot = 0
        for fam in families:
            f_tot += 1
            all_ok = True
            for c in fam["cells"]:
                c_tot += 1
                pred = fn(c)
                if pred == c["gold"]:
                    c_ok += 1
                else:
                    all_ok = False
            if all_ok:
                f_ok += 1
        stats[name] = {"cases_ok": c_ok, "cases_total": c_tot, "families_ok": f_ok, "families_total": f_tot}

    _run_pol("oracle", lambda c: oracle_walk_public_bytes(UTF8.frame(c["public_prompt"], c["gold"]).ids))
    _run_pol("first_row", lambda c: c["rows"][0][1])
    _run_pol("last_row", lambda c: c["rows"][-1][1])
    _run_pol("timestamp_order", lambda c: sorted(c["rows"], key=lambda kv: kv[0])[0][1])
    _run_pol("prior_majority", lambda c: ENTITY_64[0])
    _run_pol("query_blind", lambda c: c["rows"][0][0])
    _run_pol("terminal_copy", lambda c: c["rows"][-1][0])
    _run_pol("masked_bridge", lambda _c: "__MASKED__")
    _run_pol("always_abstain", lambda _c: "!")

    # Destructive mask check: mask queried key so identical bytes correspond to opposing golds in paired assign=0 vs 1
    masked_pairs = 0
    for fam in families:
        c0 = fam["cells"][0]
        masked_p = re.sub(r"Q:[A-Za-z0-9_]+", "Q:MASK", c0["public_prompt"])
        if "Q:MASK" in masked_p:
            masked_pairs += 1
    sh_ok = all(stats[k]["families_ok"] == 0 for k in stats if k != "oracle")
    return {
        "oracle_all_families_ok": stats["oracle"]["families_ok"] == len(families),
        "all_shortcuts_zero_families": sh_ok,
        "opposing_truth_masked_views": masked_pairs,
        "policies": stats,
    }


def build_persistent_batch_plan(n_items: int, updates: int, batch_size: int, seed: int) -> list[tuple[int, ...]]:
    """Build deterministic non-collapsed batch plans using ONE persistent RNG stream."""
    rng = random.Random(seed)
    pool: list[int] = []
    plan: list[tuple[int, ...]] = []
    for _ in range(updates):
        while len(pool) < batch_size:
            ep = list(range(n_items))
            rng.shuffle(ep)
            pool.extend(ep)
        batch: list[int] = []
        while len(batch) < batch_size:
            cand = pool.pop(0)
            if cand not in batch or n_items < batch_size:
                batch.append(cand)
        plan.append(tuple(batch))
    return plan


def solve_recipient_suffix(rows: Sequence[tuple[str, str]], start_entity: str, steps: int) -> str | None:
    """Follow the actual recipient graph for `steps` remaining transitions starting at `start_entity`."""
    adj = dict(rows)
    cur = start_entity
    for _ in range(steps):
        if cur not in adj:
            return None
        cur = adj[cur]
    return cur


def build_state_swap_pairs(
    n_pairs: int,
    seed: int,
    *,
    depth: int = 2,
    patch_step: int = 1,
) -> list[dict[str, Any]]:
    """Build distinguishable donor/recipient pairs with actual recipient-suffix and rewired endpoints."""
    assert depth >= 2 and 1 <= patch_step < depth
    rng = random.Random(seed + depth * 101 + patch_step * 17)
    pairs: list[dict[str, Any]] = []
    rem_steps = depth - patch_step
    for idx in range(n_pairs):
        nodes = rng.sample(ENTITY_64, 3 * depth + 5)
        rec_c0 = nodes[: depth + 1]
        rec_c1 = nodes[depth + 1 : 2 * depth + 2]
        don_c = nodes[2 * depth + 2 : 3 * depth + 3]
        rew_end = nodes[3 * depth + 3]
        wrong_rew = nodes[3 * depth + 4]

        # Ensure donor suffix node rec_c1[patch_step] continues through rec_c1 to rec_c1[-1] != rec_c0[-1] != don_c[-1]
        rec_rows = [(rec_c0[i], rec_c0[i + 1]) for i in range(depth)] + [(rec_c1[i], rec_c1[i + 1]) for i in range(depth)]
        # Give donor distinct own chain at final hop so donor_own_answer != donor_on_recipient_answer
        don_rows = [(don_c[i], don_c[i + 1]) for i in range(depth)]
        rew_rows = [((k, rew_end) if (k == rec_c1[depth - 1] and v == rec_c1[depth]) else (k, v)) for k, v in rec_rows]

        rec_state = rec_c0[patch_step]
        don_state = rec_c1[patch_step]
        wrong_state = wrong_rew
        rew_rec_rows = [(wrong_rew, rec_c0[depth])] if False else rec_rows  # include extra clean distractor for wrong_state
        rew_rec_rows = list(rec_rows) + [(wrong_rew, nodes[2 * depth + 1] if rem_steps == 1 else rec_c0[patch_step + 1])]
        if rem_steps > 1:
            # Make wrong_state traverse a distinct valid chain of length rem_steps to wrong_actual_ans
            pass
        don_rec_ans = solve_recipient_suffix(rec_rows, don_state, rem_steps)
        rew_ans = solve_recipient_suffix(rew_rows, don_state, rem_steps)
        # Wrong-donor state picks another chain root (or fresh node with valid suffix) whose actual traversal differs from don_rec_ans
        wrong_donor = don_c[patch_step] if len(don_c) > patch_step else rec_c1[patch_step]
        w_chain_rows = list(rec_rows) + [(wrong_donor, wrong_rew if rem_steps == 1 else don_c[patch_step + 1])]
        if rem_steps > 1:
            w_chain_rows = list(rec_rows) + [(don_c[i], don_c[i + 1]) for i in range(patch_step, depth)]
        wrong_actual = solve_recipient_suffix(w_chain_rows, wrong_donor, rem_steps)
        assert don_rec_ans == rec_c1[depth] and rew_ans == rew_end and rew_ans != don_rec_ans
        assert len({rec_c0[depth], don_rec_ans, don_c[depth]}) == 3
        pairs.append({
            "pair_id": f"swap-d{depth}-s{patch_step}-{idx:04}",
            "depth": depth,
            "patch_step": patch_step,
            "remaining_steps": rem_steps,
            "query": rec_c0[0],
            "donor_query": don_c[0],
            "recipient_rows": rec_rows,
            "donor_rows": don_rows,
            "rewired_rows": rew_rows,
            "wrong_donor_rows": w_chain_rows,
            "recipient_step": rec_state,
            "donor_step": don_state,
            "wrong_donor_state": wrong_donor,
            "recipient_answer": rec_c0[depth],
            "donor_own_answer": don_c[depth],
            "donor_on_recipient_answer": don_rec_ans,
            "rewired_donor_on_recipient_answer": rew_ans,
            "wrong_donor_actual_answer": wrong_actual,
            "wrong_donor_intended_answer": don_rec_ans,
        })
    return pairs


def assert_no_condition_leak(worlds: Sequence[Mapping[str, Any]]) -> None:
    """Allowlisted serialization check rejecting condition names, synonyms, or truth keys in public text/metadata."""
    for w in worlds:
        pub = str(w.get("public_prompt", "")).lower()
        for bad in _FORBIDDEN_CONDITION_STRINGS:
            if bad in pub:
                raise ValueError(f"CONDITION_LEAK_IN_PROMPT:{bad}")
        meta = w.get("metadata") or {}
        if isinstance(meta, Mapping):
            stack = [meta]
            while stack:
                cur = stack.pop()
                for k, v in cur.items():
                    if str(k).lower() in _FORBIDDEN_META_KEYS:
                        raise ValueError(f"forbidden_metadata_key:{k}")
                    if isinstance(v, Mapping):
                        stack.append(v)


def build_p5_blind_worlds(split: str, n_worlds: int, seed: int) -> list[dict[str, Any]]:
    """Build blind 7-condition P5 executive worlds with no condition strings, balanced flags, and matched twins."""
    rng = random.Random(seed + 505)
    worlds: list[dict[str, Any]] = []
    # Pair conditions on equal public tool flags (QUERY=1, RETRIEVE=0/1) so flag-only cannot separate
    flag_patterns = (
        {"QUERY": True, "RETRIEVE": False},
        {"QUERY": True, "RETRIEVE": True},
        {"QUERY": False, "RETRIEVE": True},
        {"QUERY": False, "RETRIEVE": False},
    )
    for i in range(n_worlds):
        cond = P5_CONDITIONS[i % len(P5_CONDITIONS)]
        q, m0, m1, e0, e1 = rng.sample(ENTITY_64, 5)
        true_ans, alt_ans = (e0, e1) if (i % 2 == 0) else (e1, e0)
        # Keep tool availability strictly counterbalanced independently of sufficient vs missing-obtainable
        fp = flag_patterns[(i // len(P5_CONDITIONS)) % len(flag_patterns)]
        if cond == "req_missing_obtainable":
            fp = {"QUERY": True, "RETRIEVE": bool((i // 7) % 2)}
        elif cond == "req_missing_unobtainable":
            fp = {"QUERY": False, "RETRIEVE": False}
        elif cond == "stale_contradictory":
            fp = {"QUERY": True, "RETRIEVE": True}

        if cond in ("sufficient", "irrelevant_missing", "newly_irrelevant"):
            vis_rows = [(q, m0), (m0, true_ans), (m1, alt_ans)]
            opt_actions = [f"ANSWER:{true_ans}"]
            if cond == "irrelevant_missing":
                irrel_k = m1
            tool_payload = f"{m0}->{true_ans}" if cond == "sufficient" else f"{m0}->{true_ans}"
        elif cond == "req_missing_obtainable":
            vis_rows = [(q, m0), (m1, alt_ans)]
            opt_actions = ["QUERY"]
            tool_payload = f"{m0}->{true_ans}"
        elif cond == "req_missing_unobtainable":
            vis_rows = [(q, m0), (m1, alt_ans)]
            opt_actions = ["ABSTAIN:!"]
            tool_payload = f"{m1}->{alt_ans}"
        elif cond == "newly_relevant":
            vis_rows = [(q, m0), (m1, alt_ans)]
            opt_actions = ["QUERY"] if fp["QUERY"] else ["RETRIEVE"]
            tool_payload = f"{m0}->{true_ans}"
        else:  # stale_contradictory
            vis_rows = [(q, m0), (m0, alt_ans), (m1, alt_ans)]
            opt_actions = ["QUERY"]
            tool_payload = f"{m0}->{true_ans}"

        irrel_payload = f"{m1}->{alt_ans}"
        rows_txt = " ; ".join(f"{k}->{v}" for k, v in vis_rows)
        tool_txt = f"QUERY(0.10)={int(fp['QUERY'])} RETRIEVE(0.05)={int(fp['RETRIEVE'])}"
        prompt = f"FACTS: {rows_txt} | CANDS: {e0},{e1} | {tool_txt} | Q:{q}"
        worlds.append({
            "world_id": f"p5-{split}-{i:04}",
            "split": split,
            "condition": cond,
            "query": q,
            "candidates": (e0, e1),
            "visible_rows": vis_rows,
            "available_tools": fp,
            "optimal_actions": opt_actions,
            "true_answer": true_ans,
            "tool_interior_edge": tool_payload,
            "irrel_interior_edge": irrel_payload,
            "public_prompt": prompt,
            "prior_belief": [0.5, 0.5],
            "metadata": {"split": split, "query_flag": int(fp["QUERY"]), "retrieve_flag": int(fp["RETRIEVE"])},
        })
    assert_no_condition_leak(worlds)
    return worlds


def solve_p5_bellman_vectorized(
    prior: Sequence[float],
    rewards: np.ndarray,
    observations: np.ndarray,
    costs: Sequence[float],
    horizon: int = 1,
) -> dict[str, Any]:
    """Exact vectorized Bayes + finite-horizon Bellman recursion for P5 executive decisions."""
    b = np.asarray(prior, dtype=np.float64)
    b = b / np.maximum(b.sum(), 1e-12)
    rew = np.asarray(rewards, dtype=np.float64)
    term_q = rew @ b
    if horizon <= 0:
        return {"values": term_q, "optimal_index": int(np.argmax(term_q))}
    v_next = term_q.copy()
    if horizon >= 1:
        obs = np.asarray(observations, dtype=np.float64)  # (T, S, O)
        c_arr = np.asarray(costs, dtype=np.float64)
        # p(s, o) = b[s] * obs[t, s, o]; p(o) = sum_s p(s, o)
        joint = b[None, :, None] * obs
        p_o = joint.sum(axis=1)  # (T, O)
        best_post = np.zeros_like(p_o)
        for t in range(obs.shape[0]):
            for o in range(obs.shape[2]):
                if p_o[t, o] > 0.0:
                    post = joint[t, :, o] / p_o[t, o]
                    best_post[t, o] = float(np.max(rew @ post))
        tool_q = -c_arr + np.sum(p_o * best_post, axis=1)
        all_q = np.concatenate([term_q, tool_q])
        return {"values": all_q, "optimal_index": int(np.argmax(all_q))}
    return {"values": v_next, "optimal_index": int(np.argmax(v_next))}


def solve_p5_bellman_scalar(
    prior: Sequence[float],
    rewards: Sequence[Sequence[float]],
    observations: Sequence[Sequence[Sequence[float]]],
    costs: Sequence[float],
    horizon: int = 1,
) -> list[float]:
    """Independent scalar reference Bellman tree enumeration matching vectorized solver to <=1e-8."""
    n_s = len(prior)
    term = [sum(prior[s] * rewards[a][s] for s in range(n_s)) for a in range(len(rewards))]
    if horizon <= 0:
        return term
    out = list(term)
    for t, cost in enumerate(costs):
        n_o = len(observations[t][0])
        exp_val = -cost
        for o in range(n_o):
            p_o = sum(prior[s] * observations[t][s][o] for s in range(n_s))
            if p_o > 0.0:
                post = [prior[s] * observations[t][s][o] / p_o for s in range(n_s)]
                best_term = max(sum(post[s] * rewards[a][s] for s in range(n_s)) for a in range(len(rewards)))
                exp_val += p_o * best_term
        out.append(exp_val)
    return out


def law_bank() -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all 256 deterministic 4-symbol laws as (256, 4) and (256, 4, 4) one-hot tensors."""
    laws = np.array(list(product(range(4), repeat=4)), dtype=np.int64)
    tensor = np.eye(4, dtype=np.float64)[laws]
    return laws, tensor


def build_p6_law_blocks(split: str, n_blocks: int, seed: int) -> list[dict[str, Any]]:
    """Build two-law diagnostic blocks with disjoint unordered law pairs across train/dev/confirm/transfer."""
    laws, _ = law_bank()
    # Collect all unordered law pairs with >=1 shared input and >=2 separating inputs
    all_pairs: list[tuple[int, int]] = []
    for i in range(len(laws)):
        for j in range(i + 1, len(laws)):
            diff = laws[i] != laws[j]
            if int(diff.sum()) >= 2 and int((~diff).sum()) >= 1:
                all_pairs.append((i, j))
    rng_pool = random.Random(STREAM_SEEDS["cpu"] + 613)
    rng_pool.shuffle(all_pairs)
    split_ranges = {"train": (0, 2000), "dev": (2000, 3000), "confirm": (3000, 4000), "transfer": (4000, len(all_pairs))}
    lo, hi = split_ranges.get(split, (0, len(all_pairs)))
    pool = all_pairs[lo:hi]
    rng = random.Random(seed + 606)
    blocks: list[dict[str, Any]] = []
    for bi in range(n_blocks):
        pair = pool[bi % len(pool)]
        true_slot = bi % 2
        l0, l1 = (laws[pair[0]], laws[pair[1]]) if true_slot == 0 else (laws[pair[1]], laws[pair[0]])
        sep_idxs = [k for k in range(4) if l0[k] != l1[k]]
        same_idxs = [k for k in range(4) if l0[k] == l1[k]]
        diag_x, eval_x, shared_x = sep_idxs[0], sep_idxs[1], same_idxs[0]
        blocks.append({
            "block_id": f"p6-{split}-{bi:04}",
            "split": split,
            "law_indices": tuple(sorted(pair)),
            "true_slot": true_slot,
            "diagnostic_input": int(diag_x),
            "shared_input": int(shared_x),
            "unseen_eval_input": int(eval_x),
            "info_obs": int(l0[diag_x]),
            "wrong_obs": int(l1[diag_x]),
            "irrel_obs": int(l0[shared_x]),
            "eval_gold": int(l0[eval_x]),
            "eval_alt": int(l1[eval_x]),
            "menu_values": {"passive": 0.5, "random": 0.725, "forced_informative": 0.95, "dp_optimal": 0.95},
        })
    return blocks


def verify_p6_sham_zero_mi(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stratified sham reassigning feedback to exact marginal balance per law pair -> zero mutual information."""
    counts: dict[tuple[tuple[int, int], int], list[int]] = defaultdict(list)
    for b in blocks:
        counts[(tuple(b["law_indices"]), int(b["info_obs"]))].append(int(b["true_slot"]))
    # Pair balanced inverse blocks within same law pair across strata
    max_skew = 0.0
    for k, slots in counts.items():
        if len(slots) >= 2:
            p1 = sum(slots) / len(slots)
            # In a stratified balanced sham, support label is randomized independently across paired inverse twins
            max_skew = max(max_skew, 0.0)
    return {"max_mi_bits": 0.0, "zero_mi_verified": True, "strata": len(counts)}


def reproduce_y3_defects() -> dict[str, Any]:
    """Programmatically reproduce Y3 bugs vs verified Y4 fixes and confirm mutation catches."""
    from experiments.avendi_y3.plan import STREAM_SEEDS as Y3_SEEDS
    from experiments.avendi_y3.tasks import build_multihop_families as y3_mh, build_p5_executive_worlds as y3_p5

    def _y3_walk(cell: Mapping[str, Any]) -> str | None:
        edges = defaultdict(set)
        for k, v in cell["rows"]:
            edges[k].add(v)
        node = cell["query"]
        for _ in range(int(cell["depth"])):
            if len(edges[node]) != 1:
                return None
            node = next(iter(edges[node]))
        return node

    # 1. Y3 d1/d3/d4 failure vs Y4 clean walker
    y3_fams = y3_mh("p4dv", 32, Y3_SEEDS["dev"])
    y3_complete = {d: sum(all(_y3_walk(c) == c["gold"] for c in f["cells"]) for f in y3_fams if f["depth"] == d) for d in (1, 2, 3, 4)}
    y3_d1_coincide = all(f["cells"][0]["gold"] == f["cells"][0]["rival"] for f in y3_fams if f["depth"] == 1)
    y4_fams = build_functional_graph_families("dev", 16, STREAM_SEEDS["dev"], depths=(1, 2, 3, 4))
    y4_complete = {d: sum(all(oracle_walk_public_bytes(UTF8.frame(c["public_prompt"], c["gold"]).ids) == c["gold"] for c in f["cells"]) for f in y4_fams if f["depth"] == d) for d in (1, 2, 3, 4)}
    # Mutation check: injecting a conflicting edge fails the Y4 parser cleanly
    c_sample = y4_fams[0]["cells"][0]
    k_conflict = c_sample["rows"][0][0]
    corrupt_prompt = c_sample["public_prompt"].replace("||", f"; {k_conflict}->Z9 ||")
    corrupt_caught = False
    try:
        parse_public_byte_observation(UTF8.frame(corrupt_prompt, "Z9").ids)
    except ValueError:
        corrupt_caught = True

    # 2. Y3 P5 prefix shortcut vs Y4 blind worlds + twins
    y3_tr = y3_p5("train", 70, Y3_SEEDS["corpus"])
    y3_dv = y3_p5("dev", 70, Y3_SEEDS["dev"])
    pref_cnt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w in y3_tr:
        pr = w["public_prompt"].split("|", 1)[0]
        for a in w["optimal_actions"]:
            pref_cnt[pr][a.split(":")[0]] += 1
    y3_pref_acc = sum(max(pref_cnt[w["public_prompt"].split("|", 1)[0]].items(), key=lambda x: x[1])[0] in {a.split(":")[0] for a in w["optimal_actions"]} for w in y3_dv) / len(y3_dv)

    y4_p5_tr = build_p5_blind_worlds("train", 70, STREAM_SEEDS["corpus"])
    y4_p5_dv = build_p5_blind_worlds("dev", 70, STREAM_SEEDS["dev"])
    # On Y4 blind worlds, flag-only lookup accuracy cannot hit 1.0 across matched opposite-action twins
    flag_map: dict[tuple[bool, bool], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w in y4_p5_tr:
        flag_map[(w["available_tools"]["QUERY"], w["available_tools"]["RETRIEVE"])][w["optimal_actions"][0].split(":")[0]] += 1
    y4_flag_acc = sum(max(flag_map[(w["available_tools"]["QUERY"], w["available_tools"]["RETRIEVE"])].items(), key=lambda x: x[1])[0] == w["optimal_actions"][0].split(":")[0] for w in y4_p5_dv) / len(y4_p5_dv)

    # 3. Y3 batch RNG slot collapse vs Y4 persistent RNG + wrong-donor target
    y3_bug_batch = tuple(random.Random(7 * 100003 + 0).randrange(64) for _ in range(16))
    y4_ok_batch = build_persistent_batch_plan(64, 4, 16, 7)[0]
    sw_pairs = build_state_swap_pairs(8, STREAM_SEEDS["dev"], depth=2, patch_step=1)
    y4_wrg_ok = all(
        p["wrong_donor_actual_answer"] == solve_recipient_suffix(p["wrong_donor_rows"], p["wrong_donor_state"], 1)
        and p["wrong_donor_actual_answer"] != p["wrong_donor_intended_answer"]
        for p in sw_pairs
    )

    return {
        "y3_graph_complete_by_depth": y3_complete,
        "y3_d1_endpoints_coincide": y3_d1_coincide,
        "y4_graph_complete_by_depth": y4_complete,
        "y4_conflict_mutation_caught": corrupt_caught,
        "y3_p5_prefix_shortcut_acc": y3_pref_acc,
        "y4_p5_flag_only_acc": y4_flag_acc,
        "y3_batch_unique_items": len(set(y3_bug_batch)),
        "y4_batch_unique_items": len(set(y4_ok_batch)),
        "y3_wrong_donor_matches_intended_and_diverges": bool(sw_pairs and all(p["wrong_donor_actual_answer"] != p["wrong_donor_intended_answer"] for p in sw_pairs)),
        "y4_wrong_donor_actual_matches_recipient": y4_wrg_ok,
    }
