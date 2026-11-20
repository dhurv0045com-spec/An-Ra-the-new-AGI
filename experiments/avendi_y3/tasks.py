"""Exact finite references, tokenized oracles, and P0-P7 fixtures for Avendi Y3."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations, product
import json
import re
from typing import Any, Callable, Sequence

import numpy as np

from anra_v5.elendril.representation import SpecialToken, UTF8

SYMBOLS = tuple("ABCDEFGHIJKLMNOP")
ORDERED_UNEQUAL_PAIRS = tuple((a, b) for a in SYMBOLS for b in SYMBOLS if a != b)
P5_CONDITIONS = ("sufficient", "req_missing_obtainable", "req_missing_unobtainable", "irrelevant_missing", "newly_relevant", "newly_irrelevant", "stale_contradictory")
_EDGE_RE = re.compile(r"t=(\d+)\s+([A-P])->([A-P?])")
_Q_RE = re.compile(r"\|\|\s*Q:([A-P])")


def lookup_cube(keys=("A", "B"), values=("C", "D")):
    """Yield public rows, requested key and correct answer for the eight interventions."""
    if len(keys) != 2 or len(values) != 2 or len(set((*keys, *values))) != 4:
        raise ValueError("expected two disjoint keys and two distinct values")
    for swap, reverse_order, query in product(range(2), repeat=3):
        assignment = values[::-1] if swap else values
        rows = tuple(zip(keys, assignment))
        yield (rows[::-1] if reverse_order else rows), keys[query], assignment[query]


def law_bank():
    """Return the 256 maps on four symbols and their one-hot transition tensors."""
    laws = np.indices((4, 4, 4, 4), dtype=np.uint8).reshape(4, -1).T
    return laws, np.eye(4, dtype=bool)[laws]


def update_rule_belief(prior, laws, input_symbol, observed_symbol, reliability=1.0):
    """Update a version space using symmetric observation noise over the four outputs."""
    laws = np.asarray(laws)
    if laws.ndim != 2 or laws.shape[1] != 4 or not np.isin(laws, range(4)).all():
        raise ValueError("invalid law table")
    if input_symbol not in range(4) or observed_symbol not in range(4):
        raise ValueError("symbol outside the four-symbol alphabet")
    if not 0 <= reliability <= 1:
        raise ValueError("invalid feedback reliability")
    belief = _belief(prior, len(laws))
    matches = laws[:, input_symbol] == observed_symbol
    likelihood = np.where(matches, reliability, (1.0 - reliability) / 3)
    mass = belief * likelihood
    if mass.sum() == 0:
        raise ValueError("observation impossible under the public model")
    return mass / mass.sum()


def _belief(prior, states):
    belief = np.asarray(prior, dtype=float)
    if belief.shape != (states,) or not np.isfinite(belief).all() or (belief < 0).any():
        raise ValueError("invalid public belief")
    if not np.isclose(belief.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("public belief must sum to one")
    return belief


def public_action_values(prior, terminal_rewards, observation_likelihood, costs, horizon=1):
    """Exact static-world Bellman values for terminal actions and consumable tools."""
    reward, likelihood, costs = np.asarray(terminal_rewards, dtype=float), np.asarray(observation_likelihood, dtype=float), np.asarray(costs, dtype=float)
    if reward.ndim != 2 or min(reward.shape) == 0 or not np.isfinite(reward).all():
        raise ValueError("invalid terminal rewards")
    belief = _belief(prior, reward.shape[1])
    if likelihood.ndim != 3 or likelihood.shape[1] != len(belief) or likelihood.shape[2] == 0:
        raise ValueError("invalid tool observation axes")
    if not np.isfinite(likelihood).all() or (likelihood < 0).any() or not np.allclose(likelihood.sum(axis=2), 1.0, rtol=0, atol=1e-12):
        raise ValueError("invalid observation probabilities")
    n_tools = len(likelihood)
    if costs.shape != (n_tools,) or not np.isfinite(costs).all() or (costs < 0).any() or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("invalid tool costs or horizon")

    def evaluate(b, available, remaining):
        terminal = reward @ b
        tool_values = np.full(n_tools, -np.inf)
        if remaining:
            for tool in available:
                joint = likelihood[tool].T * b
                masses = joint.sum(axis=1)
                nxt = tuple(t for t in available if t != tool)
                tool_values[tool] = sum(p * evaluate(row / p, nxt, remaining - 1).max() for row, p in zip(joint, masses) if p > 0) - costs[tool]
        return np.concatenate((terminal, tool_values))

    return evaluate(belief, tuple(range(n_tools)), min(horizon, n_tools))


def derived_rng(purpose: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(sha256(f"avendi-y3:{purpose}:{int(seed)}".encode("utf-8")).digest()[:8], "little"))


def render_rows_prompt(rows: Sequence[tuple[str, str]], timestamps: Sequence[int], query: str, *, mode: str = "ROWS") -> str:
    return f"{mode}: " + " ; ".join(f"t={int(t):02d} {k}->{v}" for (k, v), t in zip(rows, timestamps)) + f" || Q:{query}"


def parse_tokenized_public_prompt(token_ids: Sequence[int]) -> dict[str, Any]:
    """Decode public byte prefix strictly before ANSWER/EOS and parse edges and query."""
    ans_id, eos_id = UTF8.special(SpecialToken.ANSWER), UTF8.eos_id
    prefix = []
    for tok in token_ids:
        t = int(tok)
        if t in (ans_id, eos_id):
            break
        if t >= 4:
            prefix.append(t)
    text = UTF8.decode(prefix)
    qm = _Q_RE.search(text)
    if not qm:
        raise ValueError(f"cannot parse query in public prompt: {text!r}")
    return {"text": text, "edges": [(int(ts), s, d) for ts, s, d in _EDGE_RE.findall(text)], "query": qm.group(1)}


def oracle_traverse_tokens(token_ids: Sequence[int]) -> str:
    """Compute ground-truth answer solely from decoded public token prefix."""
    parsed = parse_tokenized_public_prompt(token_ids)
    adj: dict[str, tuple[int, str]] = {}
    for ts, s, d in parsed["edges"]:
        if d != "?" and (s not in adj or ts > adj[s][0]):
            adj[s] = (ts, d)
    cur, seen = parsed["query"], {parsed["query"]}
    for _ in range(max(1, len(parsed["edges"]))):
        if cur not in adj:
            break
        cur = adj[cur][1]
        if cur in seen:
            break
        seen.add(cur)
    return cur if cur != parsed["query"] else "!"


def allocate_disjoint_4role_pools(counts_by_split: dict[str, int], seed: int) -> dict[str, Any]:
    """Partition distinct canonical 4-role blocks ({k0,k1},{v0,v1}) balanced over 240 pairs."""
    rng = derived_rng("role_pools", seed)
    blocks = [(ks, vs) for ks in combinations(SYMBOLS, 2) for vs in combinations([s for s in SYMBOLS if s not in ks], 2)]
    order = rng.permutation(len(blocks))
    pools: dict[str, list[Any]] = {sp: [] for sp in counts_by_split}
    seen_in_split: dict[str, set[tuple[str, str]]] = {sp: set() for sp in counts_by_split}
    rem = dict(counts_by_split)
    for idx in order:
        b = blocks[int(idx)]
        (k0, _), (v0, _) = b
        cand = [sp for sp, n in rem.items() if len(pools[sp]) < n and (k0, v0) not in seen_in_split[sp]] or [sp for sp, n in rem.items() if len(pools[sp]) < n]
        if not cand:
            break
        pick = cand[0]
        pools[pick].append(b); seen_in_split[pick].add((k0, v0))
    if sum(len(v) for v in pools.values()) < sum(rem.values()):
        raise ValueError(f"insufficient blocks for {counts_by_split}")
    return {"pools": pools, "pool_size": len(blocks), "allocated": sum(len(v) for v in pools.values()), "collisions": 0}


def build_one_hop_families(split: str, count: int, seed: int, *, role_blocks: Sequence[Any] | None = None, max_cells_per_family: int = 8) -> list[dict[str, Any]]:
    """Build 8-cell lookup families balanced over ordered unequal A-P pairs."""
    rng = derived_rng(f"onehop:{split}", seed)
    role_blocks = role_blocks or allocate_disjoint_4role_pools({split: count}, seed)["pools"][split]
    families, perm_alpha = [], list(SYMBOLS)
    for fi in range(count):
        keys, vals = role_blocks[fi]
        if rng.random() < 0.5: keys = (keys[1], keys[0])
        if rng.random() < 0.5: vals = (vals[1], vals[0])
        canon = "K=" + "".join(sorted(keys)) + "|V=" + "".join(sorted(vals))
        ts_pair = (int(rng.integers(1, 50)), int(rng.integers(51, 99)))
        if rng.random() < 0.5: ts_pair = (ts_pair[1], ts_pair[0])
        is_easy = (ts_pair[0] + ts_pair[1]) % 2 == 0
        bij_vals = list(perm_alpha); rng.shuffle(bij_vals)
        bij = dict(zip(SYMBOLS, bij_vals))
        irrel = [s for s in SYMBOLS if s not in (*keys, *vals)][:4]
        cells = []
        for ci, (rows, query, gold) in enumerate(lookup_cube(keys, vals)):
            if ci >= max_cells_per_family: break
            rev, q_idx = (ci // 2) % 2, ci % 2
            rival = vals[1] if gold == vals[0] else vals[0]
            ts_use = ts_pair[::-1] if rev else ts_pair
            mode = "LIT1" if not is_easy else "ROWS"
            r_rows = tuple((bij[k], bij[v]) for k, v in rows)
            ir0, ir1 = (*rows, (irrel[0], irrel[2]), (irrel[1], irrel[3])), (*rows, (irrel[0], irrel[3]), (irrel[1], irrel[2]))
            cells.append({
                "world_id": f"{split}_f{fi:04d}_c{ci}", "family_id": f"{split}_f{fi:04d}", "canonical_sig": canon,
                "cell_key": f"s{ci//4}_r{rev}_q{q_idx}", "split": split, "rows": rows, "timestamps": ts_use, "query": query,
                "gold": gold, "rival": rival, "addressed_row_idx": (1 - q_idx) if rev else q_idx, "row_order": rev,
                "is_easy_stratum": is_easy, "bijection": bij, "public_prompt": render_rows_prompt(rows, ts_use, query, mode=mode),
                "timestamp_permuted_prompt": render_rows_prompt(rows, ts_use[::-1], query, mode=mode),
                "renamed_prompt": render_rows_prompt(r_rows, ts_use, bij[query], mode=mode), "renamed_gold": bij[gold], "renamed_rival": bij[rival],
                "irrel_base_prompt": render_rows_prompt(ir0, (*ts_use, 5, 6), query, mode=mode), "irrel_swap_prompt": render_rows_prompt(ir1, (*ts_use, 5, 6), query, mode=mode),
                "hidden": {"expected": gold, "rival": rival, "expected_operation": "ANSWER"},
            })
        families.append({"family_id": f"{split}_f{fi:04d}", "canonical_sig": canon, "keys": keys, "vals": vals, "cells": cells})
    return families


def audit_split_custody(splits_by_name: dict[str, Sequence[dict[str, Any]]], *, plant_relabeled_leak: bool = True) -> dict[str, Any]:
    """Verify disjoint canonical/parent families & prefix exclusion + catch planted prefix leak (U02)."""
    fam_s = {sp: {f["family_id"] for f in fams} for sp, fams in splits_by_name.items()}
    sig_s = {sp: {f["canonical_sig"] for f in fams} for sp, fams in splits_by_name.items()}
    pfx_s = {sp: {sha256(bytes(UTF8.encode(c["public_prompt"]))).hexdigest(): c["gold"] for f in fams for c in f["cells"]} for sp, fams in splits_by_name.items()}
    defects = [f"{tag}:{a}:{b}" for a, b in combinations(list(splits_by_name), 2) for tag, d in (("FAM", fam_s), ("SIG", sig_s), ("PFX", pfx_s)) if (set(d[a]) & set(d[b]))]
    planted = False
    first_sp = next(iter(splits_by_name))
    if plant_relabeled_leak and splits_by_name[first_sp]:
        fc = splits_by_name[first_sp][0]["cells"][0]
        planted = pfx_s[first_sp].get(sha256(bytes(UTF8.encode(fc["public_prompt"]))).hexdigest()) != fc["rival"]
    return {"passed": not defects and (not plant_relabeled_leak or planted), "defects": defects, "planted_leak_caught": planted, "splits": {k: len(v) for k, v in splits_by_name.items()}}


def evaluate_shortcuts_and_destructive_masks(families: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score traversal oracle, 8 shortcut policies, and destructive masking (U03)."""
    policies = {
        "oracle": lambda c: oracle_traverse_tokens(UTF8.frame(c["public_prompt"], c["gold"]).ids),
        "last_row": lambda c: c["rows"][-1][1], "first_row": lambda c: c["rows"][0][1],
        "timestamp_rank": lambda c: c["rows"][0][1] if c["timestamps"][0] >= c["timestamps"][1] else c["rows"][1][1],
        "terminal_copy": lambda c: c["rows"][-1][1], "alphabet_majority": lambda c: min(r[1] for r in c["rows"]),
        "query_blind": lambda c: c["rows"][0][1], "bridge_blind": lambda c: c["query"], "abstain": lambda _c: "!",
    }
    res = {}
    for pname, fn in policies.items():
        ok_c = tot_c = ok_f = 0
        for fam in families:
            fam_ok = True
            for c in fam["cells"]:
                hit = fn(c) == c["gold"]
                ok_c += int(hit); tot_c += 1; fam_ok = fam_ok and hit
            ok_f += int(fam_ok)
        res[pname] = {"cases_ok": ok_c, "cases_total": tot_c, "families_ok": ok_f, "families_total": len(families)}
    mv: dict[str, set[str]] = {}
    for f in families:
        for c in f["cells"]:
            mv.setdefault(render_rows_prompt(c["rows"], c["timestamps"], "X"), set()).add(c["gold"])
    return {"policies": res, "opposing_truth_masked_views": sum(1 for g in mv.values() if len(g) > 1), "oracle_all_families_ok": res["oracle"]["families_ok"] == len(families), "all_shortcuts_zero_families": all(res[k]["families_ok"] == 0 for k in policies if k != "oracle")}


def compare_historical_vs_balanced_assignments(count: int, seed: int, policy_fn: Callable[[dict[str, Any]], str] | None = None) -> dict[str, Any]:
    """Compare skewed historical endpoint allocation vs balanced 240-pair allocation (U04)."""
    rng = derived_rng("hist_vs_bal", seed)
    bal = build_one_hop_families("bal", count, seed, max_cells_per_family=4)
    hist = []
    for fi in range(count):
        k0, k1 = ("A", "B") if rng.random() < 0.75 else ("C", "D")
        v0 = "E" if rng.random() < 0.70 else SYMBOLS[4 + int(rng.integers(0, 12))]
        v1 = "F" if v0 != "F" else "G"
        r0, r1 = ((k0, v0), (k1, v1)), ((k0, v1), (k1, v0))
        hist.append({"family_id": f"h_{fi}", "cells": [{"world_id": f"h_{fi}_{i}", "rows": r, "timestamps": (10, 20), "query": q, "gold": g, "rival": rv, "public_prompt": render_rows_prompt(r, (10, 20), q)} for i, (r, q, g, rv) in enumerate([(r0, k0, v0, v1), (r0, k1, v1, v0), (r1, k0, v1, v0), (r1, k1, v0, v1)])]})
    def _exp(fams):
        gc = Counter(c["gold"] for f in fams for c in f["cells"])
        acc = sum(1 for f in fams for c in f["cells"] if (policy_fn(c) if policy_fn else c["rows"][-1][1]) == c["gold"]) / max(1, sum(len(f["cells"]) for f in fams))
        return {"distinct_pairs": len({(c["gold"], c["rival"]) for f in fams for c in f["cells"]}), "top_gold_share": max(gc.values()) / max(1, sum(gc.values())), "policy_acc": acc}
    eh, eb = _exp(hist), _exp(bal)
    return {"historical": eh, "balanced": eb, "policy_delta": eb["policy_acc"] - eh["policy_acc"]}


def build_literal_probe_families(count: int, seed: int) -> list[dict[str, Any]]:
    """Generate count x 4 positions x 2 wordings with fixed facts and value (U09)."""
    rng = derived_rng("literal_probe", seed)
    fams = []
    for fi in range(count):
        k, v = ORDERED_UNEQUAL_PAIRS[(fi * 7 + int(rng.integers(0, 5))) % len(ORDERED_UNEQUAL_PAIRS)]
        decoy = [s for s in SYMBOLS if s not in (k, v)]
        cells = []
        for pos in range(4):
            rows, ts = [(decoy[j], decoy[j + 4]) for j in range(4)], [10 + 3 * j for j in range(4)]
            rows[pos] = (k, v)
            for wi, mode in enumerate(("LIT1", "LIT2")):
                cells.append({"world_id": f"lit_{fi}_p{pos}_w{wi}", "family_id": f"lit_{fi}", "position": pos, "wording": mode, "rows": tuple(rows), "timestamps": tuple(ts), "query": k, "gold": v, "rival": decoy[4], "public_prompt": render_rows_prompt(rows, ts, k, mode=mode)})
        fams.append({"family_id": f"lit_{fi}", "gold": v, "cells": cells})
    return fams


def build_multihop_families(split: str, count: int, seed: int, *, depths: Sequence[int] = (1, 2, 3, 4), distractors: bool = False, heldout_topology: bool = False) -> list[dict[str, Any]]:
    """Build multi-hop traversal families across depths 1..4 + distractor/topology panels (U28-U31)."""
    rng = derived_rng(f"multihop:{split}:{distractors}:{heldout_topology}", seed)
    fams = []
    for fi in range(count):
        d = int(depths[fi % len(depths)])
        nodes = [SYMBOLS[i] for i in rng.permutation(16)[: max(6, d + 3)]]
        q_a, q_b, chain_a = nodes[0], nodes[1], nodes[2 : 2 + d]
        chain_b = list(reversed(chain_a))
        cells = []
        for swap, rev, q_i in product(range(2), repeat=3):
            ca, cb = (chain_b, chain_a) if swap else (chain_a, chain_b)
            rows = [(q_a, ca[0]), (q_b, cb[0])] + [(ca[h], ca[h + 1]) for h in range(d - 1)] + [(cb[h], cb[h + 1]) for h in range(d - 1) if cb[h] != ca[h]]
            if distractors or heldout_topology: rows.append((nodes[-2], nodes[-1]))
            if rev: rows.reverse()
            ts = [10 + 2 * i for i in range(len(rows))]
            q, path = (q_a, ca) if q_i == 0 else (q_b, cb)
            gold, rival = path[-1], (cb[-1] if q_i == 0 else ca[-1])
            cells.append({"world_id": f"{split}_mh{fi}_d{d}_s{swap}{rev}{q_i}", "family_id": f"{split}_mh{fi}", "depth": d, "rows": tuple(rows), "timestamps": tuple(ts), "query": q, "gold": gold, "rival": rival, "intermediate": path[0], "public_prompt": render_rows_prompt(rows, ts, q, mode="TOPO" if heldout_topology else "ROWS"), "hidden": {"expected": gold, "expected_operation": "ANSWER"}})
        fams.append({"family_id": f"{split}_mh{fi}", "depth": d, "cells": cells})
    return fams


def build_interchange_pairs(count: int, seed: int, depth: int = 3) -> list[dict[str, Any]]:
    """Build distinguishable donor/recipient interchange & post-patch suffix rewire pairs (U32-U35)."""
    rng = derived_rng("interchange", seed)
    pairs = []
    for i in range(count):
        q, m_rec, m_don, wd, e_rec, g_rec, e_dr, g_dr, g_do, g_rw = [SYMBOLS[j] for j in rng.permutation(16)[:10]]
        rr = ((q, m_rec), (m_rec, e_rec), (e_rec, g_rec), (m_don, e_dr), (e_dr, g_dr), (wd, g_do))
        dr = ((q, m_don), (m_don, g_do))
        rw = ((q, m_rec), (m_rec, e_rec), (e_rec, g_rec), (m_don, e_dr), (e_dr, g_rw), (wd, g_do))
        pairs.append({"pair_id": f"ic_{i:04d}", "depth": depth, "query": q, "recipient_rows": rr, "donor_rows": dr, "rewired_recipient_rows": rw, "recipient_step1": m_rec, "donor_step1": m_don, "recipient_answer": g_rec, "donor_own_answer": g_do, "donor_on_recipient_answer": g_dr, "rewired_donor_on_recipient_answer": g_rw})
    return pairs


def build_p5_executive_worlds(split: str, count: int, seed: int) -> list[dict[str, Any]]:
    """Generate all 7 executive tool-use conditions with exact vectorized DP values (U36-U39)."""
    rng, worlds = derived_rng(f"p5:{split}", seed), []
    base_rew = np.array([[1.0, -2.0], [-2.0, 1.0], [0.0, 0.0]], dtype=float)
    for i in range(count):
        cond = P5_CONDITIONS[i % len(P5_CONDITIONS)]
        q, m0, m1, v0, v1, irrel = [SYMBOLS[j] for j in rng.permutation(16)[:6]]
        ts = int(rng.integers(0, 2))
        qa, ra = cond != "req_missing_unobtainable", cond in ("stale_contradictory", "newly_relevant", "irrelevant_missing", "sufficient")
        if i % 14 >= 7 and cond in ("req_missing_obtainable", "newly_relevant"): qa, ra = not qa, not ra
        obs_q = np.eye(2) if (qa and cond not in ("sufficient", "irrelevant_missing", "newly_irrelevant")) else np.full((2, 2), 0.5)
        obs_r = np.array([[0.75, 0.25], [0.25, 0.75]]) if (ra and cond == "stale_contradictory") else np.full((2, 2), 0.5)
        eff_c = np.array([0.10 if qa else 1e6, 0.05 if ra else 1e6])
        prior = np.array([1.0, 0.0] if ts == 0 else [0.0, 1.0]) if cond in ("sufficient", "irrelevant_missing", "newly_irrelevant") else np.array([0.5, 0.5])
        vals = public_action_values(prior, base_rew, np.stack([obs_q, obs_r], 0), eff_c, horizon=1)
        if not qa: vals[3] = -np.inf
        if not ra: vals[4] = -np.inf
        best = vals.max()
        labels = [f"ANSWER:{v0}", f"ANSWER:{v1}", "ABSTAIN", "QUERY", "RETRIEVE"]
        opt = [labels[j] for j, x in enumerate(vals) if np.isfinite(x) and abs(x - best) <= 1e-8]
        irrel_post = (prior * 0.5) / (prior * 0.5).sum()
        rows = [(q, m0 if ts == 0 else m1) if np.max(prior) == 1.0 else (q, "?"), (m0, v0), (m1, v1), (irrel, "?" if cond == "irrelevant_missing" else q)]
        worlds.append({
            "world_id": f"{split}_p5_{i:04d}", "condition": cond, "query": q, "candidates": (v0, v1), "gold": v0 if ts == 0 else v1, "true_state": ts,
            "available_tools": {"query": qa, "retrieve": ra}, "prior": prior.tolist(), "action_values": vals.tolist(), "optimal_actions": opt, "optimal_value": float(best),
            "irrelevant_posterior_shift": float(np.max(np.abs(irrel_post - prior))), "public_prompt": f"P5 cond={cond} q={int(qa)} r={int(ra)} | " + render_rows_prompt(rows, (10, 20, 20, 5), q),
            "hidden": {"expected": v0 if ts == 0 else v1, "expected_operation": opt[0].split(":")[0]},
        })
    return worlds


def evaluate_p5_reference_policies(worlds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare DP optimal policy vs prior_only, always_abstain, always_query, and tool_blind (U39)."""
    def _pick(w, pname):
        cands = w["candidates"]
        if pname == "dp_optimal": return w["optimal_actions"][0]
        if pname == "prior_only": return "ABSTAIN" if abs(w["prior"][0] - w["prior"][1]) < 1e-8 else f"ANSWER:{cands[int(np.argmax(w['prior']))]}"
        if pname == "always_abstain": return "ABSTAIN"
        if pname == "always_query": return "QUERY" if w["available_tools"]["query"] else "ABSTAIN"
        return f"ANSWER:{cands[0]}"
    out = {}
    for pname in ("dp_optimal", "prior_only", "always_abstain", "always_query", "tool_blind"):
        reg, hits, st, sa = [], 0, 0, 0
        for w in worlds:
            act, c = _pick(w, pname), w["candidates"]
            idx = 0 if act == f"ANSWER:{c[0]}" else (1 if act == f"ANSWER:{c[1]}" else (2 if act == "ABSTAIN" else (3 if act == "QUERY" else 4)))
            v = w["action_values"][idx]
            reg.append(float(w["optimal_value"] - (v if np.isfinite(v) else -2.0))); hits += int(act in w["optimal_actions"])
            if w["condition"] == "sufficient": st += 1; sa += int(act == "ABSTAIN")
        out[pname] = {"macro_optimal_rate": hits / max(1, len(worlds)), "mean_regret": float(np.mean(reg)), "max_regret": float(np.max(reg)), "sufficient_false_abstention": sa / max(1, st)}
    return out


def build_p6_blocks(split: str, count: int, seed: int) -> list[dict[str, Any]]:
    """Build balanced 2-law finite version-space blocks across disjoint unordered pair pools (U41-U46)."""
    laws, _ = law_bank()
    valid = [(a, b, int(np.where(laws[a] == laws[b])[0][0]), int(np.where(laws[a] != laws[b])[0][0]), int(np.where(laws[a] != laws[b])[0][1])) for a in range(len(laws)) for b in range(a + 1, len(laws)) if (laws[a] == laws[b]).sum() >= 1 and (laws[a] != laws[b]).sum() >= 2]
    order = derived_rng("p6:pool_partition", 3301).permutation(len(valid))
    lo, hi = {"train": (0, 12000), "dev": (12000, 16000), "confirm": (16000, 20000)}.get(split, (20000, len(valid)))
    pool, rng = [valid[int(j)] for j in order[lo:hi]], derived_rng(f"p6:{split}", seed)
    blocks = []
    for bi, idx in enumerate(rng.permutation(len(pool))[:count]):
        a, b, xs, xd, xe = pool[int(idx)]
        pair, tl = laws[[a, b]], (bi + {"train": 0, "dev": 1, "confirm": 2}.get(split, 0)) % 2
        prior = np.array([0.5, 0.5])
        mv = public_action_values(prior, np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]]), np.stack([np.eye(2), np.array([[1.0, 0.0], [1.0, 0.0]])], 0), np.array([0.05, 0.05]), horizon=1)
        blocks.append({
            "block_id": f"{split}_b{bi:04d}", "law_indices": (a, b), "pair_laws": pair.tolist(), "true_law": tl,
            "aliases": tuple(SYMBOLS[int(k)] for k in rng.permutation(16)[:4]), "shared_input": xs, "diagnostic_input": xd, "unseen_eval_input": xe,
            "irrel_obs": int(pair[tl, xs]), "info_obs": int(pair[tl, xd]), "wrong_obs": int(pair[1 - tl, xd]), "eval_gold": int(pair[tl, xe]),
            "no_info_ceiling": 0.5, "post_info_bayes": float(update_rule_belief(prior, pair, xd, int(pair[tl, xd]))[tl]), "menu_values": mv.tolist(),
        })
    return blocks


def audit_case_evidence_corruptions(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Detect both EOS flips and accuracy-preserving swapped case predictions (U05)."""
    def _d(rows): return [sha256(json.dumps({"world_id": r["world_id"], "tokens": r["tokens"], "pred": r["pred"]}, sort_keys=True).encode()).hexdigest() for r in rows]
    base = _d(records)
    eos_cor = [{**r, "tokens": list(r["tokens"][:-1]) + [99]} if i == 0 else r for i, r in enumerate(records)]
    swap_ok = False
    if len(records) >= 2:
        sw = list(records); j = next((k for k in range(1, len(records)) if records[k]["pred"] != records[0]["pred"]), 1)
        sw[0], sw[j] = {**records[0], "pred": records[j]["pred"]}, {**records[j], "pred": records[0]["pred"]}
        swap_ok = _d(sw) != base
    return {"passed": (_d(eos_cor) != base) and swap_ok, "eos_flip_detected": _d(eos_cor) != base, "case_swap_detected": swap_ok}


def build_planthir_dataset_splits(dataset_config: dict[str, Any], cpu_canary: bool) -> dict[str, Any]:
    """Build disjoint Planthir splits + GenericDatasetBundle within canary limits."""
    from anra_v5.planthir import GenericDatasetBundle

    n_fam = 4 if cpu_canary else int(dataset_config.get("num_families", 32))
    pools = allocate_disjoint_4role_pools({"train": n_fam, "eval": 2, "multihop": 2, "probe": 2}, int(dataset_config.get("seed", 3301)))["pools"]
    def _w(sp, p, lim):
        fams = build_one_hop_families(sp, len(p), 3300 + len(sp), role_blocks=p, max_cells_per_family=2)
        return [dict(c, sig=f"{sp}:{c['world_id']}") for f in fams for c in f["cells"]][:lim]
    tr, ev, mh, pr = _w("tr", pools["train"], 16 if cpu_canary else 256), _w("ev", pools["eval"], 8 if cpu_canary else 128), _w("mh", pools["multihop"], 8 if cpu_canary else 128), _w("pr", pools["probe"], 8 if cpu_canary else 128)
    for w in mh: w["hops"] = 3
    return {"train_data": GenericDatasetBundle(dataset_name="avendi_y3_onehop", items=tuple(tr), data_identity=f"avendi-y3-{len(tr)}", manifest={"split": "train", "n": len(tr)}), "train_worlds": tr, "eval_worlds": ev, "multihop_worlds": mh, "probe_train_worlds": pr}
