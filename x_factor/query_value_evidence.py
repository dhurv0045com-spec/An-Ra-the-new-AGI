"""Query-value evidence matrix (Triquetra Mission A/B DEV).

GRAND QUESTION: does the raw Core contain query-conditioned evidence for the
correct value that an answer-blind Connector can extract, or does performance
improve only when an oracle supplies the answer?

For each fact set F = {entity_i -> value_i} and queries q_i, score (no gold
inserted; every candidate visible in the original task):

  S[i,j] = log P(value_j | F, q_i)

Metrics: raw rank/rank1, QCS (same value queried vs not), VDM (correct vs
other values under same query), diagonal advantage, query-swap sensitivity,
permutation invariance, counterfactual query normalization, value-prior and
position decomposition, answer-blind extraction ladder E0-E8.

Unit: semantic fact set (queries are repeated measures; bootstrap by set).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

import sys as _sys

_RUNTIME = Path(__file__).resolve().parent / "_runtime"
if str(_RUNTIME) not in _sys.path:
    _sys.path.insert(0, str(_RUNTIME))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from provenance import git_head, param_sha256_from_state_dict, sha256_file, sha256_json  # noqa: E402
from observed import (  # noqa: E402
    EvaluatorTruth,
    VisibleTask,
    assert_answer_blind,
    make_truth,
    make_visible,
)

from anra_core.config import CoreConfig, CANONICAL_CONFIG  # noqa: E402
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/anra-v4-current-full-resume.pt"
SEED = 71717
N_SETS = 80
K = 4
ADDR_SUBSET = 40
PERM_SUBSETS = 30
MAX_NEW = 12
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
STRICT_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")


def _stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _gen_sets(seed: int, n: int, k: int = K):
    OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "gaol", "impound",
               "lancet", "nave", "oratory", "portcullis")
    PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")
    rng = random.Random(seed)
    sets = []
    for s in range(n):
        objs = rng.sample(OBJECTS, k)
        codes = [f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}" for _ in objs]
        block = "\n".join(f"{o.capitalize()} keeps ref {c}." for o, c in zip(objs, codes))
        sets.append({"id": f"qv-{s:03d}", "objs": objs, "codes": codes, "block": block})
    return sets


def _query(obj: str) -> str:
    return f"Return ONLY the ref of {obj.capitalize()}."


# ---------------- answer-blind addressing (VisibleTask only) ----------------

def _visible_entity_of(vt: VisibleTask) -> str:
    m = re.search(r"ref of\s+([A-Za-z]+)", vt.query, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def _visible_fact_line(vt: VisibleTask, entity_norm: str) -> str:
    for line in vt.context.splitlines():
        if entity_norm in re.sub(r"[^a-z0-9]", "", line.lower()):
            return line
    return ""


def e5_dup_matched(vt: VisibleTask) -> str:
    ent = _visible_entity_of(vt)
    line = _visible_fact_line(vt, ent)
    return f"{vt.context}\n{line}\n{vt.query}\nAnswer:"


def e5_dup_sham(vt: VisibleTask) -> str:
    ent = _visible_entity_of(vt)
    mine = _visible_fact_line(vt, ent)
    others = [l for l in vt.context.splitlines() if l.strip() and l != mine]
    rng = random.Random(_stable_seed(vt.task_id, "sham"))
    pick = rng.choice(others) if others else mine
    return f"{vt.context}\n{pick}\n{vt.query}\nAnswer:"


def e6_mark_matched(vt: VisibleTask) -> str:
    ent = _visible_entity_of(vt)
    mine = _visible_fact_line(vt, ent)
    marked = "\n".join(f">>> {l}" if l == mine else l for l in vt.context.splitlines())
    return f"{marked}\n{vt.query}\nAnswer:"


def e7_select_matched(vt: VisibleTask) -> str:
    ent = _visible_entity_of(vt)
    line = _visible_fact_line(vt, ent)
    return f"{line}\n{vt.query}\nAnswer:"


def e8_oracle_value(gold_value: str, context: str, query: str) -> str:
    return f"{context}\nRecall: {gold_value}.\n{query}\nAnswer:"


# ---------------- neural helpers ----------------

@torch.no_grad()
def _greedy(model, tok, prompt, device, max_new=MAX_NEW) -> str:
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    cur, out = list(ids), []
    for _ in range(max_new):
        logits = model(torch.tensor([cur], dtype=torch.long, device=device))[:, -1, :]
        nxt = int(logits.argmax(dim=-1))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        cur.append(nxt)
    return tok.decode(out)


@torch.no_grad()
def _cand_lp(model, tok, prompt, cand, device) -> float:
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(f" {cand}.")
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long, device=device)
    lp = torch.log_softmax(model(ids)[0].float(), -1)
    return sum(float(lp[pos - 1, ids[0, pos]]) for pos in range(1 + len(p_ids), ids.shape[1]))


def _strict(out: str, gold: str) -> int:
    c = STRICT_RE.findall(out)
    return int(len(c) == 1 and c[0] == gold)


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _boot_ci(vals, n_boot=10000, seed=9901):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return [0.0, 0.0]
    ms = [float(rng.choice(v, size=len(v), replace=True).mean()) for _ in range(n_boot)]
    return [round(float(np.percentile(ms, 2.5)), 4), round(float(np.percentile(ms, 97.5)), 4)]


def run(checkpoint: str, seed: int, n_sets: int, device: str):
    torch.manual_seed(seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = CoreConfig(**{k: payload["model_config"][k] for k in CANONICAL_CONFIG.__dataclass_fields__})
    model = AnRaCore(cfg)
    model.load_state_dict({k: v for k, v in payload["model_state_dict"].items() if k != "lm_head.weight"},
                          strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()

    # firewall self-check: blind arms pass, oracle fails
    for fn in (e5_dup_matched, e5_dup_sham, e6_mark_matched, e7_select_matched):
        assert_answer_blind(fn)
    try:
        assert_answer_blind(e8_oracle_value)
        raise AssertionError("oracle passed blind guard (must fail)")
    except ValueError:
        pass

    param_sha = param_sha256_from_state_dict(model.state_dict())
    ckpt_sha = sha256_file(checkpoint)
    cfg_sha = sha256_json(payload["model_config"])
    try:
        tok_ident = tok.identity()
    except Exception:
        tok_ident = {"vocab": "canonical-v4-32k"}
    exp_sha = sha256_file(str(Path(__file__).resolve()))

    sets = _gen_sets(seed, n_sets)
    print(f"[qv] {len(sets)} sets x {K} queries on {device}", flush=True)

    per_set: dict = {}
    # accumulators over queries
    raw_rank1_ok: list[int] = []
    norm_rank1_ok: list[int] = []
    gen_ok: list[int] = []
    qcs_all: list[float] = []
    vdm_all: list[float] = []
    set_raw_acc: list[float] = []
    set_norm_acc: list[float] = []

    for si, s in enumerate(sets):
        queries = [_query(o) for o in s["objs"]]
        prompts = [f"{s['block']}\n{q}\nAnswer:" for q in queries]
        # S[i,j]
        S = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                S[i, j] = _cand_lp(model, tok, prompts[i], s["codes"][j], device)
        # ranks
        raw_pred = S.argmax(axis=1)
        raw_ok = [(1 if raw_pred[i] == i else 0) for i in range(K)]
        # normalization: BASE[j] over r != i
        NORM = np.zeros_like(S)
        for i in range(K):
            others = [r for r in range(K) if r != i]
            base = S[others, :].mean(axis=0)
            NORM[i, :] = S[i, :] - base
        norm_pred = NORM.argmax(axis=1)
        norm_ok = [(1 if norm_pred[i] == i else 0) for i in range(K)]
        # greedy generation
        outs = [_greedy(model, tok, p, device) for p in prompts]
        g_ok = [_strict(o, s["codes"][i]) for i, o in enumerate(outs)]
        # QCS_i = S[i,i] - mean_{r!=i} S[r,i]; VDM_i = S[i,i]-mean_{j!=i} S[i,j]
        qcs, vdm = [], []
        for i in range(K):
            qcs.append(float(S[i, i] - np.mean([S[r, i] for r in range(K) if r != i])))
            vdm.append(float(S[i, i] - np.mean([S[i, j] for j in range(K) if j != i])))
        raw_rank1_ok.extend(raw_ok)
        norm_rank1_ok.extend(norm_ok)
        gen_ok.extend(g_ok)
        qcs_all.extend(qcs)
        vdm_all.extend(vdm)
        set_raw_acc.append(float(np.mean(raw_ok)))
        set_norm_acc.append(float(np.mean(norm_ok)))
        per_set[s["id"]] = {
            "objs": s["objs"], "codes": s["codes"],
            "S": [[round(float(x), 3) for x in row] for row in S.tolist()],
            "NORM": [[round(float(x), 3) for x in row] for row in NORM.tolist()],
            "raw_pred": [int(x) for x in raw_pred.tolist()],
            "norm_pred": [int(x) for x in norm_pred.tolist()],
            "raw_ok": raw_ok, "norm_ok": norm_ok, "gen_ok": g_ok,
            "outputs": [o[:120] for o in outs],
            "qcs": [round(float(x), 3) for x in qcs],
            "vdm": [round(float(x), 3) for x in vdm],
        }
        if (si + 1) % 20 == 0:
            print(f"  ... {si + 1}/{len(sets)} sets", flush=True)

    n_q = len(raw_rank1_ok)
    chance = 1.0 / K
    # paired norm-vs-raw by query
    b = sum(1 for a, x in zip(norm_rank1_ok, raw_rank1_ok) if a == 1 and x == 0)
    c = sum(1 for a, x in zip(norm_rank1_ok, raw_rank1_ok) if a == 0 and x == 1)
    # heuristics on raw scores: first/last/highest-prior
    # prior[j] = mean_i S[i,j]
    Pall = np.array([per_set[tid]["S"] for tid in sorted(per_set)])
    prior = Pall.mean(axis=0).mean(axis=0)  # mean over sets+queries -> per-candidate-pos? positional
    # per-set heuristic acc: predict argmax prior-position, first, last
    h_first = float(np.mean([1.0 if 0 == i else 0.0 for s in sets for i in range(K)]) )  # placeholder
    # simpler: heuristic "predict position p" accuracy = fraction queries with target at p
    tgt_pos = [i for s in sets for i in range(K)]  # targets cycle uniformly by construction? no: all i queried
    # every position queried equally -> first/last acc = 1/K exactly; report chance + prior-based:
    # highest-prior-position rule: pick argmax over mean-S-per-position
    pos_mean = Pall.mean(axis=0)  # avg over sets: (queries x candpos)
    # for each query-position i, prior rule predicts argmax_j pos_mean[i,j]
    prior_pred = pos_mean.argmax(axis=1)
    prior_ok = float(np.mean([(1.0 if prior_pred[i % K] == (i % K) else 0.0) for i in range(n_q)]))
    # position regression: S ~ query_match + cand_pos + dist_to_answer
    Y, M, Ppos, D = [], [], [], []
    for tid in sorted(per_set):
        Sm = np.array(per_set[tid]["S"])
        for i in range(K):
            for j in range(K):
                Y.append(Sm[i, j])
                M.append(1.0 if i == j else 0.0)
                Ppos.append(float(j))
                D.append(float(abs(K - 1 - j)))  # distance from answer region (end)
    Y = np.array(Y)
    def ols_coef(col):
        col = np.array(col)
        col = (col - col.mean()) / (col.std() + 1e-12)
        y = (Y - Y.mean()) / (Y.std() + 1e-12)
        return float((col * y).mean())
    reg = {"query_match_std": round(ols_coef(M), 4),
           "cand_position_std": round(ols_coef(Ppos), 4),
           "dist_to_answer_std": round(ols_coef(D), 4)}

    # ---- addressing + ladder on subset ----
    sub = sets[:ADDR_SUBSET]
    lad = {"e0_raw": [], "e1_rawrank": [], "e2_norm": [], "e3_lenient": [],
           "e4_proj": [], "e5_dup": [], "e5_sham": [], "e6_mark": [], "e7_sel": [], "e8_oracle": []}
    for s in sub:
        for i, o in enumerate(s["objs"]):
            q = _query(o)
            base_p = f"{s['block']}\n{q}\nAnswer:"
            truth_gold = s["codes"][i]
            vt = make_visible(f"{s['id']}-q{i}", s["block"], q, list(s["codes"]))
            _t = make_truth(vt.task_id, truth_gold, o, truth_gold)
            out0 = _greedy(model, tok, base_p, device)
            lad["e0_raw"].append(_strict(out0, truth_gold))
            # E1: raw rank on this query
            row = [_cand_lp(model, tok, base_p, cd, device) for cd in s["codes"]]
            lad["e1_rawrank"].append(1 if int(np.argmax(row)) == i else 0)
            # E2: normalized (need other-query rows: compute on the fly)
            other_rows = []
            for r in range(K):
                if r == i:
                    continue
                pr = f"{s['block']}\n{_query(s['objs'][r])}\nAnswer:"
                other_rows.append([_cand_lp(model, tok, pr, cd, device) for cd in s["codes"]])
            base_vec = np.mean(other_rows, axis=0)
            norm_row = np.array(row) - base_vec
            lad["e2_norm"].append(1 if int(np.argmax(norm_row)) == i else 0)
            # E3 lenient: gold appears anywhere
            lad["e3_lenient"].append(1 if truth_gold in out0 else 0)
            # E4 projection: any visible code in output; correct if gold present
            vis = [cd for cd in s["codes"] if cd in out0]
            lad["e4_proj"].append(1 if vis == [truth_gold] else 0)
            # E5/E6/E7/E8
            lad["e5_dup"].append(_strict(_greedy(model, tok, e5_dup_matched(vt), device), truth_gold))
            lad["e5_sham"].append(_strict(_greedy(model, tok, e5_dup_sham(vt), device), truth_gold))
            lad["e6_mark"].append(_strict(_greedy(model, tok, e6_mark_matched(vt), device), truth_gold))
            lad["e7_sel"].append(_strict(_greedy(model, tok, e7_select_matched(vt), device), truth_gold))
            lad["e8_oracle"].append(_strict(_greedy(model, tok, e8_oracle_value(truth_gold, s["block"], q), device), truth_gold))
    lad_rates = {k: round(sum(v) / max(len(v), 1), 4) for k, v in lad.items()}

    def paired(a, b):
        aa, bb = lad[a], lad[b]
        ao = sum(1 for x, y in zip(aa, bb) if x == 1 and y == 0)
        bo = sum(1 for x, y in zip(aa, bb) if x == 0 and y == 1)
        eff = (ao - bo) / len(aa)
        return {"paired_effect": round(eff, 4), "mcnemar_exact_p": round(_mcnemar_exact(ao, bo), 4),
                "a_rate": round(sum(aa) / len(aa), 4), "b_rate": round(sum(bb) / len(bb), 4),
                "discord": [ao, bo]}

    lad_paired = {
        "e2_vs_e1": paired("e2_norm", "e1_rawrank"),
        "e5dup_vs_sham": paired("e5_dup", "e5_sham"),
        "e7sel_vs_e0": paired("e7_sel", "e0_raw"),
        "e8_vs_e2": paired("e8_oracle", "e2_norm"),
    }

    # ---- permutation subset ----
    perm_sets = sets[:PERM_SUBSETS]
    perm_raw, perm_norm = [], []
    for s in perm_sets:
        for rep in (1, 2):
            rng = random.Random(_stable_seed(s["id"], "perm", rep))
            idx = list(range(K))
            rng.shuffle(idx)
            objs2 = [s["objs"][i] for i in idx]
            codes2 = [s["codes"][i] for i in idx]
            block2 = "\n".join(f"{o.capitalize()} keeps ref {c}." for o, c in zip(objs2, codes2))
            S2 = np.zeros((K, K))
            for i2, o2 in enumerate(objs2):
                p2 = f"{block2}\n{_query(o2)}\nAnswer:"
                for j2, c2 in enumerate(codes2):
                    S2[i2, j2] = _cand_lp(model, tok, p2, c2, device)
            # raw rank1 on permuted (diagonal = correct by construction)
            perm_raw.append(float(np.mean([1.0 if S2[i].argmax() == i else 0.0 for i in range(K)])))
            N2 = np.zeros_like(S2)
            for i in range(K):
                N2[i] = S2[i] - S2[[r for r in range(K) if r != i]].mean(axis=0)
            perm_norm.append(float(np.mean([1.0 if N2[i].argmax() == i else 0.0 for i in range(K)])))

    diag = float(np.mean([np.mean(np.diag(np.array(per_set[t]["S"]))) for t in per_set]))
    offd = float(np.mean([np.mean(np.array(per_set[t]["S"])[~np.eye(K, dtype=bool)]) for t in per_set]))

    receipt = {
        "schema": "anra-query-value-evidence/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "DEV (same-generator development; NOT fresh)",
        "provenance": {
            "checkpoint": checkpoint, "checkpoint_sha256": ckpt_sha,
            "parameter_sha256": param_sha, "config_sha256": cfg_sha,
            "tokenizer_identity": tok_ident,
            "tokenizer_sha256": sha256_json(tok_ident),
            "runtime_commit": git_head(Path(__file__).resolve().parents[1]),
            "experiment_source_sha256": exp_sha,
        },
        "design": {"seed": seed, "n_sets": n_sets, "k": K, "chance": chance,
                   "addr_subset": ADDR_SUBSET, "perm_subset": PERM_SUBSETS,
                   "score": "log P(' VALUE.') continuation"},
        "matrix": {
            "n_queries": n_q,
            "raw_rank1": round(sum(raw_rank1_ok) / n_q, 4),
            "norm_rank1": round(sum(norm_rank1_ok) / n_q, 4),
            "gen_acc": round(sum(gen_ok) / n_q, 4),
            "norm_vs_raw": {"n01": b, "n10": c,
                            "paired_effect": round((b - c) / n_q, 4),
                            "mcnemar_exact_p": round(_mcnemar_exact(b, c), 4),
                            "ci95_byset": _boot_ci(set_norm_acc and
                                                   [b2 - a2 for a2, b2 in zip(set_raw_acc, set_norm_acc)])},
            "qcs": {"mean": round(float(np.mean(qcs_all)), 4),
                    "median": round(float(np.median(qcs_all)), 4),
                    "frac_pos": round(float(np.mean([1.0 if x > 0 else 0.0 for x in qcs_all])), 4),
                    "ci95": _boot_ci(qcs_all)},
            "vdm": {"mean": round(float(np.mean(vdm_all)), 4),
                    "median": round(float(np.median(vdm_all)), 4),
                    "frac_pos": round(float(np.mean([1.0 if x > 0 else 0.0 for x in vdm_all])), 4),
                    "ci95": _boot_ci(vdm_all)},
            "diag_advantage": round(diag - offd, 4),
            "prior_position_rule": round(prior_ok, 4),
            "position_regression_std": reg,
            "perm_raw_rank1": round(float(np.mean(perm_raw)), 4) if perm_raw else 0.0,
            "perm_norm_rank1": round(float(np.mean(perm_norm)), 4) if perm_norm else 0.0,
        },
        "ladder_rates": lad_rates,
        "ladder_paired": lad_paired,
        "ladder_n": len(lad["e0_raw"]),
        "per_set": per_set,
    }
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_SETS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="output/query_value_evidence_dev.json")
    args = ap.parse_args()
    from checkpoint_identity import resolve_checkpoint  # strict: no silent fallback
    receipt = run(str(resolve_checkpoint(args.checkpoint)), args.seed, args.n, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"matrix": receipt["matrix"], "ladder": receipt["ladder_rates"],
                      "paired": receipt["ladder_paired"]}, indent=2))
    print(f"wrote {out}")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
