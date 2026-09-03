"""Structural-OOD DEV of E5 duplication-assist (Triquetra next-gated-step).

QUESTION: does visible-query fact-duplication assistance generalize across
surface shift, or is it template-bound to the DEV distribution?

H1 (generalizes): E5dup-vs-sham stays positive with compatible magnitude.
H0 (template-bound): E5dup-vs-sham collapses to ~0.
DECISION: holds -> internalization training becomes justifiable; dies ->
report E5 as format hack and stop.

OOD shifts (all disjoint from DEV generator):
  entities : keep/jamb/transept/apse/scriptorium/refectory/belfry/chancel/...
  codes    : new prefixes {JKL,MNP,QRS,TVW,XYZ,KQR} + 4-digit numbers
  grammar  : "Ref {CODE} is kept by the {entity}." + "reference || holder" table
  queries  : 2 new templates, rotated
  order    : seeded shuffle per set
Arms (VisibleTask firewall for E5): E0 raw, E2 normalized rank,
E5dup matched-dup vs E5sham, E8 oracle ceiling.
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
from observed import assert_answer_blind, make_visible  # noqa: E402

from anra_core.config import CoreConfig, CANONICAL_CONFIG  # noqa: E402
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/anra-v4-current-full-resume.pt"
SEED = 91919
N_SETS = 60
K = 4
MAX_NEW = 12
CODE_RE = re.compile(r"\b[A-Z]{2,4}-\d{3,4}\b")
ENTITIES = ("keep", "jamb", "transept", "apse", "scriptorium",
            "refectory", "belfry", "chancel", "sacristy", "undercroft")
PREFIXES = ("JKL", "MNP", "QRS", "TVW", "XYZ", "KQR")
QTPL = ("Which ref belongs to the {X}? Respond with only the ref.",
        "Give only the ref held by the {X}.")


def _stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _gen_sets(seed: int, n: int, k: int = K):
    rng = random.Random(seed)
    sets = []
    for s in range(n):
        objs = rng.sample(ENTITIES, k)
        codes = [f"{rng.choice(PREFIXES)}-{rng.randrange(1000, 10000)}" for _ in objs]
        order = list(range(k))
        r2 = random.Random(_stable_seed(seed, f"ood-{s:03d}", "order"))
        r2.shuffle(order)
        objs = [objs[i] for i in order]
        codes = [codes[i] for i in order]
        if s % 2 == 0:
            block = "\n".join(f"Ref {c} is kept by the {o}." for o, c in zip(objs, codes))
        else:
            block = "reference || holder\n" + "\n".join(
                f"{c} || {o}" for o, c in zip(objs, codes))
        sets.append({"id": f"ood-{s:03d}", "objs": objs, "codes": codes, "block": block,
                     "qtpl": QTPL[s % len(QTPL)]})
    return sets


def _query(s, o: str) -> str:
    return s["qtpl"].format(X=o)


def _ent_of(query: str) -> str:
    m = re.search(r"(?:ref belongs to the|ref held by the)\s+([A-Za-z]+)", query, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def e5_dup_matched(vt) -> str:  # VisibleTask only
    ent = _ent_of(vt.query)
    mine = next((l for l in vt.context.splitlines()
                 if ent in re.sub(r"[^a-z0-9]", "", l.lower())), "")
    return f"{vt.context}\n{mine}\n{vt.query}\nAnswer:"


def e5_dup_sham(vt) -> str:  # VisibleTask only
    ent = _ent_of(vt.query)
    mine = next((l for l in vt.context.splitlines()
                 if ent in re.sub(r"[^a-z0-9]", "", l.lower())), "")
    others = [l for l in vt.context.splitlines() if l.strip() and l != mine]
    rng = random.Random(_stable_seed(vt.task_id, "sham"))
    pick = rng.choice(others) if others else mine
    return f"{vt.context}\n{pick}\n{vt.query}\nAnswer:"


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
    c = CODE_RE.findall(out)
    return int(len(c) == 1 and c[0] == gold)


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _boot_ci(vals, n_boot=10000, seed=31313):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
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
    for fn in (e5_dup_matched, e5_dup_sham):
        assert_answer_blind(fn)

    param_sha = param_sha256_from_state_dict(model.state_dict())
    ckpt_sha = sha256_file(checkpoint)
    try:
        tok_ident = tok.identity()
    except Exception:
        tok_ident = {"vocab": "canonical-v4-32k"}
    exp_sha = sha256_file(str(Path(__file__).resolve()))

    sets = _gen_sets(seed, n_sets)
    print(f"[ood] {len(sets)} sets x {K} on {device}", flush=True)
    arms = {"e0": [], "e2": [], "e5dup": [], "e5sham": [], "e8": []}
    per_set = {}
    for si, s in enumerate(sets):
        row = []
        for i, o in enumerate(s["objs"]):
            q = _query(s, o)
            base = f"{s['block']}\n{q}\nAnswer:"
            gold = s["codes"][i]
            vt = make_visible(f"{s['id']}-q{i}", s["block"], q, list(s["codes"]))
            e0 = _strict(_greedy(model, tok, base, device), gold)
            # E2 normalized rank
            r0 = [_cand_lp(model, tok, base, cd, device) for cd in s["codes"]]
            orows = []
            for r in range(K):
                if r == i:
                    continue
                pr = f"{s['block']}\n{_query(s, s['objs'][r])}\nAnswer:"
                orows.append([_cand_lp(model, tok, pr, cd, device) for cd in s["codes"]])
            nrow = np.array(r0) - np.mean(orows, axis=0)
            e2 = 1 if int(np.argmax(nrow)) == i else 0
            e5d = _strict(_greedy(model, tok, e5_dup_matched(vt), device), gold)
            e5s = _strict(_greedy(model, tok, e5_dup_sham(vt), device), gold)
            e8 = _strict(_greedy(model, tok, f"{s['block']}\nRecall: {gold}.\n{q}\nAnswer:", device), gold)
            for k_, v_ in (("e0", e0), ("e2", e2), ("e5dup", e5d), ("e5sham", e5s), ("e8", e8)):
                arms[k_].append(v_)
            row.append({"i": i, "e0": e0, "e2": e2, "e5dup": e5d, "e5sham": e5s, "e8": e8})
        per_set[s["id"]] = row
        if (si + 1) % 15 == 0:
            print(f"  ... {si + 1}/{len(sets)}", flush=True)

    rates = {k: round(sum(v) / len(v), 4) for k, v in arms.items()}

    def paired(a, b):
        aa, bb = arms[a], arms[b]
        ao = sum(1 for x, y in zip(aa, bb) if x == 1 and y == 0)
        bo = sum(1 for x, y in zip(aa, bb) if x == 0 and y == 1)
        d = [x - y for x, y in zip(aa, bb)]
        return {"a_rate": round(sum(aa) / len(aa), 4), "b_rate": round(sum(bb) / len(bb), 4),
                "paired_effect": round((ao - bo) / len(aa), 4),
                "mcnemar_exact_p": round(_mcnemar_exact(ao, bo), 4),
                "ci95": _boot_ci(d), "discord": [ao, bo]}

    receipt = {
        "schema": "anra-structural-ood-e5/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "STRUCTURAL_OOD_DEV (held out during DEV; NOT fresh)",
        "provenance": {
            "checkpoint": checkpoint, "checkpoint_sha256": ckpt_sha,
            "parameter_sha256": param_sha,
            "tokenizer_sha256": sha256_json(tok_ident),
            "runtime_commit": git_head(Path(__file__).resolve().parents[1]),
            "experiment_source_sha256": exp_sha,
        },
        "design": {"seed": seed, "n_sets": n_sets, "k": K,
                   "shifts": ["entity lexicon", "code prefixes+4-digit", "fact grammar",
                              "query templates", "fact order"]},
        "rates": rates, "n_queries": len(arms["e0"]),
        "paired": {"e5dup_vs_sham": paired("e5dup", "e5sham"),
                   "e2_vs_e0": paired("e2", "e0"),
                   "e8_vs_e2": paired("e8", "e2"),
                   "e5dup_vs_e0": paired("e5dup", "e0")},
        "per_set": per_set,
    }
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_SETS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="output/structural_ood_e5.json")
    args = ap.parse_args()
    receipt = run(args.checkpoint, args.seed, args.n, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"rates": receipt["rates"], "paired": receipt["paired"]}, indent=2))
    print(f"wrote {out}")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
