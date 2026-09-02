"""X1-REAL: harvest a real checkpoint's interventional cognitive geometry.

Preregistered rung X1 (x_factor/ladder.py) executed on the LOCAL GPU with
operator authorization. Protocol:

  - 60 fresh selective-binding tasks (seed disjoint from all training data);
  - baseline greedy attempt -> failures only proceed;
  - 5 single-variable runtime interventions per failure: NO_CHANGE,
    KNOWLEDGE_RESTATED (fact repeated in a separate line — not value supply),
    FORMAT_NORMALIZED (facts re-rendered as a clean list),
    QUERY_NEAR_ANSWER (query relocated), DECODE_SEARCH (best-of-4 sampling);
  - neutral observations per failure: greedy confidence (mean token
    log-prob), first-position entropy, top-2 margin, output length,
    distinct-token ratio, prompt length. NO factor names, NO family ids.
  - outcomes: strict single-code verifier match.

Analysis: outcome-matrix effective rank (compressibility, X0-real),
prospective ridge prediction o -> R_x with held-out split (X1), fixed-policy
baselines, oracle ceiling. Receipt saved; fresh fixtures untouched.
"""

from __future__ import annotations

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
    _sys.path.insert(0, str(_RUNTIME))  # core-exp lineage runtime: the
    # checkpoint was produced by that trainer; provenance requires its code.
from anra_core.config import CoreConfig  # noqa: E402
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

CHECKPOINT = "checkpoints/anra-v4-20k-sft3-accumulate.pt"
SEED = 31337
N_TASKS = 60
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
INTERVENTIONS = ("NO_CHANGE", "KNOWLEDGE_RESTATED", "FORMAT_NORMALIZED",
                 "QUERY_NEAR_ANSWER", "DECODE_SEARCH")


def _strict(out: str, gold: str) -> bool:
    c = CODE_RE.findall(out)
    return len(c) == 1 and c[0] == gold


_OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "entablature",
            "gaol", "hypostyle", "impound", "jamb", "keep", "lancet",
            "machicolation", "nave", "oratory", "portcullis")
_PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")


def build_eval_tasks(seed: int, n: int):
    """Fresh selective-binding tasks: new seed, bank vocabulary (the same
    distribution the checkpoint was trained on — development evidence)."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        k = 2 + (i % 4)
        objs = rng.sample(_OBJECTS, k)
        codes = [f"{rng.choice(_PREFIXES)}-{rng.randrange(100, 1000)}" for _ in objs]
        fmt = "prose" if i % 2 == 0 else "table"
        if fmt == "prose":
            block = "\n".join(f"{o.capitalize()} keeps ref {c}." for o, c in zip(objs, codes))
        else:
            block = "item | ref\n" + "\n".join(f"{o.capitalize()} | {c}" for o, c in zip(objs, codes))
        target = i % k
        q = f"Return ONLY the ref of {objs[target].capitalize()}."
        tasks.append({"id": f"x1-{i:03d}", "block": block, "query": q,
                      "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[target],
                      "facts": list(zip(objs, codes)), "target": objs[target],
                      "target_code": codes[target], "n_facts": k})
    return tasks


def variants(task) -> dict[str, str]:
    facts = [f"{o.capitalize()} keeps ref {c}." for o, c in task["facts"]]
    listing = "Facts:\n" + "\n".join(f"- {o.capitalize()} => {c}"
                                     for o, c in task["facts"])
    base_query = task["query"]
    return {
        "NO_CHANGE": task["prompt"],
        "KNOWLEDGE_RESTATED": (f"Reminder of the question: {base_query}\n"
                               + task["prompt"]),
        "FORMAT_NORMALIZED": f"{listing}\n{base_query}\nAnswer:",
        "QUERY_NEAR_ANSWER": f"{task['block']}\nAnswer:\n{base_query}\nAnswer:",
        "DECODE_SEARCH": task["prompt"],
    }


@torch.no_grad()
def _forward_stats(model, tok, prompt: str, device: str, sample_seeds=()):
    """Greedy output + neutral observation vector."""
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    logits0 = model(torch.tensor([ids], dtype=torch.long, device=device))[:, -1, :]
    probs = torch.softmax(logits0[0].float(), dim=-1)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    top2 = torch.topk(probs, 2)
    margin = float(top2.values[0] - top2.values[1])
    out, lp_sum = [], 0.0
    cur = list(ids)
    for _ in range(12):
        logits = model(torch.tensor([cur], dtype=torch.long, device=device))[:, -1, :]
        lps = torch.log_softmax(logits[0].float(), dim=-1)
        nxt = int(lps.argmax())
        if nxt == tok.eos_token_id:
            break
        lp_sum += float(lps[nxt])
        out.append(nxt)
        cur.append(nxt)
    text = tok.decode(out)
    n_out = max(len(out), 1)
    distinct = len(set(out)) / n_out
    return text, {
        "confidence": round(lp_sum / n_out, 4),
        "entropy": round(entropy, 4),
        "margin": round(margin, 4),
        "output_len": n_out,
        "distinct_ratio": round(distinct, 3),
        "prompt_tokens": len(ids),
    }


@torch.no_grad()
def _best_of_4(model, tok, prompt: str, device: str) -> str:
    """Inline temperature sampling (stays on the model's device; the core-exp
    generate() helper would rebuild a CPU executor around the raw model)."""
    best = ""
    for seed in (1, 2, 3, 4):
        g = torch.Generator(device=device).manual_seed(seed)
        ids = [tok.bos_token_id, *tok.encode(prompt)]
        out = []
        for _ in range(12):
            logits = model(torch.tensor([ids], dtype=torch.long, device=device))[:, -1, :]
            probs = torch.softmax(logits[0].float() / 0.8, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=g))
            if nxt == tok.eos_token_id:
                break
            out.append(nxt)
            ids.append(nxt)
        text = tok.decode(out)
        if len(text) > len(best):
            best = text
    return best


def harvest(device: str = "cuda") -> dict:
    torch.manual_seed(SEED)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    from dataclasses import asdict
    from anra_core.config import CANONICAL_CONFIG
    cfg = CoreConfig(**{k: payload["model_config"][k]
                        for k in CANONICAL_CONFIG.__dataclass_fields__})
    model = AnRaCore(cfg)
    state = payload["model_state_dict"]
    model.load_state_dict({k: v for k, v in state.items() if k != "lm_head.weight"},
                          strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    identity = type("I", (), {"parameter_sha256": payload.get("parameter_sha256"),
                              "global_step": payload.get("global_step")})
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()
    tasks = build_eval_tasks(SEED, N_TASKS)

    failures = []
    for t in tasks:
        out = _forward_stats(model, tok, t["prompt"], device)[0]
        if not _strict(out, t["gold"]):
            failures.append((t, out))
    print(f"[harvest] baseline failures: {len(failures)}/{len(tasks)}", flush=True)

    matrix, obs, meta = {}, {}, {}
    for t, base_out in failures:
        row, ovec = {}, None
        for iv in INTERVENTIONS:
            prompt = variants(t)[iv]
            if iv == "DECODE_SEARCH":
                out = _best_of_4(model, tok, prompt, device)
                stats = _forward_stats(model, tok, prompt, device)[1]
            else:
                out, stats = _forward_stats(model, tok, prompt, device)
            row[iv] = _strict(out, t["gold"])
            if ovec is None:
                ovec = stats
        matrix[t["id"]] = row
        obs[t["id"]] = ovec
        meta[t["id"]] = {"gold": t["gold"], "n_facts": t["n_facts"],
                         "format": "prose" if "keeps ref" in t["prompt"] else "table",
                         "baseline_output": base_out[:60]}

    del model
    import gc
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    return {"failures": len(failures), "n_tasks": len(tasks),
            "matrix": matrix, "observations": obs, "meta": meta,
            "checkpoint": CHECKPOINT,
            "parameter_sha256": getattr(identity, "parameter_sha256", None)}


def analyze(h: dict) -> dict:
    ids = sorted(h["matrix"])
    M = np.array([[int(h["matrix"][i][iv]) for iv in INTERVENTIONS] for i in ids])
    O = np.array([[h["observations"][i][k] for k in
                   ("confidence", "entropy", "margin", "output_len",
                    "distinct_ratio", "prompt_tokens")] for i in ids])
    # X0-real: compressibility of the real outcome matrix (informative cols).
    def eff_rank(m):
        m = m - m.mean(0, keepdims=True)
        s = np.linalg.svd(m, compute_uv=False)
        e = s ** 2
        return float((e.sum() ** 2) / (e ** 2).sum())
    er_all, er_info = eff_rank(M), eff_rank(M[:, 1:])
    # X1-real: prospective prediction, even/odd split (observed-only).
    tr, te = list(range(0, len(ids), 2)), list(range(1, len(ids), 2))
    def ridge(A, B, lam=1e-2):
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)
    center = O[tr].mean(0)
    W = ridge(O[tr] - center, M[tr].astype(float))
    pred = (O[te] - center) @ W
    acc = float(((pred > 0.5) == (M[te] > 0.5)).mean())
    fixed = {iv: float(M[te][:, INTERVENTIONS.index(iv)].mean()) for iv in INTERVENTIONS}
    best_fixed = max(fixed.values())
    oracle = float(M[te].max(axis=1).mean())
    return {
        "n_failures": len(ids),
        "X0_effective_rank_all": round(er_all, 3),
        "X0_effective_rank_informative": round(er_info, 3),
        "X1_prospective_cell_accuracy": round(acc, 4),
        "baselines_heldout": {k: round(v, 4) for k, v in fixed.items()},
        "best_fixed_policy_accuracy": round(best_fixed, 4),
        "oracle_repair_rate": round(oracle, 4),
        "prospective_beats_best_fixed": bool(acc > best_fixed),
        "verdict": ("X1-REAL PASS: observed state predicts intervention outcomes "
                    "above every fixed policy" if acc > best_fixed else
                    "X1-REAL FAIL: observed state does not beat fixed policies "
                    "(honest negative; fresh replication not attempted)"),
    }


def main() -> int:
    t0 = time.time()
    h = harvest()
    analysis = analyze(h)
    receipt = {"schema": "anra-x1-real/v1",
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               **h, "analysis": analysis,
               "wall_seconds": round(time.time() - t0, 1)}
    out = Path("output/x1_real_receipt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    main()
