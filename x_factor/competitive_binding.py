"""Clean competitive binding load (Triquetra Experiment 2, DEV).

Tests whether failure is caused by competing entity-value bindings.

SAME-TASK nested loads (deterministic order, never resampled per load):
  L0: target only
  L1: target + D1
  L2: target + D1 + D2
  L3: target + D1 + D2 + D3
  L4: target + D1 + D2 + D3 + D4 (full, k=5)

At each load L1..L4, two matched variants:
  COMPETING: real distractor facts with plausible codes
  FILLER:    code-free sentences, approximately token-matched

Question: harm from context length generally, or specifically from
competing entity-value mappings?

Margin coherence (precommitted, option B):
  - gold logprob reported at L0..L4 for both arms
  - competitive margin (gold - mean co-present distractor LPs) at L1..L4 only
  - L0 margin not defined against absent distractors (no incompatible pooling)

Candidate CBL(x) object: number of plausible competing entity-value
mappings. Retained only if COMPETING degrades faster than FILLER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from anra_core.config import CoreConfig, CANONICAL_CONFIG  # noqa: E402
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/anra-v4-current-full-resume.pt"
SEED = 61616
N_TASKS = 80
K_FACTS = 5
MAX_NEW = 12
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
FILLER_POOL = (
    "The {o} stands beside the quiet garden.",
    "A narrow path runs past the old {o}.",
    "The {o} keeps its lamps lit at dusk.",
    "Visitors often rest near the calm {o}.",
    "The {o} overlooks the still water.",
)


def _strict(out: str, gold: str) -> int:
    c = CODE_RE.findall(out)
    return int(len(c) == 1 and c[0] == gold)


def _stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _tasks(seed: int, n: int, k: int = K_FACTS):
    OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "gaol", "impound",
               "lancet", "nave", "oratory", "portcullis")
    PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        objs = rng.sample(OBJECTS, k)
        codes = [f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}" for _ in objs]
        tgt = rng.randrange(k)
        q = f"Return ONLY the ref of {objs[tgt].capitalize()}."
        out.append({"id": f"cb-{i:03d}", "objs": objs, "codes": codes,
                    "target_idx": tgt, "target": objs[tgt], "gold": codes[tgt], "query": q})
    return out


def _nested_order(task, seed: int):
    idx = [i for i in range(len(task["objs"])) if i != task["target_idx"]]
    rng = random.Random(_stable_seed(seed, task["id"], "nested-order"))
    rng.shuffle(idx)
    return idx


def _filler_line(obj: str, j: int) -> str:
    return FILLER_POOL[j % len(FILLER_POOL)].format(o=obj)


def build_prompts(task, order, seed: int):
    t = task["target_idx"]
    tgt_line = f"{task['objs'][t].capitalize()} keeps ref {task['codes'][t]}."
    prompts: dict[str, str] = {}
    prompts["L0"] = f"{tgt_line}\n{task['query']}\nAnswer:"
    for load in range(1, len(order) + 1):
        ds = order[:load]
        comp_lines = [tgt_line] + [f"{task['objs'][i].capitalize()} keeps ref {task['codes'][i]}." for i in ds]
        fill_lines = [tgt_line] + [_filler_line(task["objs"][i], j) for j, i in enumerate(ds)]
        prompts[f"L{load}c"] = "\n".join(comp_lines) + f"\n{task['query']}\nAnswer:"
        prompts[f"L{load}f"] = "\n".join(fill_lines) + f"\n{task['query']}\nAnswer:"
    return prompts


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
def _gold_lp(model, tok, prompt, gold, device) -> float:
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(f" {gold}.")
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long, device=device)
    lp = torch.log_softmax(model(ids)[0].float(), -1)
    return sum(float(lp[pos - 1, ids[0, pos]]) for pos in range(1 + len(p_ids), ids.shape[1]))


@torch.no_grad()
def _margin(model, tok, prompt, gold, others, device):
    lg = _gold_lp(model, tok, prompt, gold, device)
    if not others:
        return None, lg
    los = [_gold_lp(model, tok, prompt, c, device) for c in others]
    return lg - sum(los) / len(los), lg


def run(checkpoint: str, seed: int, n_tasks: int, device: str):
    torch.manual_seed(seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = CoreConfig(**{k: payload["model_config"][k] for k in CANONICAL_CONFIG.__dataclass_fields__})
    model = AnRaCore(cfg)
    model.load_state_dict({k: v for k, v in payload["model_state_dict"].items() if k != "lm_head.weight"},
                          strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()

    param_sha = param_sha256_from_state_dict(model.state_dict())
    ckpt_sha = sha256_file(checkpoint)
    cfg_sha = sha256_json(payload["model_config"])
    try:
        tok_ident = tok.identity()
    except Exception:
        tok_ident = {"vocab": "canonical-v4-32k"}
    exp_sha = sha256_file(str(Path(__file__).resolve()))

    tasks = _tasks(seed, n_tasks)
    keys = ["L0", "L1c", "L1f", "L2c", "L2f", "L3c", "L3f", "L4c", "L4f"]
    per_task: dict = {}
    for ii, t in enumerate(tasks):
        order = _nested_order(t, seed)
        prompts = build_prompts(t, order, seed)
        row: dict = {"gold": t["gold"], "target": t["target"], "order": order, "conds": {}}
        for k in keys:
            out = _greedy(model, tok, prompts[k], device)
            ok = _strict(out, t["gold"])
            glp = _gold_lp(model, tok, prompts[k], t["gold"], device)
            if k == "L0":
                mg = None
            else:
                load = int(k[1])
                co = [t["codes"][i] for i in order[:load]]
                mg, _ = _margin(model, tok, prompts[k], t["gold"], co, device)
            row["conds"][k] = {"correct": ok, "output": out[:200],
                               "gold_lp": round(glp, 3),
                               "margin": (round(mg, 3) if mg is not None else None)}
        per_task[t["id"]] = row
        if (ii + 1) % 10 == 0:
            print(f"  ... {ii + 1}/{len(tasks)} done", flush=True)

    def col(k):
        return [per_task[tid]["conds"][k]["correct"] for tid in sorted(per_task)]

    def glp_col(k):
        return [per_task[tid]["conds"][k]["gold_lp"] for tid in sorted(per_task)]

    acc = {k: round(sum(col(k)) / len(per_task), 4) for k in keys}
    mean_lp = {k: round(float(np.mean(glp_col(k))), 3) for k in keys}
    # paired competing-vs-filler at each load
    import math as _m

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2.0 * sum(_m.comb(n, i) for i in range(k + 1)) / 2**n)

    paired = {}
    for load in (1, 2, 3, 4):
        a, b = col(f"L{load}c"), col(f"L{load}f")
        bpass = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
        aonly = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        bonly = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        bfail = len(a) - bpass - aonly - bonly
        paired[f"L{load}c_vs_L{load}f"] = {
            "competing_rate": round(sum(a) / len(a), 4),
            "filler_rate": round(sum(b) / len(b), 4),
            "paired_effect_comp_minus_fill": round((aonly - bonly) / len(a), 4),
            "both_pass": bpass, "comp_only": aonly, "fill_only": bonly, "both_fail": bfail,
            "mcnemar_exact_p": round(mcnemar(aonly, bonly), 4),
        }
    receipt = {
        "schema": "anra-competitive-binding/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "DEV (same-generator development; NOT fresh)",
        "provenance": {
            "checkpoint": checkpoint, "checkpoint_sha256": ckpt_sha,
            "parameter_sha256": param_sha, "config_sha256": cfg_sha,
            "tokenizer_identity": tok_ident,
            "tokenizer_sha256": sha256_json(tok_ident),
            "runtime_commit": git_head(Path(__file__).resolve().parents[1]),
            "experiment_source_sha256": exp_sha,
            "margin_rule": "gold LP at L0-L4; competitive margin vs co-present distractors at L1-L4 only",
        },
        "generator": {"seed": seed, "n_tasks": n_tasks, "k": K_FACTS, "nested": True},
        "accuracy": acc,
        "mean_gold_lp": mean_lp,
        "paired_competing_vs_filler": paired,
        "per_task": per_task,
    }
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_TASKS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="output/competitive_binding_dev.json")
    args = ap.parse_args()
    from checkpoint_identity import resolve_checkpoint  # strict: no silent fallback
    receipt = run(str(resolve_checkpoint(args.checkpoint)), args.seed, args.n, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": receipt["accuracy"], "paired": receipt["paired_competing_vs_filler"]}, indent=2))
    print(f"wrote {out}")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
