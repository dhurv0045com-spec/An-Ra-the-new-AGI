"""HISTORICAL_TOOL — checkpoint comparison (software only).

Do NOT use for mechanism claims. The "routing gap" concept is not established:
target-fact duplication may work via recency/proximity/repetition, not routing.
Future use: compare stronger developmental Cymek checkpoints.

Original docstring:

Runs the binding factorial on every available checkpoint and measures:
  - raw accuracy
  - target-duplication repair rate (address-elicited capability)
  - routing gap = address-elicited minus raw
  - minimal assistance level distribution
  - binding margin change under target marking

If raw accuracy stays flat while the routing gap shrinks, that means
training is internalizing the routing computation (not just learning answers).

Run: py -3 -m x_factor.checkpoint_comparison
"""

from __future__ import annotations

import json, math, random, re, time, hashlib
from pathlib import Path

import numpy as np
import torch

_RUNTIME = Path(__file__).resolve().parent / "_runtime"
if str(_RUNTIME) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_RUNTIME))

from anra_core.config import CoreConfig, CANONICAL_CONFIG
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer

CHECKPOINTS = [
    ("parent_20k", r"C:\Users\ankit\Downloads\anra-v4-current-full-resume.pt", True),
    ("context_child", "checkpoints/anra-v4-20k-sft-context-binding.pt", False),
    ("accumulate", "checkpoints/anra-v4-20k-sft3-accumulate.pt", False),
    ("queryswap", "checkpoints/anra-v4-20k-sft5-queryswap.pt", False),
]
SEED = 42424
N_TASKS = 60
CODE_RE = __import__("re").compile(r"\b[A-Z]{3}-\d{3}\b")


def _strict(out, gold):
    c = CODE_RE.findall(out)
    return len(c) == 1 and c[0] == gold


def _tasks(seed, n):
    OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "gaol", "impound",
               "lancet", "nave", "oratory", "portcullis")
    PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = 2 + (i % 4)
        objs = rng.sample(OBJECTS, k)
        codes = [f"{rng.choice(PREFIXES)}-{rng.randrange(100,1000)}" for _ in objs]
        block = "\n".join(f"{o.capitalize()} keeps ref {c}." for o,c in zip(objs,codes))
        tgt = i % k
        q = f"Return ONLY the ref of {objs[tgt].capitalize()}."
        out.append({"id": f"cc-{i:03d}", "block": block, "query": q,
                    "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[tgt],
                    "target": objs[tgt], "target_code": codes[tgt],
                    "facts": list(zip(objs, codes)), "n_facts": k})
    return out


def _param_sha(model):
    h = hashlib.sha256()
    for name in sorted(model.state_dict().keys()):
        t = model.state_dict()[name].detach().cpu().contiguous()
        h.update(f"{name}\0{tuple(t.shape)}\0{t.dtype}\0".encode())
        h.update(t.view(torch.uint8).reshape(-1).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def _greedy(model, tok, prompt, device, max_new=12):
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    out = []
    for _ in range(max_new):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[:, -1, :]
        nxt = int(logits.argmax(dim=-1))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out)


@torch.no_grad()
def _gold_lp(model, tok, prompt, gold, device):
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(f" {gold}.")
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long, device=device)
    lp = torch.log_softmax(model(ids)[0].float(), -1)
    return sum(float(lp[pos-1, ids[0, pos]]) for pos in range(1+len(p_ids), ids.shape[1]))


def _dup_prompt(t):
    tgt_line = next((l for l in t["block"].splitlines()
                     if _norm(t["target"]) in _norm(l.lower())), "")
    return f"{t['block']}\n{tgt_line}\n{t['query']}\nAnswer:"


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def evaluate_checkpoint(label, path, *, legacy, device="cuda"):
    torch.manual_seed(SEED)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = payload.get("model_config", {})
    cfg = CoreConfig(**{k: cfg_dict[k] for k in CANONICAL_CONFIG.__dataclass_fields__
                        if k in cfg_dict}) if cfg_dict else CANONICAL_CONFIG
    model = AnRaCore(cfg)
    state = payload.get("model_state_dict", payload.get("model", payload))
    model.load_state_dict({k: v for k, v in state.items() if k != "lm_head.weight"},
                          strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()
    param_sha = _param_sha(model)
    tasks = _tasks(SEED, N_TASKS)

    raw_correct = dup_correct = 0
    task_results = []
    raw_lps, dup_lps = [], []

    for t in tasks:
        raw = _greedy(model, tok, t["prompt"], device)
        raw_ok = _strict(raw, t["gold"])
        raw_correct += int(raw_ok)

        dup_prompt = _dup_prompt(t)
        dup = _greedy(model, tok, dup_prompt, device)
        dup_ok = _strict(dup, t["gold"])
        dup_correct += int(dup_ok)
        task_results.append({"id": t["id"], "raw_ok": raw_ok, "dup_ok": dup_ok})

        raw_lps.append(_gold_lp(model, tok, t["prompt"], t["gold"], device))
        dup_lps.append(_gold_lp(model, tok, dup_prompt, t["gold"], device))

    n = len(tasks)
    raw_acc = raw_correct / n
    dup_acc = dup_correct / n
    routing_gap = dup_acc - raw_acc
    lp_gain = (sum(dup_lps) - sum(raw_lps)) / n
    # Per-task: raw_fail AND assisted_fail = unrepairable by duplication
    unrepairable = sum(1 for r in task_results if not r["raw_ok"] and not r["dup_ok"]) / n

    del model
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)

    return {
        "label": label, "checkpoint": path,
        "parameter_sha256": param_sha,
        "raw_accuracy": round(raw_acc, 4),
        "duplication_accuracy": round(dup_acc, 4),
        "duplication_assistance_gap": round(routing_gap, 4),
        "mean_gold_lp_raw": round(sum(raw_lps)/n, 3),
        "mean_gold_lp_duplicated": round(sum(dup_lps)/n, 3),
        "lp_gain_from_duplication": round(lp_gain, 3),
        "failures_unrepairable_by_duplication": round(unrepairable, 4),
    }


def main():
    import time
    t0 = time.time()
    results = []
    for label, path, legacy in CHECKPOINTS:
        if not Path(path).exists():
            print(f"[skip] {label}: {path} not found")
            continue
        print(f"[eval] {label}...", flush=True)
        r = evaluate_checkpoint(label, path, legacy=legacy)
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    receipt = {
        "schema": "anra-checkpoint-comparison/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_seed": SEED, "n_tasks": N_TASKS,
        "checkpoints": results,
        "wall_seconds": round(time.time() - t0, 1),
    }

    # Cross-checkpoint analysis
    if len(results) >= 2:
        by_label = {r["label"]: r for r in results}
        analysis = {}
        for r in results:
            analysis[r["label"]] = {
                "raw": r["raw_accuracy"],
                "addressed": r["duplication_accuracy"],
                "duplication_assistance_gap": r["duplication_assistance_gap"],
                "lp_gain": r["lp_gain_from_duplication"],
            }
        receipt["comparison"] = analysis
        # Routing gap trajectory
        gaps = [(r["label"], r["duplication_assistance_gap"]) for r in results]
        receipt["routing_gap_trajectory"] = gaps

    out = Path("output/checkpoint_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"wall: {receipt['wall_seconds']}s")


if __name__ == "__main__":
    main()
