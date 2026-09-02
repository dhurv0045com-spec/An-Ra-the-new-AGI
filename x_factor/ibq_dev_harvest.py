"""IBQ-DEV harvest: apply the v2 probe basis to real model failures.

Executes the qualification stage of the X-factor program: harvest the
intervention-outcome matrix using the v2 basis (7 probes with matched
controls), then run IBQ qualification + geometry-vs-null analysis.
Development evidence only — does not qualify for promotion without fresh
replication on a frozen basis.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_RUNTIME = Path(__file__).resolve().parent / "_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from anra_core.config import CoreConfig, CANONICAL_CONFIG  # noqa: E402
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

from x_factor.ibq import basis_qualified, basis_quality, geometry_vs_nulls  # noqa: E402
from x_factor.ibq_v2 import (  # noqa: E402
    BASIS_V2_IDS, V2_BASIS, apply_probe, COSTS_V2, qualify_basis_v2,
)

CHECKPOINT = "checkpoints/anra-v4-20k-sft3-accumulate.pt"
SEED = 41414
N_TASKS = 80
CODE_RE = __import__("re").compile(r"\b[A-Z]{3}-\d{3}\b")


def _strict(out, gold):
    c = CODE_RE.findall(out)
    return len(c) == 1 and c[0] == gold


def _tasks(seed, n):
    OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "entablature",
               "gaol", "hypostyle", "impound", "jamb", "keep", "lancet",
               "machicolation", "nave", "oratory", "portcullis")
    PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")
    rng = __import__("random").Random(seed)
    out = []
    for i in range(n):
        k = 2 + (i % 4)
        objs = rng.sample(OBJECTS, k)
        codes = [f"{rng.choice(PREFIXES)}-{rng.randrange(100,1000)}" for _ in objs]
        fmt = "prose" if i % 2 == 0 else "table"
        block = ("\n".join(f"{o.capitalize()} keeps ref {c}." for o,c in zip(objs,codes))
                 if fmt=="prose" else
                 "item | ref\n" + "\n".join(f"{o.capitalize()} | {c}" for o,c in zip(objs,codes)))
        tgt = i % k
        q = f"Return ONLY the ref of {objs[tgt].capitalize()}."
        out.append({"id": f"ibq-{i:03d}", "block": block, "query": q,
                    "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[tgt],
                    "facts": list(zip(objs, codes)), "target": objs[tgt]})
    return out


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
def _stats(model, tok, prompt, device):
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    t = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(t)[0, -1, :]
    probs = torch.softmax(logits.float(), -1)
    ent = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    t2 = torch.topk(probs, 2)
    return {"entropy": round(ent, 4),
            "margin": round(float(t2.values[0] - t2.values[1]), 4),
            "confidence": 0.0, "output_len": 0, "distinct_ratio": 0.0,
            "prompt_tokens": len(ids)}


def main():
    device = "cuda"
    torch.manual_seed(SEED)
    print("[load] checkpoint", flush=True)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = CoreConfig(**{k: payload["model_config"][k]
                        for k in CANONICAL_CONFIG.__dataclass_fields__})
    model = AnRaCore(cfg)
    model.load_state_dict({k: v for k, v in payload["model_state_dict"].items()
                           if k != "lm_head.weight"}, strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()
    tasks = _tasks(SEED, N_TASKS)

    # Baseline pass.
    failures = []
    baseline_pass = 0
    for t in tasks:
        out = _greedy(model, tok, t["prompt"], device)
        if _strict(out, t["gold"]):
            baseline_pass += 1
        else:
            failures.append(t)
    print(f"[baseline] {baseline_pass}/{len(tasks)} pass, {len(failures)} failures", flush=True)

    # Harvest outcome matrix using v2 probes.
    M, obs, meta = [], [], []
    for t in failures:
        row = []
        for pid in BASIS_V2_IDS:
            probe_task = dict(t, query=t["query"], block=t["block"])
            prompt = apply_probe(pid, probe_task)
            if pid == "QUERY_FRONTLOAD":
                prompt = apply_probe("QUERY_FRONTLOAD", probe_task)
            out = _greedy(model, tok, prompt, device)
            row.append(1 if _strict(out, t["gold"]) else 0)
            if pid == BASIS_V2_IDS[1]:  # capture obs once (from second probe)
                stats = _stats(model, tok, prompt, device)
                obs.append(stats)
        M.append(row)
        meta.append({"gold": t["gold"], "n_facts": len(t["facts"]),
                     "baseline_output": _greedy(model, tok, t["prompt"], device)[:50]})
    print(f"[matrix] {len(M)} failures x {len(BASIS_V2_IDS)} probes", flush=True)

    # IBQ qualification.
    gate = qualify_basis_v2(list(V2_BASIS.values()), M)
    quality = basis_quality(M)

    # Geometry vs nulls.
    nulls = geometry_vs_nulls(M, n_nulls=200, seed=SEED)

    oracle_coverage = sum(1 for r in M if any(r)) / len(M) if M else 0
    receipt = {
        "schema": "anra-ibq-dev-harvest/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": CHECKPOINT,
        "parameter_sha256": payload.get("parameter_sha256"),
        "n_tasks": N_TASKS, "baseline_pass": baseline_pass,
        "n_failures": len(failures),
        "basis_ids": list(BASIS_V2_IDS),
        "basis_sha": __import__("hashlib").sha256(
            json.dumps(sorted(BASIS_V2_IDS)).encode()).hexdigest()[:16],
        "outcome_matrix": {t["id"]: dict(zip(BASIS_V2_IDS, row))
                           for t, row in zip(failures, M)},
        "observations": dict(zip([t["id"] for t in failures], obs)),
        "meta": dict(zip([t["id"] for t in failures], meta)),
        "ibq_gate": gate,
        "ibq_quality": quality,
        "geometry_nulls": nulls,
        "oracle_coverage": round(oracle_coverage, 4),
    }
    out = Path("output/ibq_dev_harvest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate["qualified"], "checks": gate["checks"],
                      "oracle_coverage": round(oracle_coverage, 4),
                      "geometry": nulls["verdict"],
                      "quality": quality}, indent=2, default=str))
    print(f"wrote {out}")

    del model
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


if __name__ == "__main__":
    main()
