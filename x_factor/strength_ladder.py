"""Intervention strength ladder: INSTRUMENT vs SUBSTRATE discrimination.

A0 null/control → A1 restructure → A2 query-conditioned hint →
A3 strong answer-blind scaffold → A4 oracle ceiling.

Decision rule:
  A3 high + A4 high  → INSTRUMENT_TOO_WEAK (latent capability exists)
  A3 low  + A4 high  → SUBSTRATE_LIMITATION (computation not elicitable)
  A3 low  + A4 low   → task/evaluator pathology
"""

from __future__ import annotations

import json, math, random, re, time
from pathlib import Path

import torch

_RUNTIME = Path(__file__).resolve().parent / "_runtime"
if str(_RUNTIME) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_RUNTIME))

from anra_core.config import CoreConfig, CANONICAL_CONFIG
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer

CHECKPOINT = "checkpoints/anra-v4-20k-sft3-accumulate.pt"
SEED = 41414
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
        fmt = "prose" if i % 2 == 0 else "table"
        block = ("\n".join(f"{o.capitalize()} keeps ref {c}." for o,c in zip(objs,codes))
                 if fmt=="prose" else
                 "item | ref\n" + "\n".join(f"{o.capitalize()} | {c}" for o,c in zip(objs,codes)))
        tgt = i % k
        q = f"Return ONLY the ref of {objs[tgt].capitalize()}."
        out.append({"id": f"sl-{i:03d}", "block": block, "query": q,
                    "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[tgt],
                    "facts": list(zip(objs, codes)), "target": objs[tgt],
                    "target_code": codes[tgt]})
    return out


def _A0_null(t):
    return t["prompt"]  # identity: baseline IS A0


def _A1_canonical(t):
    lines = sorted(t["block"].splitlines())
    return "ITEMS:\n" + "\n".join(f"* {l.strip()}" for l in lines if l.strip()) \
        + f"\n{t['query']}\nAnswer:"


def _A2_lexical_hint(t):
    qtok = set(t["query"].lower().split()) - {"return", "only", "ref", "the",
                                              "of", "for", "answer:", "code"}
    lines = []
    for line in t["block"].splitlines():
        head = line.strip().split(" ")[0].strip(":*|-").lower()
        marker = " >>" if any(tok in head for tok in qtok) and len(head) > 2 else ""
        lines.append(f"{line} {marker}".rstrip())
    return "\n".join(lines) + f"\n{t['query']}\nAnswer:"


def _A3_scaffold(t):
    """Strong answer-blind scaffold: extract the queried entity, index all
    facts, point to the matching slot, provide structured output slots.
    Performs deterministic external structuring WITHOUT supplying the answer."""
    qtok = set(t["query"].lower().split()) - {"return", "only", "ref", "the",
                                              "of", "for", "answer:", "code"}
    lines = t["block"].splitlines()
    indexed = []
    for j, line in enumerate(lines):
        head = line.strip().split(" ")[0].strip(":*|-").lower()
        match = "MATCH" if any(tok in head for tok in qtok) and len(head) > 2 else "---"
        indexed.append(f"Fact {j+1}: {line.strip()} [{match}]")
    target_name = next((e for e in t["facts"][0] and
                        [o for o, c in t["facts"] if any(tok in o.lower() for tok in qtok)]
                        ), t["facts"][0][0])
    scaffold = (
        "Task decomposition:\n"
        "Step 1: Identify the queried entity.\n"
        f"Queried entity: {target_name.capitalize()}\n"
        "Step 2: Locate the fact containing this entity.\n"
        + "\n".join(indexed) + "\n"
        + f"Step 3: Extract the ref associated with {target_name.capitalize()}.\n"
        + f"{t['query']}\nAnswer:"
    )
    return scaffold


def _A4_oracle(t):
    """Oracle ceiling: directly supplies the answer. NEVER counted as
    native cognition. Only estimates whether the task is externally solvable."""
    return f"The answer is {t['target_code']}.\n{t['block']}\n{t['query']}\nAnswer:"


LEVELS = [
    ("A0_null", _A0_null),
    ("A1_canonical", _A1_canonical),
    ("A2_lexical_hint", _A2_lexical_hint),
    ("A3_scaffold", _A3_scaffold),
    ("A4_oracle", _A4_oracle),
]


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


def main():
    device = "cuda"
    torch.manual_seed(SEED)
    print("[load]", flush=True)
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

    # Baseline failures only.
    failures = []
    base_pass = 0
    for t in tasks:
        out = _greedy(model, tok, t["prompt"], device)
        if _strict(out, t["gold"]):
            base_pass += 1
        else:
            failures.append(t)
    print(f"[baseline] {base_pass}/{len(tasks)} pass, {len(failures)} failures", flush=True)

    # Strength ladder harvest.
    results = {name: {"repairs": 0, "n": 0} for name, _ in LEVELS}
    per_task = []
    for t in failures:
        row = {}
        for name, fn in LEVELS:
            prompt = fn(t)
            out = _greedy(model, tok, prompt, device)
            ok = _strict(out, t["gold"])
            row[name] = ok
            results[name]["repairs"] += int(ok)
            results[name]["n"] += 1
        per_task.append({"id": t["id"], **{k: int(v) for k, v in row.items()},
                         "gold": t["gold"], "n_facts": len(t["facts"])})

    # Repairability curve.
    curve = {}
    for name, _ in LEVELS:
        r = results[name]
        curve[name] = round(r["repairs"] / max(r["n"], 1), 4)

    # Substrate vs instrument classification.
    a3 = curve["A3_scaffold"]
    a4 = curve["A4_oracle"]
    if a3 >= 0.30 and a4 >= 0.50:
        diagnosis = "INSTRUMENT_TOO_WEAK"
        interpretation = "latent capability exists; stronger probes expose it"
    elif a3 < 0.15 and a4 >= 0.50:
        diagnosis = "SUBSTRATE_LIMITATION_CANDIDATE"
        interpretation = "answer-blind assistance cannot recover correct behavior"
    elif a3 < 0.05 and a4 < 0.20:
        diagnosis = "TASK_OR_EVALUATOR_PATHOLOGY"
        interpretation = "even oracle assistance fails; check task validity"
    else:
        diagnosis = "MIXED_PARTIAL"
        interpretation = "partial repairability at intermediate strength"

    # Monotonicity check.
    levels_list = [curve[name] for name, _ in LEVELS]
    violations = [(LEVELS[i][0], LEVELS[i+1][0])
                  for i in range(len(levels_list)-1)
                  if levels_list[i+1] < levels_list[i]]

    receipt = {
        "schema": "anra-strength-ladder/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": CHECKPOINT,
        "parameter_sha256": payload.get("parameter_sha256"),
        "n_tasks": N_TASKS, "baseline_pass": base_pass,
        "n_failures": len(failures),
        "repairability_curve": curve,
        "monotonicity_violations": violations,
        "diagnosis": diagnosis,
        "interpretation": interpretation,
        "per_task": per_task,
        "note": "A4 oracle is ceiling estimate only, never native cognition",
    }
    out = Path("output/strength_ladder.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"curve": curve, "diagnosis": diagnosis,
                      "violations": violations}, indent=2))
    print(f"wrote {out}")

    del model
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


if __name__ == "__main__":
    main()
