"""Binding factorial: separates addressing from interference from realization.

The +31pp selection effect was confounded — it simultaneously removed
distractors, shortened context, and added explicit selection. This experiment
manipulates each independently with matched controls:

  PRIMARY 1: target-mark vs random-mark (full context, same marker)
  PRIMARY 2: target-duplicate vs distractor-duplicate (full context)
  PRIMARY 3: entity-mark vs fact-mark (addressing without code repetition)
  PRIMARY 4: distractor-load curve (target + 0..k-1 distractors)
  PRIMARY 5: competitive binding load (code vs non-code distractors)

All interventions are answer-blind: they use only visible query text.
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

CHECKPOINT = "checkpoints/anra-v4-20k-sft3-accumulate.pt"
SEED = 41414
N_TASKS = 120
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")


def _strict(out, gold):
    c = CODE_RE.findall(out)
    return len(c) == 1 and c[0] == gold

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _qentity(query):
    m = re.search(r"ref of\s+([A-Za-z]+)", query, re.IGNORECASE)
    return _norm(m.group(1)) if m else ""


def _stable_seed(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


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
        out.append({"id": f"bf-{i:03d}", "block": block, "query": q,
                    "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[tgt],
                    "facts": list(zip(objs, codes)), "target": objs[tgt],
                    "n_facts": k, "target_pos": tgt})
    return out


def _seeded_rng(*parts):
    return random.Random(_stable_seed(*parts))


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


@torch.no_grad()
def _all_code_lps(model, tok, prompt, codes, device):
    """Binding margin: logP(gold) - mean(logP(other visible codes))."""
    lps = {}
    for c in codes:
        p_ids = tok.encode(prompt)
        c_ids = tok.encode(f" {c}.")
        ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long,
                           device=device)
        lp = torch.log_softmax(model(ids)[0].float(), -1)
        lps[c] = sum(float(lp[pos-1, ids[0, pos]]) for pos in range(1+len(p_ids), ids.shape[1]))
    return lps


def main():
    device = "cuda"
    torch.manual_seed(SEED)
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

    failures = []
    base_pass = 0
    for t in tasks:
        out = _greedy(model, tok, t["prompt"], device)
        if _strict(out, t["gold"]):
            base_pass += 1
        else:
            failures.append(t)
    print(f"[baseline] {base_pass}/{N_TASKS} pass, {len(failures)} failures", flush=True)

    # ---- PRIMARY 1-3: mark/duplicate contrasts with full context
    p1_mark_tgt, p1_mark_rand = [], []
    p2_dup_tgt, p2_dup_rand = [], []
    p3_entity_mark, p3_fact_mark = [], []
    p3_entity_dup, p3_fact_dup = [], []
    binding_margins = {}
    distractor_curve = {}

    for t in failures:
        entity = _qentity(t["query"])
        lines = t["block"].splitlines()
        tgt_line = next((l for l in lines if entity in _norm(l.lower())), lines[0])
        rng = _seeded_rng(SEED, t["id"], "control")
        others = [l for l in lines if l != tgt_line and l.strip()]
        rand_line = rng.choice(others) if others else lines[0]
        codes_visible = [c for _, c in t["facts"]]
        rand_code = rng.choice([c for _, c in t["facts"] if c != t["gold"]] or [t["gold"]])

        # PRIMARY 1: target-mark vs random-mark (full context)
        m_tgt = f"\n".join(f">>> {l}" if l == tgt_line else l for l in lines) \
            + f"\n{t['query']}\nAnswer:"
        m_rand = f"\n".join(f">>> {l}" if l == rand_line else l for l in lines) \
            + f"\n{t['query']}\nAnswer:"
        p1_mark_tgt.append(int(_strict(_greedy(model, tok, m_tgt, device), t["gold"])))
        p1_mark_rand.append(int(_strict(_greedy(model, tok, m_rand, device), t["gold"])))

        # PRIMARY 2: target-duplicate vs distractor-duplicate (full context)
        d_tgt = f"{t['block']}\n{tgt_line}\n{t['query']}\nAnswer:"
        d_rand = f"{t['block']}\n{rand_line}\n{t['query']}\nAnswer:"
        p2_dup_tgt.append(int(_strict(_greedy(model, tok, d_tgt, device), t["gold"])))
        p2_dup_rand.append(int(_strict(_greedy(model, tok, d_rand, device), t["gold"])))

        # PRIMARY 3: entity-only vs fact-level (addressing without code repeat)
        e_mark = f"\n".join(
            f">>> {l.split(' keeps')[0].strip()} (queried)" if entity in _norm(l.lower())
            else l for l in lines) + f"\n{t['query']}\nAnswer:"
        f_mark = f"\n".join(f">>> {l}" if l == tgt_line else l for l in lines) \
            + f"\n{t['query']}\nAnswer:"
        p3_entity_mark.append(int(_strict(_greedy(model, tok, e_mark, device), t["gold"])))
        p3_fact_mark.append(int(_strict(_greedy(model, tok, f_mark, device), t["gold"])))

        # Binding margin under baseline and target-mark
        lp_base = _all_code_lps(model, tok, t["prompt"], codes_visible, device)
        lp_mark = _all_code_lps(model, tok, m_tgt, codes_visible, device)
        gold = t["gold"]
        others_lp = [lp_base[c] for c in codes_visible if c != gold]
        margin_base = lp_base[gold] - (sum(others_lp) / len(others_lp) if others_lp else 0)
        others_mk = [lp_mark[c] for c in codes_visible if c != gold]
        margin_mark = lp_mark[gold] - (sum(others_mk) / len(others_mk) if others_mk else 0)
        binding_margins[t["id"]] = {"base": round(margin_base, 3),
                                     "marked": round(margin_mark, 3),
                                     "delta": round(margin_mark - margin_base, 3)}

        # PRIMARY 4: distractor-load curve (for a subset)
        if t["n_facts"] >= 3:
            tgt_fact = (t["target"], t["gold"])
            distractors = [(o, c) for o, c in t["facts"] if o != t["target"]]
            curve = {}
            for d_load in range(len(distractors) + 1):
                rng2 = _seeded_rng(SEED, t["id"], "distractor", d_load)
                selected = rng2.sample(distractors, d_load)
                all_facts = [tgt_fact] + selected
                if t["id"][3:].isdigit() and int(t["id"][3:]) % 2 == 0:
                    blk = "\n".join(f"{o.capitalize()} keeps ref {c}." for o, c in all_facts)
                else:
                    blk = "item | ref\n" + "\n".join(
                        f"{o.capitalize()} | {c}" for o, c in all_facts)
                p = f"{blk}\n{t['query']}\nAnswer:"
                out = _greedy(model, tok, p, device)
                lps = _all_code_lps(model, tok, p, [c for _, c in all_facts], device)
                others_l = [lps[c] for c in lps if c != t["gold"]]
                margin = lps[t["gold"]] - (sum(others_l)/len(others_l) if others_l else 0)
                curve[d_load] = {"correct": int(_strict(out, t["gold"])),
                                 "margin": round(margin, 3)}
            distractor_curve[t["id"]] = {"n_facts": t["n_facts"], "curve": curve}

    # Aggregate.
    def mean(x):
        return round(sum(x) / max(len(x), 1), 4)

    results = {
        "P1_target_mark_rate": mean(p1_mark_tgt),
        "P1_random_mark_rate": mean(p1_mark_rand),
        "P1_marking_contrast": round(mean(p1_mark_tgt) - mean(p1_mark_rand), 4),
        "P2_target_dup_rate": mean(p2_dup_tgt),
        "P2_distractor_dup_rate": mean(p2_dup_rand),
        "P2_duplication_contrast": round(mean(p2_dup_tgt) - mean(p2_dup_rand), 4),
        "P3_entity_mark_rate": mean(p3_entity_mark),
        "P3_fact_mark_rate": mean(p3_fact_mark),
        "P3_entity_vs_fact": round(mean(p3_entity_mark) - mean(p3_fact_mark), 4),
        "P3_entity_dup_rate": mean(p3_entity_dup),
        "P3_fact_dup_rate": mean(p3_fact_dup),
        "margin_delta_mean": round(np.mean([b["delta"] for b in binding_margins.values()]), 4),
        "margin_delta_positive_fraction": round(
            sum(1 for b in binding_margins.values() if b["delta"] > 0)
            / len(binding_margins), 4),
    }

    # Distractor-load interference slope.
    load_data = {}
    for tid, dc in distractor_curve.items():
        for d, v in dc["curve"].items():
            load_data.setdefault(int(d), []).append(v["margin"])
    interference = {d: round(np.mean(v), 4) for d, v in sorted(load_data.items())}
    if len(interference) >= 2:
        loads = sorted(interference.keys())
        slope = (interference[loads[-1]] - interference[loads[0]]) / (loads[-1] - loads[0])
    else:
        slope = 0.0

    receipt = {
        "schema": "anra-binding-factorial/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": CHECKPOINT,
        "parameter_sha256": payload.get("parameter_sha256"),
        "n_tasks": N_TASKS, "baseline_pass": base_pass, "n_failures": len(failures),
        "results": results,
        "interference_curve": interference,
        "interference_slope_per_distractor": round(slope, 4),
        "binding_margins": binding_margins,
        "distractor_curves": distractor_curve,
    }
    out = Path("output/binding_factorial.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"results": results,
                      "interference_curve": interference,
                      "interference_slope": round(slope, 4)}, indent=2))
    print(f"wrote {out}")

    del model
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


if __name__ == "__main__":
    main()
