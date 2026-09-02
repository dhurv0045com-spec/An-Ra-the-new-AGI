"""Causal elicitation decomposition: orthogonal intervention axes on real failures.

Fixes the strength-ladder confounds: (1) query parsing now normalizes
punctuation/case; (2) position, addressing, structure, and realization are
manipulated independently; (3) every mechanistic probe has a matched control;
(4) realization is decomposed from content access.
"""

from __future__ import annotations

import json, math, random, re, time, hashlib
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
N_TASKS = 80
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")


def _strict(out, gold):
    c = CODE_RE.findall(out)
    return len(c) == 1 and c[0] == gold


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _extract_query_entity(query: str) -> str:
    """Answer-blind query-entity extraction with punctuation/case normalization."""
    m = re.search(r"ref of\s+([A-Za-z]+)", query, re.IGNORECASE)
    return _norm(m.group(1)) if m else ""


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
        out.append({"id": f"cd-{i:03d}", "block": block, "query": q,
                    "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[tgt],
                    "facts": list(zip(objs, codes)), "target": objs[tgt],
                    "target_code": codes[tgt], "n_facts": k, "target_pos": tgt,
                    "format": fmt})
    return out


# ---------------------------------------------------------------------------
# Orthogonal intervention axes. Each manipulates ONE dimension.
# ---------------------------------------------------------------------------

def _reorder(block_lines, facts, target_entity, new_target_pos, rng):
    """Move target fact to new_target_pos; return new block + new target pos."""
    tgt_line = None
    others = []
    for line in block_lines:
        if _norm(target_entity) in _norm(line.lower()):
            tgt_line = line
        else:
            others.append(line)
    if tgt_line is None:
        return block_lines, None
    result = others[:]
    result.insert(min(new_target_pos, len(result)), tgt_line)
    return result, new_target_pos


def build_prompts(t):
    """Build all intervention prompts for one task."""
    rng = random.Random(hash(t["id"]) % 10000)
    lines = t["block"].splitlines()
    entity = _extract_query_entity(t["query"])
    k = t["n_facts"]

    # --- POSITION AXIS (same content, different target position)
    pos = {}
    for new_pos in range(k):
        reorder_lines, _ = _reorder(lines, t["facts"], t["target"], new_pos, rng)
        pos[f"P{new_pos}"] = f"\n".join(reorder_lines) + f"\n{t['query']}\nAnswer:"

    # --- ADDRESSING AXIS (mark target vs random non-target)
    tgt_line = next((l for l in lines if _norm(t["target"]) in _norm(l.lower())), "")
    nontgt = [l for l in lines if l != tgt_line and l.strip()]
    ctrl_line = rng.choice(nontgt) if nontgt else tgt_line
    addr = {
        "A_mark_target": "\n".join(
            f">>> {l}" if l == tgt_line else l for l in lines) + f"\n{t['query']}\nAnswer:",
        "A_mark_random": "\n".join(
            f">>> {l}" if l == ctrl_line else l for l in lines) + f"\n{t['query']}\nAnswer:",
        "A_relevant_fact_selected": (
            f"Queried entity: {t['target'].capitalize()}\n"
            f"Matching fact: {tgt_line}\n"
            f"{t['query']}\nAnswer:"),
        "A_distractor_selected": (
            f"Queried entity: {_norm(ctrl_line.split(' ')[0]).capitalize() or 'Unknown'}\n"
            f"Matching fact: {ctrl_line}\n"
            f"{t['query']}\nAnswer:"),
    }

    # --- STRUCTURE AXIS (same order, different format)
    struct = {
        "S_original": t["prompt"],
        "S_bullet": ("ITEMS:\n" + "\n".join(f"* {l.strip()}" for l in lines if l.strip())
                     + f"\n{t['query']}\nAnswer:"),
        "S_numbered": ("Facts:\n" + "\n".join(
            f"{j+1}. {l.strip()}" for j, l in enumerate(lines)) + f"\n{t['query']}\nAnswer:"),
    }

    # --- QUERY POSITION AXIS
    qpos = {
        "Q_after": t["prompt"],
        "Q_before": f"{t['query']}\n{t['block']}\nAnswer:",
        "Q_both": f"{t['query']}\n{t['block']}\n{t['query']}\nAnswer:",
    }

    # --- REALIZATION AXIS (evaluator-side decomposition)
    real = {
        "R_raw_greedy": t["prompt"],
        "R_copy_test": f"Return exactly: {t['target_code']}\nAnswer:",
        "R_oracle_near": f"Correct code: {t['target_code']}\n{t['block']}\n{t['query']}\nAnswer:",
    }

    return {"pos": pos, "addr": addr, "struct": struct, "qpos": qpos, "real": real}


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
def _gold_logprob(model, tok, prompt, gold, device):
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(f" {gold}.")
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long, device=device)
    lp = torch.log_softmax(model(ids)[0].float(), dim=-1)
    return sum(float(lp[pos - 1, ids[0, pos]]) for pos in range(1 + len(p_ids), ids.shape[1]))


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

    # Compute provenance.
    param_sha = hashlib.sha256()
    for name in sorted(model.state_dict().keys()):
        t = model.state_dict()[name].detach().cpu().contiguous()
        param_sha.update(f"{name}\0{tuple(t.shape)}\0{t.dtype}\0".encode())
        param_sha.update(t.view(torch.uint8).reshape(-1).numpy().tobytes())

    failures = []
    base_pass = 0
    for t in tasks:
        out = _greedy(model, tok, t["prompt"], device)
        if _strict(out, t["gold"]):
            base_pass += 1
        else:
            failures.append(t)
    print(f"[baseline] {base_pass}/{len(tasks)} pass, {len(failures)} failures", flush=True)

    # Causal decomposition harvest.
    profiles = {}
    for t in failures:
        prompts = build_prompts(t)
        profile = {"id": t["id"], "gold": t["gold"], "n_facts": t["n_facts"],
                   "format": t["format"], "target_pos": t["target_pos"]}

        # Position axis
        for key, prompt in prompts["pos"].items():
            out = _greedy(model, tok, prompt, device)
            profile[f"pos_{key}"] = int(_strict(out, t["gold"]))
        # Addressing axis
        for key, prompt in prompts["addr"].items():
            out = _greedy(model, tok, prompt, device)
            profile[f"addr_{key}"] = int(_strict(out, t["gold"]))
        # Structure axis
        for key, prompt in prompts["struct"].items():
            out = _greedy(model, tok, prompt, device)
            profile[f"struct_{key}"] = int(_strict(out, t["gold"]))
        # Query position axis
        for key, prompt in prompts["qpos"].items():
            out = _greedy(model, tok, prompt, device)
            profile[f"qpos_{key}"] = int(_strict(out, t["gold"]))
        # Realization axis
        for key, prompt in prompts["real"].items():
            out = _greedy(model, tok, prompt, device)
            profile[f"real_{key}"] = int(_strict(out, t["gold"]))
        # Gold logprob under key conditions
        lp_base = _gold_logprob(model, tok, t["prompt"], t["gold"], device)
        tgt_first = prompts["pos"]["P0"]
        lp_tgt_first = _gold_logprob(model, tok, tgt_first, t["gold"], device)
        sel = prompts["addr"]["A_relevant_fact_selected"]
        lp_sel = _gold_logprob(model, tok, sel, t["gold"], device)
        profile["lp_base"] = round(lp_base, 3)
        profile["lp_tgt_first"] = round(lp_tgt_first, 3)
        profile["lp_selected"] = round(lp_sel, 3)
        profile["lp_delta_selected"] = round(lp_sel - lp_base, 3)
        profiles[t["id"]] = profile

    # Aggregate causal contrasts.
    n = len(profiles)
    contrasts = {}
    for key in ("pos_P0", "pos_P1", "pos_P2", "pos_P3",
                "addr_A_mark_target", "addr_A_mark_random",
                "addr_A_relevant_fact_selected", "addr_A_distractor_selected",
                "struct_S_bullet", "struct_S_numbered",
                "qpos_Q_before", "qpos_Q_both",
                "real_R_copy_test", "real_R_oracle_near"):
        vals = [p[key] for p in profiles.values() if key in p]
        contrasts[key] = round(sum(vals) / max(len(vals), 1), 4)

    # Causal contrasts (matched pairs).
    causal = {
        "addressing_contrast": round(
            sum(p["addr_A_mark_target"] - p["addr_A_mark_random"]
                for p in profiles.values()) / n, 4),
        "selection_contrast": round(
            sum(p["addr_A_relevant_fact_selected"] - p["addr_A_distractor_selected"]
                for p in profiles.values()) / n, 4),
        "position_contrast_P0_vs_P_last": round(
            sum(p.get("pos_P0", 0) - p.get(f"pos_P{p['n_facts']-1}", 0)
                for p in profiles.values()) / n, 4),
        "lp_delta_selected_mean": round(
            sum(p["lp_delta_selected"] for p in profiles.values()) / n, 3),
    }

    receipt = {
        "schema": "anra-causal-decomposition/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": CHECKPOINT,
        "parameter_sha256": param_sha.hexdigest(),
        "n_tasks": N_TASKS, "baseline_pass": base_pass, "n_failures": n,
        "profiles": profiles,
        "marginal_repair_rates": contrasts,
        "causal_contrasts": causal,
    }
    out = Path("output/causal_decomposition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"n_failures": n, "contrasts": contrasts,
                      "causal": causal}, indent=2))
    print(f"wrote {out}")

    del model
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


if __name__ == "__main__":
    main()
