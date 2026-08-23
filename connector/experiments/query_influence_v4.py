"""Query Influence Matrix v4: fresh replication fixture for the margin
experiment (tp-margin-queryswap-003).

Role: DEVELOPMENT_REPLICATION_ONLY — frozen BEFORE the margin child exists;
evaluated ONCE on the selected child; never used for selection; never
triggers retraining. Same role contract as QIM-v3.

Fresh vocabulary (prefixes JBR/KZM/LQN/MWD/NFX verified absent from every
consumed corpus and sealed OOD suite; entities are a brand-new set, checked
programmatically at import time via vocabulary_disjointness()).

Statistical unit: the FACT GROUP. Primary inference = paired sign-flip test
on per-group mean candidate-normalized query lift, parent vs child. Greedy
uses each target's OWN prompt via build_query_prompt (the v3 stale-prompt
bug class is structurally impossible here). Monte Carlo p uses the plus-one
estimator (never literal zero).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path

import torch

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
SEED_V4 = 20260912
PREFIXES_V4 = ("JBR", "KZM", "LQN", "MWD", "NFX")
ENTITIES_V4 = ("gargoyle", "spirelet", "lucarne", "crocket", "finial",
               "gablet", "housel", "piscina", "sedilia", "reredos-arc",
               "squint", "parclose")
N_GROUPS_V4 = 40
DIAGNOSTIC_VERSION = "anra-query-influence/v4-margin-replication"
PERM_DRAWS = 20000
PERM_SEED = 7


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES_V4)}-{rng.randrange(100, 1000)}"


def build_groups() -> list[dict]:
    rng = random.Random(SEED_V4)
    groups = []
    for gi in range(N_GROUPS_V4):
        k = 2 + (gi % 3)
        ents = rng.sample(ENTITIES_V4, k)
        codes = [_code(rng) for _ in ents]
        records = [{"entity": e, "code": c,
                    "line": f"The {e} is marked {c}."}
                   for e, c in zip(ents, codes)]
        rng.shuffle(records)
        fmt = "prose" if gi % 2 == 0 else "table"
        groups.append({"displayed_facts": records, "format": fmt})
    return groups


def fixture_hash() -> str:
    return hashlib.sha256(
        json.dumps(build_groups(), sort_keys=True).encode("utf-8")).hexdigest()


def _query(rec: dict) -> str:
    return f"Return the tag of the {rec['entity']}."


def _prompt(block: str, query: str) -> str:
    return f"{block}\n{query}\nAnswer:"


def build_query_prompt(group: dict, target_index: int) -> str:
    """Pure per-target prompt builder (stale-prompt bug impossible)."""
    recs = group["displayed_facts"]
    if group.get("format") == "table":
        block = ("item | tag\n"
                 + "\n".join(f"the {r['entity']} | {r['code']}" for r in recs))
    else:
        block = "\n".join(r["line"] for r in recs)
    return _prompt(block, _query(recs[target_index]))


def vocabulary_disjointness() -> dict:
    consumed_prefixes = set()
    corpus = ""
    for p in ("data/grouped_queryswap/train.jsonl",
              "data/grouped_queryswap/heldout.jsonl",
              "data/capability_bank/train.jsonl",
              "data/capability_bank/dev.jsonl",
              "connector/experiments/ood_battery/items.json",
              "connector/experiments/ood2_battery/items.json",
              "connector/experiments/ood3_battery/items.json",
              "connector/experiments/ood4_battery/items.json"):
        f = Path(p)
        if f.exists():
            text = f.read_text(encoding="utf-8")
            corpus += text
            consumed_prefixes |= set(re.findall(r"\b([A-Z]{3})-\d{3}\b", text))
    pref_hits = sorted(p for p in PREFIXES_V4
                       if p in consumed_prefixes
                       or re.search(rf"\b{p}-\d{{3}}\b", corpus))
    ent_hits = sorted(e for e in ENTITIES_V4
                      if e in corpus or e.replace("-", " ") in corpus
                      or e.capitalize() in corpus)
    return {"prefix_hits": pref_hits, "entity_hits": ent_hits,
            "disjoint": not (pref_hits or ent_hits)}


# ---- statistics (same contracts as v3: plus-one MC, exact small-n) ------

def sign_flip_p(values: list[float]) -> float:
    n = len(values)
    if not values:
        raise ValueError("sign_flip_p requires at least one paired value")
    obs = sum(values)
    rng = random.Random(PERM_SEED)
    if n <= 20:
        import itertools
        count = total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if sum(s * v for s, v in zip(signs, values)) >= obs:
                count += 1
        return count / total
    count = 0
    for _ in range(PERM_DRAWS):
        signs = [rng.choice((1, -1)) for _ in range(n)]
        if sum(s * v for s, v in zip(signs, values)) >= obs:
            count += 1
    return (count + 1) / (PERM_DRAWS + 1)


def bootstrap_ci(values: list[float], draws: int = 10000) -> tuple[float, float]:
    rng = random.Random(PERM_SEED + 1)
    n = len(values)
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(draws))
    return stats[int(0.025 * draws)], stats[min(draws - 1, int(0.975 * draws))]


def dz(values: list[float]) -> float:
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    return m / math.sqrt(max(var, 1e-12))


@torch.no_grad()
def _completion_logprob(model, tok, prompt: str, completion: str) -> float:
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion)
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long,
                       device=next(model.parameters()).device)
    logits = model(ids)[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return sum(float(logprobs[pos - 1, ids[0, pos]].item())
               for pos in range(1 + len(p_ids), ids.shape[1]))


@torch.no_grad()
def _greedy(model, tok, prompt: str, max_new_tokens: int = 10) -> str:
    device = next(model.parameters()).device
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    out = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[:, -1, :]
        nxt = int(logits.argmax(dim=-1).item())
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out)


def _strict(out: str, gold: str) -> bool:
    c = CODE_RE.findall(out)
    return len(c) == 1 and c[0] == gold


def run_model(label: str, checkpoint: str, *,
              parent_report: dict | None = None,
              device: str = "cuda",
              evaluation_class: str | None = None) -> dict:
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    model, _, identity = load_core_checkpoint(checkpoint, legacy_unverified=True)
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()
    groups = build_groups()

    group_means: list[float] = []
    cand_lifts: list[float] = []
    ranks: list[int] = []
    greedy_ok: list[bool] = []

    for g in groups:
        recs = g["displayed_facts"]
        prompts = [build_query_prompt(g, qi) for qi in range(len(recs))]
        L = []
        group_lifts = []
        for qi in range(len(recs)):
            L.append([_completion_logprob(model, tok, prompts[qi],
                                          f" {r['code']}.") for r in recs])
        for qi in range(len(recs)):
            others = [L[j][qi] for j in range(len(recs)) if j != qi]
            group_lifts.append(L[qi][qi] - sum(others) / len(others))
            ranks.append(1 + sum(1 for j in range(len(recs))
                                 if L[qi][j] > L[qi][qi]))
            greedy_ok.append(_strict(_greedy(model, tok, prompts[qi]),
                                     recs[qi]["code"]))
        cand_lifts.extend(group_lifts)
        group_means.append(sum(group_lifts) / len(group_lifts))

    report = {
        "schema": DIAGNOSTIC_VERSION,
        "role": "DEVELOPMENT_REPLICATION_ONLY",
        "evaluation_class": evaluation_class or "ORIGINAL_EVALUATION",
        "label": label, "checkpoint": checkpoint,
        "global_step": identity.global_step,
        "parameter_sha256": getattr(identity, "parameter_sha256", None),
        "fixture_sha256": fixture_hash(),
        "vocab_disjointness": vocabulary_disjointness(),
        "primary_unit": "fact_group",
        "evaluator_version": "v4.0-margin-experiment",
        "per_group_query_lift": [round(x, 4) for x in group_means],
        "group_level": {
            "n_groups": len(group_means),
            "mean_group_lift": round(sum(group_means) / len(group_means), 4),
            "median_group_lift": round(sorted(group_means)[len(group_means) // 2], 4),
            "groups_positive": f"{sum(1 for x in group_means if x > 0)}/{len(group_means)}",
            "sign_flip_p_mean_gt_0": round(sign_flip_p(group_means), 5),
        },
        "candidate_diagnostic_only": {
            "mean_candidate_lift": round(sum(cand_lifts) / len(cand_lifts), 4),
            "correct_rank1_fraction": f"{sum(1 for r in ranks if r == 1)}/{len(ranks)}",
            "mean_correct_rank": round(sum(ranks) / len(ranks), 2),
            "greedy_corresponding_accuracy":
                f"{sum(1 for x in greedy_ok if x)}/{len(greedy_ok)}",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass

    if parent_report is not None:
        pg = parent_report["per_group_query_lift"]
        assert len(pg) == len(group_means), "parent fixture mismatch"
        deltas = [c - p for c, p in zip(group_means, pg)]
        lo, hi = bootstrap_ci(deltas)
        report["paired_vs_parent"] = {
            "parent_label": parent_report.get("label"),
            "parent_parameter_sha256": parent_report.get("parameter_sha256"),
            "parent_mean_group_lift": parent_report["group_level"]["mean_group_lift"],
            "child_mean_group_lift": report["group_level"]["mean_group_lift"],
            "mean_paired_delta": round(sum(deltas) / len(deltas), 4),
            "median_paired_delta": round(sorted(deltas)[len(deltas) // 2], 4),
            "positive_delta_groups": f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}",
            "paired_sign_flip_p": round(sign_flip_p(deltas), 5),
            "p_estimator": "plus-one MC: (count+1)/(draws+1); never literal 0",
            "bootstrap95CI_mean_delta": [round(lo, 4), round(hi, 4)],
            "effect_size_dz": round(dz(deltas), 3),
        }

    del model
    import gc
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)
    return report


if __name__ == "__main__":
    print(json.dumps({"schema": DIAGNOSTIC_VERSION,
                      "fixture_sha256": fixture_hash(),
                      "vocab_disjointness": vocabulary_disjointness(),
                      "n_groups": N_GROUPS_V4}, indent=2))
