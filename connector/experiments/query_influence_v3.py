"""Query Influence Matrix v3: REPLICATION fixture (DEVELOPMENT_REPLICATION_ONLY).

Role separation (P5):
  - anra-query-influence/v2 (query_influence.py) is DEVELOPMENT data: it
    informed the SFT5 hypothesis and stays untouched for trajectory work.
  - THIS module is a fresh frozen replication instrument: 40 independent
    fact groups, fresh entity vocabulary, fresh code prefixes, fresh seed,
    controlled fact counts (k = 2..4 deliberately), balanced target
    position, two formats. Its vocabulary is disjoint from the grouped-
    queryswap train AND heldout data, from the capability bank, and from
    QIM-v2. It must be frozen BEFORE any replication child is evaluated
    on it, and it must never influence training or checkpoint selection.

Statistical unit fix (P3): the independent unit is the FACT GROUP.
Candidate lifts within one group are correlated and are NEVER fed to the
primary test as if independent. Primary statistics are computed over
per-group mean query lifts:

    lift_cand(i)   = logP(v_i | own query_i) - mean_{j != i} logP(v_i | q_j)
    group_lift(g)  = mean_i lift_cand within g
    delta(g)       = group_lift_child(g) - group_lift_parent(g)

Primary inference: paired sign-flip permutation test over independent
group deltas (+ bootstrap CI, standardized d_z). Candidate-level numbers
are reported as DIAGNOSTIC_ONLY.
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
SEED_V3 = 20260907
PREFIXES_V3 = ("FRC", "LXM", "PVG", "TQH", "VZB")
ENTITIES_V3 = ("sconce", "pinnacle", "vault", "transept", "crypt",
               "clerestory", "triforium", "ambulatory", "apse", "narthex",
               "pulpitum", "rood")
N_GROUPS_V3 = 40
DIAGNOSTIC_VERSION = "anra-query-influence/v3-replication"
PERM_DRAWS = 20000
PERM_SEED = 7


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES_V3)}-{rng.randrange(100, 1000)}"


def build_groups() -> list[dict]:
    """Structured records in DISPLAY order; ordinal targets index display."""
    rng = random.Random(SEED_V3)
    groups = []
    for gi in range(N_GROUPS_V3):
        k = 2 + (gi % 3)                       # deliberate k in {2,3,4}
        ents = rng.sample(ENTITIES_V3, k)
        codes = [_code(rng) for _ in ents]
        records = [{"entity": e, "code": c,
                    "line": f"{e.capitalize()} bears tag {c}."}
                   for e, c in zip(ents, codes)]
        rng.shuffle(records)                   # display order is the only order
        fmt = "prose" if gi % 2 == 0 else "table"
        groups.append({"displayed_facts": records, "format": fmt})
    return groups


def fixture_hash() -> str:
    text = json.dumps(build_groups(), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vocabulary_disjointness() -> dict:
    """Prove the v3 vocab touches no training/heldout/diagnostic vocab."""
    ents = set(ENTITIES_V3)
    pres = set(PREFIXES_V3)
    overlaps: dict[str, list[str]] = {}
    # grouped-queryswap train+heldout rows
    for name in ("train", "heldout"):
        p = Path(f"data/grouped_queryswap/{name}.jsonl")
        if p.exists():
            blob = p.read_text(encoding="utf-8")
            o_e = {w for w in ents if w.capitalize() in blob}
            o_p = {w for w in pres if re.search(rf"\b{w}-\d{{3}}\b", blob)}
            overlaps[f"grouped_queryswap_{name}"] = sorted(o_e | o_p)
    # capability bank (train + dev)
    for name in ("train", "dev"):
        p = Path(f"data/capability_bank/{name}.jsonl")
        if p.exists():
            blob = p.read_text(encoding="utf-8")
            o_e = {w for w in ents if w.capitalize() in blob}
            o_p = {w for w in pres if re.search(rf"\b{w}-\d{{3}}\b", blob)}
            overlaps[f"capability_bank_{name}"] = sorted(o_e | o_p)
    # QIM-v2 fixture (import lazily; that module imports torch at top level)
    from connector.experiments.query_influence import (
        ENTITIES as E2, PREFIXES as P2)
    overlaps["qim_v2"] = sorted((ents & set(E2)) | (pres & set(P2)))
    flat = [x for v in overlaps.values() for x in v]
    return {"overlaps": overlaps, "disjoint": not flat}


# ---------------- model plumbing (identical numerics to v2) ---------------

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


def _query(rec: dict) -> str:
    return f"Return the tag of {rec['entity'].capitalize()}."


def _prompt(block: str, query: str) -> str:
    return f"{block}\n{query}\nAnswer:"


# ---------------- statistics -----------------------------------------------

def sign_flip_p(values: list[float]) -> float:
    """Paired one-sided sign-flip permutation p for mean > 0.

    Exact enumeration when n <= 20 (2^20 = 1,048,576); else seeded random
    flips. The INPUTS must already be independent units (per-group deltas),
    never correlated candidate rows.
    """
    n = len(values)
    obs = sum(values)
    rng = random.Random(PERM_SEED)
    total = 0
    count = 0
    if n <= 20:
        import itertools
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if sum(s * v for s, v in zip(signs, values)) >= obs:
                count += 1
    else:
        for _ in range(PERM_DRAWS):
            total += 1
            signs = [rng.choice((1, -1)) for _ in range(n)]
            if sum(s * v for s, v in zip(signs, values)) >= obs:
                count += 1
    return count / total


def bootstrap_ci(values: list[float], draws: int = 10000) -> tuple[float, float]:
    rng = random.Random(PERM_SEED + 1)
    n = len(values)
    stats = []
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)
    stats.sort()
    lo = stats[int(0.025 * draws)]
    hi = stats[min(draws - 1, int(0.975 * draws))]
    return lo, hi


def dz(values: list[float]) -> float:
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    return m / math.sqrt(max(var, 1e-12))


# ---------------- evaluation ------------------------------------------------

def run_model(label: str, checkpoint: str, *,
              parent_report: dict | None = None,
              device: str = "cuda") -> dict:
    """Evaluate one checkpoint. Pass parent_report (produced by an earlier
    call with include_per_group=True) to get the PRIMARY paired deltas."""
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
        if g["format"] == "table":
            block = ("item | tag\n"
                     + "\n".join(f"{r['entity'].capitalize()} | {r['code']}"
                                 for r in recs))
        else:
            block = "\n".join(r["line"] for r in recs)
        L = []
        for qi in range(len(recs)):
            prompt = _prompt(block, _query(recs[qi]))
            L.append([_completion_logprob(model, tok, prompt, f" {r['code']}.")
                      for r in recs])
        lifts = []
        for qi in range(len(recs)):
            others = [L[j][qi] for j in range(len(recs)) if j != qi]
            lifts.append(L[qi][qi] - sum(others) / len(others))
            ranks.append(1 + sum(1 for j in range(len(recs))
                                 if L[qi][j] > L[qi][qi]))
            greedy_ok.append(_strict(_greedy(model, tok, prompt), recs[qi]["code"]))
        cand_lifts.extend(lifts)
        group_means.append(sum(lifts) / len(lifts))

    report = {
        "schema": DIAGNOSTIC_VERSION,
        "role": "DEVELOPMENT_REPLICATION_ONLY",
        "label": label, "checkpoint": checkpoint,
        "global_step": identity.global_step,
        "parameter_sha256": getattr(identity, "parameter_sha256", None),
        "fixture_sha256": fixture_hash(),
        "vocab_disjointness": vocabulary_disjointness(),
        "primary_unit": "fact_group",
        "per_group_query_lift": [round(x, 4) for x in group_means],
        "group_level": {
            "n_groups": len(group_means),
            "mean_group_lift": round(sum(group_means) / len(group_means), 4),
            "median_group_lift": round(sorted(group_means)[len(group_means) // 2], 4),
            "groups_positive": f"{sum(1 for x in group_means if x > 0)}/{len(group_means)}",
            "sign_flip_p_mean_gt_0": round(sign_flip_p(group_means), 4),
        },
        # DIAGNOSTIC_ONLY: candidate rows within a group are correlated.
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
        assert len(pg) == len(group_means), \
            "parent evaluated on a different fixture"
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
            "bootstrap95CI_mean_delta": [round(lo, 4), round(hi, 4)],
            "effect_size_dz": round(dz(deltas), 3),
        }
    del model
    import gc
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = run_model(args.label, args.checkpoint, device=args.device)
    print(json.dumps({k: report[k] for k in (
        "label", "fixture_sha256", "vocab_disjointness", "group_level",
        "candidate_diagnostic_only")}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
