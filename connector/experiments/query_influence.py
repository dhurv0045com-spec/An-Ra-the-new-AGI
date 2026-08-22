"""Query Influence Matrix v2: corrected controls, clean mechanism evidence.

v2 fixes over v1 (each invalidated a previous conclusion):
  - ORDINAL INDEXING: displayed facts are structured records in display
    order; ordinal/pointer targets index the DISPLAYED fact, and entity /
    ordinal / pointer provably designate the same gold.
  - RELOCATION PURITY: prompts are built from components (facts block,
    query, single Answer marker); relocation variants permute components
    only — exactly one "Answer:" in every prompt, same line multiset.
  - MARK vs REPEAT separated: MARK_ONLY annotates the fact in place;
    REPEAT_ONLY duplicates the fact near the answer; MARK_AND_REPEAT is
    reported but never treated as single-variable.
  - CANDIDATE-NORMALIZED QUERY LIFT: lift_i = logP(value_i|own query) -
    mean_j!=i logP(value_i|query_j) — the query's effect on ITS value,
    cleansed of candidate priors. With a paired permutation p-value.
  - Stable JS (max-shift normalization; no exp underflow).
  - Fixture hash: identical items across models, asserted in receipts.

Probe status: the earlier residual probe is EXPLORATORY_ONLY (n=6) and is
not part of any classification.
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
SEED = 20260902
PREFIXES = ("HGR", "JPL", "KSN", "MBT", "NWD")
ENTITIES = ("tarn", "crease", "hollow", "spindle", "gable", "wicket",
            "louver", "quoinx", "crib", "drift")
N_GROUPS = 10
DIAGNOSTIC_VERSION = "anra-query-influence/v2"


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


def build_groups() -> list[dict]:
    """Structured fact records in DISPLAY order; ordinal/pointer index this
    order; the gold is always the designated record's code."""
    rng = random.Random(SEED)
    groups = []
    for _ in range(N_GROUPS):
        ents = rng.sample(ENTITIES, 3)
        codes = [_code(rng) for _ in ents]
        records = [{"entity": e, "code": c, "line": f"{e.capitalize()} bears tag {c}."}
                   for e, c in zip(ents, codes)]
        rng.shuffle(records)  # display order is the ONLY order that matters
        groups.append({"displayed_facts": records})
    return groups


def fixture_hash() -> str:
    text = json.dumps(build_groups(), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _query(record: dict) -> str:
    return f"Return the tag of {record['entity'].capitalize()}."


def _prompt(facts_block: str, query: str) -> str:
    return f"{facts_block}\n{query}\nAnswer:"


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
def _greedy(model, tok, prompt: str, max_new_tokens: int = 12) -> str:
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


def _stable_js(logp: list[float], logq: list[float]) -> float:
    def norm(v):
        m = max(v)
        w = [math.exp(x - m) for x in v]
        s = sum(w)
        return [x / s for x in w]
    p, q = norm(logp), norm(logq)
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a, b):
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)
    return 0.5 * (kl(p, m) + kl(q, m))


def _permutation_p(values: list[float]) -> float:
    """Paired permutation test for mean(values) > 0 (sign flips, exact for
    small n via full enumeration when n <= 12, else 2000 random flips)."""
    n = len(values)
    obs = sum(values)
    rng = random.Random(7)

    def flips():
        if n <= 12:
            import itertools
            for signs in itertools.product((1, -1), repeat=n):
                yield signs
        else:
            for _ in range(2000):
                yield tuple(rng.choice((1, -1)) for _ in range(n))

    count = total = 0
    for signs in flips():
        total += 1
        if sum(s * v for s, v in zip(signs, values)) >= obs:
            count += 1
    return count / total


def run_model(label: str, checkpoint: str, *, legacy: bool, device="cuda") -> dict:
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    model, _, identity = load_core_checkpoint(checkpoint, legacy_unverified=True)
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()
    groups = build_groups()

    # ---------- 1. QIM + candidate-normalized query lift
    margins, ranks, greedy_ok, js_all = [], [], [], []
    lifts: list[float] = []
    per_group_adv: list[float] = []
    for g in groups:
        facts = [r["line"] for r in g["displayed_facts"]]
        block = "\n".join(facts)
        L = {}
        for qi in range(3):
            prompt = _prompt(block, _query(g["displayed_facts"][qi]))
            L[qi] = [_completion_logprob(model, tok, prompt, f" {r['code']}.")
                     for r in g["displayed_facts"]]
        for qi in range(3):
            margins.append(L[qi][qi] - max(L[qi][j] for j in range(3) if j != qi))
            ranks.append(1 + sum(1 for j in range(3) if L[qi][j] > L[qi][qi]))
            greedy_ok.append(_strict(_greedy(model, tok,
                                             _prompt(block, _query(g["displayed_facts"][qi]))),
                                     g["displayed_facts"][qi]["code"]))
        for i in range(3):
            others = [L[j][i] for j in range(3) if j != i]
            lifts.append(L[i][i] - sum(others) / len(others))
        adv = sum(L[i][i] - sum(L[j][i] for j in range(3) if j != i) / 2
                  for i in range(3)) / 3
        per_group_adv.append(adv)
        for a in range(3):
            for b in range(a + 1, 3):
                js_all.append(_stable_js(L[a], L[b]))
    qim = {
        "raw_mean_diagonal_margin": round(sum(margins) / len(margins), 3),
        "mean_query_lift": round(sum(lifts) / len(lifts), 4),
        "median_query_lift": round(sorted(lifts)[len(lifts) // 2], 4),
        "query_lift_positive_rate": f"{sum(1 for x in lifts if x > 0)}/{len(lifts)}",
        "query_lift_permutation_p": round(_permutation_p(lifts), 4),
        "query_lift_effect_size": round(
            (sum(lifts) / len(lifts)) /
            (max(math.sqrt(sum((x - sum(lifts) / len(lifts)) ** 2 for x in lifts) / len(lifts)), 1e-9)), 3),
        "correct_rank1_fraction": f"{sum(1 for r in ranks if r == 1)}/{len(ranks)}",
        "mean_correct_rank": round(sum(ranks) / len(ranks), 2),
        "mean_js_across_queries": round(sum(js_all) / len(js_all), 5),
        "fraction_groups_positive_query_effect": f"{sum(1 for a in per_group_adv if a > 0)}/{len(per_group_adv)}",
        "greedy_corresponding_accuracy": f"{sum(greedy_ok)}/{len(greedy_ok)}",
        "n_groups": len(groups),
    }

    # ---------- 2. corrected entity vs ordinal vs pointer (same target)
    rng = random.Random(SEED + 1)
    cond = {"entity": [], "ordinal": [], "pointer": []}
    for g in groups[:8]:
        t = rng.randrange(3)
        rec = g["displayed_facts"][t]
        block = "\n".join(r["line"] for r in g["displayed_facts"])
        variants = {
            "entity": _query(rec),
            "ordinal": f"Return the tag from fact {t + 1}.",
            "pointer": f"Fact {t + 1} is relevant. Return its tag.",
        }
        for name, q in variants.items():
            cond[name].append(_strict(_greedy(model, tok, _prompt(block, q)), rec["code"]))
    eqoq = {k: f"{sum(1 for x in v if x)}/{len(v)}" for k, v in cond.items()}

    # ---------- 3. corrected interventions on failing items
    failing = []
    for g in groups:
        block = "\n".join(r["line"] for r in g["displayed_facts"])
        for qi, rec in enumerate(g["displayed_facts"]):
            q = _query(rec)
            out = _greedy(model, tok, _prompt(block, q))
            if not _strict(out, rec["code"]):
                failing.append({"g": g, "qi": qi, "q": q, "gold": rec["code"],
                                "block": block})
    failing = failing[:12]
    rescues: dict[str, list[bool]] = {k: [] for k in
                                      ("query_relocation", "fact_relocation",
                                       "mark_only", "repeat_only",
                                       "mark_and_repeat", "single_fact_control",
                                       "query_duplication")}
    for f in failing:
        g, rec = f["g"], f["g"]["displayed_facts"][f["qi"]]
        facts = [r["line"] for r in g["displayed_facts"]]
        q, gold = f["q"], f["gold"]
        others = [l for l in facts if l != rec["line"]]
        variants = {
            # BASE layout is facts+query+Answer; relocation = query moved to
            # the FRONT (far from the answer). One Answer marker everywhere;
            # identical line multisets.
            "query_relocation": f"{q}\n" + "\n".join(facts) + "\nAnswer:",
            # fact relocation: target fact moved to the position immediately
            # before the query/answer; same multiset, one marker.
            "fact_relocation": "\n".join(others) + f"\n{q}\n{rec['line']}\nAnswer:",
            # MARK_ONLY: annotation in place, no duplication.
            "mark_only": "\n".join(
                f"[RELEVANT] {l}" if l == rec["line"] else l for l in facts)
                + f"\n{q}\nAnswer:",
            # REPEAT_ONLY: duplication near the answer, no annotation.
            "repeat_only": "\n".join(facts) + f"\n{q}\n{rec['line']}\nAnswer:",
            "mark_and_repeat": "\n".join(
                f"[RELEVANT] {l}" if l == rec["line"] else l for l in facts)
                + f"\n{q}\n[RELEVANT] {rec['line']}\nAnswer:",
            "single_fact_control": f"{rec['line']}\n{q}\nAnswer:",
            "query_duplication": "\n".join(facts) + f"\n{q}\n{q}\nAnswer:",
        }
        for name, prompt in variants.items():
            rescues[name].append(_strict(_greedy(model, tok, prompt), gold))
    interventions = {k: {"rescued": f"{sum(1 for x in v if x)}/{len(v)}",
                         "n_failing": len(v)} for k, v in rescues.items()}

    # ---------- 4. protected retention on the bank dev split
    bank_dev = [json.loads(l) for l in
                Path("data/capability_bank/dev.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    prot = {}
    for fam in ("single_fact", "tool_result", "copy", "protocol_transfer"):
        rows = [b for b in bank_dev if b["family"] == fam][:20]
        hits = 0
        for b in rows:
            out = _greedy(model, tok, b["prompt"], max_new_tokens=10)
            gold = b.get("gold") or b.get("answer", "")
            cands = CODE_RE.findall(out)
            ok = ((len(cands) == 1 and cands[0] == gold)
                  if CODE_RE.fullmatch(gold or "") else
                  bool(re.search(rf"(?<!\w){re.escape(gold.lower())}(?!\w)",
                                 re.sub(r"[^0-9a-z ]", " ", out.lower()))))
            hits += bool(ok)
        prot[fam] = f"{hits}/{len(rows)}"

    report = {
        "schema": DIAGNOSTIC_VERSION, "label": label, "checkpoint": checkpoint,
        "global_step": identity.global_step,
        "parameter_sha256": getattr(identity, "parameter_sha256", None),
        "fixture_sha256": fixture_hash(),
        "query_influence_matrix": qim,
        "entity_vs_ordinal_vs_pointer_corrected": eqoq,
        "interventions_corrected": interventions,
        "protected_retention_bank_dev": prot,
        "n_failing_items": len(failing),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    del model
    import gc
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = run_model(args.label, args.checkpoint,
                       legacy=args.legacy, device=args.device)
    print(json.dumps({k: report[k] for k in
                      ("label", "query_influence_matrix",
                       "entity_vs_ordinal_vs_pointer_corrected",
                       "interventions_corrected",
                       "protected_retention_bank_dev")}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
