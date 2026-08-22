"""Query Influence Matrix: where does the query stop controlling the answer?

Instruments (DEV-only; no training consumes them):

  1. QIM       — controlled fact blocks (A/B/C → codes); only the query
                changes; for every (query, candidate) pair we score the
                FULL-SEQUENCE conditional log-probability of the complete
                answer string " CODE.". Reports diagonal margin, correct
                rank, rank-1 fraction, corresponding-gain on swaps, JS
                divergence across query-conditioned candidate distributions,
                greedy accuracy.
  2. EQ/OQ/XP  — same block, same answer type: ENTITY query vs ORDINAL
                ("fact 2") vs EXPLICIT POINTER ("Fact 2 is relevant…").
  3. RECENCY   — single-variable interventions on failing items: query moved
                near Answer, target fact moved near Answer, neutral relevance
                pointer, distractor removal, query duplication. Verifier
                flips under exactly-one-moved-variable conditions can become
                VerifiedInterventionExperience entries (the first real ones).
  4. RETENTION — protected-family strict accuracy on the bank dev split
                (retrospective PR5 evidence for SFT4).

Run per checkpoint:
  py -3 -m connector.experiments.query_influence --checkpoint <pt> \
      --label <name> --legacy --out output/qim_<name>.json
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


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


def build_groups() -> list[dict]:
    rng = random.Random(SEED)
    groups = []
    for g in range(N_GROUPS):
        ents = rng.sample(ENTITIES, 3)
        codes = [_code(rng) for _ in ents]
        lines = [f"{e.capitalize()} bears tag {c}." for e, c in zip(ents, codes)]
        rng.shuffle(lines)
        groups.append({"entities": ents, "codes": codes, "lines": lines,
                       "target_line": lines[[l for l in lines]
                                            .index(f"{ents[i].capitalize()} bears tag {codes[i]}.")]
                       if False else None})
        # store per-target its line for the fact-move intervention
        groups[-1]["line_of"] = {e: f"{e.capitalize()} bears tag {c}."
                                 for e, c in zip(ents, codes)}
    return groups


def _prompt_block(group: dict, query_line: str) -> str:
    return "\n".join(group["lines"]) + f"\n{query_line}\nAnswer:"


@torch.no_grad()
def _completion_logprob(model, tok, prompt: str, completion: str) -> float:
    """Full-sequence conditional logP(completion | prompt): teacher-forced,
    summed over completion tokens (codes are multi-token, so first-token
    scoring would be wrong)."""
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion)
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long,
                       device=next(model.parameters()).device)
    logits = model(ids)[0]  # [seq, vocab]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for pos in range(1 + len(p_ids), ids.shape[1]):
        total += float(logprobs[pos - 1, ids[0, pos]].item())
    return total


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


def _rank(value: float, others: list[float]) -> int:
    return 1 + sum(1 for o in others if o > value)


def _js(p: list[float], q: list[float]) -> float:
    def norm(v):
        s = sum(v)
        return [x / s for x in v]
    p, q = norm(p), norm(q)
    m = [(a + b) / 2 for a, b in zip(p, q)]
    def kl(a, b):
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)
    return 0.5 * (kl(p, m) + kl(q, m))


def run_model(label: str, checkpoint: str, *, legacy: bool, device="cuda") -> dict:
    from anra_core.checkpoint import load_core_checkpoint
    from anra_core.tokenizer import V4Tokenizer
    model, _, identity = load_core_checkpoint(checkpoint, legacy_unverified=True)
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()
    groups = build_groups()

    # ---------- 1. Query Influence Matrix
    margins, ranks, gains, js_all, greedy_ok = [], [], [], [], []
    for g in groups:
        L = {}
        for qi in range(3):
            q = f"Return the tag of {g['entities'][qi].capitalize()}."
            prompt = _prompt_block(g, q)
            L[qi] = [_completion_logprob(model, tok, prompt, f" {c}.")
                     for c in g["codes"]]
        for qi in range(3):
            margins.append(L[qi][qi] - max(L[qi][j] for j in range(3) if j != qi))
            ranks.append(_rank(L[qi][qi], L[qi]))
            greedy_ok.append(_strict(_greedy(model, tok,
                                             _prompt_block(g, f"Return the tag of {g['entities'][qi].capitalize()}.")),
                                     g["codes"][qi]))
        for a in range(3):
            for b in range(3):
                if a != b:
                    gains.append(L[b][b] > L[a][b])  # correct cand gains under its query
                    pa = [math.exp(x) for x in L[a]]
                    pb = [math.exp(x) for x in L[b]]
                    js_all.append(_js(pa, pb))
    qim = {
        "mean_diagonal_margin": round(sum(margins) / len(margins), 3),
        "median_diagonal_margin": round(sorted(margins)[len(margins)//2], 3),
        "correct_rank1_fraction": f"{sum(1 for r in ranks if r == 1)}/{len(ranks)}",
        "mean_correct_rank": round(sum(ranks) / len(ranks), 2),
        "corresponding_gain_on_swap": f"{sum(1 for x in gains if x)}/{len(gains)}",
        "mean_js_divergence_across_queries": round(sum(js_all) / len(js_all), 4),
        "greedy_corresponding_accuracy": f"{sum(greedy_ok)}/{len(greedy_ok)}",
        "n_groups": len(groups),
    }

    # ---------- 2. entity vs ordinal vs explicit pointer
    rng = random.Random(SEED + 1)
    cond_acc = {"entity": [], "ordinal": [], "pointer": []}
    for g in groups[:8]:
        target = rng.randrange(3)
        ent = g["entities"][target].capitalize()
        variants = {
            "entity": f"Return the tag of {ent}.",
            "ordinal": f"Return the tag from fact {target + 1}.",
            "pointer": f"Fact {target + 1} is the relevant fact. Return its tag.",
        }
        for name, q in variants.items():
            prompt = _prompt_block(g, q)
            cond_acc[name].append(_strict(_greedy(model, tok, prompt), g["codes"][target]))
    eqoq = {k: f"{sum(1 for x in v if x)}/{len(v)}" for k, v in cond_acc.items()}

    # ---------- 3. recency interventions on failing items
    failing = []
    for g in groups:
        for qi in range(3):
            q = f"Return the tag of {g['entities'][qi].capitalize()}."
            base_prompt = _prompt_block(g, q)
            out = _greedy(model, tok, base_prompt)
            if not _strict(out, g["codes"][qi]):
                failing.append({"group": g, "qi": qi, "q": q, "base_out": out,
                                "gold": g["codes"][qi]})
    failing = failing[:12]
    rescues = {k: [] for k in ("query_near_answer", "fact_near_answer",
                               "relevance_pointer", "distractor_removal",
                               "query_duplication")}
    for f in failing:
        g, qi, q, gold = f["group"], f["qi"], f["q"], f["gold"]
        lines = list(g["lines"])
        target_line = g["line_of"][g["entities"][qi]]
        variants = {
            "query_near_answer": "\n".join(lines) + f"\nAnswer:\n{q}\nAnswer:",
            "fact_near_answer": "\n".join(l for l in lines if l != target_line)
                                + f"\n{q}\n{target_line}\nAnswer:",
            "relevance_pointer": "\n".join(lines)
                                 + f"\n[RELEVANT FACT]\n{target_line}\n{q}\nAnswer:",
            "distractor_removal": target_line + f"\n{q}\nAnswer:",
            "query_duplication": "\n".join(lines) + f"\n{q}\n{q}\nAnswer:",
        }
        for name, prompt in variants.items():
            rescues[name].append(_strict(_greedy(model, tok, prompt), gold))
    interventions = {k: {"rescued": f"{sum(1 for x in v if x)}/{len(v)}",
                         "n_failing": len(v)} for k, v in rescues.items()}

    # ---------- 4. PR5 retrospective: protected retention on bank dev
    bank_dev = [json.loads(l) for l in
                Path("data/capability_bank/dev.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    prot = {}
    for fam in ("single_fact", "tool_result", "copy", "protocol_transfer"):
        rows = [b for b in bank_dev if b["family"] == fam][:20]
        if not rows:
            continue
        hits = 0
        for b in rows:
            out = _greedy(model, tok, b["prompt"], max_new_tokens=10)
            gold = b.get("gold") or b.get("answer", "")
            cands = CODE_RE.findall(out)
            ok = (len(cands) == 1 and cands[0] == gold) if CODE_RE.fullmatch(gold or "") \
                else bool(re.search(rf"(?<!\w){re.escape(gold.lower())}(?!\w)",
                                    re.sub(r"[^0-9a-z ]", " ", out.lower())))
            hits += bool(ok)
        prot[fam] = f"{hits}/{len(rows)}"

    report = {
        "schema": "anra-query-influence/v1", "label": label,
        "checkpoint": checkpoint,
        "global_step": identity.global_step,
        "parameter_sha256": getattr(identity, "parameter_sha256", None),
        "query_influence_matrix": qim,
        "entity_vs_ordinal_vs_pointer": eqoq,
        "recency_interventions": interventions,
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
                       "entity_vs_ordinal_vs_pointer", "recency_interventions",
                       "protected_retention_bank_dev")}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
