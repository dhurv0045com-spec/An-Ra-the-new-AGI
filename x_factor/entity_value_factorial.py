"""Entity x Value factorial (Triquetra Experiment 1).

WHY DOES QUERY-MATCHED FACT DUPLICATION OUTPERFORM DISTRACTOR DUPLICATION?

Separates, for the SAME baseline failure and SAME full original context:
  ENTITY PRESENT? VALUE PRESENT? PAIR CORRECT?

INSERT family (fixed location: immediately before final query/Answer;
original facts untouched, order unchanged, query unchanged):
  C0 NEUTRAL            Memo: use only the listed refs.
  C1 ENTITY_ONLY        Recall: {Entity}.
  C2 VALUE_ONLY         Recall: {Code}.
  C3 CORRECT_PAIR       Recall: {Entity} = {Code}.
  C4 WRONG_ENT_OK_VAL   Recall: {DistrEntity} = {Code}.
  C5 OK_ENT_WRONG_VAL   Recall: {Entity} = {DistrCode}.
  C6 WRONG_VALID_PAIR   Recall: {DistrEntity} = {DistrCode}. (real other fact)
  C7 FULL_TARGET_FACT   {Entity} keeps ref {Code}.
  C8 FULL_DISTR_FACT    {DistrEntity} keeps ref {DistrCode}. (same D* as C6)

MARK family (in-place, original fact intact, code never deleted):
  C9 ENTITY_MARKED      [[{Entity}]] keeps ref {Code}.
  C10 VALUE_MARKED      {Entity} keeps ref [[{Code}]].
  C11 WHOLE_MARKED      >>> {Entity} keeps ref {Code}.

Distractor D* is chosen deterministically per task via SHA256 seed
(SEED, task_id, "distractor-choice"); its entity/code are reused for
C4/C5/C6/C8 so controls stay matched.

Mechanistic read:
  C1-C0 strong            -> query/entity addressing matters
  C2-C0 ~= full effect    -> value-token recency/readout explains most
  C3/C7 >> C1,C2          -> correct ENTITY<->VALUE relation matters
  C4 ~= C3                -> value recency, not relational binding
  C5 hurts                -> entity routes value selection
  C6 small                -> query-conditioned pair relevance matters

Primary unit: TASK. Paired stats vs C0 neutral control (and vs baseline).
Outcomes per condition: strict correctness, gold LP, binding margin
(gold - mean(other ORIGINAL codes)), gold rank among original codes.
Reference pool is fixed to original fact codes so margins stay comparable.

Provenance: checkpoint file SHA256 + ordered-parameter SHA256 are computed
from bytes (never metadata). No null SHAs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from provenance import (  # noqa: E402
    git_head,
    param_sha256_from_state_dict,
    sha256_bytes,
    sha256_file,
    sha256_json,
)

from anra_core.config import CoreConfig, CANONICAL_CONFIG  # noqa: E402
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/anra-v4-current-full-resume.pt"
LEGACY_CHECKPOINT = "checkpoints/anra-v4-20k-sft3-accumulate.pt"
SEED = 41414
N_TASKS = 120
MAX_NEW = 12
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")

CONDITION_IDS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
CONDITION_DESCR = {
    "C0": "NEUTRAL control: no relevant entity/value repeated",
    "C1": "ENTITY_ONLY: queried entity, no code",
    "C2": "VALUE_ONLY: correct code, no entity",
    "C3": "CORRECT_PAIR: entity = code",
    "C4": "WRONG_ENTITY_CORRECT_VALUE: distractor entity = gold code",
    "C5": "CORRECT_ENTITY_WRONG_VALUE: target entity = distractor code",
    "C6": "WRONG_VALID_PAIR: distractor entity = distractor code (real other fact)",
    "C7": "FULL_TARGET_FACT: natural surface duplicate",
    "C8": "FULL_DISTRACTOR_FACT: matched distractor duplicate",
    "C9": "ENTITY_MARKED intact: [[Entity]] keeps ref CODE.",
    "C10": "VALUE_MARKED intact: Entity keeps ref [[CODE]].",
    "C11": "WHOLE_FACT_MARKED: >>> Entity keeps ref CODE.",
}


def _strict(out: str, gold: str) -> int:
    c = CODE_RE.findall(out)
    return int(len(c) == 1 and c[0] == gold)


def _stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _tasks(seed: int, n: int):
    """Same generator as binding_factorial (DEV comparability)."""
    OBJECTS = ("aviary", "barbican", "cloister", "dolmen", "gaol", "impound",
               "lancet", "nave", "oratory", "portcullis")
    PREFIXES = ("AVR", "BQW", "CTY", "DZN", "EKH", "FMP", "GQS", "HUB")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = 2 + (i % 4)
        objs = rng.sample(OBJECTS, k)
        codes = [f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}" for _ in objs]
        block = "\n".join(f"{o.capitalize()} keeps ref {c}." for o, c in zip(objs, codes))
        tgt = i % k
        q = f"Return ONLY the ref of {objs[tgt].capitalize()}."
        out.append({"id": f"ev-{i:03d}", "block": block, "query": q,
                    "prompt": f"{block}\n{q}\nAnswer:", "gold": codes[tgt],
                    "facts": list(zip(objs, codes)), "target": objs[tgt],
                    "target_code": codes[tgt], "n_facts": k, "target_pos": tgt})
    return out


def _choose_distractor(task, seed: int):
    """Deterministic matched distractor fact D* (entity, code)."""
    cands = [(o, c) for o, c in task["facts"] if o != task["target"]]
    if not cands:
        return task["target"], task["target_code"]
    rng = random.Random(_stable_seed(seed, task["id"], "distractor-choice"))
    return rng.choice(cands)


def build_conditions(task, seed: int):
    ent = task["target"].capitalize()
    code = task["target_code"]
    d_ent, d_code = _choose_distractor(task, seed)
    d_ent_c = d_ent.capitalize()
    inserts = {
        "C0": "Memo: use only the listed refs.",
        "C1": f"Recall: {ent}.",
        "C2": f"Recall: {code}.",
        "C3": f"Recall: {ent} = {code}.",
        "C4": f"Recall: {d_ent_c} = {code}.",
        "C5": f"Recall: {ent} = {d_code}.",
        "C6": f"Recall: {d_ent_c} = {d_code}.",
        "C7": f"{ent} keeps ref {code}.",
        "C8": f"{d_ent_c} keeps ref {d_code}.",
    }
    prompts: dict[str, str] = {}
    for cid, ins in inserts.items():
        prompts[cid] = f"{task['block']}\n{ins}\n{task['query']}\nAnswer:"
    # Mark family: in-place edits, code never deleted.
    lines = task["block"].splitlines()
    tgt_line = next((l for l in lines if task["target"].capitalize() in l), lines[0])

    def _swap(old: str, new: str):
        return "\n".join(new if l == old else l for l in lines)

    ent_cap = task["target"].capitalize()
    c9_line = tgt_line.replace(ent_cap, f"[[{ent_cap}]]", 1)
    c10_line = tgt_line.replace(code, f"[[{code}]]", 1)
    c11_line = f">>> {tgt_line}"
    prompts["C9"] = f"{_swap(tgt_line, c9_line)}\n{task['query']}\nAnswer:"
    prompts["C10"] = f"{_swap(tgt_line, c10_line)}\n{task['query']}\nAnswer:"
    prompts["C11"] = f"{_swap(tgt_line, c11_line)}\n{task['query']}\nAnswer:"
    meta = {"distractor_entity": d_ent, "distractor_code": d_code, "inserts": inserts}
    return prompts, meta


@torch.no_grad()
def _greedy(model, tok, prompt, device, max_new=MAX_NEW) -> str:
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    out = []
    cur = list(ids)
    for _ in range(max_new):
        logits = model(torch.tensor([cur], dtype=torch.long, device=device))[:, -1, :]
        nxt = int(logits.argmax(dim=-1))
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        cur.append(nxt)
    return tok.decode(out)


@torch.no_grad()
def _code_lps(model, tok, prompt, codes, device) -> dict[str, float]:
    """Single-forward-per-code logprob of ' CODE.' continuation."""
    lps: dict[str, float] = {}
    p_ids = tok.encode(prompt)
    for c in codes:
        c_ids = tok.encode(f" {c}.")
        ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]], dtype=torch.long, device=device)
        lp = torch.log_softmax(model(ids)[0].float(), -1)
        lps[c] = sum(float(lp[pos - 1, ids[0, pos]]) for pos in range(1 + len(p_ids), ids.shape[1]))
    return lps


def _margin_rank(lps: dict[str, float], gold: str):
    others = [v for k, v in lps.items() if k != gold]
    margin = lps[gold] - (sum(others) / len(others) if others else 0.0)
    rank = 1 + sum(1 for v in others if v > lps[gold])
    return margin, rank


# ---------------- statistics ----------------

def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p-value on discordant cells."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _boot_ci(vals, n_boot=10000, seed=7707):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return [0.0, 0.0]
    means = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(n_boot)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def paired_binary(a_treat: list[int], a_ctrl: list[int], seed: int):
    n = len(a_treat)
    both_pass = sum(1 for x, y in zip(a_treat, a_ctrl) if x == 1 and y == 1)
    treat_only = sum(1 for x, y in zip(a_treat, a_ctrl) if x == 1 and y == 0)
    ctrl_only = sum(1 for x, y in zip(a_treat, a_ctrl) if x == 0 and y == 1)
    both_fail = n - both_pass - treat_only - ctrl_only
    eff = (treat_only - ctrl_only) / n if n else 0.0
    diffs = [x - y for x, y in zip(a_treat, a_ctrl)]
    return {
        "n": n,
        "both_pass": both_pass,
        "treatment_only": treat_only,
        "control_only": ctrl_only,
        "both_fail": both_fail,
        "treat_rate": round(sum(a_treat) / n, 4) if n else 0.0,
        "ctrl_rate": round(sum(a_ctrl) / n, 4) if n else 0.0,
        "paired_effect": round(eff, 4),
        "mcnemar_exact_p": round(_mcnemar_exact(treat_only, ctrl_only), 4),
        "effect_ci95": _boot_ci(diffs, seed=seed),
    }


def paired_continuous(deltas: list[float], seed: int):
    arr = np.asarray(deltas, dtype=float)
    pos = int((arr > 0).sum())
    neg = int((arr < 0).sum())
    n_nz = pos + neg
    sign_p = round(min(1.0, 2.0 * sum(math.comb(n_nz, i) for i in range(min(pos, neg) + 1)) / 2**n_nz), 4) if n_nz else 1.0
    return {
        "n": len(deltas),
        "mean": round(float(arr.mean()) if len(arr) else 0.0, 4),
        "median": round(float(np.median(arr)) if len(arr) else 0.0, 4),
        "frac_positive": round(pos / len(arr), 4) if len(arr) else 0.0,
        "mean_ci95": _boot_ci(deltas, seed=seed),
        "sign_p": sign_p,
    }


def run(checkpoint: str, seed: int, n_tasks: int, device: str):
    torch.manual_seed(seed)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = CoreConfig(**{k: payload["model_config"][k] for k in CANONICAL_CONFIG.__dataclass_fields__})
    model = AnRaCore(cfg)
    model.load_state_dict({k: v for k, v in payload["model_state_dict"].items() if k != "lm_head.weight"},
                          strict=False)
    model.lm_head.weight = model.token_embedding_table.weight
    model = model.to(device).eval()
    tok = V4Tokenizer.load_canonical()

    # ---- provenance (bytes, never metadata) ----
    param_sha = param_sha256_from_state_dict(model.state_dict())
    ckpt_sha = sha256_file(str(ckpt_path))
    cfg_sha = sha256_json(payload["model_config"])
    try:
        tok_ident = tok.identity() if hasattr(tok, "identity") else {"vocab": "canonical-v4-32k"}
    except Exception:
        tok_ident = {"vocab": "canonical-v4-32k"}
    tok_sha = sha256_json(tok_ident)
    this_file = Path(__file__).resolve()
    exp_sha = sha256_file(str(this_file))
    prov_file = this_file.parent / "provenance.py"
    prov_sha = sha256_file(str(prov_file)) if prov_file.exists() else "missing"
    gen_sha = exp_sha  # generator lives in this file (_tasks)
    cond_reg = {cid: CONDITION_DESCR[cid] for cid in CONDITION_IDS}
    cond_sha = sha256_json(cond_reg)
    runtime_commit = git_head(Path(__file__).resolve().parents[1])
    try:
        import anra_core.config as _cfgmod  # noqa

        runtime_src = sha256_file(str(Path(_cfgmod.__file__)))
    except Exception:
        runtime_src = "unknown"

    tasks = _tasks(seed, n_tasks)
    print(f"[baseline] running {len(tasks)} tasks on {device} ckpt={checkpoint}", flush=True)
    base_correct: dict[str, int] = {}
    fail_tasks = []
    for t in tasks:
        out = _greedy(model, tok, t["prompt"], device)
        ok = _strict(out, t["gold"])
        base_correct[t["id"]] = ok
        if not ok:
            fail_tasks.append(t)
    n_pass = sum(base_correct.values())
    print(f"[baseline] {n_pass}/{len(tasks)} pass, {len(fail_tasks)} failures", flush=True)

    # insert token lengths (approximate matching audit)
    tok_lens = {}
    probe, _ = build_conditions(tasks[0], seed), None
    for cid, ins in probe[1]["inserts"].items():
        tok_lens[cid] = len(tok.encode(ins))
    tok_lens["C7"] = len(tok.encode(probe[1]["inserts"]["C7"]))
    print(f"[inserts] token lens: {tok_lens}", flush=True)

    per_task: dict = {}
    for idx, t in enumerate(fail_tasks):
        prompts, meta = build_conditions(t, seed)
        codes = [c for _, c in t["facts"]]
        row: dict = {"gold": t["gold"], "n_facts": t["n_facts"],
                     "target_pos": t["target_pos"],
                     "distractor_entity": meta["distractor_entity"],
                     "distractor_code": meta["distractor_code"],
                     "baseline_correct": 0, "conds": {}}
        # baseline metrics for this failure (correct=0 by construction)
        lps0 = _code_lps(model, tok, t["prompt"], codes, device)
        m0, r0 = _margin_rank(lps0, t["gold"])
        row["baseline_gold_lp"] = round(lps0[t["gold"]], 3)
        row["baseline_margin"] = round(m0, 3)
        row["baseline_rank"] = r0
        for cid in CONDITION_IDS:
            p = prompts[cid]
            out = _greedy(model, tok, p, device)
            ok = _strict(out, t["gold"])
            lps = _code_lps(model, tok, p, codes, device)
            m, r = _margin_rank(lps, t["gold"])
            row["conds"][cid] = {"correct": ok,
                                 "output": out[:200],
                                 "gold_lp": round(lps[t["gold"]], 3),
                                 "margin": round(m, 3),
                                 "rank": r,
                                 "gold_lp_delta_vs_base": round(lps[t["gold"]] - lps0[t["gold"]], 3),
                                 "margin_delta_vs_base": round(m - m0, 3)}
        per_task[t["id"]] = row
        if (idx + 1) % 10 == 0:
            print(f"  ... {idx + 1}/{len(fail_tasks)} failures done", flush=True)

    # ---- paired analysis on failures ----
    def col(cid):
        return [per_task[tid]["conds"][cid]["correct"] for tid in sorted(per_task)]

    def lp_delta(cid, ref):
        if ref == "BASE":
            return [per_task[tid]["conds"][cid]["gold_lp_delta_vs_base"] for tid in sorted(per_task)]
        return [per_task[tid]["conds"][cid]["gold_lp"] - per_task[tid]["conds"][ref]["gold_lp"]
                for tid in sorted(per_task)]

    def mg_delta(cid, ref):
        if ref == "BASE":
            return [per_task[tid]["conds"][cid]["margin_delta_vs_base"] for tid in sorted(per_task)]
        return [per_task[tid]["conds"][cid]["margin"] - per_task[tid]["conds"][ref]["margin"]
                for tid in sorted(per_task)]

    c0 = col("C0")
    contrasts = [
        ("C1_vs_C0", "C1", "C0"), ("C2_vs_C0", "C2", "C0"), ("C3_vs_C0", "C3", "C0"),
        ("C4_vs_C0", "C4", "C0"), ("C5_vs_C0", "C5", "C0"), ("C6_vs_C0", "C6", "C0"),
        ("C7_vs_C0", "C7", "C0"), ("C8_vs_C0", "C8", "C0"),
        ("C3_vs_C1", "C3", "C1"), ("C3_vs_C2", "C3", "C2"),
        ("C3_vs_C4", "C3", "C4"), ("C3_vs_C5", "C3", "C5"),
        ("C7_vs_C8", "C7", "C8"), ("C7_vs_C3", "C7", "C3"), ("C8_vs_C6", "C8", "C6"),
        ("C9_vs_C0", "C9", "C0"), ("C10_vs_C0", "C10", "C0"), ("C11_vs_C0", "C11", "C0"),
    ]
    binary = {}
    for name, a, b in contrasts:
        binary[name] = paired_binary(col(a), col(b), seed=_stable_seed(seed, name) % (2**31))
    continuous_lp = {f"{n}_goldLP": paired_continuous(lp_delta(a, b), seed=9100 + i)
                     for i, (n, a, b) in enumerate(contrasts)}
    continuous_mg = {f"{n}_margin": paired_continuous(mg_delta(a, b), seed=4200 + i)
                     for i, (n, a, b) in enumerate(contrasts)}

    cond_rates = {cid: round(sum(col(cid)) / max(len(c0), 1), 4) for cid in CONDITION_IDS}

    receipt = {
        "schema": "anra-entity-value-factorial/v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "DEV (same-generator development; NOT fresh)",
        "provenance": {
            "checkpoint": checkpoint,
            "checkpoint_sha256": ckpt_sha,
            "parameter_sha256": param_sha,
            "config_sha256": cfg_sha,
            "tokenizer_identity": tok_ident,
            "tokenizer_sha256": tok_sha,
            "runtime_commit": runtime_commit,
            "runtime_source_sha256": runtime_src,
            "generator_sha256": gen_sha,
            "experiment_source_sha256": exp_sha,
            "analysis_source_sha256": exp_sha,
            "provenance_source_sha256": prov_sha,
            "condition_registry_sha256": cond_sha,
            "legacy_accumulate_checkpoint_missing": LEGACY_CHECKPOINT,
            "lineage_note": ("DEV run on locally available pretraining checkpoint; "
                             "legacy SFT accumulation child file absent. "
                             "Effect sizes are NOT directly comparable to +15.22pp history."),
        },
        "generator": {"name": "binding-compatible selective binding", "seed": seed, "n_tasks": n_tasks},
        "conditions": {cid: {"description": CONDITION_DESCR[cid],
                             "family": "INSERT" if cid in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8") else "MARK"}
                       for cid in CONDITION_IDS},
        "condition_inserts_probe_task0": probe[1]["inserts"],
        "insert_token_lens_probe": tok_lens,
        "baseline": {"n_tasks": n_tasks, "n_pass": n_pass, "n_failures": len(fail_tasks),
                     "pass_ids": sorted([tid for tid, v in base_correct.items() if v == 1])},
        "condition_repair_rates_on_failures": cond_rates,
        "paired_binary": binary,
        "paired_gold_lp": continuous_lp,
        "paired_margin": continuous_mg,
        "per_task": per_task,
    }
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N_TASKS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="output/entity_value_factorial_dev.json")
    args = ap.parse_args()
    ckpt = args.checkpoint
    if not Path(ckpt).exists() and Path(LEGACY_CHECKPOINT).exists():
        ckpt = LEGACY_CHECKPOINT
    receipt = run(ckpt, args.seed, args.n, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": receipt["baseline"],
                      "rates": receipt["condition_repair_rates_on_failures"],
                      "C2_vs_C0": receipt["paired_binary"]["C2_vs_C0"],
                      "C1_vs_C0": receipt["paired_binary"]["C1_vs_C0"],
                      "C7_vs_C8": receipt["paired_binary"]["C7_vs_C8"]}, indent=2))
    print(f"wrote {out}")
    del receipt
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
