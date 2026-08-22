"""Entity-matching diagnostic: WHERE does query-conditioned fact binding fail?

Development-only probe (never sealed, never imported by training). OOD-4
showed paired-CF selective binding at 0/26 on both models while retention
replicated — the question changed from "can it select?" to "which primitive
is missing?". This battery decomposes the operation into a measurable
ladder:

  P0 query entity recognition   ("Which entity is being queried?" -> copy it)
  P1 fact entity recognition    (distractor-resilient naming)
  P2 fact value extraction      ("Return ANY one supplied code.")
  P3 query<->fact entity match  (exact / case / alias / distractor conditions)
  P4 matched fact -> value      (targeted query, strict single code)
  P5 counterfactual dependence  (byte-pure twins, SHA-derived replacements)

Crossed with: fact counts 2-5 (balanced), target position first/middle/last
(balanced), two formats. Plus:

  query_dependence   same facts, query swapped across entities: what
                     fraction of swaps change the output at all (and to the
                     corresponding value)?
  decomposition      every targeted item classified observationally:
                     INVALID_OUTPUT / NON_FACT_VALUE / WRONG_FACT_VALUE /
                     TARGET_VALUE_BASE_ONLY / PAIRED_CORRECT
  intervention arm   the FIRST real cognitive-credit battery: on failing
                     items, one variable at a time — explicit entity-match
                     hint, direct value supply, format normalization, decode
                     search. Single-variable flips may become legitimate
                     VerifiedInterventionExperience entries (kept in the
                     report; the causal bank stays empty unless they
                     qualify).

Run:
  py -3 -m connector.experiments.entity_matching_diagnostic \
      --checkpoint <pt> --label anchor --out output/emd_anchor.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
FREEZE_SEED = 20260901

# Fresh vocabularies (disjoint from every suite/bank/generator so far).
PREFIXES = ("WKC", "XDN", "YRM", "ZTS", "BVF", "CGJ")
ENTITIES = ("ferrum", "cupra", "zincum", "stannum", "aurum", "argent",
            "plumbum", "hydrarg", "wolfram", "niccolum")
DISTRACTOR_SETS = {
    "ferrum": ("ferrum2", "ferrumx", "ferrum-9"),
    "cupra": ("cupra2", "cuprax", "cupra-9"),
    "aurum": ("aurum2", "aurumx", "aurum-9"),
}
ALIASES = {"ferrum": "iron-marker", "cupra": "copper-marker", "aurum": "gold-marker"}


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


def _stable_cf(value: str, salt: str) -> str:
    """Deterministic replacement from SHA-256 — never Python hash()."""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return f"QVK-{200 + int(digest[:8], 16) % 700}"


def _strict(out: str, gold: str) -> bool:
    cands = CODE_RE.findall(out)
    return len(cands) == 1 and cands[0] == gold


def _word(out: str, gold: str) -> bool:
    n = re.sub(r"[^0-9a-z]+", " ", out.lower()).strip()
    g = re.sub(r"[^0-9a-z]+", " ", gold.lower()).strip()
    return re.fullmatch(rf".*(^|\s){re.escape(g)}(\s|$).*", n) is not None


def _facts_block(entities, codes, fmt: str) -> str:
    lines = [f"{e.capitalize()} stores tag {c}." for e, c in zip(entities, codes)]
    if fmt == "table":
        return "item | tag\n" + "\n".join(
            f"{l.split(' stores tag ')[0]} | {l.split(' stores tag ')[1].rstrip('.')}"
            for l in lines)
    return "\n".join(lines)


def build_items() -> dict[str, list]:
    rng = random.Random(FREEZE_SEED)
    items: dict[str, list] = defaultdict(list)

    def targeted(condition: str, entity_case: str, n: int, alias: bool,
                 distractors: bool):
        """Build n targeted items for one matching-difficulty condition,
        balanced over fact count and target position."""
        pos_counter: dict[int, int] = {}
        for i in range(n):
            k = 2 + (i % 4)
            if alias:
                # target must be an alias-bearing entity; fillers are not
                alias_pool = [e for e in ENTITIES if e in ALIASES]
                others = [e for e in ENTITIES if e not in ALIASES]
                rng.shuffle(alias_pool)
                rng.shuffle(others)
                pool = (alias_pool + others)[:k]
                target = 0 if i % 2 == 0 else min(len(alias_pool) - 1, k - 1)
                target = min(target, len(pool) - 1)
                # ensure pool[target] is alias-bearing
                if pool[target] not in ALIASES:
                    pool[0], pool[target] = pool[target], pool[0]
                    target = 0
            elif distractors:
                base = rng.choice(list(DISTRACTOR_SETS))
                pool = [base, *DISTRACTOR_SETS[base], *rng.sample(
                    [e for e in ENTITIES if e != base], 2)][:k]
                pool = pool[:k]
                target = pos_counter.get(k, 0)
                pos_counter[k] = (target + 1) % k
            else:
                pool = rng.sample(ENTITIES, k)
                target = pos_counter.get(k, 0)
                pos_counter[k] = (target + 1) % k
            codes = [_code(rng) for _ in pool]
            fmt = "plain" if i % 2 == 0 else "table"
            block = _facts_block(pool, codes, fmt)
            if alias:
                block = (f"Alias: {ALIASES[pool[target]]} = {pool[target].capitalize()}\n"
                         + block)
                q_entity = ALIASES[pool[target]]
            else:
                q_entity = {"exact": pool[target].capitalize(),
                            "lower": pool[target],
                            "upper": pool[target].upper()}[entity_case]
            q = f"Return the tag of {q_entity}."
            prompt = f"{block}\n{q}\nAnswer:"
            gold = codes[target]
            cf_gold = _stable_cf(gold, f"{condition}-{i}")
            items["targeted"].append({
                "condition": condition, "id": f"T-{condition}-{i:02d}",
                "prompt": prompt, "gold": gold, "all_codes": codes,
                "target_index": target, "n_facts": k, "format": fmt,
                "cf_prompt": prompt.replace(gold, cf_gold), "cf_gold": cf_gold,
                "target_entity": pool[target]})

    # P3/P4/P5 conditions: A exact, B case, C alias, D distractors.
    targeted("exact", "exact", 12, alias=False, distractors=False)
    targeted("case", "lower", 4, alias=False, distractors=False)
    targeted("case_upper", "upper", 4, alias=False, distractors=False)
    targeted("alias", "exact", 8, alias=True, distractors=False)
    targeted("distract", "exact", 8, alias=False, distractors=True)

    # P0 query entity recognition: name the queried entity.
    for i in range(10):
        k = 2 + (i % 4)
        pool = rng.sample(ENTITIES, k)
        codes = [_code(rng) for _ in pool]
        t = i % k
        prompt = (f"{_facts_block(pool, codes, 'plain')}\n"
                  f"Which entity is being queried if the task is: "
                  f"return the tag of {pool[t].capitalize()}? "
                  f"Answer with the entity name only.\nAnswer:")
        items["query_recognition"].append(
            {"id": f"P0-{i:02d}", "prompt": prompt, "gold": pool[t]})

    # P2 ANY-fact extraction: any supplied code counts.
    for i in range(10):
        k = 2 + (i % 4)
        pool = rng.sample(ENTITIES, k)
        codes = [_code(rng) for _ in pool]
        prompt = (f"{_facts_block(pool, codes, 'plain')}\n"
                  f"Return ANY one of the supplied tags.\nAnswer:")
        items["any_fact"].append(
            {"id": f"P2-{i:02d}", "prompt": prompt, "valid_codes": codes})

    # Query-swap sets: same facts, each entity queried once.
    for s in range(6):
        pool = rng.sample(ENTITIES, 3)
        codes = [_code(rng) for _ in pool]
        block = _facts_block(pool, codes, "plain")
        for j, (e, c) in enumerate(zip(pool, codes)):
            items["query_swap"].append(
                {"id": f"QS-{s}-{j}", "set": s, "block": block,
                 "prompt": f"{block}\nReturn the tag of {e.capitalize()}.\nAnswer:",
                 "gold": c, "all_codes": codes})
    return items


def _gen(ex, tok, prompt, max_new_tokens=12, temperature=0.0, seed=0):
    from anra_core.generate import generate
    return generate(ex, tok, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, seed=seed,
                    repetition_penalty=1.0, no_repeat_ngram_size=0)


def _classify(base_out, cf_out, item) -> str:
    """Observational decomposition; evaluator descriptors, not causes."""
    base_cands = CODE_RE.findall(base_out)
    if not base_cands:
        return "INVALID_OUTPUT"
    if not (set(base_cands) & set(item["all_codes"])):
        return "NON_FACT_VALUE"
    if base_cands[0] == item["gold"]:
        cf_cands = CODE_RE.findall(cf_out)
        if cf_cands and cf_cands[0] == item["cf_gold"]:
            return "PAIRED_CORRECT"
        return "TARGET_VALUE_BASE_ONLY"
    return "WRONG_FACT_VALUE"


def run_diagnostic(label: str, checkpoint: str, *, legacy: bool, device="cuda") -> dict:
    from anra_core.executor import CoreExecutor
    ex = CoreExecutor.from_checkpoint(checkpoint, device=device,
                                      allow_legacy_unverified=legacy)
    tok = ex.tokenizer
    items = build_items()
    report: dict = {"schema": "anra-entity-matching-diagnostic/v1",
                    "label": label, "checkpoint": checkpoint,
                    "global_step": ex.checkpoint_identity.global_step,
                    "parameter_sha256": getattr(ex.checkpoint_identity,
                                                "parameter_sha256", None),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass

    # ---- primitives
    p0 = [1 if _word(_gen(ex, tok, it["prompt"]), it["gold"]) else 1 - 1 + 0
          for it in items["query_recognition"]]
    p0 = [bool(_word(_gen(ex, tok, it["prompt"]), it["gold"]))
          for it in items["query_recognition"]]
    any_fact = []
    for it in items["any_fact"]:
        cands = CODE_RE.findall(_gen(ex, tok, it["prompt"]))
        any_fact.append(len(cands) == 1 and cands[0] in it["valid_codes"])

    # ---- targeted conditions + decomposition
    by_cond: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    decomposition: dict[str, str] = {}
    raw = {}
    for it in items["targeted"]:
        base_out = _gen(ex, tok, it["prompt"])
        cf_out = _gen(ex, tok, it["cf_prompt"])
        cls = _classify(base_out, cf_out, it)
        decomposition[it["id"]] = cls
        raw[it["id"]] = {"base": base_out, "cf": cf_out, "cls": cls,
                         "gold": it["gold"], "cf_gold": it["cf_gold"],
                         "target_entity": it["target_entity"]}
        c = by_cond[it["condition"]]
        c["base"].append(_strict(base_out, it["gold"]))
        c["cf"].append(_strict(cf_out, it["cf_gold"]))
        c["paired"].append(_strict(base_out, it["gold"])
                           and _strict(cf_out, it["cf_gold"]))
        c["n_facts"].append(it["n_facts"])
        c["pos"].append("first" if it["target_index"] == 0 else
                        "last" if it["target_index"] == it["n_facts"] - 1
                        else "middle")

    def acc(rows):
        return f"{sum(1 for x in rows if x)}/{len(rows)}"

    conditions = {}
    for cond, c in sorted(by_cond.items()):
        conditions[cond] = {
            "base": acc(c["base"]), "cf": acc(c["cf"]), "paired": acc(c["paired"]),
            "by_n_facts": {str(k): acc([p for p, n in zip(c["paired"], c["n_facts"]) if n == k])
                           for k in sorted(set(c["n_facts"]))},
            "by_position": {p: acc([x for x, q in zip(c["paired"], c["pos"]) if q == p])
                            for p in ("first", "middle", "last")},
        }

    # ---- query dependence
    sets = defaultdict(list)
    for it in items["query_swap"]:
        out = _gen(ex, tok, it["prompt"])
        cands = CODE_RE.findall(out)
        correct = len(cands) == 1 and cands[0] == it["gold"]
        changed = None  # filled after group assembly
        sets[it["set"]].append({"gold": it["gold"], "out": out,
                                "cands": cands, "correct": correct,
                                "all_codes": it["all_codes"]})
    swap_changed = swap_correct = 0
    total_pairs = 0
    for s, rows in sets.items():
        outs = [r["out"] for r in rows]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                total_pairs += 1
                if outs[a] != outs[b]:
                    swap_changed += 1
                if rows[a]["correct"] and rows[b]["correct"]:
                    swap_correct += 1
    query_dependence = {
        "sets": len(sets),
        "per_query_correct": acc([r["correct"] for rows in sets.values() for r in rows]),
        "output_changed_on_swap": f"{swap_changed}/{total_pairs}",
        "both_swapped_queries_correct": f"{swap_correct}/{total_pairs}"}

    # ---- intervention arm (first real cognitive-credit battery)
    failing = [it for it in items["targeted"]
               if decomposition[it["id"]] in ("WRONG_FACT_VALUE",
                                              "TARGET_VALUE_BASE_ONLY",
                                              "NON_FACT_VALUE")][:10]
    interventions = defaultdict(list)
    for it in failing:
        gold = it["gold"]
        ent = it["target_entity"].capitalize()
        block_fmt = it["format"]
        variants = {
            "entity_match_hint":
                it["prompt"].replace("Answer:",
                                     f"(The queried entity {ent} is the one that "
                                     f"stores a tag. Use its tag.)\nAnswer:"),
            "value_supplied":
                it["prompt"].replace("Answer:", f"(Correct tag: {gold})\nAnswer:"),
            "format_normalized":
                (f"Facts:\n" + "\n".join(
                    l for l in it["prompt"].splitlines() if "stores tag" in l or "|" in l)
                 + f"\nReturn the tag of {ent}.\nAnswer:"),
        }
        for name, prompt in variants.items():
            out = _gen(ex, tok, prompt)
            interventions[name].append(_strict(out, gold))
        sampled = [_gen(ex, tok, it["prompt"], temperature=0.8, seed=s)
                   for s in (1, 2, 3, 4)]
        interventions["decode_search"].append(
            any(_strict(o, gold) for o in sampled))

    report["primitives"] = {
        "P0_query_entity_recognition": acc(p0),
        "P2_any_fact_extraction": acc(any_fact),
        "P3_P4_P5_targeted": conditions}
    report["decomposition_counts"] = dict(Counter(decomposition.values()))
    report["query_dependence"] = query_dependence
    report["interventions_on_failures"] = {
        k: {"rescued": acc(v), "n_failures_tested": len(v)}
        for k, v in interventions.items()} or {"note": "no failing items"}
    report["raw_outputs"] = raw

    del ex
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = run_diagnostic(args.label, args.checkpoint,
                            legacy=args.legacy, device=args.device)
    text = json.dumps(report, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("primitives", "decomposition_counts", "query_dependence",
                       "interventions_on_failures")}, indent=2))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
