"""OOD-3: sealed anti-game selective-binding + retention suite (frozen 2026-08-22).

OOD-2 is now development data (it informed the next curriculum design).
OOD-3 is frozen BEFORE the next serious child exists, with every OOD-2
property plus stronger anti-gaming controls:

  - STRICT answer parsing: for code tasks, extract candidate codes from the
    output via regex; pass iff exactly one candidate and it equals gold.
    Regurgitating several codes — one of them right — FAILS.
  - POSITION BASELINES computed at eval time: first-fact, last-fact, and
    random heuristics, so "learned selection" is distinguishable from
    "changed favorite position".
  - CONDITIONAL reporting: by fact count (2/3/4/5), target position
    (first/middle/last), and format (paragraph/table/json/dialogue/kv/mixed).
  - CAUSAL DEPENDENCE decomposition per item: base correct, counterfactual
    correct, paired, and output-actually-changed.
  - Byte-pure counterfactuals asserted at freeze and at run.
  - RAW + ASSISTED; raw generations preserved; SHA-pinned; receipts.

The training code must never import this package. Do not inspect a future
child's OOD-3 outputs until its training recipe is finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
FROZEN = HERE / "items.json"
FREEZE_SEED = 20260831
SCHEMA = "anra-ood4-battery/v1"

# Disjoint from OOD-1, OOD-2, and both SFT generators.
PREFIXES = ("MKP", "NDR", "OFS", "PGL", "QHN", "RJW", "SKZ", "TVB")
OBJECTS = ("amble", "bight", "coombe", "dell", "eyot", "firn",
           "gill", "holt", "inge", "karst", "lea", "mere",
           "ness", "plen", "ruse", "shaw")
DISTRACTORS = ("The ledger was archived in March.",
               "A courier arrives each morning.",
               "The stairwell was repainted.",
               "Two windows face the courtyard.")
CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")

RENDERERS = {
    "plain": lambda facts, q: "\n".join(facts) + f"\n{q}\nAnswer:",
    "table": lambda facts, q: "item | code\n" + "\n".join(
        f"{_p(f)[0]} | {_p(f)[1]}" if _p(f) else f for f in facts) + f"\n\n{q}\nAnswer:",
    "json": lambda facts, q: "{" + ", ".join(
        f'"{_p(f)[0]}": "{_p(f)[1]}"' if _p(f) else f'"{f}"' for f in facts)
        + "}\n" + q + "\nAnswer:",
    "dialogue": lambda facts, q: "H: one code needed.\nLOG:\n" + "\n".join(
        f"- {f}" for f in facts) + f"\nH: {q}\nANRA:",
    "kv": lambda facts, q: "\n".join(
        f"{_p(f)[0]} :: {_p(f)[1]}" if _p(f) else f for f in facts) + f"\n\n{q}\nAnswer:",
    "mixed": lambda facts, q: "REGISTRY\n" + "\n".join(
        f"* {_p(f)[0]} -> {_p(f)[1]}" if _p(f) else f"* {f}" for f in facts)
        + f"\nQ: {q}\nA:",
}
QUERIES = {
    "plain": "Reply with the tag of {t} alone.",
    "table": "The tag of {t}:",
    "json": "Output the tag for {t}.",
    "dialogue": "Give me {t} tag.",
    "kv": "Tag for {t}:",
    "mixed": "Which tag belongs to {t}?",
}


def _p(line: str):
    if " maps to " in line:
        obj, code = line.split(" maps to ")
        return obj, code.rstrip(".")
    return None


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


def _fact(obj: str, code: str) -> str:
    return f"{obj.capitalize()} holds code {code}."


def _pure_cf(prompt: str, old: str, new: str, item_id: str) -> str:
    if prompt.count(old) != 1:
        raise SystemExit(f"{item_id}: gold not unique in base")
    return prompt.replace(old, new)


def _strict_code_match(output: str, gold: str) -> bool:
    """Exactly one code candidate in the output, equal to gold."""
    candidates = CODE_RE.findall(output)
    return len(candidates) == 1 and candidates[0] == gold


def _loose_match(output: str, gold: str) -> bool:
    norm = re.sub(r"[^0-9a-z]+", " ", output.lower()).strip()
    return re.search(rf"(?<!\w){re.escape(norm if norm else gold.lower())}(?!\w)",
                     re.sub(r"[^0-9a-z]+", " ", gold.lower())) is not None


def build_items() -> list[dict]:
    rng = random.Random(FREEZE_SEED)
    items: list[dict] = []

    # SEL: 18 items. Balanced fact counts 2/3/4/5 and target positions, all
    # six formats, pure CF twins, position metadata for baselines/conditionals.
    plan = []
    for fmt in RENDERERS:
        for k in (2, 3, 4):
            plan.append((fmt, k))
    for fmt in ("plain", "table", "kv", "json", "dialogue", "mixed", "plain", "table"):
        plan.append((fmt, 5))
    for i, (fmt, k) in enumerate(plan):
        objs = rng.sample(OBJECTS, k)
        codes = [_code(rng) for _ in objs]
        if k == 2:
            target = i % 2
        elif k == 3:
            target = i % 3
        else:
            target = rng.randrange(k)
        lines = [_fact(o, c) for o, c in zip(objs, codes)]
        if i % 5 == 4:
            lines.insert(rng.randrange(len(lines) + 1), rng.choice(DISTRACTORS))
        q = QUERIES[fmt].format(t=objs[target].capitalize())
        prompt = RENDERERS[fmt](lines, q)
        new_code = _code(rng)
        cf = _pure_cf(prompt, codes[target], new_code, f"SEL{i:02d}")
        items.append({
            "capability": "SEL_selective_binding", "id": f"SEL{i:02d}-{fmt}",
            "prompt": prompt, "gold": codes[target],
            "cf_prompt": cf, "cf_gold": new_code,
            "n_facts": k, "target_position": target,
            "target_position_class": ("first" if target == 0 else
                                       "last" if target == k - 1 else "middle"),
            "format": fmt, "all_codes": codes})

    # Retention axes (strict parsing where applicable).
    protocols = ("The code for {o} is {c}. State the code.\nAnswer:",
                 "<k>{o} => {c}</k>\n<q>Report the code.</q>\n<answer>",
                 "H: code for {o}?\nLOG: {o} = {c}\nANRA:")
    for i in range(6):
        o = rng.choice(OBJECTS).capitalize()
        c = _code(rng)
        prompt = protocols[i % 3].format(o=o, c=c)
        item = {"capability": "RET_single_fact", "id": f"RET1-{i:02d}",
                "prompt": prompt, "gold": c, "strict": True,
                "cf_prompt": None, "cf_gold": None}
        if i % 2 == 0:
            new_c = _code(rng)
            item["cf_prompt"] = _pure_cf(prompt, c, new_c, item["id"])
            item["cf_gold"] = new_c
        items.append(item)
    for i, w in enumerate(("lanyard", "plumbline", "grindstone", "casement")):
        items.append({"capability": "RET_copy", "id": f"RETC-{i:02d}",
                      "prompt": f"Reference word: {w}\nRepeat the word verbatim.\nAnswer:",
                      "gold": w, "strict": False,
                      "cf_prompt": None, "cf_gold": None})
    for i in range(4):
        c, new_c = _code(rng), _code(rng)
        prompt = f"Tool response: {c}\nReturn the exact response.\nAnswer:"
        items.append({"capability": "RET_tool_result", "id": f"RETT-{i:02d}",
                      "prompt": prompt, "gold": c, "strict": True,
                      "cf_prompt": _pure_cf(prompt, c, new_c, f"RETT{i:02d}"),
                      "cf_gold": new_c})
    return items


def freeze() -> None:
    items = build_items()
    for it in items:
        if it.get("cf_prompt"):
            assert it["prompt"].replace(it["gold"], it["cf_gold"]) == it["cf_prompt"]
            assert it["prompt"].count(it["gold"]) == 1
    payload = {"schema": SCHEMA, "frozen_at": time.strftime("%Y-%m-%d %H:%M"),
               "freeze_seed": FREEZE_SEED,
               "counterfactual_purity": "byte-exact single-substring replacement",
               "answer_parsing": "strict single-code extraction for code tasks",
               "items": items}
    text = json.dumps(payload, indent=2, sort_keys=True)
    FROZEN.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (HERE / "items.sha256").write_text(digest + "\n", encoding="utf-8")
    sel = [i for i in items if i["capability"].startswith("SEL")]
    from collections import Counter
    print(f"frozen: {len(items)} items, sha256={digest[:16]}…")
    print("SEL fact-count histogram:", dict(Counter(i["n_facts"] for i in sel)))
    print("SEL target-position histogram:", dict(Counter(i["target_position_class"] for i in sel)))
    print("SEL format histogram:", dict(Counter(i["format"] for i in sel)))


def _verify_frozen() -> str:
    text = FROZEN.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != (HERE / "items.sha256").read_text(encoding="utf-8").strip():
        raise SystemExit("FROZEN OOD-3 SUITE MODIFIED — refusing to evaluate")
    return digest


PROFILES = {
    "RAW": {"temperature": 0.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
    "ASSISTED": {"temperature": 0.0, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
}


def _score(item, output: str) -> bool:
    return _strict_code_match(output, item["gold"]) if item.get("strict", True) \
        else _match_word(output, item["gold"])


def _match_word(output: str, gold: str) -> bool:
    n = re.sub(r"[^0-9a-z]+", " ", output.lower()).strip()
    g = re.sub(r"[^0-9a-z]+", " ", gold.lower()).strip()
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", n) is not None


def evaluate_checkpoint(label: str, path: str, *, legacy: bool, device: str = "cuda") -> dict:
    from anra_core.executor import CoreExecutor
    ex = CoreExecutor.from_checkpoint(path, device=device, allow_legacy_unverified=legacy)
    tok = ex.tokenizer
    items = json.loads(FROZEN.read_text(encoding="utf-8"))["items"]
    suite_sha = _verify_frozen()

    results, outputs = {}, {}
    for profile_name, profile in PROFILES.items():
        rows: dict[str, dict[str, bool]] = {}
        outputs[profile_name] = {}
        for item in items:
            out = _gen(ex, tok, item["prompt"], profile)
            base_ok = _score(item, out)
            rec = {"id": item["id"], "gold": item["gold"], "output": out,
                   "base_passed": base_ok}
            if item.get("cf_prompt"):
                cf_out = _gen(ex, tok, item["cf_prompt"], profile)
                # Score the twin against the TWIN's gold. (An earlier version
                # compared cf_out to the base gold — a scorer defect that
                # produced false negatives; fixed and recomputed 2026-08-22.)
                cf_ok = (_strict_code_match(cf_out, item["cf_gold"])
                         if item.get("strict", True)
                         else _match_word(cf_out, item["cf_gold"]))
                rec.update({"cf_output": cf_out, "cf_passed": cf_ok,
                            "paired": base_ok and cf_ok,
                            "output_changed": out != cf_out})
                ok = base_ok and cf_ok
            else:
                ok = base_ok
            if "n_facts" in item:  # position baselines need raw fact codes
                rec["position_baselines"] = {
                    "first": item["all_codes"][0] == item["gold"],
                    "last": item["all_codes"][-1] == item["gold"]}
            rows.setdefault(item["capability"], {})[item["id"]] = ok
            outputs[profile_name][item["id"]] = rec
        results[profile_name] = {c: {"pass": sum(1 for v in r.values() if v), "n": len(r)}
                                 for c, r in rows.items()}

    # Conditional breakdowns + baselines (RAW only — claims cite RAW).
    sel = [i for i in items if "n_facts" in i]
    raw_out = outputs["RAW"]
    def cond(key_fn):
        groups: dict[str, list] = {}
        for i in sel:
            groups.setdefault(key_fn(i), []).append(raw_out[i["id"]]["paired"
                                  if "cf_prompt" in i else "base_passed"])
        return {k: f"{sum(v)}/{len(v)}" for k, v in sorted(groups.items())}
    first_base = sum(1 for i in sel if i["all_codes"][0] == i["gold"])
    last_base = sum(1 for i in sel if i["all_codes"][-1] == i["gold"])
    conditionals = {
        "by_fact_count": cond(lambda i: str(i["n_facts"])),
        "by_target_position": cond(lambda i: i["target_position_class"]),
        "by_format": cond(lambda i: i["format"]),
        "position_baselines": {"first_fact_heuristic": f"{first_base}/{len(sel)}",
                               "last_fact_heuristic": f"{last_base}/{len(sel)}",
                               "random": f"{sum(i['n_facts'] for i in sel) and round(sum(1/i['n_facts'] for i in sel), 2)} expected"},
        "causal_dependence": {
            "base_correct": sum(1 for i in sel if raw_out[i["id"]]["base_passed"]),
            "cf_correct": sum(1 for i in sel if raw_out[i["id"]].get("cf_passed")),
            "paired": sum(1 for i in sel if raw_out[i["id"]].get("paired")),
            "output_changed_on_cf": sum(1 for i in sel if raw_out[i["id"]].get("output_changed")),
            "n": len(sel)},
    }

    receipt = {
        "experiment_schema": SCHEMA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label, "checkpoint_path": path,
        "global_step": ex.checkpoint_identity.global_step,
        "parameter_sha256": getattr(ex.checkpoint_identity, "parameter_sha256", None),
        "suite_sha256": suite_sha,
        "profiles": PROFILES,
        "device": device, "precision": "float32",
        "capability_vector_RAW": results["RAW"],
        "capability_vector_ASSISTED": results["ASSISTED"],
        "conditionals_RAW": conditionals,
        "raw_outputs": outputs,
    }
    try:
        receipt["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        receipt["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        pass
    del ex
    import gc

    import torch
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)
    return receipt


def _gen(ex, tok, prompt: str, profile: dict, max_new_tokens: int = 14) -> str:
    from anra_core.generate import generate
    return generate(ex, tok, prompt, max_new_tokens=max_new_tokens,
                    temperature=profile["temperature"],
                    repetition_penalty=profile["repetition_penalty"],
                    no_repeat_ngram_size=profile["no_repeat_ngram_size"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--label", default="run")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return 0
    if not args.checkpoint:
        raise SystemExit("--checkpoint required (or --freeze)")
    receipt = evaluate_checkpoint(args.label, args.checkpoint,
                                  legacy=args.legacy, device=args.device)
    text = json.dumps(receipt, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
        print("RAW:", {c: f"{v['pass']}/{v['n']}" for c, v in receipt["capability_vector_RAW"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
