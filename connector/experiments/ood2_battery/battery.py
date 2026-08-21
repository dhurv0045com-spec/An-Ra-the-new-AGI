"""OOD-2: sealed selective-binding + retention battery (frozen 2026-08-22).

Why this exists: OOD-1's A-family counterfactuals were impure (fact order
reshuffled between base and CF — audited: 10/10 impure), and OOD-1 results
informed the selective-binding curriculum, so for the grandchild OOD-1 is
development evidence, not blinded validation.

OOD-2 rules, enforced in code at freeze time:
  - PURE counterfactuals: for every pair, base.replace(old, new) == cf
    byte-for-byte, and old occurs EXACTLY ONCE in base. Position/order never
    changes between base and CF — positional-binding hypotheses stay clean.
  - Completely new object vocabulary, code prefixes, and wording versus
    OOD-1 and both SFT generators.
  - Six representations for the target capability (paragraph, table, JSON,
    dialogue, key-value, mixed) plus four never-used formats for generality.
  - Retention axes: single-fact binding, copying, opaque tool results.
  - Capability-vector scoring (never one number), RAW + ASSISTED profiles.
  - Frozen with SHA-256; the runner refuses a modified suite.

The training code must never import this package.
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
FREEZE_SEED = 20260823
SCHEMA = "anra-ood2-battery/v1"

# Disjoint from OOD-1 (GXT/MVB/… codes; beacon/compass/… objects) and from
# both SFT generators.
PREFIXES = ("AQF", "BZJ", "CKW", "DMV", "EYP", "FHR", "GLT", "INX")
OBJECTS = ("tundra", "mirage", "cobalt-vein", "archway", "pinnacle", "estuary",
           "mosaic", "cavern", "harborwall", "trellis", "summit", "fresco",
           "gazebo", "outcrop", "parapet", "rotunda")
DISTRACTORS = ("Shipping occurs on weekdays only.",
               "The inspector visits each spring.",
               "Two crates were damaged in transit.",
               "The ledger was rebound last year.")


def _code(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)}-{rng.randrange(100, 1000)}"


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def _match(text: str, gold: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(_norm(gold))}(?!\w)", _norm(text)) is not None


def _fact_line(obj: str, code: str) -> str:
    return f"{obj.capitalize()} is filed under {code}."


def _parts(line: str):
    """(object, code) for fact lines; None for distractors (pass-through)."""
    if " is filed under " in line:
        obj, code = line.split(" is filed under ")
        return obj, code.rstrip(".")
    return None


RENDERERS = {
    "paragraph": lambda facts, q: "\n".join(facts) + f"\n{q}\nAnswer:",
    "table": lambda facts, q: "entry | code\n" + "\n".join(
        f"{p[0]} | {p[1]}" if p else f for f in facts if (p := _parts(f)) or True) + f"\n\n{q}\nAnswer:",
    "json": lambda facts, q: "{" + ", ".join(
        f'"{p[0]}": "{p[1]}"' if p else f'"{f}"' for f in facts for p in [_parts(f)]) + "}\n" + q + "\nAnswer:",
    "dialogue": lambda facts, q: "H: I need one code.\nSYSTEM:\n" + "\n".join(
        f"* {f}" for f in facts) + f"\nH: {q}\nANRA:",
    "kv": lambda facts, q: "\n".join(
        f"{p[0]} :: {p[1]}" if p else f for f in facts for p in [_parts(f)]) + f"\n\n{q}\nAnswer:",
    "mixed": lambda facts, q: "RECORDS:\n" + "\n".join(
        f"- [{p[0]}] => {p[1]}" if p else f"- {f}" for f in facts for p in [_parts(f)])
        + f"\nQUERY: {q}\nANSWER:",
}
FORMAT_QUERIES = {
    "paragraph": "Return ONLY the code filed for {t}.",
    "table": "Which code belongs to {t}? Code only.",
    "json": "State the code value for {t}.",
    "dialogue": "Tell me the code for {t}, nothing else.",
    "kv": "Report the value for {t}.",
    "mixed": "Give the code of {t}.",
}


def _pure_cf(prompt: str, old: str, new: str, item_id: str) -> str:
    """Counterfactual by exactly one substring replacement, asserted pure."""
    if prompt.count(old) != 1:
        raise SystemExit(f"{item_id}: gold {old!r} not unique in base prompt")
    cf = prompt.replace(old, new)
    if cf == prompt or prompt.replace(old, new) != cf:
        raise SystemExit(f"{item_id}: impure counterfactual construction")
    return cf


def build_items() -> list[dict]:
    rng = random.Random(FREEZE_SEED)
    items: list[dict] = []

    # ---- SEL: selective binding, pure CF, six representations (12 pairs)
    layouts = ["paragraph"] * 3 + ["table"] * 2 + ["json"] * 2 + \
              ["dialogue"] * 2 + ["kv"] * 2 + ["mixed"]
    for i, layout in enumerate(layouts):
        k = rng.choice((2, 3, 4, 5))
        objs = rng.sample(OBJECTS, k)
        codes = [_code(rng) for _ in objs]
        target = rng.choice(range(k))
        lines = [_fact_line(o, c) for o, c in zip(objs, codes)]
        if i % 4 == 3:
            lines.insert(rng.randrange(len(lines) + 1), rng.choice(DISTRACTORS))
        q = FORMAT_QUERIES[layout].format(t=objs[target].capitalize())
        prompt = RENDERERS[layout](lines, q)
        new_code = _code(rng)
        cf_prompt = _pure_cf(prompt, codes[target], new_code, f"SEL{i:02d}")
        items.append({"capability": "SEL_selective_binding", "id": f"SEL{i:02d}-{layout}",
                      "prompt": prompt, "gold": codes[target],
                      "cf_prompt": cf_prompt, "cf_gold": new_code})

    # ---- RET-single: single-fact binding retention (6, 3 with pure CF)
    protocols = ("The code for {o} is {c}. State the code.\nAnswer:",
                 "<k>{o} => {c}</k>\n<q>Report the code.</q>\n<answer>",
                 "H: what is {o}'s code?\nSYSTEM: {o} = {c}\nANRA:")
    for i in range(6):
        o = rng.choice(OBJECTS).capitalize()
        c = _code(rng)
        prompt = protocols[i % 3].format(o=o, c=c)
        item = {"capability": "RET_single_fact", "id": f"RET1-{i:02d}",
                "prompt": prompt, "gold": c, "cf_prompt": None, "cf_gold": None}
        if i % 2 == 0:
            new_c = _code(rng)
            item["cf_prompt"] = _pure_cf(prompt, c, new_c, item["id"])
            item["cf_gold"] = new_c
        items.append(item)

    # ---- RET-copy: verbatim copying retention (4)
    words = ("obscure", "tundra2", "windlass", "caltrop")
    for i, w in enumerate(words):
        items.append({"capability": "RET_copy", "id": f"RETC-{i:02d}",
                      "prompt": f"Reference word: {w}\nRepeat the requested word verbatim.\nAnswer:",
                      "gold": w, "cf_prompt": None, "cf_gold": None})

    # ---- RET-tool: opaque tool results, pure CF (4 pairs)
    for i in range(4):
        c, new_c = _code(rng), _code(rng)
        prompt = f"Tool response: {c}\nReturn the exact tool response.\nAnswer:"
        items.append({"capability": "RET_tool_result", "id": f"RETT-{i:02d}",
                      "prompt": prompt, "gold": c,
                      "cf_prompt": _pure_cf(prompt, c, new_c, f"RETT{i:02d}"),
                      "cf_gold": new_c})

    # ---- GEN-format: never-used representations (4)
    o = rng.choice(OBJECTS).capitalize()
    c = _code(rng)
    items.append({"capability": "GEN_unseen_formats", "id": "GEN-yaml",
                  "prompt": f"entry:\n  name: {o}\n  code: {c}\n\nThe code?\nAnswer:",
                  "gold": c, "cf_prompt": None, "cf_gold": None})
    o2, c2 = rng.choice(OBJECTS).capitalize(), _code(rng)
    items.append({"capability": "GEN_unseen_formats", "id": "GEN-csv",
                  "prompt": f"name,code\n{o2},{c2}\n\nOutput the code.\nAnswer:",
                  "gold": c2, "cf_prompt": None, "cf_gold": None})
    o3, c3 = rng.choice(OBJECTS).capitalize(), _code(rng)
    items.append({"capability": "GEN_unseen_formats", "id": "GEN-bold",
                  "prompt": f"**{o3}** — code **{c3}**.\nReply with the code only.\nAnswer:",
                  "gold": c3, "cf_prompt": None, "cf_gold": None})
    o4, c4 = rng.choice(OBJECTS).capitalize(), _code(rng)
    items.append({"capability": "GEN_unseen_formats", "id": "GEN-paren",
                  "prompt": f"( {o4} / {c4} )\nWhat is listed?\nAnswer:",
                  "gold": c4, "cf_prompt": None, "cf_gold": None})
    return items


def freeze() -> None:
    items = build_items()
    # purity gate over the whole suite before pinning
    for it in items:
        if it.get("cf_prompt"):
            assert it["prompt"].replace(it["gold"], it["cf_gold"]) == it["cf_prompt"], it["id"]
            assert it["prompt"].count(it["gold"]) == 1, it["id"]
    payload = {"schema": SCHEMA, "frozen_at": time.strftime("%Y-%m-%d %H:%M"),
               "freeze_seed": FREEZE_SEED,
               "counterfactual_purity": "base.replace(gold, cf_gold) == cf, byte-exact",
               "items": items}
    text = json.dumps(payload, indent=2, sort_keys=True)
    FROZEN.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (HERE / "items.sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"frozen: {len(items)} items, sha256={digest[:16]}…")


def _verify_frozen() -> str:
    text = FROZEN.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    recorded = (HERE / "items.sha256").read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise SystemExit("FROZEN OOD-2 SUITE MODIFIED — refusing to evaluate")
    return digest


PROFILES = {
    "RAW": {"temperature": 0.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
    "ASSISTED": {"temperature": 0.0, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
}


def _gen(ex, tok, prompt: str, profile: dict, max_new_tokens: int = 16) -> str:
    from anra_core.generate import generate
    return generate(ex, tok, prompt, max_new_tokens=max_new_tokens,
                    temperature=profile["temperature"],
                    repetition_penalty=profile["repetition_penalty"],
                    no_repeat_ngram_size=profile["no_repeat_ngram_size"])


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
            ok = _match(out, item["gold"])
            rec = {"id": item["id"], "gold": item["gold"], "output": out, "passed": ok}
            if item.get("cf_prompt"):
                cf_out = _gen(ex, tok, item["cf_prompt"], profile)
                cf_ok = _match(cf_out, item["cf_gold"])
                ok = ok and cf_ok
                rec["cf_output"] = cf_out
                rec["cf_passed"] = cf_ok
            rows.setdefault(item["capability"], {})[item["id"]] = ok
            outputs[profile_name][item["id"]] = rec
        results[profile_name] = {
            c: {"pass": sum(1 for v in r.values() if v), "n": len(r),
                "acc": round(sum(1 for v in r.values() if v) / max(len(r), 1), 3)}
            for c, r in rows.items()}

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
        "raw_outputs": outputs,
    }
    try:
        receipt["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        receipt["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        receipt["source_commit"] = None

    del ex
    import gc

    import torch
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)
    return receipt


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
        print(f"wrote {args.out}: capability vectors —")
        for p in ("RAW", "ASSISTED"):
            print(p, {c: f"{v['pass']}/{v['n']}" for c, v in receipt[f"capability_vector_{p}"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
