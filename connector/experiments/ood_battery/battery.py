"""Frozen out-of-distribution capability battery (never seen by training).

Design contract (frozen 2026-08-22, before any further training):
  - Items are generated once from FREEZE_SEED and pinned in ``items.json``
    with a SHA-256 manifest. The runner REFUSES to run on a modified suite.
  - The SFT generator's templates, scaffolds ("Fact:/Instructions:/Question:
    /Answer:" and ``<k>/<plan>/<q>/<answer>``), and vocabularies must not
    appear here. New protocols and new operations only.
  - Counterfactual scoring: a pair passes only if the base item passes AND
    its counterfactual (one fact changed, everything else identical) flips
    to the new correct answer. This proves dependence on supplied
    information, not surface priors.
  - Two decode profiles are always reported:
      RAW       temperature=0, repetition_penalty=1.0, no_repeat_ngram=0
      ASSISTED  temperature=0, repetition_penalty=1.15, no_repeat_ngram=4
    Learned-capability claims cite RAW. ASSISTED is decoder engineering.

Capabilities:
  A selective_multi_fact   3+ facts, retrieve one, counterfactual flip
  B novel_composition      operation pairs absent from the SFT curriculum
  C causal_tool_result     opaque tool outputs, counterfactual flip
  D perturbation_controls  correct / irrelevant / replaced / removed context
  E protocol_transfer      prose / JSON-ish / table / key-value / dialogue
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
FREEZE_SEED = 20260822
SCHEMA = "anra-ood-battery/v1"

# Disjoint from the SFT generator (QRV/LMN/XDP/JKW/TFZ/BCYG/NRH/VQS/PLM/ZWXT
# codes; materials; echo words; word pairs) and from the P1-P6 probes.
CODES = ("GXT", "MVB", "RQN", "HZL", "PDW", "KJS", "WFM", "TBR", "CNV", "LYG",
         "XRQ", "LMP", "NKD", "SGY", "VHT")
OBJECTS = ("beacon", "compass", "drum", "engine", "furnace", "gadget",
           "hammer", "instrument", "junction", "kettle", "lamp", "motor",
           "nozzle", "piston", "quarry")
IRRELEVANT = ("The warehouse closes at 6 pm.",
              "A nearby river floods every spring.",
              "The supervisor drinks tea, not coffee.",
              "Blue paint was ordered last month.")
WORDS = ("cedar", "birch", "falcon", "otter", "granite", "slate", "juniper",
         "heron", "basalt", "silex")


def _code(rng: random.Random) -> str:
    return f"{rng.choice(CODES)}-{rng.randrange(100, 999)}"


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).strip()


def _match(text: str, gold: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(_norm(gold))}(?!\w)", _norm(text)) is not None


def build_items() -> list[dict]:
    rng = random.Random(FREEZE_SEED)
    items: list[dict] = []

    # ---- A: selective multi-fact binding (10 counterfactual pairs + recency)
    for i in range(10):
        objs = rng.sample(OBJECTS, 3)
        codes = [_code(rng) for _ in objs]
        target = rng.randrange(3)
        lines = [f"Object {o.capitalize()} has code {c}." for o, c in zip(objs, codes)]
        rng.shuffle(lines)
        if i % 3 == 1:
            question = (f"Which code is assigned to {objs[target].capitalize()}? "
                        f"Reply with the code only.")
        else:
            question = f"Return ONLY the code assigned to {objs[target].capitalize()}."
        if i % 3 == 2:
            lines.insert(rng.randrange(len(lines) + 1), rng.choice(IRRELEVANT))
        prompt = "\n".join(lines) + f"\n{question}\nAnswer:"
        cf_codes = list(codes)
        cf_codes[target] = _code(rng)
        cf_lines = [f"Object {o.capitalize()} has code {c}." for o, c in zip(objs, cf_codes)]
        shuf = random.Random(FREEZE_SEED + 100 + i)
        shuf.shuffle(cf_lines)
        cf_prompt = "\n".join(cf_lines) + f"\n{question}\nAnswer:"
        items.append({"capability": "A_selective_multi_fact", "id": f"A{i:02d}",
                      "prompt": prompt, "gold": codes[target],
                      "cf_prompt": cf_prompt, "cf_gold": cf_codes[target]})
    o = rng.choice(OBJECTS).capitalize()
    c1, c2 = _code(rng), _code(rng)
    items.append({
        "capability": "A_selective_multi_fact", "id": "A10-recency",
        "prompt": (f"Object {o} has code {c1}.\nCorrection: the registry now "
                   f"lists {o} with code {c2}.\nReturn ONLY the current code "
                   f"for {o}.\nAnswer:"),
        "gold": c2, "cf_prompt": None, "cf_gold": None})

    # ---- B: novel instruction composition (12; three-item sequences)
    def b_item(i: int, instruction: str, seq: str, gold: str) -> dict:
        return {"capability": "B_novel_composition", "id": f"B{i:02d}",
                "prompt": f"Sequence: {seq}\nInstruction: {instruction}\nAnswer:",
                "gold": gold, "cf_prompt": None, "cf_gold": None}

    for i in range(12):
        w = rng.sample(WORDS, 3)
        kind = i % 6
        if kind == 0:
            items.append(b_item(i, "TAKE the second item, then DUPLICATE it.",
                                f"{w[0]} {w[1]} {w[2]}", f"{w[1]} {w[1]}"))
        elif kind == 1:
            items.append(b_item(i, "MOVE the first item to the end.",
                                f"{w[0]} {w[1]} {w[2]}", f"{w[1]} {w[2]} {w[0]}"))
        elif kind == 2:
            items.append(b_item(i, "REMOVE the second item.",
                                f"{w[0]} {w[1]} {w[2]}", f"{w[0]} {w[2]}"))
        elif kind == 3:
            items.append(b_item(i, "APPLY the mapping A->Z and B->Y, then output the sequence.",
                                f"{w[0]} A {w[1]} B", f"{w[0]} Z {w[1]} Y"))
        elif kind == 4:
            items.append(b_item(i, "REVERSE the order, then append MAPLE.",
                                f"{w[0]} {w[1]}", f"{w[1]} {w[0]} maple"))
        else:
            items.append(b_item(i, "MOVE the second item to the front.",
                                f"{w[0]} {w[1]} {w[2]}", f"{w[1]} {w[0]} {w[2]}"))

    # ---- C: causal tool-result dependence (8 counterfactual pairs)
    for i in range(6):
        code, cf = _code(rng), _code(rng)
        prompt = f"Tool response: {code}\nReturn the exact tool response.\nAnswer:"
        items.append({"capability": "C_causal_tool_result", "id": f"C{i:02d}",
                      "prompt": prompt, "gold": code,
                      "cf_prompt": prompt.replace(code, cf), "cf_gold": cf})
    for i in range(2):
        code, cf = _code(rng), _code(rng)
        prompt = f"H: Report the tool response exactly.\nTOOL: {code}\nANRA:"
        items.append({"capability": "C_causal_tool_result", "id": f"C{i + 6:02d}-chat",
                      "prompt": prompt, "gold": code,
                      "cf_prompt": prompt.replace(code, cf), "cf_gold": cf})

    # ---- D: perturbation controls (5 groups × 4 conditions)
    for i in range(5):
        o = rng.choice(OBJECTS).capitalize()
        code, wrong = _code(rng), _code(rng)
        q = f"Return ONLY the code for {o}.\nAnswer:"
        items.append({
            "capability": "D_perturbation_controls", "id": f"D{i:02d}",
            "conditions": {
                "correct": {"prompt": f"{o} is registered with code {code}.\n{q}",
                            "gold": code},
                "irrelevant": {"prompt": f"{o} is registered with code {code}.\n"
                                         f"{rng.choice(IRRELEVANT)}\n{q}", "gold": code},
                "replaced": {"prompt": f"{o} is registered with code {wrong}.\n{q}",
                             "gold": wrong},
                "removed": {"prompt": q, "not_gold": code},
            }})

    # ---- E: protocol transfer (3 tasks × 5 untrained protocols)
    for i in range(3):
        o = rng.choice(OBJECTS).capitalize()
        code = _code(rng)
        for suffix, prompt in (
            ("prose", f"The registry lists {o} under code {code}. State the code.\nAnswer:"),
            ("json", f'{{"object": "{o}", "code": "{code}"}}\nWhat is the code?\nAnswer:'),
            ("table", f"object | code\n{o} | {code}\n\nCode for {o}?\nAnswer:"),
            ("kv", f"{o} :: {code}\n\nReport the value.\nAnswer:"),
            ("dialogue", f"H: I need the code for {o}.\nSYSTEM: {o} => {code}\nANRA:"),
        ):
            items.append({"capability": "E_protocol_transfer", "id": f"E{i}-{suffix}",
                          "prompt": prompt, "gold": code,
                          "cf_prompt": None, "cf_gold": None})
    return items


def freeze() -> None:
    payload = {"schema": SCHEMA, "frozen_at": time.strftime("%Y-%m-%d %H:%M"),
               "freeze_seed": FREEZE_SEED, "items": build_items()}
    text = json.dumps(payload, indent=2, sort_keys=True)
    FROZEN.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (HERE / "items.sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"frozen: {len(payload['items'])} items, sha256={digest[:16]}…")


def _verify_frozen() -> str:
    text = FROZEN.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    recorded = (HERE / "items.sha256").read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise SystemExit("FROZEN SUITE MODIFIED — refusing to evaluate")
    if json.loads(text)["schema"] != SCHEMA:
        raise SystemExit("schema drift")
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

    results = {}
    for profile_name, profile in PROFILES.items():
        rows: dict[str, dict[str, bool]] = {}
        for item in items:
            cap, item_id = item["capability"], item["id"]
            if cap == "D_perturbation_controls":
                conds = {}
                for cond in ("correct", "irrelevant", "replaced"):
                    out = _gen(ex, tok, item["conditions"][cond]["prompt"], profile)
                    conds[cond] = _match(out, item["conditions"][cond]["gold"])
                out = _gen(ex, tok, item["conditions"]["removed"]["prompt"], profile)
                # With the fact removed the model must NOT still emit the code:
                # emitting it proves recall/leakage rather than context use.
                conds["removed"] = not _match(out, item["conditions"]["removed"]["not_gold"])
                rows.setdefault(cap, {})[item_id] = all(conds.values())
                continue
            out = _gen(ex, tok, item["prompt"], profile)
            ok = _match(out, item["gold"])
            if item.get("cf_prompt"):
                cf_out = _gen(ex, tok, item["cf_prompt"], profile)
                ok = ok and _match(cf_out, item["cf_gold"])
            rows.setdefault(cap, {})[item_id] = ok
        results[profile_name] = {
            c: {"pass": sum(1 for v in r.values() if v), "n": len(r),
                "acc": round(sum(1 for v in r.values() if v) / max(len(r), 1), 3),
                "detail": r}
            for c, r in rows.items()
        }

    receipt = {
        "experiment_schema": SCHEMA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "checkpoint_path": path,
        "global_step": ex.checkpoint_identity.global_step,
        "parameter_sha256": getattr(ex.checkpoint_identity, "parameter_sha256", None),
        "checkpoint_file_sha256": getattr(ex.checkpoint_identity, "checkpoint_sha256", None),
        "suite_sha256": suite_sha,
        "profiles": PROFILES,
        "device": device, "precision": "float32",
        "results": results,
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
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
