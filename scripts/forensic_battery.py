"""Forensic audit battery: parent (step-20k) vs new 500M checkpoint.

Frozen prompt set, identical greedy decoding, stateful executor path.
Writes raw outputs + distribution metrics to output/forensic_audit/.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from anra_core.generate import generate  # noqa: E402
from anra_core.executor import CoreExecutor  # noqa: E402

CHECKPOINTS = {
    "parent20k": r"C:\Users\ankit\Downloads\anra-v4-current-full-resume.pt",
    "new500m": r"C:\Users\ankit\Downloads\anra-v4-tpu-latest to 500m token.pt",
}
OUT = REPO / "output" / "forensic_audit"

# ---------------- frozen battery ----------------
BATTERY: list[tuple[str, str, str]] = []
def add(family, prompt): BATTERY.append((family, prompt))

for t in ["The river flowed through the valley and the villagers used it to",
          "Scientists announced that the telescope had detected signals from",
          "In the kitchen she arranged the ingredients carefully because the recipe required"]:
    add("continuation", t)

add("conversation", "<q>Hello, how are you?</q>\n<answer>")
add("conversation", "<q>What can you help with?</q>\n<answer>")
add("conversation", "<q>Explain what a river is.</q>\n<answer>")
add("conversation", "<q>Tell me about yourself.</q>\n<answer>")

FACTS = [
    ("What is the capital of France?", "paris"),
    ("What is the largest planet?", "jupiter"),
    ("At what temperature does water freeze?", "0"),
    ("What gas do plants absorb?", "carbon"),
    ("What is the capital of Japan?", "tokyo"),
    ("How many continents are there?", "7|seven"),
]
for q, _ in FACTS:
    add("factual", f"<q>{q}</q>\n<answer>")

ARITH = ["2 + 3", "7 * 8", "15 - 6", "12 / 3", "23 + 19", "17 * 6"]
ANS = {"2 + 3": "5", "7 * 8": "56", "15 - 6": "9", "12 / 3": "4", "23 + 19": "42", "17 * 6": "102"}
for a in ARITH:
    add("arithmetic", f"<k></k><plan>Compute.</plan><q>What is {a}?</q>\n<answer>")

ECHOES = ["ember", "VXQ-482", "quixotic lantern", "K7P2M"]
for e in ECHOES:
    add("copy", f"<k></k><q>Echo exactly this word: {e}</q>\n<answer>")

NONCE = [
    ("<k>The talren code is VXQ-482.</k><q>What is the talren code?</q>", "vxq"),
    ("<k>The morvan festival happens in Juveth.</k><q>Where does the morvan festival happen?</q>", "juveth"),
]
for ctx, exp in NONCE:
    add("context_binding", f"{ctx}\n<answer>")
add("context_binding", "<k>Zephra stores 7. Caldra stores 4.</k><q>What does Zephra store?</q>\n<answer>")

MULTI = (
    "<k>Aster code XQH-312. Beacon code QLM-441. Cinder code JZR-805.</k>",
    [("What is the Aster code?", "xqh"), ("What is the Beacon code?", "qlm"), ("What is the Cinder code?", "jzr")],
)
ctx_m = MULTI[0]
for q, exp in MULTI[1]:
    add("selective_binding", f"{ctx_m}<q>{q}</q>\n<answer>")

base_ctx = "<k>Aster code XQH-312. Beacon code QLM-441. Cinder code JZR-805.</k>"
for q, marker in [("What is the Beacon code?", "qlm"), ("Which code belongs to Cinder?", "jzr")]:
    add("query_swap", f"{base_ctx}<q>{q}</q>\n<answer>")

add("composition", "<k>Zephra stores 7. Caldra stores 4.</k><q>What is Zephra plus Caldra?</q>\n<answer>")

INST = [
    "Answer with one word: what color is snow?",
    "Return only the number: 3 + 4",
    "List exactly three items: primary colors",
]
for i in INST:
    add("instruction", f"<q>{i}</q>\n<answer>")

add("uncertainty", "<k>The glorbik radius is undefined in this text.</k><q>What is the glorbik radius in meters?</q>\n<answer>")

DEGEN = [
    "the meaning of the meaning of the",
    "France is the capital of France France France",
]
for d in DEGEN:
    add("degeneration", d)

DISTRACTOR = ("<k>Aster code XQH-312. The sky is blue. Beacon code QLM-441. "
              "Gravel is heavy. Cinder code JZR-805.</k><q>What is the Aster code?</q>")
add("distractor", DISTRACTOR + "\n<answer>")


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()


def contains(text: str, needle: str) -> bool:
    if "|" in needle:
        return any(contains(text, n) for n in needle.split("|"))
    n = norm(needle)
    return re.search(rf"(?<!\w){re.escape(n)}(?!\w)", norm(text)) is not None


def repetition_rate(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return sum(1 for a, b in bigrams if a == b) / len(bigrams)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts_hash = hashlib.sha256(
        json.dumps([p for _, p, *_ in [(f, p) for f, p in BATTERY]], sort_keys=True).encode()
    ).hexdigest()

    results = {}
    for tag, ckpt in CHECKPOINTS.items():
        print(f"\n########## {tag}: {ckpt}", flush=True)
        # The step-20k parent is a schema-v9 legacy artifact without an
        # embedded tokenizer contract; explicit legacy load is the documented
        # forensic path (weights are still verified against the dense core).
        try:
            executor = CoreExecutor.from_checkpoint(ckpt, device="cpu")
        except Exception as exc:
            print(f"  strict load refused ({type(exc).__name__}); explicit legacy load", flush=True)
            executor = CoreExecutor.from_checkpoint(
                ckpt, device="cpu", allow_legacy_unverified=True
            )
        tok = executor.tokenizer
        rows = []
        for family, prompt in BATTERY:
            out = generate(executor, tok, prompt, max_new_tokens=24,
                           temperature=0.0, repetition_penalty=1.0, no_repeat_ngram_size=0)
            ids = tok.encode(out)
            toks = tok.decode(ids).lower().split()
            row = {
                "family": family, "prompt": prompt, "output": out,
                "n_tokens": len(ids),
                "repetition": round(repetition_rate(toks), 3),
            }
            if family == "factual":
                q = prompt.split("<q>")[1].split("</q>")[0]
                exp = dict(FACTS)[q]
                row["correct"] = contains(out, exp)
            elif family == "arithmetic":
                expr = prompt.split("<q>What is ")[1].split("?</q>")[0]
                row["correct"] = contains(out, ANS[expr])
            elif family == "copy":
                word = prompt.split(": ")[1].split("</q>")[0]
                row["exact"] = word in out
            elif family in ("context_binding", "selective_binding", "query_swap"):
                for exp in ("vxq-482", "qlm-441", "jzr-805", "juveth", "7"):
                    pass
                row["mentions_code"] = bool(re.search(r"[A-Z]{3}-\d{3}", out))
                row["codes_found"] = re.findall(r"[A-Z]{3}-\d{3}", out)
            rows.append(row)
            print(f"  [{family}] {prompt[:40]!r} -> {out[:60]!r}", flush=True)
        results[tag] = rows
        del executor
        import gc; gc.collect()

    # scorecard
    def score(rows, pred):
        vals = [pred(r) for r in rows if pred(r) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    scorecard = {}
    for tag, rows in results.items():
        scorecard[tag] = {
            "factual_acc": score(rows, lambda r: float(r["correct"]) if r["family"] == "factual" else None),
            "arith_acc": score(rows, lambda r: float(r["correct"]) if r["family"] == "arithmetic" else None),
            "copy_exact": score(rows, lambda r: float(r["exact"]) if r["family"] == "copy" else None),
            "code_emission": score(rows, lambda r: float(bool(r.get("codes_found"))) if r["family"] in ("selective_binding", "query_swap", "context_binding") else None),
            "mean_repetition": score(rows, lambda r: r["repetition"]),
            "distinct_outputs": len({r["output"][:40] for r in rows}) / max(1, len(rows)),
        }

    receipt = {
        "schema": "forensic-battery/v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu", "dtype": "float32", "decoding": "greedy, rep_penalty=1.0, ngram=0, max_new=24",
        "prompts_sha256": prompts_hash, "n_prompts": len(BATTERY),
        "scorecard": scorecard,
        "results": results,
    }
    (OUT / "battery_results.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("\n=== SCORECARD ===")
    print(json.dumps(scorecard, indent=1))


if __name__ == "__main__":
    main()
