"""CPU evaluation of the soup checkpoint using the fixed generate().

Compares soup vs both parents on the core battery, all CPU, sequential.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from anra_core.config import CANONICAL_CONFIG  # noqa: E402
from anra_core.executor import CoreExecutor  # noqa: E402
from anra_core.generate import generate  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

CASES = (
    ("fact_capital", "The capital of France is", "Paris"),
    ("fact_capital_ctx", "<k>The capital of Japan is Tokyo.</k>\n<q>What is the capital of Japan?</q>\n<answer>", "Tokyo"),
    ("echo_ember", "Echo exactly this word: ember", "ember"),
    ("copy_ctx", "<k>the magic word is lantern</k>\n<q>What is the magic word?</q>\n<answer>", "lantern"),
    ("arith_add", "Compute 7 + 5.", "12"),
    ("arith_toolresult", "Use the calculator to add 20 and 22.\n<tool_output>42</tool_output>\nWhat is 20 + 22?", "42"),
    ("chat_greeting", "Hello! How are you today?", None),
    ("story", "Once upon a time, there was a little girl who", None),
)


def repetition_ratio(text: str) -> float:
    import re

    words = re.sub(r"[^0-9a-z]+", " ", text.lower()).split()
    if len(words) < 8:
        return 0.0
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return round(1.0 - len(set(grams)) / len(grams), 3)


def evaluate(tag: str, ckpt: str) -> dict:
    from anra_core.checkpoint import load_core_checkpoint

    print(f"\n=== [{tag}] ===", flush=True)
    try:
        model, _meta, identity = load_core_checkpoint(ckpt)
    except Exception:
        from anra_core.checkpoint import load_core_checkpoint as lc

        model, _meta, identity = lc(ckpt, legacy_unverified=True)
    executor = CoreExecutor(model, tokenizer=V4Tokenizer.load_canonical())
    tok = executor.tokenizer
    rows = []
    hits = total = 0
    for name, prompt, gold in CASES:
        text = generate(executor, tok, prompt, max_new_tokens=24)
        match = None
        if gold:
            import re

            norm = lambda s: re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()
            match = bool(re.search(rf"(?<!\w){re.escape(norm(gold))}(?!\w)", norm(text)))
            total += 1
            hits += 1 if match else 0
        rep = repetition_ratio(text)
        rows.append({"name": name, "output": text, "match": match, "repetition": rep})
        print(f"  [{name}] {'PASS' if match else ('FAIL' if gold else '----')} rep={rep:.2f} :: {text[:64]!r}", flush=True)
    del executor
    gc.collect()
    return {
        "checkpoint": ckpt,
        "global_step": int(identity.global_step or -1),
        "exact_accuracy": round(hits / total, 3) if total else None,
        "mean_repetition": round(sum(r["repetition"] for r in rows) / len(rows), 3),
        "cases": rows,
    }


if __name__ == "__main__":
    targets = {
        "step20000": r"C:\Users\ankit\Downloads\anra-v4-current-full-resume.pt",
        "step30400": r"C:\Users\ankit\Downloads\anra-v4-tpu-latest.pt",
        "soup": str(REPO / "output" / "ckpt_eval" / "soup_20k_30k.pt"),
    }
    report = {}
    for tag, ckpt in targets.items():
        report[tag] = evaluate(tag, ckpt)

    print("\n=== COMPARISON ===", flush=True)
    for tag, data in report.items():
        print(
            f"{tag}: exact={data['exact_accuracy']} rep={data['mean_repetition']}",
            flush=True,
        )
    dest = REPO / "output" / "ckpt_eval" / "soup_comparison.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
