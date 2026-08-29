"""Fast, strict, stateful smoke evaluation for one An-Ra checkpoint."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_core.executor import CoreExecutor
from anra_core.generate import generate


CASES = (
    ("context", "<k>The talren code is VXQ-482.</k><q>What is the talren code?</q><answer>", "vxq-482"),
    ("fact", "<q>What is the capital of France?</q><answer>", "paris"),
    ("arithmetic", "<q>Return only the number: 7 + 5</q><answer>", "12"),
    ("instruction", "<q>Echo exactly this word: ember</q><answer>", "ember"),
    ("explanation", "<q>Explain photosynthesis in one short sentence.</q><answer>", None),
    ("creative", "The moonlit library opened its doors and", None),
)


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", " ", text.lower()).strip()


def repetition_ratio(text: str) -> float:
    words = normalized(text).split()
    if len(words) < 4:
        return 0.0
    grams = list(zip(words, words[1:], words[2:]))
    return round(1.0 - len(set(grams)) / len(grams), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    executor = CoreExecutor.from_checkpoint(args.checkpoint, device=args.device)
    loaded = time.perf_counter()
    tokenizer = executor.tokenizer
    rows = []
    for name, prompt, expected in CASES:
        t0 = time.perf_counter()
        output = generate(
            executor,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
        seconds = time.perf_counter() - t0
        rows.append({
            "name": name,
            "output": output,
            "expected": expected,
            "match": expected in normalized(output) if expected else None,
            "output_tokens": len(tokenizer.encode(output)),
            "seconds": round(seconds, 3),
            "repetition": repetition_ratio(output),
        })

    creative_prompt = CASES[-1][1]
    torch.manual_seed(7)
    intervention = generate(
        executor,
        tokenizer,
        creative_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )
    identity = executor.checkpoint_identity
    report = {
        "checkpoint_step": identity.global_step,
        "schema": identity.artifact_schema_version,
        "artifact_class": identity.artifact_class,
        "tokenizer_verified": identity.tokenizer_contract_verified,
        "device": str(next(executor.model.parameters()).device),
        "load_seconds": round(loaded - started, 3),
        "cases": rows,
        "creative_intervention": {
            "output": intervention,
            "repetition": repetition_ratio(intervention),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
