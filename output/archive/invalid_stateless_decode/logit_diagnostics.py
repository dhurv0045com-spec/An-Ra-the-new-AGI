"""Reverse-engineer the generation failure: logit diagnostics at token 1.

For each checkpoint, measures on identical prompts:
  - top-1 vs top-2 logit margin (how confident is the greedy choice?)
  - entropy of the softmax (how flat is the distribution?)
  - whether the top-1 token is punctuation/whitespace (collapse precursors)

Sequential with full VRAM release between checkpoints.
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
from anra_core.model import AnRaCore  # noqa: E402
from anra_core.tokenizer import V4Tokenizer  # noqa: E402

PROMPTS = (
    "The capital of France is",
    "Echo exactly this word: ember",
    "Hello! How are you today?",
    "Once upon a time, there was a little girl who",
    "def add_numbers(a, b):\n    return",
)


def analyze(ckpt: str) -> dict:
    from anra_core.checkpoint import load_core_checkpoint

    try:
        model, _meta, identity = load_core_checkpoint(ckpt)
    except Exception:
        from anra_core.checkpoint import load_core_checkpoint as lc

        model, _meta, identity = lc(ckpt, legacy_unverified=True)
    model = model.to("cuda").eval()
    tok = V4Tokenizer.load_canonical()
    rows = []
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = torch.tensor(
                [[tok.bos_token_id, *tok.encode(prompt)]], dtype=torch.long, device="cuda"
            )
            logits = model(ids)[0, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            top2 = torch.topk(logits, 2)
            margin = float((top2.values[0] - top2.values[1]).item())
            entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
            max_prob = float(probs.max().item())
            top1_token = tok.decode([int(top2.indices[0].item())])
            top2_token = tok.decode([int(top2.indices[1].item())])
            rows.append({
                "prompt": prompt[:40],
                "top1": top1_token,
                "top2": top2_token,
                "margin": round(margin, 3),
                "max_prob": round(max_prob, 4),
                "entropy_nats": round(entropy, 3),
                "entropy_bits": round(entropy / 0.693147, 3),
            })
            print(
                f"  {prompt[:34]!r:38} top1={top1_token!r:12} top2={top2_token!r:12} "
                f"margin={margin:.3f} p1={max_prob:.4f} H={entropy/0.693147:.2f}bits",
                flush=True,
            )
    margins = [r["margin"] for r in rows]
    entropies = [r["entropy_bits"] for r in rows]
    summary = {
        "checkpoint": ckpt,
        "global_step": int(identity.global_step or -1),
        "rows": rows,
        "mean_margin": round(sum(margins) / len(margins), 3),
        "mean_entropy_bits": round(sum(entropies) / len(entropies), 3),
        "uniform_entropy_bits": 15.0,  # log2(32768)
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  VRAM released: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB free", flush=True)
    return summary


if __name__ == "__main__":
    out = {}
    for tag, ckpt in (
        ("step20000", r"C:\Users\ankit\Downloads\anra-v4-current-full-resume.pt"),
        ("step30400", r"C:\Users\ankit\Downloads\anra-v4-tpu-latest.pt"),
    ):
        print(f"\n=== {tag} ===", flush=True)
        out[tag] = analyze(ckpt)
    dest = REPO / "output" / "ckpt_eval" / "logit_diagnostics.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
