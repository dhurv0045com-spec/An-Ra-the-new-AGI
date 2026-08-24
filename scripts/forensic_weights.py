"""Weight-delta + logit forensics: parent vs new500m.

Answers: did weights actually move? Where? Do logits differ at all?
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FILES = {
    "parent20k": r"C:\Users\ankit\Downloads\anra-v4-current-full-resume.pt",
    "new500m": r"C:\Users\ankit\Downloads\anra-v4-tpu-latest to 500m token.pt",
}


def load_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or payload.get("model")
    return state, payload


def main() -> None:
    state_a, _ = load_state(FILES["parent20k"])
    state_b, payload_b = load_state(FILES["new500m"])

    # ---- weight deltas grouped ----
    groups: dict[str, list] = {}
    identical = 0
    total = 0
    for key in sorted(set(state_a) & set(state_b)):
        a, b = state_a[key].float(), state_b[key].float()
        if a.shape != b.shape:
            continue
        total += 1
        if torch.equal(a, b):
            identical += 1
            continue
        delta = (b - a)
        rel = delta.norm() / max(a.norm().item(), 1e-12)
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
        if "token_embedding" in key or "lm_head" in key:
            g = "embedding_head"
        elif ".blocks." in key:
            layer = key.split(".blocks.")[1].split(".")[0]
            kind = ("attn" if any(s in key for s in ("q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo"))
                    else "mlp" if any(s in key for s in ("mlp", "w1", "w2", "w3", "gate", "up", "down"))
                    else "norm")
            g = f"block{int(layer):02d}_{kind}"
        else:
            g = "other"
        groups.setdefault(g, []).append(
            {"key": key, "abs_l2": float(delta.norm()), "rel_l2": float(rel),
             "cos": cos, "max_abs": float(delta.abs().max())}
        )

    print(f"tensors compared: {total} | bit-identical: {identical} | changed: {total - identical}")
    if identical == total:
        print("!!! WEIGHTS ARE BITWISE IDENTICAL - no training happened or wrong artifact")

    # aggregate by group
    agg = []
    for g, items in groups.items():
        rel_mean = sum(i["rel_l2"] for i in items) / len(items)
        cos_min = min(i["cos"] for i in items)
        agg.append((g, len(items), rel_mean, cos_min))
    agg.sort(key=lambda x: -x[2])
    print("\ntop groups by mean relative L2 drift:")
    for g, n, rel, cos in agg[:12]:
        print(f"  {g:<18} n={n:<4} relL2={rel:.5f} min_cos={cos:.6f}")
    print("\nlowest groups:")
    for g, n, rel, cos in agg[-5:]:
        print(f"  {g:<18} n={n:<4} relL2={rel:.5f} min_cos={cos:.6f}")

    all_changed = [i for items in groups.values() for i in items]
    all_changed.sort(key=lambda i: -i["rel_l2"])
    print("\nTOP 10 most-changed tensors:")
    for i in all_changed[:10]:
        print(f"  {i['key'][:60]:<60} rel={i['rel_l2']:.5f} cos={i['cos']:.6f}")

    # ---- logit comparison on controlled prompts ----
    from anra_core.executor import CoreExecutor
    from anra_core.generate import generate
    from anra_core.tokenizer import V4Tokenizer

    tok = V4Tokenizer.load_canonical()
    probes = [
        "<k>The talren code is VXQ-482.</k><q>What is the talren code?</q>\n<answer>",
        "<k>Aster code XQH-312. Beacon code QLM-441.</k><q>What is the Aster code?</q>\n<answer>",
        "<q>What is the capital of France?</q>\n<answer>",
    ]
    execs = {}
    for tag, path in FILES.items():
        try:
            execs[tag] = CoreExecutor.from_checkpoint(path, device="cpu")
        except Exception:
            execs[tag] = CoreExecutor.from_checkpoint(path, device="cpu",
                                                      allow_legacy_unverified=True)

    print("\nlogit divergence per probe (top-1 token + KL parent->child):")
    for prompt in probes:
        ids = torch.tensor([[tok.bos_token_id, *tok.encode(prompt)]])
        with torch.inference_mode():
            la = execs["parent20k"].model(ids)[0][:, -1, :].float()
            lb = execs["new500m"].model(ids)[0][:, -1, :].float()
        pa = torch.softmax(la, dim=-1)
        pb = torch.softmax(lb, dim=-1)
        kl = (pa * (pa.add(1e-12).log() - pb.add(1e-12).log())).sum(-1).item()
        top_a = int(la.argmax()); top_b = int(lb.argmax())
        ent_a = -(pa * pa.clamp_min(1e-12).log()).sum(-1).item()
        print(f"  {prompt[:44]!r:<50} top1_same={top_a == top_b} "
              f"KL(p||q)={kl:.4f} H(parent)={ent_a:.3f}")

    (REPO / "output" / "forensic_audit" / "weight_delta.json").write_text(
        json.dumps({"groups": {g: i[:5] for g, i in groups.items()},
                    "bit_identical": identical, "total": total}, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
