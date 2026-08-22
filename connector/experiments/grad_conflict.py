"""Gradient-conflict and layerwise-drift diagnostics (pre-training evidence).

Question (H1 vs H2): does the grandchild's forgetting happen because the
selective-binding objective was merely under-accompanied by rehearsal (H1),
or because its gradients actively push parameters against retained behavior
(H2)?

Method:
  1. Load the context-binding child on CUDA.
  2. Take TARGET batches (selective binding, dev bank) and RETENTION batches
     (single-fact, tool, copy, protocol-transfer from the dev bank).
  3. Average gradients per class over equal batch counts.
  4. Report cosine(target, retention) globally and per module group
     (embedding, blocks 0-17, final norm/head).
  5. Independently: layerwise ||dW||/||W|| between parent->child and
     child->grandchild checkpoints, to see where training moved weights.

Interpretation guide:
  cos >> 0   : compatible — forgetting likely under-sampling (H1); replay fix
  cos ~ 0    : neutral — mixture/data-balance question
  cos << 0   : conflict — active interference (H2); anchoring/regularization
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from anra_core.checkpoint import load_core_checkpoint
from anra_core.tokenizer import V4Tokenizer
from training.sft_context_binding import encode_item

TARGET_FAMS = {"selective", "selective_cf"}
RETENTION_FAMS = {"single_fact", "tool_result", "copy", "protocol_transfer"}


def _load_items(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def _grad_of_batch(model, batch, tok, device) -> dict[str, torch.Tensor]:
    """Average gradient of the masked LM loss over one batch (batch size 1
    accumulated; sequences are short so this is fast)."""
    model.zero_grad(set_to_none=True)
    for item in batch:
        ids_t, labels_t = encode_item(tok, item)
        logits = model(ids_t.to(device))
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels_t.view(-1).to(device), ignore_index=-100)
        (loss / len(batch)).backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0))


def _group(name: str) -> str:
    if name.startswith("token_embedding"):
        return "embedding"
    if name.startswith("blocks."):
        return f"block{int(name.split('.')[1]):02d}"
    return "head/norm"


def run_gradient_conflict(checkpoint: str, bank: str, device: str,
                          n_batches: int = 6, batch: int = 12) -> dict:
    model, _, identity = load_core_checkpoint(checkpoint, legacy_unverified=True)
    model = model.to(device).train()
    tok = V4Tokenizer.load_canonical()
    items = _load_items(bank)
    rng = torch.Generator().manual_seed(0)
    target_pool = [i for i in items if i["family"] in TARGET_FAMS]
    retention_pool = [i for i in items if i["family"] in RETENTION_FAMS]
    print(f"pool sizes: target={len(target_pool)} retention={len(retention_pool)}")

    def avg_grads(pool):
        acc = defaultdict(lambda: None)
        for b in range(n_batches):
            idx = torch.randperm(len(pool), generator=rng)[:batch].tolist()
            g = _grad_of_batch(model, [pool[i] for i in idx], tok, device)
            for n, t in g.items():
                acc[n] = t if acc[n] is None else acc[n] + t
        return {n: t / n_batches for n, t in acc.items()}

    t0 = time.time()
    g_target = avg_grads(target_pool)
    g_ret = avg_grads(retention_pool)

    global_cos = _cosine(torch.cat([v.flatten() for v in g_target.values()]),
                         torch.cat([v.flatten() for v in g_ret.values()]))

    per_group = defaultdict(list)
    for n in g_target:
        if n in g_ret and g_target[n].numel() > 1:
            per_group[_group(n)].append(_cosine(g_target[n], g_ret[n]))
    per_group = {k: round(sum(v) / len(v), 4) for k, v in sorted(per_group.items())}

    model.zero_grad(set_to_none=True)
    del model, g_target, g_ret
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    return {"checkpoint": checkpoint,
            "global_cosine_target_vs_retention": round(global_cos, 4),
            "per_group_cosine": per_group,
            "n_batches": n_batches, "batch_size": batch,
            "seconds": round(time.time() - t0, 1)}


def run_layerwise_drift(pairs: list[tuple[str, str]], device: str = "cpu") -> dict:
    """||dW||/||W|| per module group for each (from, to) checkpoint pair.
    lm_head.weight is skipped: it is tied to the embedding and would
    double-count the same tensor in the head/norm group."""
    out = {}
    cached = {}
    for src, dst in pairs:
        if src not in cached:
            payload = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
            cached[src] = payload.get("model_state_dict", payload.get("model", payload))
        if dst not in cached:
            payload = torch.load(dst, map_location="cpu", weights_only=True, mmap=True)
            cached[dst] = payload.get("model_state_dict", payload.get("model", payload))
        a, b = cached[src], cached[dst]
        groups = defaultdict(lambda: [0.0, 0.0])
        for n in a:
            if n == "lm_head.weight":
                continue  # tied to token_embedding_table.weight
            if n not in b or not torch.is_floating_point(a[n]):
                continue
            d = (b[n].float() - a[n].float()).norm().item()
            w = a[n].float().norm().item()
            g = _group(n)
            groups[g][0] += d * d
            groups[g][1] += w * w
        out[f"{Path(src).stem}->{Path(dst).stem}"] = {
            g: round((dv ** 0.5) / max(ww ** 0.5, 1e-9), 5)
            for g, (dv, ww) in sorted(groups.items())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",
                        default="checkpoints/anra-v4-20k-sft-context-binding.pt")
    parser.add_argument("--bank", default="data/capability_bank/train.jsonl")
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()
    report = {"schema": "anra-grad-conflict/v1",
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    report["gradient_conflict"] = run_gradient_conflict(
        args.checkpoint, args.bank, "cuda")
    print(json.dumps(report["gradient_conflict"], indent=2))
    if args.drift:
        report["layerwise_drift"] = run_layerwise_drift([
            (r"C:/Users/ankit/Downloads/anra-v4-current-full-resume.pt",
             "checkpoints/anra-v4-20k-sft-context-binding.pt"),
            ("checkpoints/anra-v4-20k-sft-context-binding.pt",
             "checkpoints/anra-v4-20k-sft2-selective.pt")])
        print(json.dumps(report["layerwise_drift"], indent=2))
    Path("output/grad_conflict.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
