"""Local context-binding SFT: fine-tune the step-20k artifact on corrective data.

This is the smallest training intervention that attacks the measured deficit
(P1-P6: context-to-answer binding 0/5 at every step). It trains the exact
formats the probes test, with nonce alphabets disjoint from the probe's — so
the P1-P6 battery remains a true held-out test.

Execution:
  - loads the step-20k checkpoint (legacy identity binding) on CUDA;
  - masked cross-entropy: loss on completion + EOS tokens only;
  - AdamW (5e-5, betas 0.9/0.95), grad-accum 8, grad-clip 1.0, 2 epochs;
  - held-out greedy eval every half epoch; best checkpoint is saved in the
    strict `model_only` schema with the canonical tokenizer contract;
  - closes cleanly: model deleted, CUDA cache emptied and verified.

Run:  py -3 -m training.sft_context_binding --checkpoint <pt> --out <pt>
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from anra_core.checkpoint import load_core_checkpoint
from anra_core.config import CANONICAL_CONFIG
from anra_core.tokenizer import V4Tokenizer


def _contains(text: str, gold: str) -> bool:
    norm = lambda s: re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()  # noqa: E731
    return re.search(rf"(?<!\w){re.escape(norm(gold))}(?!\w)", norm(text)) is not None


def load_items(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def encode_item(tok, item):
    """Tokenize one SFT item with correctly shifted labels.

    Causal semantics: logits at position j predict token j+1, so the label
    for position j must be ids[j+1]. Supervision starts at the LAST prompt
    token (which predicts the first completion token — the binding step)
    and ends at the last completion token (which predicts EOS).
    """
    prompt_ids = tok.encode(item["prompt"])
    completion_ids = tok.encode(item["completion"])
    ids = [tok.bos_token_id, *prompt_ids, *completion_ids, tok.eos_token_id]
    labels = [-100] * len(ids)
    last_prompt_index = len(prompt_ids)  # index of the final prompt token
    for j in range(last_prompt_index, len(ids) - 1):
        labels[j] = ids[j + 1]
    return torch.tensor([ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


@torch.no_grad()
def greedy_decode(model, tok, prompt: str, max_new_tokens: int = 12) -> str:
    """No-cache greedy decode for eval (sequences are tiny)."""
    device = next(model.parameters()).device
    ids = [tok.bos_token_id, *tok.encode(prompt)]
    out = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[:, -1, :]
        nxt = int(logits.argmax(dim=-1).item())
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        ids.append(nxt)
    return tok.decode(out)


def evaluate(model, tok, items) -> dict:
    model.eval()
    per: dict[str, list[int]] = {}
    for it in items:
        text = greedy_decode(model, tok, it["prompt"])
        key = f"{it['family']}:{it['protocol']}"
        per.setdefault(key, []).append(1 if _contains(text, it["gold"]) else 0)
    model.train()
    return {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in sorted(per.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=r"C:/Users/ankit/Downloads/anra-v4-current-full-resume.pt")
    parser.add_argument("--data", default="data/sft_context_binding")
    parser.add_argument("--out", default="checkpoints/anra-v4-20k-sft-context-binding.pt")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=470)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for this local SFT run"
    torch.manual_seed(3407)
    device = "cuda"

    train = load_items(f"{args.data}/train.jsonl")
    held = load_items(f"{args.data}/heldout.jsonl")
    print(f"[load] {args.checkpoint}", flush=True)
    model, metadata, identity = load_core_checkpoint(
        args.checkpoint, legacy_unverified=True)
    model = model.to(device)
    model.train()
    tok = V4Tokenizer.load_canonical()
    print(f"[load] step={identity.global_step} params="
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    encoded = [encode_item(tok, it) for it in train]

    steps_total = int(len(encoded) * args.epochs)
    best_acc, history = -1.0, []
    t0 = time.time()
    print(f"[train] {steps_total} micro-steps "
          f"(effective batch {args.accum})", flush=True)

    step = 0
    while step < steps_total:
        opt.zero_grad(set_to_none=True)
        loss_sum, denom = 0.0, 0
        for _ in range(args.accum):
            if step >= steps_total:
                break
            ids, labels = encoded[step % len(encoded)]
            logits = model(ids.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1).to(device), ignore_index=-100)
            (loss / args.accum).backward()
            loss_sum += float(loss.detach())
            denom += 1
            step += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 < args.accum:
            print(f"  step {step}/{steps_total} loss={loss_sum / max(denom, 1):.4f} "
                  f"({step / (time.time() - t0):.1f} it/s)", flush=True)
        if step % args.eval_every < args.accum or step >= steps_total:
            report = evaluate(model, tok, held)
            acc = sum(v["acc"] * v["n"] for v in report.values()) / len(held)
            history.append({"step": step, "heldout_acc": acc, "detail": report})
            print(f"  [eval @ {step}] heldout_acc={acc:.3f} "
                  + " ".join(f"{k}={v['acc']:.2f}" for k, v in report.items()), flush=True)
            if acc >= best_acc:
                best_acc = acc
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                state["lm_head.weight"] = state["token_embedding_table.weight"]
                try:
                    commit = subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
                except Exception:
                    commit = None
                torch.save({
                    "checkpoint_artifact_class": "model_only",
                    "checkpoint_schema_version": 1,
                    "global_step": identity.global_step,
                    "training_stage": "context_binding_sft",
                    "source_commit": commit,
                    "source_checkpoint": str(args.checkpoint),
                    "model_config": asdict(CANONICAL_CONFIG),
                    "model_state_dict": state,
                    "tokenizer_contract": {"available": True, **tok.identity()},
                    "metrics": {"heldout_acc": acc, "sft_micro_steps": step,
                                "base_global_step": identity.global_step},
                }, args.out)
                print(f"  [save] best={acc:.3f} -> {args.out}", flush=True)

    print(f"[done] best heldout acc={best_acc:.3f} "
          f"wall={time.time() - t0:.0f}s history={json.dumps(history[-1]['detail'])}", flush=True)

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    print(f"[free] reserved={torch.cuda.memory_reserved() / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
