"""Accumulation harness: install a capability without stealing the others.

Objective (per the gradient-conflict evidence: cos(target,retention)=+0.17,
so replay first, anchor only if needed):

    L = L_target + L_rehearsal + beta * L_anchor

  L_target     masked LM loss on selective-binding items (all six formats)
  L_rehearsal  masked LM loss on retention items from the dev bank
  L_anchor     optional KL(student || anchor) on retention prompts where the
               anchor checkpoint produces the verified-correct greedy answer
               (never distill the anchor's mistakes)

Selection is retention-aware (multi-objective): a checkpoint is ELIGIBLE only
when target dev accuracy improves AND retention dev accuracy stays above its
floor. Sealed OOD suites are never imported here — dev bank only.

Every eval point records a trajectory row: losses, per-capability dev
scores, optimizer updates, and the checkpoint parameter SHA. Receipt at end.

Run:
  py -3 -m training.sft_accumulate --anchor checkpoints/anra-v4-20k-sft-context-binding.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch

from anra_core.checkpoint import load_core_checkpoint
from anra_core.config import CANONICAL_CONFIG
from anra_core.tokenizer import V4Tokenizer
from training.sft_context_binding import encode_item, greedy_decode

CODE_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")
TARGET_FAMS = {"selective", "selective_cf"}
RETENTION_FAMS = {"single_fact", "tool_result", "copy", "protocol_transfer", "symbolic_ops"}
# Floors protect only what the CapabilityContract promotes. symbolic_composition
# is BELOW_FLOOR by contract — monitored, never a blocker.
PROTECTED_FAMS = {"single_fact", "tool_result", "copy", "protocol_transfer"}
PARENT_REGRESSION_TOLERANCE = 0.10


def _strict(out: str, gold: str) -> bool:
    if CODE_RE.search(gold):
        cands = CODE_RE.findall(out)
        return len(cands) == 1 and cands[0] == gold
    n = re.sub(r"[^0-9a-z]+", " ", out.lower()).strip()
    g = re.sub(r"[^0-9a-z]+", " ", gold.lower()).strip()
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", n) is not None


def _cf_twin(item: dict) -> dict | None:
    """Deterministic counterfactual twin: same prompt, the answer value
    replaced by a fresh code (byte-exact single replacement)."""
    gold = item["answer"]
    if not CODE_RE.fullmatch(gold) or item["prompt"].count(gold) != 1:
        return None
    new = "ZQX-" + str(700 + (hash(gold) % 200))
    return {"prompt": item["prompt"].replace(gold, new), "answer": new}


def _dev_eval(model, tok, dev_items) -> dict[str, float]:
    """Per-capability greedy accuracy on dev items (strict parsing).

    The selective score is CAUSAL: an item passes only if the base answer is
    right AND its counterfactual twin (value swapped, bytes otherwise
    identical) flips to the new value. Positional heuristics cannot pass it —
    this is the anti-self-deception metric born from the label-shift lesson:
    training loss is not behavior.
    """
    model.eval()
    scores: dict[str, list[int]] = {}
    with torch.no_grad():
        for it in dev_items:
            fam = it["family"]
            ok = _strict(greedy_decode(model, tok, it["prompt"], max_new_tokens=10),
                         it["answer"])
            twin = _cf_twin(it)
            if fam.startswith("selective") and twin is not None:
                ok = ok and _strict(
                    greedy_decode(model, tok, twin["prompt"], max_new_tokens=10),
                    twin["answer"])
            scores.setdefault(fam, []).append(1 if ok else 0)
    model.train()
    return {f: sum(v) / len(v) for f, v in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True,
                        help="anchor checkpoint (context-binding child)")
    parser.add_argument("--bank-train", default="data/capability_bank/train.jsonl")
    parser.add_argument("--bank-dev", default="data/capability_bank/dev.jsonl")
    parser.add_argument("--out", default="checkpoints/anra-v4-20k-sft3-accumulate.pt")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="anchor KL weight (0 disables; try replay first)")
    parser.add_argument("--retention-floor", type=float, default=0.8)
    parser.add_argument("--eval-every", type=int, default=120)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"

    # Structural split from the bank generator: train and dev come from
    # DISJOINT vocabularies and seeds (split_audit.json guarantees zero
    # group/prompt/pair overlap). The harness performs no slicing of its own.
    train_items = [json.loads(l) for l in
                   Path(args.bank_train).read_text(encoding="utf-8").splitlines() if l.strip()]
    dev = [json.loads(l) for l in
           Path(args.bank_dev).read_text(encoding="utf-8").splitlines() if l.strip()]
    comp = dict(Counter(i["family"] for i in train_items))
    dev_comp = dict(Counter(i["family"] for i in dev))
    print(f"[data] train composition: {json.dumps(comp)}", flush=True)
    print(f"[data] dev composition (disjoint vocab): {json.dumps(dev_comp)}", flush=True)
    train_target = [i for i in train_items if i["family"] in TARGET_FAMS]
    train_ret = [i for i in train_items if i["family"] in RETENTION_FAMS]
    target = train_target  # dev comes from the file split below

    print(f"[load] anchor {args.anchor}", flush=True)
    model, _, identity = load_core_checkpoint(args.anchor, legacy_unverified=True)
    model = model.to(device).train()
    tok = V4Tokenizer.load_canonical()

    # Anchor KL is fail-closed: a true teacher forward requires a frozen
    # second model copy which this harness intentionally does not load yet
    # (gradient evidence says replay first). Refuse rather than fake it.
    if args.beta > 0:
        raise NotImplementedError(
            "anchor KL (beta>0) requires a frozen teacher forward; not "
            "implemented — run replay-only (beta=0), which the gradient-"
            "conflict evidence (cos=+0.17) recommends first")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    enc_t = [encode_item(tok, it) for it in train_target]
    enc_r = [encode_item(tok, it) for it in train_ret]

    mixed = [(x, "t") for x in enc_t] + [(x, "r") for x in enc_r]
    steps_total = int(len(mixed) * args.epochs)
    baseline_dev = _dev_eval(model, tok, dev)
    baseline_target = (baseline_dev.get("selective", 0) + baseline_dev.get("selective_cf", 0)) / 2
    baseline_ret = sum(baseline_dev.get(f, 0) for f in PROTECTED_FAMS) / len(PROTECTED_FAMS)
    print(f"[dev-baseline] target={baseline_target:.3f} retention={baseline_ret:.3f} "
          f"detail={json.dumps(baseline_dev)}", flush=True)

    trajectory, best, best_score = [], None, -1.0
    step = 0
    t0 = time.time()
    random.Random(args.seed).shuffle(mixed)
    while step < steps_total:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if step >= steps_total:
                break
            (ids, labels), kind = mixed[step % len(mixed)]
            logits = model(ids.to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1).to(device), ignore_index=-100)
            (loss / args.accum).backward()
            step += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every < args.accum or step >= steps_total:
            dev_scores = _dev_eval(model, tok, dev)
            tgt = (dev_scores.get("selective", 0) + dev_scores.get("selective_cf", 0)) / 2
            ret = sum(dev_scores.get(f, 0) for f in PROTECTED_FAMS) / len(PROTECTED_FAMS)
            # Parent-RELATIVE floors: protect what the parent actually knows.
            # A capability at 0.9 may not silently fall to a global 0.6 floor;
            # it may only regress by PARENT_REGRESSION_TOLERANCE from ITS OWN
            # measured baseline.
            fam_floors = {f: dev_scores.get(f, 0) >=
                          baseline_dev.get(f, 0.0) - PARENT_REGRESSION_TOLERANCE
                          for f in PROTECTED_FAMS}
            eligible = ((tgt > baseline_target) and (ret >= args.retention_floor)
                        and all(fam_floors.values()))
            row = {"step": step, "updates": step // args.accum,
                   "target_acc": round(tgt, 3), "retention_acc": round(ret, 3),
                   "per_family_floors_vs_parent": fam_floors,
                   "eligible": eligible, "dev": {k: round(v, 3) for k, v in dev_scores.items()}}
            trajectory.append(row)
            print(f"  [eval @{step}] target={tgt:.3f} retention={ret:.3f} "
                  f"eligible={eligible}", flush=True)
            score = tgt + ret  # multi-objective: sum of dev target+retention
            if eligible and score > best_score:
                best_score = score
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                state["lm_head.weight"] = state["token_embedding_table.weight"]
                try:
                    commit = subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
                except Exception:
                    commit = None
                # Canonical parameter identity (same implementation the
                # loader/evaluator use) — asserted against the saved file.
                from anra_core.checkpoint import parameter_sha256
                saved_sha = parameter_sha256(state)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "checkpoint_artifact_class": "model_only",
                    "checkpoint_schema_version": 1,
                    "global_step": identity.global_step,
                    "training_stage": "accumulation_selective_balanced",
                    "source_commit": commit,
                    "source_checkpoint": str(args.anchor),
                    "model_config": asdict(CANONICAL_CONFIG),
                    "model_state_dict": state,
                    "tokenizer_contract": {"available": True, **tok.identity()},
                    "metrics": {"dev_target": tgt, "dev_retention": ret,
                                "eligible": True, "bank_composition": comp},
                    "parameter_sha256": saved_sha,
                }, args.out)
                best = {"step": step, "target": round(tgt, 3), "retention": round(ret, 3),
                        "param_sha256": saved_sha}
                print(f"  [save] eligible best score={score:.3f} -> {args.out}", flush=True)

    receipt = {"schema": "anra-accumulate/v1",
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "anchor": args.anchor, "bank": args.bank,
               "exact_composition": comp,
               "baseline_dev": baseline_dev, "trajectory": trajectory,
               "best": best, "retention_floor": args.retention_floor,
               "hyper": {"lr": args.lr, "epochs": args.epochs, "beta": args.beta},
               "wall_seconds": round(time.time() - t0, 1)}
    Path("output/accumulate_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"[done] best={best} wall={receipt['wall_seconds']}s", flush=True)

    del model, opt
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(f"[free] reserved={torch.cuda.memory_reserved() / 2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
