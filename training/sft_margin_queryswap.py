"""Margin SFT: group CE + sampled-competitor hinge (tp-margin-queryswap-003).

Targets the SELECTION regime measured on SFT6 (56/75 failures where gold
was NOT candidate-ranked #1). The hypothesis (preregistered): adding a
margin term that pushes the correct code's full-sequence logprob above the
other codes FROM THE SAME FACT BLOCK converts selection misses without
breaking realization or protection floors.

Objective per target micro-unit:
    L = CE(completion | prompt)                      [realization anchor]
        + margin_weight * max(0, gamma - (lp_gold - max lp_competitor))

where lp_* are FULL-SEQUENCE completion logprobs. The hinge is computed
per candidate pair and summed (not just the argmax competitor) so every
competitor gets gradient pressure, weighted by violation.

Mix with replay is the same explicit Bernoulli unit draw as the replication
(alpha_group_loss), realized counts in receipt. Trajectory, gates, fallback
candidate, and receipt structure mirror training/sft_grouped_queryswap.py.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from anra_core.checkpoint import load_core_checkpoint, parameter_sha256
from anra_core.config import CANONICAL_CONFIG
from anra_core.tokenizer import V4Tokenizer
from connector.experiments.context_value_extraction_v2 import (
    evaluate as cve2_eval, extraction_floor_v2_ok)
from training.sft_context_binding import encode_item, greedy_decode


@torch.no_grad()
def _noop():
    pass


def _diff_completion_logprob(model, tok, prompt: str, completion: str) -> torch.Tensor:
    """DIFFERENTIABLE full-sequence completion logprob (no no_grad).

    The margin hinge needs gradients through gold AND competitor logprobs;
    the evaluators' `_completion_logprob` is @torch.no_grad() and would
    silently detach the hinge (caught by smoke-run gradient inspection).
    """
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion)
    ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]],
                       dtype=torch.long, device=next(model.parameters()).device)
    logits = model(ids)[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    total = None
    for pos in range(1 + len(p_ids), ids.shape[1]):
        term = logprobs[pos - 1, ids[0, pos]]
        total = term if total is None else total + term
    return total if total is not None else torch.tensor(0.0)

REPLAY_FAMS = ("single_fact", "tool_result", "copy",
               "protocol_transfer", "symbolic_ops")
PROTECTED_FAMS = ("single_fact", "tool_result", "copy", "protocol_transfer")
PARENT_REGRESSION_TOLERANCE = 0.10
TRAJECTORY_UPDATES = [5, 10, 20, 30, 40, 50]


def _contains(text: str, gold: str) -> bool:
    import re
    norm = lambda s: re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()  # noqa: E731
    return re.search(rf"(?<!\w){re.escape(norm(gold))}(?!\w)", norm(text)) is not None


def _save_checkpoint(model, identity, tok, args, tag, metrics) -> str:
    out_path = (Path(args.out).parent /
                (Path(args.out).stem + "_fallback.pt")) \
        if str(tag).startswith("fallback") else Path(args.out)
    state = {k: v.detach().cpu().clone()
             for k, v in model.state_dict().items()}
    state["lm_head.weight"] = state["token_embedding_table.weight"]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    sha = parameter_sha256(state)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_artifact_class": "model_only",
        "checkpoint_schema_version": 1,
        "global_step": identity.global_step,
        "training_stage": f"margin_queryswap_{tag}",
        "source_commit": commit,
        "source_checkpoint": str(args.parent),
        "model_config": asdict(CANONICAL_CONFIG),
        "model_state_dict": state,
        "tokenizer_contract": {"available": True, **tok.identity()},
        "metrics": {**metrics,
                    "margin_gamma": args.gamma,
                    "margin_weight": args.margin_weight,
                    "alpha_group_loss": args.alpha,
                    "candidate_label":
                        "FALLBACK_LAST_POINT" if str(tag).startswith("fallback")
                        else "GATED_DEVELOPMENT_BEST"},
        "parameter_sha256": sha,
    }, out_path)
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="checkpoints/anra-v4-20k-sft6-queryswap-replication.pt",
                        help="SFT6 child is the parent: this experiment targets ITS measured selection misses")
    parser.add_argument("--data", default="data/grouped_queryswap")
    parser.add_argument("--out", default="checkpoints/anra-v4-20k-sft7-margin.pt")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="required nats gap between gold and best competitor")
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2801)
    parser.add_argument("--receipt", default="output/margin_receipt.json")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows = [json.loads(l) for l in
                  Path(args.data, "train.jsonl").read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    held_rows = [json.loads(l) for l in
                 Path(args.data, "heldout.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    audit = json.loads(Path(args.data, "split_audit.json").read_text(encoding="utf-8"))
    assert audit["group_overlap"] == 0, "split audit fails"

    groups: dict[str, list[dict]] = {}
    replay_rows: list[dict] = []
    for r in train_rows:
        if r.get("family") == "queryswap_group":
            groups.setdefault(r["group_id"], []).append(r)
        else:
            replay_rows.append(r)
    assert len(groups) >= 40 and replay_rows

    by_fam: dict[str, list[dict]] = {}
    for r in replay_rows:
        by_fam.setdefault(r["family"], []).append(r)
    floor = min(len(v) for v in by_fam.values())
    balanced_replay = [r for fam in REPLAY_FAMS for r in by_fam[fam][:floor]]
    dropped = len(replay_rows) - len(balanced_replay)

    print(f"[load] parent {args.parent}", flush=True)
    model, _, identity = load_core_checkpoint(args.parent, legacy_unverified=True)
    model = model.to(device).train()
    tok = V4Tokenizer.load_canonical()
    parent_param_sha = getattr(identity, "parameter_sha256", None)
    print(f"[load] step={identity.global_step} params="
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)

    # ---- baselines at update 0 ----
    def heldout_eval():
        from training.sft_context_binding import greedy_decode as gd
        model.eval()
        per: dict[str, list[int]] = {}
        for it in held_rows:
            text = gd(model, tok, it["prompt"])
            key = f"{it['family']}:{it['protocol']}"
            per.setdefault(key, []).append(1 if _contains(text, it["gold"]) else 0)
        model.train()
        rep = {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in sorted(per.items())}
        total = sum(v["n"] for v in rep.values())
        return (sum(v["acc"] * v["n"] for v in rep.values()) / max(total, 1)), rep

    base_held_acc, base_report = heldout_eval()
    base_ext = cve2_eval(model, tok)
    bank_dev = [json.loads(l) for l in
                Path("data/capability_bank/dev.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    from training.sft_accumulate import _strict as _acc_strict
    baseline_dev = {}
    for fam in PROTECTED_FAMS:
        rows_ = [b for b in bank_dev if b["family"] == fam][:20]
        baseline_dev[fam] = sum(
            1 for b in rows_
            if _acc_strict(greedy_decode(model, tok, b["prompt"], 10),
                           b.get("gold") or b.get("answer", ""))) / len(rows_)
    print(f"[baseline @upd0] heldout={base_held_acc:.3f} "
          f"cve2={base_ext['passed']} fams={baseline_dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    total_micro = args.updates * args.accum
    mix_rng = random.Random(args.seed + 999)

    queues = {0: [], 1: []}
    epochs_served = {0: -1, 1: -1}
    realized = {"group_units": 0, "replay_units": 0}

    def make_queue(kind: int, epoch: int):
        rng = random.Random(args.seed + 31 * epoch + kind)
        if kind == 0:
            gids = sorted(groups)
            rng.shuffle(gids)
            return [(gid, groups[gid]) for gid in gids]
        rep = list(balanced_replay)
        rng.shuffle(rep)
        return [("replay", [x]) for x in rep]

    def next_unit():
        nonlocal queues, epochs_served
        kind = 0 if mix_rng.random() < args.alpha else 1
        if not queues[kind]:
            epochs_served[kind] += 1
            queues[kind] = make_queue(kind, epochs_served[kind])
        realized["group_units" if kind == 0 else "replay_units"] += 1
        return queues[kind].pop()

    def run_group_unit(gid, members):
        """Group CE (mean member completion loss) + sampled-competitor hinge.

        For each member m with gold value g_m and competitors C_m = other
        codes in the SAME fact block:
            hinge_m = sum_{c in C_m} max(0, gamma - (lp(g_m) - lp(c)))
        L_unit = mean_m CE_m + margin_weight * mean_m hinge_m
        """
        ce_terms = []
        hinge_terms = []
        for item in members:
            ids, labels = encode_item(tok, item)
            logits = model(ids.to(device))
            ce_terms.append(torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1).to(device), ignore_index=-100))
            gold_lp = _diff_completion_logprob(model, tok, item["prompt"],
                                               f" {item['gold']}.")
            hinge_acc = None
            for other in groups[gid]:
                if other["gold"] == item["gold"]:
                    continue
                comp_lp = _diff_completion_logprob(model, tok, item["prompt"],
                                                   f" {other['gold']}.")
                gap = args.gamma - (gold_lp - comp_lp)
                g = torch.clamp(gap, min=0.0)
                hinge_acc = g if hinge_acc is None else hinge_acc + g
            if hinge_acc is not None:
                hinge_terms.append(
                    hinge_acc / max(len(groups[gid]) - 1, 1))
        unit = torch.stack(ce_terms).mean()
        if hinge_terms:
            unit = unit + args.margin_weight * torch.stack(hinge_terms).mean()
        return unit, torch.stack(ce_terms).mean().detach(), (
            torch.stack(hinge_terms).mean().detach() if hinge_terms
            else torch.tensor(0.0))

    def run_replay_unit(members):
        ids, labels = encode_item(tok, members[0])
        logits = model(ids.to(device))
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1).to(device), ignore_index=-100)
        return loss, loss.detach(), torch.tensor(0.0)

    trajectory = []
    eval_points = sorted(set(TRAJECTORY_UPDATES +
                             list(range(0, args.updates + 1, 10))))
    best = None
    best_score = -1.0
    t0 = time.time()
    micro = 0
    update = 0
    loss_trace = []

    def record(update_idx):
        nonlocal best, best_score
        ha, hr = heldout_eval()
        ex = cve2_eval(model, tok)
        fd = {}
        for fam in PROTECTED_FAMS:
            rows_ = [b for b in bank_dev if b["family"] == fam][:20]
            fd[fam] = sum(
                1 for b in rows_
                if _acc_strict(greedy_decode(model, tok, b["prompt"], 10),
                               b.get("gold") or b.get("answer", ""))) / len(rows_)
        floors = {f: fd.get(f, 0.0) >= baseline_dev.get(f, 0.0) - PARENT_REGRESSION_TOLERANCE
                  for f in PROTECTED_FAMS}
        ext_ok = ex["fraction"] >= base_ext["fraction"] - 0.10  # CVE-v2 floor: >= parent - 2.4 items of 24
        eligible = (ha > base_held_acc) and all(floors.values()) and ext_ok
        score = ha + sum(fd.get(f, 0) for f in PROTECTED_FAMS) / len(PROTECTED_FAMS)
        param_note = None
        fallback_note = None
        if eligible and score > best_score:
            best_score = score
            param_note = _save_checkpoint(
                model, identity, tok, args, f"u{update_idx}",
                {"dev_heldout_acc": round(ha, 4),
                 "extraction_cve2": ex["passed"], "families": fd,
                 "optimizer_update": update_idx})
            best = {"update": update_idx, "param_sha256": param_note,
                    "dev_heldout_acc": round(ha, 4),
                    "extraction_cve2": ex["passed"]}
            print(f"  [save] gated candidate score={score:.3f}", flush=True)
        if update_idx == args.updates:
            fallback_note = _save_checkpoint(
                model, identity, tok, args, f"fallback_u{update_idx}",
                {"optimizer_update": update_idx,
                 "candidate_label": "FALLBACK_LAST_POINT"})
        trajectory.append({
            "optimizer_update": update_idx,
            "dev_heldout_acc": round(ha, 4),
            "context_value_extraction_v2": ex["passed"],
            "protected_families": fd,
            "floors_vs_parent": floors,
            "extraction_floor_ok": ext_ok,
            "eligible": bool(eligible),
            "saved_param_sha256": param_note,
            "fallback_param_sha256": fallback_note,
        })
        print(f"  [eval @upd {update_idx}] heldout={ha:.3f} "
              f"cve2={ex['passed']} fams={json.dumps(fd)}", flush=True)

    while micro < total_micro:
        opt.zero_grad(set_to_none=True)
        ce_sum = hinge_sum = 0.0
        for _ in range(args.accum):
            if micro >= total_micro:
                break
            gid, members = next_unit()
            if gid == "replay":
                loss, ce_d, h_d = run_replay_unit(members)
            else:
                loss, ce_d, h_d = run_group_unit(gid, members)
            (loss / args.accum).backward()
            ce_sum += float(ce_d); hinge_sum += float(h_d)
            micro += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update += 1
        loss_trace.append({"update": update,
                           "ce": round(ce_sum / args.accum, 4),
                           "hinge": round(hinge_sum / args.accum, 4)})
        if update % 10 == 0:
            print(f"  upd {update}/{args.updates} ce={ce_sum/args.accum:.3f} "
                  f"hinge={hinge_sum/args.accum:.3f}", flush=True)
        if update in eval_points or update >= args.updates:
            record(update)

    def file_sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    receipt = {
        "schema": "anra-margin-queryswap/v1",
        "proposal_id": "tp-margin-queryswap-003",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parent_checkpoint": args.parent,
        "parent_parameter_sha256": parent_param_sha,
        "objective": {
            "form": "CE + margin_weight * mean clamp(gamma - (lp_gold - lp_comp), 0)",
            "margin_gamma": args.gamma,
            "margin_weight": args.margin_weight,
            "alpha_group_loss": args.alpha,
        },
        "realized_mix": {**realized,
                         "group_unit_share": round(realized["group_units"] /
                                                   max(sum(realized.values()), 1), 4)},
        "hyper": {"lr": args.lr, "accum": args.accum,
                  "optimizer_updates": args.updates, "seed": args.seed},
        "loss_trace_tail": loss_trace[-10:],
        "baseline": {"heldout_acc": round(base_held_acc, 4),
                     "cve2_extraction": base_ext["passed"],
                     "protected_families": baseline_dev},
        "trajectory": trajectory,
        "best": best,
        "selection_policy": "DEVELOPMENT ONLY; QIM-v4 never consulted pre-freeze",
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[done] best={best} -> {args.receipt}", flush=True)

    del model, opt
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)


if __name__ == "__main__":
    main()
