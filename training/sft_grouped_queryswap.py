"""Group-unit SFT trainer: the query-swap GROUP is the optimization unit.

This is the clean replication of tp-grouped-queryswap-001 (proposal v2:
tp-grouped-queryswap-replication-002). Differences from the row-level
harness that produced sft5-queryswap (P2):

  OPTIMIZATION UNIT  one optimizer micro-batch slot consumes ONE ENTIRE
                     GROUP: every member's masked LM loss is computed and
                     averaged, so the gradient from a target unit sees
                     SAME CONTEXT / DIFFERENT QUERY / DIFFERENT VALUE.

  MIX (P13)          L = alpha * L_group + (1 - alpha) * L_replay with
                     alpha = 0.58 at the UNIT level. Replay units are
                     single rows. Receipt records BOTH unit-level alpha
                     and row-level percentages — they are different
                     numbers and neither silently decides the mix.

  REPLAY BALANCE     each protected family contributes equally per epoch
                     (capped at its smallest pool), symbolic monitored.

  TRAJECTORY (P6)    dev evaluation at updates 0, 5, 10, 20, 30, 40, 50,
                     then every 10 to the end. Optimizer-update counted,
                     never micro-steps. Every point records parameter SHA.
                     Selection uses DEVELOPMENT metrics only.

  GATES              parent-relative protected-family floors + context-
                     value-extraction floor; a checkpoint is eligible only
                     if all gates hold AND dev group-heldout accuracy
                     improved over update 0.
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
from connector.experiments.context_value_extraction import (
    EXTRACTION_TOLERANCE, evaluate as extraction_eval,
    extraction_floor_ok)
from connector.experiments.grouped_queryswap import ALPHA_GROUP_LOSS
from connector.experiments.query_influence_v3 import (
    _completion_logprob as _v3_completion_logprob)
from training.sft_context_binding import encode_item, greedy_decode

REPLAY_FAMS = ("single_fact", "tool_result", "copy",
               "protocol_transfer", "symbolic_ops")
PROTECTED_FAMS = ("single_fact", "tool_result", "copy", "protocol_transfer")
PARENT_REGRESSION_TOLERANCE = 0.10   # same tolerance as prior lineage
TRAJECTORY_UPDATES = [5, 10, 20, 30, 40, 50]  # preregistered (P6); 0 implicit


def _contains(text: str, gold: str) -> bool:
    import re
    norm = lambda s: re.sub(r"[^0-9a-z]+", " ", s.lower()).strip()  # noqa: E731
    return re.search(rf"(?<!\w){re.escape(norm(gold))}(?!\w)", norm(text)) is not None


def evaluate_heldout(model, tok, held_rows) -> tuple[float, dict]:
    """Group-heldout greedy accuracy: per-row contains-gold, aggregated."""
    model.eval()
    per: dict[str, list[int]] = {}
    for it in held_rows:
        text = greedy_decode(model, tok, it["prompt"])
        key = f"{it['family']}:{it['protocol']}"
        per.setdefault(key, []).append(1 if _contains(text, it["gold"]) else 0)
    model.train()
    report = {k: {"acc": sum(v) / len(v), "n": len(v)}
              for k, v in sorted(per.items())}
    total = sum(v["n"] for v in report.values())
    acc = sum(v["acc"] * v["n"] for v in report.values()) / max(total, 1)
    return acc, report


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
        "training_stage": f"grouped_queryswap_replication_{tag}",
        "source_commit": commit,
        "source_checkpoint": str(args.parent),
        "model_config": asdict(CANONICAL_CONFIG),
        "model_state_dict": state,
        "tokenizer_contract": {"available": True, **tok.identity()},
        "metrics": {**metrics,
                    "alpha_group_loss": args.alpha,
                    "candidate_label":
                        "FALLBACK_LAST_POINT" if str(tag).startswith("fallback")
                        else "GATED_DEVELOPMENT_BEST"},
        "parameter_sha256": sha,
    }, out_path)
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="checkpoints/anra-v4-20k-sft3-accumulate.pt")
    parser.add_argument("--data", default="data/grouped_queryswap")
    parser.add_argument("--out", default="checkpoints/anra-v4-20k-sft6-queryswap-replication.pt")
    parser.add_argument("--alpha", type=float, default=ALPHA_GROUP_LOSS)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--accum", type=int, default=8,
                        help="micro-units per optimizer update")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--receipt", default="output/replication_receipt.json")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for this local SFT run"
    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows = [json.loads(l) for l in
                  Path(args.data, "train.jsonl").read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    held_rows = [json.loads(l) for l in
                 Path(args.data, "heldout.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    audit = json.loads(
        Path(args.data, "split_audit.json").read_text(encoding="utf-8"))
    assert audit["group_overlap"] == 0 and audit["prompt_overlap"] == 0, \
        "split audit fails; refusing to train"

    # ---- build UNITS ------------------------------------------------------
    # NOTE: bank rows also carry a group_id (bank-level twin grouping), so
    # target units are identified by FAMILY, not by the mere presence of
    # a group_id field.
    groups: dict[str, list[dict]] = {}
    replay_rows: list[dict] = []
    for r in train_rows:
        if r.get("family") == "queryswap_group":
            groups.setdefault(r["group_id"], []).append(r)
        else:
            replay_rows.append(r)
    assert len(groups) >= 40, "unexpectedly few target groups in train data"
    assert replay_rows, "no replay rows found in train data"

    # balance replay across families per epoch (P14): cap at smallest family
    by_fam: dict[str, list[dict]] = {}
    for r in replay_rows:
        by_fam.setdefault(r["family"], []).append(r)
    floor = min(len(v) for v in by_fam.values())
    balanced_replay = [r for fam in REPLAY_FAMS for r in by_fam[fam][:floor]]
    dropped = len(replay_rows) - len(balanced_replay)

    # ---- load model -------------------------------------------------------
    print(f"[load] parent {args.parent}", flush=True)
    model, _, identity = load_core_checkpoint(args.parent, legacy_unverified=True)
    model = model.to(device).train()
    tok = V4Tokenizer.load_canonical()
    parent_param_sha = getattr(identity, "parameter_sha256", None)
    print(f"[load] step={identity.global_step} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    # ---- baselines (update 0) --------------------------------------------
    base_held_acc, base_report = evaluate_heldout(model, tok, held_rows)
    base_ext = extraction_eval(model, tok)
    baseline_dev = {}
    bank_dev = [json.loads(l) for l in
                Path("data/capability_bank/dev.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    from training.sft_accumulate import _strict as _acc_strict
    for fam in PROTECTED_FAMS:
        rows = [b for b in bank_dev if b["family"] == fam][:20]
        hits = sum(1 for b in rows
                   if _acc_strict(greedy_decode(model, tok, b["prompt"], 10),
                                  b.get("gold") or b.get("answer", "")))
        baseline_dev[fam] = hits / len(rows)
    print(f"[baseline @update0] heldout={base_held_acc:.3f} "
          f"extraction={base_ext['passed']} families={baseline_dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    # ---- schedule: one UNIT per micro-slot, alpha-weighted mix (P13) -------
    # The unit-level objective is  L = alpha * L_group + (1-alpha) * L_replay.
    # Realized stochastically: each micro-slot draws kind ~ Bernoulli(alpha)
    # (seeded), then pops the next group / replay unit from that kind's
    # shuffled epoch queue. Expected gradient weight matches alpha exactly;
    # realized counts are recorded in the receipt.
    n_updates = args.updates
    total_micro = n_updates * args.accum
    mix_rng = random.Random(args.seed + 999)

    def make_queue(kind: int, epoch: int):
        rng = random.Random(args.seed + 31 * epoch + kind)
        if kind == 0:
            gids = sorted(groups)
            rng.shuffle(gids)
            return [("g", groups[g]) for g in gids]
        rep = list(balanced_replay)
        rng.shuffle(rep)
        return [("r", [x]) for x in rep]

    queues = {0: [], 1: []}
    epochs_served = {0: -1, 1: -1}
    realized = {"group_units": 0, "replay_units": 0}

    def next_unit():
        nonlocal queues, epochs_served
        kind = 0 if mix_rng.random() < args.alpha else 1
        if not queues[kind]:
            epochs_served[kind] += 1
            queues[kind] = make_queue(kind, epochs_served[kind])
            if not queues[kind]:      # empty pool safety
                raise RuntimeError(f"empty unit queue for kind={kind}")
        realized["group_units" if kind == 0 else "replay_units"] += 1
        return queues[kind].pop()

    def run_unit(unit):
        """Mean masked-LM loss over ALL members of one unit."""
        losses = []
        for item in unit[1]:
            ids, labels = encode_item(tok, item)
            logits = model(ids.to(device))
            losses.append(torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1).to(device), ignore_index=-100))
        return torch.stack(losses).mean()

    trajectory = []
    eval_points = sorted(set(TRAJECTORY_UPDATES +
                             list(range(0, n_updates + 1, 10))))
    best = None
    best_score = -1.0
    t0 = time.time()
    micro = 0
    update = 0

    def record(update_idx, held_acc, held_rep, ext, fams_dev):
        nonlocal best, best_score
        # P6: live QIM-v2 group-lift at every trajectory point (development
        # instrument; cheap: 10 groups x 3 queries x 3 candidate scorings).
        from connector.experiments.query_influence import build_groups, _prompt, _query
        qim_groups = build_groups()
        model.eval()
        per_group_lift = []
        with torch.no_grad():
            for g in qim_groups:
                recs = g["displayed_facts"]
                block = "\n".join(r["line"] for r in recs)
                lifts = []
                for qi in range(len(recs)):
                    prompt = _prompt(block, _query(recs[qi]))
                    Lq = [_v3_completion_logprob(model, tok, prompt,
                                                 f" {r['code']}.") for r in recs]
                    others = [Lq[j] for j in range(len(recs)) if j != qi]
                    lifts.append(Lq[qi] - sum(others) / len(others))
                per_group_lift.append(sum(lifts) / len(lifts))
        model.train()
        mean_group_lift = sum(per_group_lift) / len(per_group_lift)
        floors = {f: fams_dev.get(f, 0.0) >= baseline_dev.get(f, 0.0) - PARENT_REGRESSION_TOLERANCE
                  for f in PROTECTED_FAMS}
        ext_ok = extraction_floor_ok(base_ext["fraction"], ext["fraction"])
        eligible = (held_acc > base_held_acc) and all(floors.values()) and ext_ok
        score = held_acc + sum(fams_dev.get(f, 0) for f in PROTECTED_FAMS) / len(PROTECTED_FAMS)
        param_note = None
        fallback_note = None
        if eligible and score > best_score:
            best_score = score
            param_note = _save_checkpoint(
                model, identity, tok, args, f"u{update_idx}",
                {"dev_heldout_acc": round(held_acc, 4),
                 "extraction": ext["passed"], "families": fams_dev,
                 "qim2_mean_group_lift": round(mean_group_lift, 4),
                 "optimizer_update": update_idx})
            best = {"update": update_idx, "param_sha256": param_note,
                    "dev_heldout_acc": round(held_acc, 4),
                    "extraction": ext["passed"],
                    "qim2_mean_group_lift": round(mean_group_lift, 4)}
            print(f"  [save] gated-eligible candidate score={score:.3f} "
                  f"-> {args.out}", flush=True)
        # Labeled FALLBACK candidate: refreshed unconditionally at the LAST
        # trajectory point. If every gated save fails the noisy heldout gate,
        # the final state is still frozen + hashed for the single QIM-v3 shot
        # instead of leaving the run with nothing to replicate. It never
        # overrides a gated save; the receipt records which one was used.
        if update_idx == n_updates:
            fallback_note = _save_checkpoint(
                model, identity, tok, args, f"fallback_u{update_idx}",
                {"optimizer_update": update_idx,
                 "candidate_label": "FALLBACK_LAST_POINT",
                 "reason": "final-state candidate if no gated-eligible "
                           "checkpoint exists"})
            print(f"  [save] fallback candidate @u{update_idx} "
                  f"sha={fallback_note[:12]}", flush=True)
        trajectory.append({
            "optimizer_update": update_idx,
            "dev_heldout_acc": round(held_acc, 4),
            "dev_detail": held_rep,
            "qim_v2_mean_group_lift": round(mean_group_lift, 4),
            "context_value_extraction": ext["passed"],
            "protected_families": fams_dev,
            "floors_vs_parent": floors,
            "extraction_floor_ok": ext_ok,
            "eligible": bool(eligible),
            "saved_param_sha256": param_note,
            "fallback_param_sha256": fallback_note,
        })
        print(f"  [eval @upd {update_idx}] heldout={held_acc:.3f} "
              f"lift={mean_group_lift:+.3f} ext={ext['passed']} "
              f"fams={json.dumps(fams_dev)}", flush=True)

    record(0, base_held_acc, base_report, base_ext, baseline_dev)

    while micro < total_micro:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            if micro >= total_micro:
                break
            loss = run_unit(next_unit())
            (loss / args.accum).backward()
            micro += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update += 1
        if update in eval_points or update >= n_updates:
            ha, hr = evaluate_heldout(model, tok, held_rows)
            ex = extraction_eval(model, tok)
            fd = {}
            for fam in PROTECTED_FAMS:
                rows = [b for b in bank_dev if b["family"] == fam][:20]
                fd[fam] = sum(1 for b in rows
                              if _acc_strict(greedy_decode(model, tok, b["prompt"], 10),
                                             b.get("gold") or b.get("answer", ""))) / len(rows)
            record(update, ha, hr, ex, fd)
        elif update % 10 == 0:
            print(f"  upd {update}/{n_updates} "
                  f"({update / max(time.time() - t0, 1e-9):.2f} upd/s)", flush=True)

    # ---- receipt ----------------------------------------------------------
    def file_sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    target_unit_count = len(groups)
    replay_unit_count = len(balanced_replay)
    receipt = {
        "schema": "anra-grouped-queryswap-replication/v1",
        "proposal_id": "tp-grouped-queryswap-replication-002",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parent_checkpoint": args.parent,
        "parent_parameter_sha256": parent_param_sha,
        "data_dir": args.data,
        "train_sha256": file_sha(f"{args.data}/train.jsonl"),
        "heldout_sha256": file_sha(f"{args.data}/heldout.jsonl"),
        "split_audit": {k: audit[k] for k in (
            "n_train_groups", "n_heldout_groups", "group_overlap",
            "prompt_overlap", "full_fact_block_overlap",
            "train_data_sha256", "heldout_data_sha256",
            "split_manifest_sha256")},
        "objective": {
            "unit": "queryswap GROUP (mean member loss per micro-unit)",
            "alpha_group_loss": args.alpha,
            "replay_weight": round(1 - args.alpha, 3),
            "note": ("alpha is the UNIT-level gradient weight (Bernoulli mix "
                     "over micro-slots); row percentages are descriptive only"),
        },
        "units_per_epoch": {"target_groups": target_unit_count,
                            "replay_rows_balanced": replay_unit_count,
                            "replay_dropped_for_balance": dropped},
        "realized_mix": {**realized,
                         "group_unit_share": round(
                             realized["group_units"] /
                             max(sum(realized.values()), 1), 4)},
        "hyper": {"lr": args.lr, "accum": args.accum,
                  "optimizer_updates": n_updates, "seed": args.seed},
        "baseline": {"heldout_acc": round(base_held_acc, 4),
                     "extraction": base_ext["passed"],
                     "protected_families": baseline_dev,
                     "qim_v2_mean_group_lift":
                         trajectory[0]["qim_v2_mean_group_lift"]},
        "trajectory": trajectory,
        "best": best,
        "fallback_candidate": {
            "path": str(Path(args.out).parent /
                        (Path(args.out).stem + "_fallback.pt")) if best is None
            else None,
            "note": "used ONLY if no gated-eligible candidate exists; "
                    "labeled FALLBACK_LAST_POINT inside the checkpoint",
        },
        "selection_policy": "DEVELOPMENT ONLY: dev group-heldout acc > update0 "
                            "AND protected-family floors AND extraction floor; "
                            "QIM-v3 was never consulted before final freeze",
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=2),
                                  encoding="utf-8")
    print(f"[done] best={best} wall={receipt['wall_seconds']}s "
          f"-> {args.receipt}", flush=True)

    del model, opt
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(2)
    print(f"[free] reserved={torch.cuda.memory_reserved() / 2**20:.0f} MiB",
          flush=True)


if __name__ == "__main__":
    main()
