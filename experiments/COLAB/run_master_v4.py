from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-001"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-007"))

from experiments.lib import ark_tasks as t
from run_ark001 import CompactVocab, Micro, greedy_exact, loss_and_positions
from run_007_v2 import detect_g90, generate_continuation_indices, order_sha256

PLAN_SHA = "3d98103cacc38177390df78ee0eff402da687fcf"
CANONICAL_T2_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
RESULTS_DIR = Path("/content/arkenstone_v4_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BUDGET_MINUTES = 240
SESSION_START = time.time()


def minutes_left():
    return BUDGET_MINUTES - (time.time() - SESSION_START) / 60.0


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. In Colab choose Runtime -> Change runtime type -> T4 GPU.")
    device = torch.device("cuda")
    print("DEVICE:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    return device


DEVICE = require_cuda()


def sync():
    torch.cuda.synchronize()


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def receipt_hash(payload):
    body = dict(payload)
    body.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def save_json(name, payload):
    payload = dict(payload)
    payload["plan_commit_sha"] = PLAN_SHA
    payload["device"] = str(DEVICE)
    payload["torch"] = torch.__version__
    payload["receipt_sha256"] = receipt_hash(payload)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return path


def snapshot_state(model, optimizer):
    return {
        "model": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().cpu(),
        "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()],
    }


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def retention_metrics(traj):
    if not traj:
        return {"status": "EMPTY"}
    vals = [x["test_exact"] for x in traj]
    streak = 0
    collapse = None
    for x in traj:
        if x["test_exact"] < 0.90:
            streak += 1
            if streak >= 3:
                collapse = x["step"]
                break
        else:
            streak = 0
    return {
        "RET90": sum(v >= 0.90 for v in vals) / len(vals),
        "RET50": sum(v >= 0.50 for v in vals) / len(vals),
        "AREA": sum(vals) / len(vals),
        "FINAL": vals[-1],
        "PEAK": max(vals),
        "T_COLLAPSE_90": collapse,
        "collapsed": collapse is not None,
    }


def train_t2_to_g90(seed, manifest, max_steps=28000, batch=64, lr=1e-3, eval_every=200):
    torch.manual_seed(seed)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    rng = torch.Generator().manual_seed(seed)
    eval_steps, eval_ood, traj = [], [], []
    supervised_tokens = 0
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        rows = [train[int(i)] for i in idx]
        loss, count = loss_and_positions(model, vocab, rows, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())
        if step % eval_every == 0 or step == 1:
            tr, _ = greedy_exact(model, vocab, train[:100], DEVICE)
            te, _ = greedy_exact(model, vocab, test, DEVICE)
            traj.append({"step": step, "train_exact": tr, "test_exact": te, "loss": float(loss.detach())})
            eval_steps.append(step)
            eval_ood.append(te)
            onset, confirm = detect_g90(eval_steps, eval_ood)
            print(f"[ARK-007R acquire s{seed}] step={step} train={tr:.3f} test={te:.3f}", flush=True)
            if confirm is not None:
                sync()
                return {
                    "seed": seed,
                    "model": model,
                    "optimizer": optimizer,
                    "snapshot": snapshot_state(model, optimizer),
                    "g90_flat": flat_params(model).detach().cpu(),
                    "g90_onset_step": onset,
                    "g90_confirmation_step": confirm,
                    "first_treated_optimizer_step": confirm + 1,
                    "acquisition_steps": step,
                    "acquisition_supervised_tokens": supervised_tokens,
                    "trajectory": traj,
                    "train": train,
                    "test": test,
                }
    return None


def load_fork(snapshot, lr):
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in snapshot["model"].items()})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    optimizer.load_state_dict(snapshot["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    torch.set_rng_state(snapshot["torch_rng"])
    torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
    return vocab, model, optimizer


def run_continuation(snapshot, g90_flat, indices, train, test, lr, steps=6000, eval_every=200, capture_collapse=False):
    vocab, model, optimizer = load_fork(snapshot, lr)
    g90_flat = g90_flat.to(DEVICE)
    traj = []
    supervised_tokens = 0
    collapse_snapshot = None
    collapse_step = None
    below_streak = 0
    for step in range(1, steps + 1):
        rows = [train[i] for i in indices[step - 1]]
        loss, count = loss_and_positions(model, vocab, rows, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())
        if step % eval_every == 0:
            te, _ = greedy_exact(model, vocab, test, DEVICE)
            disp = (flat_params(model) - g90_flat).norm().item()
            rel = disp / max(g90_flat.norm().item(), 1e-12)
            traj.append({"step": step, "test_exact": te, "l2_displacement": disp, "relative_displacement": rel})
            if te < 0.90:
                below_streak += 1
                if capture_collapse and below_streak >= 3 and collapse_snapshot is None:
                    sync()
                    collapse_snapshot = snapshot_state(model, optimizer)
                    collapse_step = step
            else:
                below_streak = 0
    return {
        "trajectory": traj,
        "retention": retention_metrics(traj),
        "supervised_tokens": supervised_tokens,
        "collapse_snapshot": collapse_snapshot,
        "collapse_confirmation_step": collapse_step,
    }


def campaign_a_ark007r():
    print("\n=== CAMPAIGN A: ARK-007R fresh-checkpoint retention replication ===", flush=True)
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"))
    assert manifest["split_sha256"] == CANONICAL_T2_SHA
    acq_seeds = [909, 1010, 1111]
    cont_seeds = [2701, 2702, 2703, 2704]
    full_len = 10000
    all_rows = []
    collapse_sources = []
    acquired_meta = []
    for seed in acq_seeds:
        if minutes_left() < 80:
            print("budget gate: stopping Campaign A before new acquisition", flush=True)
            break
        acq = train_t2_to_g90(seed, manifest)
        if acq is None:
            acquired_meta.append({"seed": seed, "status": "NO_G90"})
            continue
        acquired_meta.append({
            "seed": seed,
            "status": "ACQUIRED",
            "g90_onset_step": acq["g90_onset_step"],
            "g90_confirmation_step": acq["g90_confirmation_step"],
            "acquisition_steps": acq["acquisition_steps"],
            "acquisition_supervised_tokens": acq["acquisition_supervised_tokens"],
        })
        for order_seed in cont_seeds:
            indices = generate_continuation_indices(order_seed, full_len, 64, len(acq["train"]))
            ohash = order_sha256(indices)
            high = run_continuation(
                acq["snapshot"], acq["g90_flat"], indices, acq["train"], acq["test"],
                1e-3, steps=6000, capture_collapse=True
            )
            low = run_continuation(
                acq["snapshot"], acq["g90_flat"], indices, acq["train"], acq["test"],
                1e-5, steps=6000, capture_collapse=False
            )
            for arm, lr, out in [("HIGH", 1e-3, high), ("LOW", 1e-5, low)]:
                all_rows.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "continuation_order_sha256": ohash,
                    "arm": arm,
                    "lr": lr,
                    "g90_onset_step": acq["g90_onset_step"],
                    "g90_confirmation_step": acq["g90_confirmation_step"],
                    "first_treated_optimizer_step": acq["first_treated_optimizer_step"],
                    "retention": out["retention"],
                    "supervised_tokens": out["supervised_tokens"],
                    "trajectory": out["trajectory"],
                })
            if high["collapse_snapshot"] is not None:
                collapse_sources.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "continuation_order_sha256": ohash,
                    "collapse_confirmation_step": high["collapse_confirmation_step"],
                    "snapshot": high["collapse_snapshot"],
                    "g90_flat": acq["g90_flat"],
                    "indices": indices,
                    "train": acq["train"],
                    "test": acq["test"],
                })
            print(
                f"s{seed} order{order_seed}: HIGH collapse={high['retention']['collapsed']} "
                f"LOW collapse={low['retention']['collapsed']}",
                flush=True,
            )
        save_json("ARK-007R_PARTIAL.json", {"acquisitions": acquired_meta, "results": all_rows})
    paired = []
    for seed in acq_seeds:
        for order_seed in cont_seeds:
            h = next((r for r in all_rows if r["acquisition_seed"] == seed and r["continuation_seed"] == order_seed and r["arm"] == "HIGH"), None)
            l = next((r for r in all_rows if r["acquisition_seed"] == seed and r["continuation_seed"] == order_seed and r["arm"] == "LOW"), None)
            if h and l:
                paired.append((h["retention"]["collapsed"], l["retention"]["collapsed"]))
    high_n = sum(h for h, _ in paired)
    low_n = sum(l for _, l in paired)
    n = len(paired)
    risk_diff = (low_n - high_n) / n if n else None
    discordant_hl = sum(h and not l for h, l in paired)
    reverse = sum((not h) and l for h, l in paired)
    payload = {
        "experiment_id": "ARK-007R",
        "manifest_sha256": CANONICAL_T2_SHA,
        "acquisitions": acquired_meta,
        "results": all_rows,
        "summary": {
            "paired_conditions": n,
            "high_collapses": high_n,
            "low_collapses": low_n,
            "risk_difference_low_minus_high": risk_diff,
            "high_collapse_low_stable": discordant_hl,
            "high_stable_low_collapse": reverse,
        },
    }
    save_json("ARK-007R_RESULT.json", payload)
    return collapse_sources, payload


def binding_candidates():
    import itertools
    keys = list(range(6))
    vals = list(range(6))
    buckets = {(q, ans): [] for q in keys for ans in vals}
    for key_tuple in itertools.combinations(keys, 3):
        for val_tuple in itertools.permutations(vals, 3):
            facts = tuple(zip(key_tuple, val_tuple))
            for qi, q in enumerate(key_tuple):
                ans = val_tuple[qi]
                buckets[(q, ans)].append((facts, q, ans))
    rng = random.Random(4242)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    train, test = [], []
    for q in keys:
        for ans in vals:
            take_train = 34 if ans in (q, (q + 1) % 6) else 33
            take_test = 9 if ans in (q, (q + 1) % 6) else 8
            items = buckets[(q, ans)]
            train.extend(items[:take_train])
            test.extend(items[take_train:take_train + take_test])
    assert len(train) == 1200 and len(test) == 300
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def binding_row(item, swap=False):
    facts, q, ans = item
    if swap:
        fact_keys = [k for k, _ in facts]
        pos = fact_keys.index(q)
        q = fact_keys[(pos + 1) % len(fact_keys)]
        ans = dict(facts)[q]
    prompt = "+".join(f"{k}={v}" for k, v in facts) + f"/{q}="
    return prompt, str(ans)


def train_binding_to_qualified(seed, max_steps=24000):
    train_items, test_items = binding_candidates()
    train = [binding_row(x) for x in train_items]
    test = [binding_row(x) for x in test_items]
    swap = [binding_row(x, swap=True) for x in test_items]
    vocab = CompactVocab()
    torch.manual_seed(seed)
    model = Micro(vocab.size, 128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(seed)
    streak = 0
    onset = None
    traj = []
    sup_tokens = 0
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (64,), generator=rng)
        rows = [train[int(i)] for i in idx]
        loss, count = loss_and_positions(model, vocab, rows, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sup_tokens += int(count.item())
        if step % 200 == 0 or step == 1:
            te, _ = greedy_exact(model, vocab, test, DEVICE)
            sw, _ = greedy_exact(model, vocab, swap, DEVICE)
            traj.append({"step": step, "test_exact": te, "query_swap_exact": sw})
            print(f"[ARK-009 acquire s{seed}] step={step} test={te:.3f} swap={sw:.3f}", flush=True)
            if te >= 0.90 and sw >= 0.85:
                if streak == 0:
                    onset = step
                streak += 1
                if streak >= 3:
                    return {
                        "seed": seed,
                        "snapshot": snapshot_state(model, optimizer),
                        "g90_flat": flat_params(model).detach().cpu(),
                        "onset": onset,
                        "confirm": step,
                        "train": train,
                        "test": test,
                        "swap": swap,
                        "trajectory": traj,
                        "supervised_tokens": sup_tokens,
                    }
            else:
                streak = 0
                onset = None
    return {"seed": seed, "status": "NO_QUALIFICATION", "trajectory": traj, "supervised_tokens": sup_tokens}


def campaign_b_ark009():
    print("\n=== CAMPAIGN B: ARK-009 non-arithmetic transfer ===", flush=True)
    results = []
    for seed in [1201, 1202]:
        if minutes_left() < 55:
            print("budget gate: skipping remaining ARK-009 acquisition", flush=True)
            break
        acq = train_binding_to_qualified(seed)
        if acq.get("status") == "NO_QUALIFICATION":
            results.append({"seed": seed, "status": "TRANSFER_BLOCKED_BY_ACQUISITION", "trajectory": acq["trajectory"]})
            save_json("ARK-009_PARTIAL.json", {"results": results})
            continue
        seed_result = {
            "seed": seed,
            "status": "QUALIFIED",
            "qualification_onset_step": acq["onset"],
            "qualification_confirmation_step": acq["confirm"],
            "acquisition_supervised_tokens": acq["supervised_tokens"],
            "forks": [],
        }
        for order_seed in range(3701, 3707):
            indices = generate_continuation_indices(order_seed, 6000, 64, len(acq["train"]))
            ohash = order_sha256(indices)
            high = run_continuation(acq["snapshot"], acq["g90_flat"], indices, acq["train"], acq["test"], 1e-3, 6000)
            low = run_continuation(acq["snapshot"], acq["g90_flat"], indices, acq["train"], acq["test"], 1e-5, 6000)
            seed_result["forks"].extend([
                {"order_seed": order_seed, "order_sha256": ohash, "arm": "HIGH", "retention": high["retention"]},
                {"order_seed": order_seed, "order_sha256": ohash, "arm": "LOW", "retention": low["retention"]},
            ])
            print(f"[ARK-009] s{seed} order{order_seed}: HIGH {high['retention']['collapsed']} LOW {low['retention']['collapsed']}", flush=True)
        results.append(seed_result)
        save_json("ARK-009_PARTIAL.json", {"results": results})
    status = "TRANSFER_BLOCKED_BY_ACQUISITION"
    qualified = [r for r in results if r["status"] == "QUALIFIED"]
    if qualified:
        status = "TRANSFER_EXECUTED"
    payload = {"experiment_id": "ARK-009", "status": status, "results": results}
    save_json("ARK-009_RESULT.json", payload)
    return payload


def run_recovery_from_collapse(source):
    start = int(source["collapse_confirmation_step"])
    tail = source["indices"][start:start + 4000]
    if len(tail) < 4000:
        return {"status": "INSUFFICIENT_FROZEN_TAIL"}
    high = run_continuation(source["snapshot"], source["g90_flat"], tail, source["train"], source["test"], 1e-3, 4000)
    low = run_continuation(source["snapshot"], source["g90_flat"], tail, source["train"], source["test"], 1e-5, 4000)

    def recovery90(traj):
        streak = 0
        onset = None
        for x in traj:
            if x["test_exact"] >= 0.90:
                if streak == 0:
                    onset = x["step"]
                streak += 1
                if streak >= 3:
                    return onset
            else:
                streak = 0
                onset = None
        return None

    return {
        "status": "EXECUTED",
        "acquisition_seed": source["acquisition_seed"],
        "continuation_seed": source["continuation_seed"],
        "collapse_confirmation_step": source["collapse_confirmation_step"],
        "HIGH_CONTINUE": {"retention": high["retention"], "recovery90_step": recovery90(high["trajectory"]), "trajectory": high["trajectory"]},
        "RECOVERY_LOW": {"retention": low["retention"], "recovery90_step": recovery90(low["trajectory"]), "trajectory": low["trajectory"]},
    }


def campaign_c_ark010(collapse_sources):
    print("\n=== CAMPAIGN C: ARK-010 recovery after collapse ===", flush=True)
    if len(collapse_sources) < 2:
        payload = {"experiment_id": "ARK-010", "status": "INCONCLUSIVE_LOW_EVENT_RATE", "collapse_sources": len(collapse_sources)}
        save_json("ARK-010_RESULT.json", payload)
        return payload
    rows = []
    for source in collapse_sources:
        if minutes_left() < 12:
            break
        out = run_recovery_from_collapse(source)
        rows.append(out)
        save_json("ARK-010_PARTIAL.json", {"results": rows})
    payload = {"experiment_id": "ARK-010", "status": "EXECUTED" if rows else "BUDGET_BLOCKED", "results": rows}
    save_json("ARK-010_RESULT.json", payload)
    return payload


def package_results():
    import zipfile
    zip_path = RESULTS_DIR / "ARKENSTONE_V4_RESULTS.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(RESULTS_DIR.glob("*.json")):
            zf.write(p, p.name)
    print("RESULT ZIP:", zip_path, flush=True)
    try:
        from google.colab import files
        files.download(str(zip_path))
    except Exception as exc:
        print("auto-download skipped:", exc, flush=True)


def main():
    print("PLAN SHA:", PLAN_SHA, flush=True)
    collapse_sources, a = campaign_a_ark007r()
    b = campaign_b_ark009()
    c = campaign_c_ark010(collapse_sources)
    summary = {
        "experiment_id": "MASTER_GPU_V4",
        "ARK-007R": a.get("summary", {}),
        "ARK-009_status": b.get("status"),
        "ARK-010_status": c.get("status"),
        "minutes_used": (time.time() - SESSION_START) / 60.0,
    }
    save_json("PROGRAM_SUMMARY.json", summary)
    package_results()


if __name__ == "__main__":
    main()
