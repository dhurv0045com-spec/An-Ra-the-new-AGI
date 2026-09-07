from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments" / "ARK-001"))

from experiments.lib import ark_tasks as ark_tasks
from run_ark001 import CompactVocab, Micro, greedy_exact, loss_and_positions

BASE_PLAN_SHA = "3d98103cacc38177390df78ee0eff402da687fcf"
PLAN_SHA = "6809bfe8ca70661a4f8b2cd42679db594b944668"
CANONICAL_T2_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
RESULTS_DIR = Path("/content/arkenstone_v5_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BUDGET_MINUTES = 240
SESSION_START = time.time()
DEVICE: torch.device | None = None


def minutes_left() -> float:
    return BUDGET_MINUTES - (time.time() - SESSION_START) / 60.0


def init_cuda() -> torch.device:
    global DEVICE
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. In Colab choose Runtime -> Change runtime type -> T4 GPU."
        )
    DEVICE = torch.device("cuda")
    print("DEVICE:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    return DEVICE


def dev() -> torch.device:
    if DEVICE is None:
        raise RuntimeError("DEVICE not initialized")
    return DEVICE


def sync() -> None:
    if DEVICE is not None and DEVICE.type == "cuda":
        torch.cuda.synchronize()


def sha_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


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


def save_json(name: str, payload: dict) -> Path:
    payload = dict(payload)
    payload["base_plan_commit_sha"] = BASE_PLAN_SHA
    payload["plan_addendum_commit_sha"] = PLAN_SHA
    payload["device"] = str(DEVICE) if DEVICE is not None else "uninitialized"
    payload["torch"] = torch.__version__
    body = dict(payload)
    body.pop("receipt_sha256", None)
    payload["receipt_sha256"] = sha_json(body)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("saved:", path, flush=True)
    return path


def package_results(download: bool = True) -> Path:
    zip_path = RESULTS_DIR / "ARKENSTONE_V5_RESULTS.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(RESULTS_DIR.glob("*.json")):
            zf.write(path, path.name)
    print("RESULT ZIP:", zip_path, flush=True)
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as exc:
            print("auto-download skipped:", repr(exc), flush=True)
    return zip_path


def snapshot_state(model, optimizer) -> dict:
    return {
        "model": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().cpu(),
        "cuda_rng": [x.cpu() for x in torch.cuda.get_rng_state_all()],
    }


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def model_state_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def detect_sustained(values: list[tuple[int, float]], bar: float, consecutive: int = 3, below: bool = False):
    streak = 0
    onset = None
    for step, value in values:
        hit = value < bar if below else value >= bar
        if hit:
            if streak == 0:
                onset = step
            streak += 1
            if streak >= consecutive:
                return onset, step
        else:
            streak = 0
            onset = None
    return None, None


def generate_continuation_indices(order_seed: int, n_batches: int, batch_size: int, pool_size: int):
    rng = torch.Generator().manual_seed(order_seed)
    return [
        torch.randint(0, pool_size, (batch_size,), generator=rng).tolist()
        for _ in range(n_batches)
    ]


def order_sha256(indices) -> str:
    return hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_t2_manifest() -> dict:
    manifest = ark_tasks.load_or_build_manifest(
        str(REPO / "experiments" / "ARK-002B" / "TASK_MANIFEST.json")
    )
    actual = manifest.get("split_sha256")
    if actual != CANONICAL_T2_SHA:
        raise RuntimeError(f"T2 manifest drift: {actual} != {CANONICAL_T2_SHA}")
    return manifest


def retention_metrics(traj: list[dict], key: str = "test_exact", bar: float = 0.90) -> dict:
    if not traj:
        return {"status": "EMPTY"}
    vals = [float(x[key]) for x in traj]
    onset, confirm = detect_sustained([(int(x["step"]), float(x[key])) for x in traj], bar, 3, below=True)
    return {
        "RET90": sum(v >= bar for v in vals) / len(vals),
        "RET50": sum(v >= 0.50 for v in vals) / len(vals),
        "AREA": sum(vals) / len(vals),
        "FINAL": vals[-1],
        "PEAK": max(vals),
        "T_COLLAPSE_90_ONSET": onset,
        "T_COLLAPSE_90_CONFIRM": confirm,
        "collapsed": confirm is not None,
    }


def load_t2_fork(snapshot: dict, lr: float):
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(dev())
    model.load_state_dict({k: v.to(dev()) for k, v in snapshot["model"].items()})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    optimizer.load_state_dict(snapshot["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    torch.set_rng_state(snapshot["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in snapshot["cuda_rng"]])
    return vocab, model, optimizer


def train_t2_to_g90(seed: int, manifest: dict, max_steps: int = 28000, batch: int = 64, eval_every: int = 200):
    torch.manual_seed(seed)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(dev())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    rng = torch.Generator().manual_seed(seed)
    evals: list[tuple[int, float]] = []
    trajectory = []
    supervised_tokens = 0
    started = time.time()
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        rows = [train[int(i)] for i in idx]
        loss, count = loss_and_positions(model, vocab, rows, dev())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())
        if step % eval_every == 0 or step == 1:
            train_exact, _ = greedy_exact(model, vocab, train[:100], dev())
            test_exact, _ = greedy_exact(model, vocab, test, dev())
            trajectory.append({
                "step": step,
                "train_exact": train_exact,
                "test_exact": test_exact,
                "loss": float(loss.detach().item()),
            })
            evals.append((step, test_exact))
            onset, confirm = detect_sustained(evals, 0.90, 3)
            print(
                f"[ARK-007R acquire s{seed}] step={step} train={train_exact:.3f} test={test_exact:.3f}",
                flush=True,
            )
            if confirm is not None:
                sync()
                return {
                    "seed": seed,
                    "status": "ACQUIRED",
                    "snapshot": snapshot_state(model, optimizer),
                    "g90_flat": flat_params(model).detach().cpu(),
                    "g90_onset_step": onset,
                    "g90_confirmation_step": confirm,
                    "first_treated_optimizer_step": confirm + 1,
                    "acquisition_steps": step,
                    "acquisition_supervised_tokens": supervised_tokens,
                    "trajectory": trajectory,
                    "train": train,
                    "test": test,
                }
        if time.time() - started > 1800:
            return {
                "seed": seed,
                "status": "ACQUISITION_WALLTIME_STOP",
                "trajectory": trajectory,
                "acquisition_steps": step,
                "acquisition_supervised_tokens": supervised_tokens,
            }
    return {
        "seed": seed,
        "status": "NO_G90",
        "trajectory": trajectory,
        "acquisition_steps": max_steps,
        "acquisition_supervised_tokens": supervised_tokens,
    }


def run_t2_continuation(
    snapshot: dict,
    g90_flat_cpu: torch.Tensor,
    indices,
    train,
    test,
    lr: float,
    steps: int = 6000,
    eval_every: int = 200,
    capture_collapse: bool = False,
):
    vocab, model, optimizer = load_t2_fork(snapshot, lr)
    g90_flat = g90_flat_cpu.to(dev())
    trajectory = []
    supervised_tokens = 0
    collapse_snapshot = None
    collapse_onset = None
    collapse_confirm = None
    evals: list[tuple[int, float]] = []
    for step in range(1, steps + 1):
        rows = [train[i] for i in indices[step - 1]]
        loss, count = loss_and_positions(model, vocab, rows, dev())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())
        if step % eval_every == 0:
            test_exact, _ = greedy_exact(model, vocab, test, dev())
            current = flat_params(model)
            l2 = float((current - g90_flat).norm().item())
            rel = l2 / max(float(g90_flat.norm().item()), 1e-12)
            trajectory.append({
                "step": step,
                "test_exact": test_exact,
                "l2_displacement": l2,
                "relative_displacement": rel,
            })
            evals.append((step, test_exact))
            onset, confirm = detect_sustained(evals, 0.90, 3, below=True)
            if capture_collapse and confirm is not None and collapse_snapshot is None:
                sync()
                collapse_onset = onset
                collapse_confirm = confirm
                collapse_snapshot = snapshot_state(model, optimizer)
    return {
        "trajectory": trajectory,
        "retention": retention_metrics(trajectory),
        "supervised_tokens": supervised_tokens,
        "collapse_snapshot": collapse_snapshot,
        "collapse_onset_step": collapse_onset,
        "collapse_confirmation_step": collapse_confirm,
    }


def campaign_a_ark007r():
    print("\n=== CAMPAIGN A: ARK-007R FRESH RETENTION REPLICATION ===", flush=True)
    manifest = load_t2_manifest()
    acq_seeds = [909, 1010, 1111]
    cont_seeds = [2701, 2702, 2703, 2704]
    all_rows = []
    acquisitions = []
    collapse_sources = []

    for seed in acq_seeds:
        if minutes_left() < 70:
            print("budget gate: no new ARK-007R acquisition", flush=True)
            break
        acq = train_t2_to_g90(seed, manifest)
        meta = {k: v for k, v in acq.items() if k in {
            "seed", "status", "g90_onset_step", "g90_confirmation_step",
            "first_treated_optimizer_step", "acquisition_steps",
            "acquisition_supervised_tokens", "trajectory"
        }}
        acquisitions.append(meta)
        save_json("ARK-007R_PARTIAL.json", {"acquisitions": acquisitions, "results": all_rows})
        if acq["status"] != "ACQUIRED":
            continue

        for order_seed in cont_seeds:
            if minutes_left() < 45:
                print("budget gate: stopping ARK-007R continuation orders", flush=True)
                break
            indices = generate_continuation_indices(order_seed, 10000, 64, len(acq["train"]))
            order_hash = order_sha256(indices)
            high = run_t2_continuation(
                acq["snapshot"], acq["g90_flat"], indices, acq["train"], acq["test"],
                1e-3, 6000, capture_collapse=True,
            )
            low = run_t2_continuation(
                acq["snapshot"], acq["g90_flat"], indices, acq["train"], acq["test"],
                1e-5, 6000, capture_collapse=False,
            )
            for arm, lr, out in (("HIGH", 1e-3, high), ("LOW", 1e-5, low)):
                all_rows.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "continuation_order_sha256": order_hash,
                    "arm": arm,
                    "lr": lr,
                    "g90_onset_step": acq["g90_onset_step"],
                    "g90_confirmation_step": acq["g90_confirmation_step"],
                    "first_treated_optimizer_step": acq["first_treated_optimizer_step"],
                    "supervised_tokens": out["supervised_tokens"],
                    "retention": out["retention"],
                    "trajectory": out["trajectory"],
                })
            if high["collapse_snapshot"] is not None:
                collapse_sources.append({
                    "acquisition_seed": seed,
                    "continuation_seed": order_seed,
                    "continuation_order_sha256": order_hash,
                    "collapse_onset_step": high["collapse_onset_step"],
                    "collapse_confirmation_step": high["collapse_confirmation_step"],
                    "snapshot": high["collapse_snapshot"],
                    "g90_flat": acq["g90_flat"],
                    "indices": indices,
                    "train": acq["train"],
                    "test": acq["test"],
                })
            print(
                f"[ARK-007R] s{seed} order{order_seed}: "
                f"HIGH collapse={high['retention']['collapsed']} "
                f"LOW collapse={low['retention']['collapsed']}",
                flush=True,
            )
            save_json("ARK-007R_PARTIAL.json", {"acquisitions": acquisitions, "results": all_rows})

    paired = []
    for seed in acq_seeds:
        for order_seed in cont_seeds:
            high = next((r for r in all_rows if r["acquisition_seed"] == seed and r["continuation_seed"] == order_seed and r["arm"] == "HIGH"), None)
            low = next((r for r in all_rows if r["acquisition_seed"] == seed and r["continuation_seed"] == order_seed and r["arm"] == "LOW"), None)
            if high and low:
                paired.append((bool(high["retention"]["collapsed"]), bool(low["retention"]["collapsed"])))
    n = len(paired)
    high_collapses = sum(int(h) for h, _ in paired)
    low_collapses = sum(int(l) for _, l in paired)
    summary = {
        "paired_conditions": n,
        "high_collapses": high_collapses,
        "low_collapses": low_collapses,
        "risk_difference_low_minus_high": ((low_collapses - high_collapses) / n) if n else None,
        "high_collapse_low_stable": sum(int(h and not l) for h, l in paired),
        "high_stable_low_collapse": sum(int((not h) and l) for h, l in paired),
    }
    payload = {
        "experiment_id": "ARK-007R",
        "manifest_sha256": CANONICAL_T2_SHA,
        "acquisitions": acquisitions,
        "results": all_rows,
        "summary": summary,
    }
    save_json("ARK-007R_RESULT.json", payload)
    return collapse_sources, payload


def fact_signature(facts) -> tuple:
    return tuple(sorted((int(k), int(v)) for k, v in facts))


def build_binding_split():
    keys = range(6)
    vals = range(6)
    factsets = []
    for key_tuple in itertools.combinations(keys, 3):
        for val_tuple in itertools.permutations(vals, 3):
            factsets.append(tuple(zip(key_tuple, val_tuple)))
    rng = random.Random(4242)
    rng.shuffle(factsets)
    train_factsets = factsets[:400]
    test_factsets = factsets[400:500]
    train_sigs = {fact_signature(f) for f in train_factsets}
    test_sigs = {fact_signature(f) for f in test_factsets}
    if train_sigs & test_sigs:
        raise RuntimeError("binding fact-set leakage")

    def expand(groups):
        rows = []
        meta = []
        for facts in groups:
            mapping = dict(facts)
            ordered_keys = [k for k, _ in facts]
            for q in ordered_keys:
                prompt = "+".join(f"{k}={v}" for k, v in facts) + f"/{q}="
                rows.append((prompt, str(mapping[q])))
                meta.append({"facts": facts, "query": q, "answer": mapping[q]})
        return rows, meta

    train, train_meta = expand(train_factsets)
    test, test_meta = expand(test_factsets)
    if len(train) != 1200 or len(test) != 300:
        raise RuntimeError(f"binding split size drift: {len(train)}, {len(test)}")

    swap = []
    for item in test_meta:
        facts = tuple(reversed(item["facts"]))
        keys_here = [k for k, _ in facts]
        old_q = int(item["query"])
        old_pos = keys_here.index(old_q)
        new_q = keys_here[(old_pos + 1) % len(keys_here)]
        mapping = dict(facts)
        prompt = "+".join(f"{k}={v}" for k, v in facts) + f"/{new_q}="
        swap.append((prompt, str(mapping[new_q])))

    manifest = {
        "task_seed": 4242,
        "train_factset_sha256": sha_json(sorted(list(train_sigs))),
        "test_factset_sha256": sha_json(sorted(list(test_sigs))),
        "train_examples": len(train),
        "test_examples": len(test),
        "train_query_counts": dict(Counter(int(x["query"]) for x in train_meta)),
        "test_query_counts": dict(Counter(int(x["query"]) for x in test_meta)),
        "train_answer_counts": dict(Counter(int(x["answer"]) for x in train_meta)),
        "test_answer_counts": dict(Counter(int(x["answer"]) for x in test_meta)),
        "factset_overlap": 0,
    }
    manifest["manifest_sha256"] = sha_json(manifest)
    return train, test, swap, manifest


def load_binding_fork(snapshot: dict, lr: float):
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(dev())
    model.load_state_dict({k: v.to(dev()) for k, v in snapshot["model"].items()})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    optimizer.load_state_dict(snapshot["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    torch.set_rng_state(snapshot["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in snapshot["cuda_rng"]])
    return vocab, model, optimizer


def train_binding_to_qualified(seed: int, train, test, swap, max_steps: int = 24000):
    vocab = CompactVocab()
    torch.manual_seed(seed)
    model = Micro(vocab.size, 128).to(dev())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )
    rng = torch.Generator().manual_seed(seed)
    qualified_evals: list[tuple[int, float]] = []
    trajectory = []
    supervised_tokens = 0
    started = time.time()
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (64,), generator=rng)
        rows = [train[int(i)] for i in idx]
        loss, count = loss_and_positions(model, vocab, rows, dev())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())
        if step % 200 == 0 or step == 1:
            test_exact, _ = greedy_exact(model, vocab, test, dev())
            swap_exact, _ = greedy_exact(model, vocab, swap, dev())
            qualified = test_exact >= 0.90 and swap_exact >= 0.85
            trajectory.append({
                "step": step,
                "test_exact": test_exact,
                "query_swap_exact": swap_exact,
                "qualified": qualified,
                "loss": float(loss.detach().item()),
            })
            qualified_evals.append((step, 1.0 if qualified else 0.0))
            onset, confirm = detect_sustained(qualified_evals, 1.0, 3)
            print(
                f"[ARK-009 acquire s{seed}] step={step} test={test_exact:.3f} swap={swap_exact:.3f}",
                flush=True,
            )
            if confirm is not None:
                sync()
                return {
                    "seed": seed,
                    "status": "QUALIFIED",
                    "snapshot": snapshot_state(model, optimizer),
                    "reference_flat": flat_params(model).detach().cpu(),
                    "qualification_onset_step": onset,
                    "qualification_confirmation_step": confirm,
                    "first_treated_optimizer_step": confirm + 1,
                    "trajectory": trajectory,
                    "supervised_tokens": supervised_tokens,
                }
        if time.time() - started > 1800:
            return {
                "seed": seed,
                "status": "ACQUISITION_WALLTIME_STOP",
                "trajectory": trajectory,
                "supervised_tokens": supervised_tokens,
            }
    return {
        "seed": seed,
        "status": "NO_QUALIFICATION",
        "trajectory": trajectory,
        "supervised_tokens": supervised_tokens,
    }


def binding_retention_metrics(traj: list[dict]) -> dict:
    if not traj:
        return {"status": "EMPTY"}
    qualified = [bool(x["qualified"]) for x in traj]
    onset, confirm = detect_sustained(
        [(int(x["step"]), 1.0 if not x["qualified"] else 0.0) for x in traj],
        1.0,
        3,
    )
    return {
        "RET_QUALIFIED": sum(qualified) / len(qualified),
        "TEST_AREA": sum(float(x["test_exact"]) for x in traj) / len(traj),
        "SWAP_AREA": sum(float(x["query_swap_exact"]) for x in traj) / len(traj),
        "FINAL_TEST": float(traj[-1]["test_exact"]),
        "FINAL_SWAP": float(traj[-1]["query_swap_exact"]),
        "T_FAILURE_ONSET": onset,
        "T_FAILURE_CONFIRM": confirm,
        "collapsed": confirm is not None,
    }


def run_binding_continuation(snapshot, reference_flat_cpu, indices, train, test, swap, lr: float, steps: int = 6000):
    vocab, model, optimizer = load_binding_fork(snapshot, lr)
    reference = reference_flat_cpu.to(dev())
    trajectory = []
    supervised_tokens = 0
    for step in range(1, steps + 1):
        rows = [train[i] for i in indices[step - 1]]
        loss, count = loss_and_positions(model, vocab, rows, dev())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        supervised_tokens += int(count.item())
        if step % 200 == 0:
            test_exact, _ = greedy_exact(model, vocab, test, dev())
            swap_exact, _ = greedy_exact(model, vocab, swap, dev())
            current = flat_params(model)
            l2 = float((current - reference).norm().item())
            trajectory.append({
                "step": step,
                "test_exact": test_exact,
                "query_swap_exact": swap_exact,
                "qualified": test_exact >= 0.90 and swap_exact >= 0.85,
                "l2_displacement": l2,
                "relative_displacement": l2 / max(float(reference.norm().item()), 1e-12),
            })
    return {
        "trajectory": trajectory,
        "retention": binding_retention_metrics(trajectory),
        "supervised_tokens": supervised_tokens,
    }


def campaign_b_ark009():
    print("\n=== CAMPAIGN B: ARK-009 NON-ARITHMETIC TRANSFER ===", flush=True)
    train, test, swap, manifest = build_binding_split()
    save_json("ARK-009_TASK_MANIFEST.json", manifest)
    results = []
    for seed in [1201, 1202]:
        if minutes_left() < 50:
            print("budget gate: no new ARK-009 acquisition", flush=True)
            break
        acq = train_binding_to_qualified(seed, train, test, swap)
        seed_result = {
            "seed": seed,
            "status": acq["status"],
            "acquisition_supervised_tokens": acq.get("supervised_tokens"),
            "trajectory": acq.get("trajectory", []),
            "forks": [],
        }
        if acq["status"] == "QUALIFIED":
            seed_result.update({
                "qualification_onset_step": acq["qualification_onset_step"],
                "qualification_confirmation_step": acq["qualification_confirmation_step"],
                "first_treated_optimizer_step": acq["first_treated_optimizer_step"],
            })
            for order_seed in range(3701, 3707):
                if minutes_left() < 25:
                    print("budget gate: stopping ARK-009 continuations", flush=True)
                    break
                indices = generate_continuation_indices(order_seed, 6000, 64, len(train))
                order_hash = order_sha256(indices)
                high = run_binding_continuation(
                    acq["snapshot"], acq["reference_flat"], indices, train, test, swap, 1e-3, 6000
                )
                low = run_binding_continuation(
                    acq["snapshot"], acq["reference_flat"], indices, train, test, swap, 1e-5, 6000
                )
                seed_result["forks"].extend([
                    {
                        "continuation_seed": order_seed,
                        "continuation_order_sha256": order_hash,
                        "arm": "HIGH",
                        "lr": 1e-3,
                        "supervised_tokens": high["supervised_tokens"],
                        "retention": high["retention"],
                        "trajectory": high["trajectory"],
                    },
                    {
                        "continuation_seed": order_seed,
                        "continuation_order_sha256": order_hash,
                        "arm": "LOW",
                        "lr": 1e-5,
                        "supervised_tokens": low["supervised_tokens"],
                        "retention": low["retention"],
                        "trajectory": low["trajectory"],
                    },
                ])
                print(
                    f"[ARK-009] s{seed} order{order_seed}: "
                    f"HIGH failure={high['retention']['collapsed']} "
                    f"LOW failure={low['retention']['collapsed']}",
                    flush=True,
                )
                save_json("ARK-009_PARTIAL.json", {"task_manifest": manifest, "results": results + [seed_result]})
        results.append(seed_result)
        save_json("ARK-009_PARTIAL.json", {"task_manifest": manifest, "results": results})

    qualified = [r for r in results if r["status"] == "QUALIFIED"]
    status = "TRANSFER_EXECUTED" if any(r["forks"] for r in qualified) else "TRANSFER_BLOCKED_BY_ACQUISITION"
    payload = {
        "experiment_id": "ARK-009",
        "status": status,
        "task_manifest": manifest,
        "results": results,
    }
    save_json("ARK-009_RESULT.json", payload)
    return payload


def run_recovery_from_collapse(source: dict):
    used = int(source["collapse_confirmation_step"])
    tail = source["indices"][used:used + 4000]
    if len(tail) < 4000:
        return {"status": "INSUFFICIENT_FROZEN_TAIL"}
    high = run_t2_continuation(
        source["snapshot"], source["g90_flat"], tail, source["train"], source["test"], 1e-3, 4000
    )
    low = run_t2_continuation(
        source["snapshot"], source["g90_flat"], tail, source["train"], source["test"], 1e-5, 4000
    )

    def recovery90(traj):
        onset, confirm = detect_sustained(
            [(int(x["step"]), float(x["test_exact"])) for x in traj], 0.90, 3
        )
        return {"onset": onset, "confirm": confirm}

    return {
        "status": "EXECUTED",
        "acquisition_seed": source["acquisition_seed"],
        "continuation_seed": source["continuation_seed"],
        "continuation_order_sha256": source["continuation_order_sha256"],
        "source_collapse_onset_step": source["collapse_onset_step"],
        "source_collapse_confirmation_step": source["collapse_confirmation_step"],
        "HIGH_CONTINUE": {
            "lr": 1e-3,
            "recovery90": recovery90(high["trajectory"]),
            "retention": high["retention"],
            "trajectory": high["trajectory"],
        },
        "RECOVERY_LOW": {
            "lr": 1e-5,
            "recovery90": recovery90(low["trajectory"]),
            "retention": low["retention"],
            "trajectory": low["trajectory"],
        },
    }


def campaign_c_ark010(collapse_sources: list[dict]):
    print("\n=== CAMPAIGN C: ARK-010 RECOVERY AFTER COLLAPSE ===", flush=True)
    if len(collapse_sources) < 2:
        payload = {
            "experiment_id": "ARK-010",
            "status": "INCONCLUSIVE_LOW_EVENT_RATE",
            "collapse_sources": len(collapse_sources),
        }
        save_json("ARK-010_RESULT.json", payload)
        return payload
    results = []
    for source in collapse_sources:
        if minutes_left() < 10:
            print("budget gate: stopping ARK-010", flush=True)
            break
        out = run_recovery_from_collapse(source)
        results.append(out)
        save_json("ARK-010_PARTIAL.json", {"results": results})
    payload = {
        "experiment_id": "ARK-010",
        "status": "EXECUTED" if results else "BUDGET_BLOCKED",
        "results": results,
    }
    save_json("ARK-010_RESULT.json", payload)
    return payload


def smoke_test() -> None:
    print("\n=== V5 SMOKE TEST ===", flush=True)
    manifest = load_t2_manifest()
    assert len(manifest["train"]) == 500 and len(manifest["test"]) > 0

    order_a = generate_continuation_indices(9991, 8, 4, len(manifest["train"]))
    order_a2 = generate_continuation_indices(9991, 8, 4, len(manifest["train"]))
    order_b = generate_continuation_indices(9992, 8, 4, len(manifest["train"]))
    assert order_sha256(order_a) == order_sha256(order_a2)
    assert order_sha256(order_a) != order_sha256(order_b)

    vocab = CompactVocab()
    torch.manual_seed(999)
    model = Micro(vocab.size, 128).to(dev())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows = [(p, a) for p, a in manifest["train"][:8]]
    loss, count = loss_and_positions(model, vocab, rows, dev())
    assert torch.isfinite(loss) and int(count.item()) > 0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    snap = snapshot_state(model, optimizer)
    restored = Micro(vocab.size, 128).to(dev())
    restored.load_state_dict({k: v.to(dev()) for k, v in snap["model"].items()})
    assert model_state_equal(model.state_dict(), restored.state_dict())
    exact, _ = greedy_exact(restored, vocab, rows[:4], dev())
    assert 0.0 <= exact <= 1.0

    train, test, swap, bmanifest = build_binding_split()
    assert len(train) == 1200 and len(test) == 300 and len(swap) == 300
    assert bmanifest["factset_overlap"] == 0
    bmodel = Micro(vocab.size, 128).to(dev())
    bopt = torch.optim.AdamW(bmodel.parameters(), lr=1e-3)
    bloss, bcount = loss_and_positions(bmodel, vocab, train[:8], dev())
    assert torch.isfinite(bloss) and int(bcount.item()) > 0
    bopt.zero_grad(set_to_none=True)
    bloss.backward()
    bopt.step()
    bexact, _ = greedy_exact(bmodel, vocab, test[:12], dev())
    sexact, _ = greedy_exact(bmodel, vocab, swap[:12], dev())
    assert 0.0 <= bexact <= 1.0 and 0.0 <= sexact <= 1.0

    receipt = {
        "experiment_id": "MASTER_GPU_V5_SMOKE",
        "status": "PASS",
        "t2_manifest_sha256": CANONICAL_T2_SHA,
        "binding_manifest_sha256": bmanifest["manifest_sha256"],
        "continuation_hash_reproducible": True,
        "snapshot_reload_exact": True,
    }
    save_json("SMOKE_TEST.json", receipt)
    print("SMOKE TEST PASS", flush=True)


def full_campaign() -> None:
    collapse_sources, a = campaign_a_ark007r()
    b = campaign_b_ark009()
    c = campaign_c_ark010(collapse_sources)
    summary = {
        "experiment_id": "MASTER_GPU_V5",
        "ARK-007R_summary": a.get("summary", {}),
        "ARK-009_status": b.get("status"),
        "ARK-010_status": c.get("status"),
        "minutes_used": (time.time() - SESSION_START) / 60.0,
    }
    save_json("PROGRAM_SUMMARY.json", summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--budget-minutes", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    global BUDGET_MINUTES, SESSION_START
    args = parse_args()
    BUDGET_MINUTES = int(args.budget_minutes)
    SESSION_START = time.time()
    init_cuda()
    print("BASE PLAN:", BASE_PLAN_SHA, flush=True)
    print("V5 ADDENDUM:", PLAN_SHA, flush=True)

    if args.smoke_test:
        smoke_test()
        package_results(download=False)
        return 0

    try:
        full_campaign()
        return 0
    except Exception as exc:
        save_json(
            "FAILURE_RECEIPT.json",
            {
                "experiment_id": "MASTER_GPU_V5",
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "minutes_used": (time.time() - SESSION_START) / 60.0,
            },
        )
        raise
    finally:
        package_results(download=True)


if __name__ == "__main__":
    raise SystemExit(main())
