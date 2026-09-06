"""ARK-007: paired continuation stochasticity experiment (corrected runner)."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments/ARK-001"))
sys.path.insert(0, str(REPO))
import torch
from experiments.lib import ark_tasks as t
from run_ark001 import CompactVocab, Micro, greedy_exact, loss_and_positions

def generate_continuation_indices(order_seed, n_batches, batch_size, pool_size):
    rng = torch.Generator().manual_seed(order_seed)
    return [torch.randint(0, pool_size, (batch_size,), generator=rng).tolist() for _ in range(n_batches)]

def order_sha256(indices):
    return hashlib.sha256(json.dumps(indices).encode()).hexdigest()

def detect_g90(eval_steps, eval_ood, bar=0.90, consec=3):
    streak, onset = 0, None
    for step, value in zip(eval_steps, eval_ood):
        if value >= bar:
            if streak == 0: onset = step
            streak += 1
            if streak >= consec: return onset, step
        else: streak, onset = 0, None
    return None, None

def _flat_params(model):
    return torch.cat([p.detach().data.reshape(-1) for p in model.parameters()])

@torch.no_grad()
def param_displacement(model, g90_flat):
    current = _flat_params(model)
    diff = (current - g90_flat).norm().item()
    base = g90_flat.norm().item()
    return {"l2_distance": round(diff, 4), "relative": round(diff / max(base, 1e-8), 6)}

def train_to_g90(*, seed, manifest, max_steps, batch, lr, device, eval_every=200):
    torch.manual_seed(seed)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    rng = torch.Generator().manual_seed(seed)
    eval_steps, eval_ood, trajectory = [], [], []
    started, acq_tokens = time.time(), 0
    model.train()
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        rows = [train[i] for i in idx]
        loss, sup = loss_and_positions(model, vocab, rows, device)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize()
        acq_tokens += sup
        if step % eval_every == 0 or step == 1:
            tr, _ = greedy_exact(model, vocab, train[:100], device)
            te, epp = greedy_exact(model, vocab, test, device)
            trajectory.append({"step": step, "train_exact": tr, "test_exact": te})
            eval_steps.append(step); eval_ood.append(te)
        onset, confirm = detect_g90(eval_steps, eval_ood)
        if confirm is not None:
            state = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
            return {"model": model, "optimizer": optimizer, "vocab": vocab, "state": state,
                    "g90_flat": _flat_params(model), "g90_onset_step": onset,
                    "g90_confirmation_step": confirm,
                    "acquisition_tokens": acq_tokens, "acquisition_steps": step,
                    "pre_trigger_trajectory": trajectory, "train": train, "test": test}
        if time.time() - started > 3600: break
    return {"model": model, "optimizer": optimizer, "vocab": vocab, "state": None,
            "g90_flat": None, "g90_onset_step": None, "g90_confirmation_step": None,
            "acquisition_tokens": acq_tokens, "acquisition_steps": step,
            "pre_trigger_trajectory": trajectory, "train": train, "test": test}

def retention_metrics(trajectory):
    if not trajectory: return {"status": "EMPTY"}
    ood = [e["test_exact"] for e in trajectory]
    ret90 = sum(1 for v in ood if v >= 0.90) / len(ood)
    ret50 = sum(1 for v in ood if v >= 0.50) / len(ood)
    area = sum(ood) / len(ood)
    collapse90, streak = None, 0
    for e in trajectory:
        if e["test_exact"] < 0.90:
            streak += 1
            if streak >= 3: collapse90 = e["step"]; break
        else: streak = 0
    peak, final = max(ood), ood[-1]
    return {"RET90": round(ret90, 4), "RET50": round(ret50, 4),
            "GENERALIZATION_AREA": round(area, 4), "T_COLLAPSE_90": collapse90,
            "PEAK_G": round(peak, 4), "STABILITY_GAP": round(peak - final, 4),
            "FINAL_OOD": round(final, 4), "collapse_binary": collapse90 is not None}

def run_fork(*, arm_lr, snapshot, g90_flat, continuation_indices, train, test, steps, eval_every, device):
    torch.manual_seed(0)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(device)
    model.load_state_dict({k: v.to(device) for k, v in snapshot["model"].items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=arm_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optimizer.load_state_dict(snapshot["optimizer"])
    for g in optimizer.param_groups: g["lr"] = arm_lr
    torch.set_rng_state(snapshot["torch_rng"].cpu())
    if device.type == "cuda" and snapshot.get("cuda_rng"):
        torch.cuda.set_rng_state_all([r.cpu() for r in snapshot["cuda_rng"]])
    gf = g90_flat.to(device)
    trajectory, sup_tokens = [], 0
    model.train()
    for step in range(1, steps + 1):
        batch_indices = continuation_indices[step - 1]
        rows = [train[i] for i in batch_indices]
        loss, sup = loss_and_positions(model, vocab, rows, device)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize()
        sup_tokens += sup
        if step % eval_every == 0:
            te, epp = greedy_exact(model, vocab, test, device)
            disp = param_displacement(model, gf)
            trajectory.append({"step": step, "test_exact": te, "test_per_position": epp,
                               "supervised_tokens": sup_tokens, **disp})
    return {"lr": arm_lr, "retention": retention_metrics(trajectory), "trajectory": trajectory, "supervised_tokens": sup_tokens}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="experiments/ARK-007/RESULT.json")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"))
    continuation_seeds = list(range(1701, 1709))
    post_steps, batch_size = 8000, 64
    continuation_orders = {}
    for cs in continuation_seeds:
        indices = generate_continuation_indices(cs, post_steps, batch_size, len(manifest["train"]))
        continuation_orders[cs] = {"indices": indices, "sha256": order_sha256(indices)}
    all_checkpoints = {}
    for seed in (707, 808):
        print(f"=== acquiring seed {seed} ===", flush=True)
        acq = train_to_g90(seed=seed, manifest=manifest, max_steps=28000, batch=batch_size, lr=1e-3, device=device)
        if acq["state"] is None:
            print(f"seed {seed}: G90 not reached"); continue
        print(f"  onset {acq['g90_onset_step']} confirmation {acq['g90_confirmation_step']}", flush=True)
        all_checkpoints[seed] = acq
    results = []
    for seed, acq in sorted(all_checkpoints.items()):
        for cs in continuation_seeds:
            indices = continuation_orders[cs]["indices"]
            for arm_name, arm_lr in (("HIGH", 1e-3), ("LOW", 1e-5)):
                print(f"  s{seed} order {cs} LR {arm_lr}", flush=True)
                fork = run_fork(arm_lr=arm_lr, snapshot=acq["state"], g90_flat=acq["g90_flat"],
                                continuation_indices=indices, train=acq["train"], test=acq["test"],
                                steps=post_steps, eval_every=200, device=device)
                results.append({"acquisition_seed": seed, "continuation_order_seed": cs,
                                "continuation_order_sha256": continuation_orders[cs]["sha256"],
                                "arm": arm_name, "lr": arm_lr,
                                "g90_onset_step": acq["g90_onset_step"],
                                "g90_confirmation_step": acq["g90_confirmation_step"], **fork})
    receipt = {"schema": "arkenstone-ark007/v2", "plan_commit_sha256": args.plan_sha,
               "manifest_split_sha256": manifest["split_sha256"],
               "continuation_order_hashes": {str(cs): continuation_orders[cs]["sha256"] for cs in continuation_seeds},
               "device": str(device), "torch": torch.__version__,
               "post_confirmation_steps": post_steps, "batch_size": batch_size, "results": results}
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("saved:", out)

if __name__ == "__main__":
    main()
