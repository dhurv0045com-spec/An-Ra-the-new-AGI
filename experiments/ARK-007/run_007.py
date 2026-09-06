"""ARK-007: multi-seed LR-threshold replication.

Tests the two extreme arms (lr x0.001 stable-predicted, lr x1.0 decay-predicted)
on two fresh seeds (707, 808). If both seeds show the same pattern — stable at
low LR, decay at full LR — the LR-threshold law upgrades from TENTATIVE to
REPLICATED for this task family.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments/ARK-001"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from experiments.lib import ark_metrics as m  # noqa: E402
from experiments.lib import ark_tasks as t  # noqa: E402



def run_lr_arm(*, seed: int, lr_multiplier: float, manifest: dict,
               box_s: float, max_steps: int, batch: int, base_lr: float,
               device) -> dict:
    import random as _random
    torch.manual_seed(seed)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    rng = torch.Generator().manual_seed(seed)
    trajectory = []
    eval_steps, eval_ood = [], []
    trigger = None
    lr_applied = False
    started = time.perf_counter()
    step = 0
    model.train()
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        rows = [train[i] for i in idx]
        loss, _ = loss_and_positions(model, vocab, rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == 1:
            train_exact, _ = greedy_exact(model, vocab, train[:100], device)
            ood_exact, epp = greedy_exact(model, vocab, test, device)
            trajectory.append({"step": step, "exposures": step * batch / len(train),
                               "train_exact": train_exact, "test_exact": ood_exact,
                               "test_per_position": epp})
            eval_steps.append(step); eval_ood.append(ood_exact)
            if trigger is None and len(eval_steps) >= 3:
                d = first_sustained_values(eval_steps, eval_ood, 0.90)
                if d is not None: trigger = d
            if trigger is not None and not lr_applied and step >= trigger:
                actual = base_lr * lr_multiplier
                for g in optimizer.param_groups: g["lr"] = actual
                lr_applied = True
                print(f"  trigger {trigger}, lr -> {actual:.6f}", flush=True)
        if time.perf_counter() - started > box_s: break
    post = [e for e in trajectory if e["step"] >= (trigger or max_steps)]
    ret90 = sum(1 for e in post if e["test_exact"] >= 0.90) / max(1, len(post))
    ret50 = sum(1 for e in post if e["test_exact"] >= 0.50) / max(1, len(post))
    area = sum(e["test_exact"] for e in post) / max(1, len(post))
    collapse = first_sustained_values(
        [e["step"] for e in post], [e["test_exact"] for e in post], 0.90, below=True)
    return {"lr_multiplier": lr_multiplier, "trigger_step": trigger,
            "post_trigger_steps": len(post), "RET90": round(ret90, 4),
            "RET50": round(ret50, 4), "GENERALIZATION_AREA": round(area, 4),
            "T_COLLAPSE_90": collapse,
            "FINAL_OOD": round(trajectory[-1]["test_exact"], 4),
            "trajectory": trajectory, "seed": seed}


def first_sustained_values(steps, values, bar, consecutive=3, below=False):
    streak, start = 0, None
    for step, value in zip(steps, values):
        hit = value < bar if below else value >= bar
        if hit:
            if streak == 0: start = step
            streak += 1
            if streak >= consecutive: return start
        else: streak, start = 0, None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="experiments/ARK-007/RESULT.json")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"))

    all_results = []
    for seed in (707, 808):
        for mult, label in ((0.001, "LOW"), (1.0, "CONTROL")):
            print(f"=== seed {seed} lr x{mult} ({label}) ===", flush=True)
            result = run_lr_arm(seed=seed, lr_multiplier=mult, manifest=manifest,
                                box_s=2400, max_steps=28000, batch=64,
                                base_lr=1e-3, device=device)
            result["label"] = label
            all_results.append(result)
            print(json.dumps({"seed": seed, "lr": mult, "RET90": result["RET90"],
                              "collapse": result["T_COLLAPSE_90"],
                              "final": result["FINAL_OOD"]}), flush=True)

    # verdict
    verdict = "REPLICATED"
    for seed in (707, 808):
        low = next(r for r in all_results if r["seed"] == seed and r["label"] == "LOW")
        ctrl = next(r for r in all_results if r["seed"] == seed and r["label"] == "CONTROL")
        if low["RET90"] < 0.9 or ctrl["RET90"] > 0.5:
            verdict = "NOT_REPLICATED"
            break

    receipt = {
        "schema": "arkenstone-ark007/v1",
        "question": "Does the LR-threshold retention law replicate on fresh seeds?",
        "seeds": (707, 808),
        "manifest_split_sha256": manifest["split_sha256"],
        "device": str(device), "torch": torch.__version__,
        "arms": all_results,
        "verdict": verdict,
        "interpretation": {
            "REPLICATED": "both seeds: low-LR stable (RET90 >= 0.9), control decays (RET90 < 0.5)",
            "NOT_REPLICATED": "at least one seed breaks the pattern; law is task/seed-specific",
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "out": str(out)}))
    return 0


if __name__ == "__main__":
    import hashlib
    main()
