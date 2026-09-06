"""ARK-006: LR dose-response on the decaying seed (606).

Preregistered: LR multipliers {0.001, 0.01, 0.1, 1.0(control)} applied at
first sustained G90. The question: is there a minimum LR that prevents
post-transition decay, or does ANY optimization pressure cause decay?

Reuses the frozen ARK-002B manifest and the ARK-004A/005 model configuration.
The control arm (x1.0) is the ARK-005 seed-606 arm A receipt (same seed,
same config, same manifest — no need to rerun).
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
from run_ark001 import CompactVocab, Micro, greedy_exact, loss_and_positions  # noqa: E402


def first_sustained_values(steps, values, bar, consecutive=3, below=False):
    streak, start = 0, None
    for step, value in zip(steps, values):
        hit = value < bar if below else value >= bar
        if hit:
            if streak == 0:
                start = step
            streak += 1
            if streak >= consecutive:
                return start
        else:
            streak, start = 0, None
    return None


def run_lr_arm(*, seed: int, lr_multiplier: float, manifest: dict,
               box_s: float, max_steps: int, batch: int, base_lr: float,
               device) -> dict:
    """Train until sustained G90, then apply lr * multiplier and continue."""
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
    tokens = 0
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
        tokens += int(batch * 10)
        if step % 200 == 0 or step == 1:
            train_exact, _ = greedy_exact(model, vocab, train[:100], device)
            ood_exact, epp = greedy_exact(model, vocab, test, device)
            trajectory.append({
                "step": step, "tokens": tokens,
                "exposures": step * batch / len(train),
                "loss": float(loss.detach()), "train_exact": train_exact,
                "test_exact": ood_exact, "test_per_position": epp,
            })
            eval_steps.append(step)
            eval_ood.append(ood_exact)
            if trigger is None and len(eval_steps) >= 3:
                detected = first_sustained_values(eval_steps, eval_ood, 0.90)
                if detected is not None:
                    trigger = detected
            if trigger is not None and not lr_applied and step >= trigger:
                actual_lr = base_lr * lr_multiplier
                for group in optimizer.param_groups:
                    group["lr"] = actual_lr
                lr_applied = True
                print(f"  trigger at {trigger}, lr -> {actual_lr:.6f}", flush=True)
        if time.perf_counter() - started > box_s:
            break
    post = [e for e in trajectory if e["step"] >= (trigger or max_steps)]
    ret90 = sum(1 for e in post if e["test_exact"] >= 0.90) / max(1, len(post))
    ret50 = sum(1 for e in post if e["test_exact"] >= 0.50) / max(1, len(post))
    area = sum(e["test_exact"] for e in post) / max(1, len(post))
    collapse90 = first_sustained_values(
        [e["step"] for e in post], [e["test_exact"] for e in post], 0.90, below=True)
    return {
        "lr_multiplier": lr_multiplier,
        "actual_lr": base_lr * lr_multiplier,
        "trigger_step": trigger,
        "post_trigger_steps": len(post),
        "RET90": round(ret90, 4),
        "RET50": round(ret50, 4),
        "GENERALIZATION_AREA": round(area, 4),
        "T_COLLAPSE_90": collapse90,
        "FINAL_OOD": round(trajectory[-1]["test_exact"], 4),
        "trajectory": trajectory,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--max-steps", type=int, default=32000)
    parser.add_argument("--box", type=float, default=2400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="experiments/ARK-006/RESULT.json")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"))
    multipliers = [0.001, 0.01, 0.1]  # 1.0 control already exists in ARK-005 receipts
    arms = []
    for mult in multipliers:
        print(f"=== lr x{mult} ===", flush=True)
        result = run_lr_arm(seed=args.seed, lr_multiplier=mult, manifest=manifest,
                            box_s=args.box, max_steps=args.max_steps, batch=64,
                            base_lr=1e-3, device=device)
        arms.append(result)
        print(json.dumps({"lr_mult": mult, "RET90": result["RET90"],
                          "collapse": result["T_COLLAPSE_90"],
                          "final": result["FINAL_OOD"]}), flush=True)
    control = {"lr_multiplier": 1.0, "actual_lr": 1e-3, "trigger_step": 18600,
               "post_trigger_steps": 5400, "RET90": 0.1429, "RET50": 1.0,
               "GENERALIZATION_AREA": 0.8223, "T_COLLAPSE_90": 19200,
               "FINAL_OOD": 0.8223, "seed": 606,
               "source": "ARK-005 RESULT_A_seed606.json (same seed/config/manifest)"}
    receipt = {
        "schema": "arkenstone-ark006/v1",
        "question": "LR dose-response: is there a minimum LR that prevents post-transition decay?",
        "seed": args.seed,
        "manifest_split_sha256": manifest["split_sha256"],
        "device": str(device), "torch": torch.__version__,
        "base_lr": 1e-3,
        "arms": arms,
        "control_from_ARK-005": control,
        "dose_response": [
            {"lr_multiplier": a["lr_multiplier"], "RET90": a["RET90"],
             "T_COLLAPSE_90": a["T_COLLAPSE_90"], "FINAL_OOD": a["FINAL_OOD"]}
            for a in arms
        ] + [{"lr_multiplier": control["lr_multiplier"], "RET90": control["RET90"],
              "T_COLLAPSE_90": control["T_COLLAPSE_90"], "FINAL_OOD": control["FINAL_OOD"]}],
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    main()
