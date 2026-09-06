"""ARK-005 runner: retention/consolidation arms, trigger-controlled.

All four arms of a seed train IDENTICALLY until the trigger (first sustained
G90 on raw weights: 3 consecutive evals >= 0.90 at the 200-step cadence), so
the trigger step is identical across arms by construction. At the trigger:
  A CONTROL        - continue ordinary training
  B LR-DECAY       - lr *= 0.1 from the trigger
  C WD-REMOVAL     - weight decay set to 0 from the trigger
  D EMA-CONSOLIDATION - an EMA (decay 0.999) of raw weights is maintained from
    the trigger; post-trigger EVALUATION uses the EMA weights (the treatment
    is what gets deployed). Pre-trigger evals use raw weights for all arms.
Retention metrics follow the ARK-005 PLAN definitions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

PLAN_SHA = None  # bound after the plan commit; see main()


def first_sustained(steps, values, bar, after_step, consecutive=3, below=False):
    streak, start = 0, None
    for step, value in zip(steps, values):
        if step <= after_step:
            continue
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


def retention_metrics(trajectory: list[dict], g90: int) -> dict:
    post = [e for e in trajectory if e["step"] >= g90]
    if not post:
        return {"status": "NO_POST_TRIGGER_WINDOW"}
    trailing = []
    for i, e in enumerate(post):
        window = post[max(0, i - 2): i + 1]
        trailing.append((e["step"], sum(x["test_exact"] for x in window) / len(window)))
    peak_step, peak_g = max(trailing, key=lambda t: t[1])
    final_sustained = trailing[-1][1]
    ret90 = sum(1 for e in post if e["test_exact"] >= 0.90) / len(post)
    ret50 = sum(1 for e in post if e["test_exact"] >= 0.50) / len(post)
    collapse90 = first_sustained([e["step"] for e in traj_steps(post)], ["x"] * len(post), 0, 0) if False else \
        first_sustained_values([e["step"] for e in post], [e["test_exact"] for e in post], 0.90, below=True)
    collapse50 = first_sustained_values([e["step"] for e in post], [e["test_exact"] for e in post], 0.50, below=True)
    return {
        "status": "MEASURED",
        "PEAK_G": round(peak_g, 4),
        "PEAK_G_step": peak_step,
        "RET90": round(ret90, 4),
        "RET50": round(ret50, 4),
        "T_COLLAPSE_90": collapse90,
        "T_COLLAPSE_50": collapse50,
        "GENERALIZATION_AREA": round(sum(e["test_exact"] for e in post) / len(post), 4),
        "STABILITY_GAP": round(peak_g - final_sustained, 4),
        "FINAL_OOD": round(post[-1]["test_exact"], 4),
    }


def traj_steps(post):
    return [e["step"] for e in post]


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


def run_arm(arm: str, *, seed: int, manifest: dict, box_s: float, steps: int,
            batch: int, lr: float, device, plan_sha: str,
            force_trigger: int | None = None) -> dict:
    torch.manual_seed(seed)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    order_gen = torch.Generator().manual_seed(seed)
    trajectory: list[dict] = []
    eval_steps, eval_ood = [], []
    trigger_step = force_trigger
    ema = None
    ema_decay = 0.999
    applied = {"lr_decayed": False, "wd_removed": False, "ema_started": False}
    started = time.perf_counter()
    tokens = 0
    step = 0
    aborted = False
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=order_gen)
        batch_rows = [train[i] for i in idx]
        loss, count = loss_and_positions(model, vocab, batch_rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += int(count.item())
        if ema is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema, model.parameters()):
                    ema_p.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        if step % 200 == 0 or step == 1:
            use_ema = arm == "D" and ema is not None
            raw_state = None
            if use_ema:
                raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(
                    {k: e for (k, _), e in zip(model.state_dict().items(), ema)}, strict=True)
            train_exact, _ = greedy_exact(model, vocab, train[:100], device)
            ood_exact, epp = greedy_exact(model, vocab, test, device)
            if use_ema:
                model.load_state_dict(raw_state, strict=True)
            trajectory.append({
                "step": step, "tokens": tokens,
                "exposures": step * batch / len(train),
                "loss": float(loss.detach()), "train_exact": train_exact,
                "test_exact": ood_exact, "test_per_position": epp,
                "evaluated_with_ema": bool(use_ema),
            })
            eval_steps.append(step)
            eval_ood.append(ood_exact)
            # trigger detection (identical raw-weight rule for every arm)
            if trigger_step is None:
                if force_trigger is not None:
                    if step >= force_trigger:
                        trigger_step = step
                elif len(eval_steps) >= 3:
                    detected = first_sustained_values(eval_steps, eval_ood, 0.90)
                    if detected is not None:
                        trigger_step = detected
            # arm application, exactly once, immediately after detection
            if trigger_step is not None and not any(applied.values()):
                if arm == "B":
                    for group in optimizer.param_groups:
                        group["lr"] = group["lr"] * 0.1
                    applied["lr_decayed"] = True
                elif arm == "C":
                    for group in optimizer.param_groups:
                        group["weight_decay"] = 0.0
                    applied["wd_removed"] = True
                elif arm == "D":
                    ema = [p.data.clone() for p in model.parameters()]
                    applied["ema_started"] = True
                else:
                    applied["control_continues"] = True
        if time.perf_counter() - started > box_s:
            aborted = True
            break
    g90_sustained = first_sustained_values(eval_steps, eval_ood, 0.90)
    retention = (retention_metrics(
        [{"step": s, "test_exact": v} for s, v in zip(eval_steps, eval_ood)], trigger_step)
        if trigger_step is not None else {"status": "NO_TRIGGER_WITHIN_BUDGET"})
    return {
        "arm": arm,
        "seed": seed,
        "init_order_seed": seed,
        "status": "ABORTED_WALL_BOX" if aborted else "COMPLETED",
        "steps_run": step,
        "wall_seconds": time.perf_counter() - started,
        "parameters": params,
        "supervised_tokens": tokens,
        "trigger_step": trigger_step,
        "applied_changes": applied,
        "ema_decay": ema_decay if arm == "D" else None,
        "sustained_G90_raw": g90_sustained,
        "retention": retention,
        "trajectory": trajectory,
        "plan_commit_sha256": plan_sha,
        "manifest_split_sha256": manifest["split_sha256"],
        "device": str(device),
        "torch": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("A", "B", "C", "D"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--box", type=float, default=2400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-trigger-step", type=int, default=None,
                        help="test-only: force the trigger at this step")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = run_arm(args.arm, seed=args.seed, manifest=manifest, box_s=args.box,
                     steps=args.steps, batch=64, lr=1e-3, device=device,
                     plan_sha=plan_sha, force_trigger=args.force_trigger_step)
    receipt = m.bind_receipt(
        experiment_id="ARK-005",
        plan_commit_sha="3d5e97a"[:40],
        code_paths={
            "runner": str(Path(__file__)),
            "harness": str(REPO / "experiments/ARK-001/run_ark001.py"),
            "metrics": str(REPO / "experiments/lib/ark_metrics.py"),
            "tasks": str(REPO / "experiments/lib/ark_tasks.py"),
        },
        config={"arm": args.arm, "seed": args.seed,
                "manifest_split_sha256": manifest["split_sha256"],
                "batch": 64, "lr": 1e-3, "wall_box_s": args.box,
                "force_trigger_step": args.force_trigger_step,
                "device": str(device), "torch": torch.__version__},
        results=result,
    )
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "seed": args.seed, "status": result["status"],
                      "trigger": result["trigger_step"],
                      "retention": result["retention"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
