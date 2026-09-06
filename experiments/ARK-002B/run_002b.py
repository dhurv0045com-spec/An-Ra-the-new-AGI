"""ARK-002B runner: frozen manifest, separated init/order seeds, sustained metrics."""
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
from run_ark001 import (  # noqa: E402
    CompactVocab,
    Micro,
    greedy_exact,
    loss_and_positions,
)


def run(*, init_seed: int, order_seed: int, manifest: dict, box_s: float,
        steps: int, batch: int, lr: float, device, plan_sha: str) -> dict:
    torch.manual_seed(init_seed)  # initialization ONLY
    vocab = CompactVocab()
    model = Micro(vocab.size, 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    order_gen = torch.Generator().manual_seed(order_seed)  # data order ONLY
    rng = torch.Generator().manual_seed(init_seed)  # batch sampling follows init stream, documented
    trajectory = []
    lift = None
    started = time.perf_counter()
    step = 0
    aborted = False
    tokens = 0
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        batch_rows = [train[i] for i in idx]
        loss, count = loss_and_positions(model, vocab, batch_rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += int(count.item()) + int(sum(len(vocab.encode(p)) for p, _ in batch_rows))
        if step % 200 == 0 or step == 1:
            train_exact, tpp = greedy_exact(model, vocab, train[:100], device)
            test_exact, epp = greedy_exact(model, vocab, test, device)
            trajectory.append({
                "step": step,
                "tokens": tokens,
                "exposures": step * batch / len(train),
                "loss": float(loss.detach()),
                "train_exact": train_exact,
                "test_exact": test_exact,
                "train_per_position": tpp,
                "test_per_position": epp,
            })
            if lift is None and train_exact >= 0.9:
                lift = step
        if time.perf_counter() - started > box_s:
            aborted = True
            break
    summary = m.sustained_summary(trajectory)
    return {
        "init_seed": init_seed,
        "order_seed": order_seed,
        "status": "ABORTED_WALL_BOX" if aborted else "COMPLETED",
        "steps_run": step,
        "wall_seconds": time.perf_counter() - started,
        "supervised_plus_prompt_tokens": tokens,
        "lift_off_step_train09": lift,
        "sustained": summary,
        "trajectory": trajectory,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--box", type=float, default=1800)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"))
    device = torch.device(args.device)
    result = run(init_seed=args.init_seed, order_seed=args.order_seed,
                 manifest=manifest, box_s=args.box, steps=args.steps,
                 batch=args.batch, lr=args.lr, device=device,
                 plan_sha="90818c165be5db7375502b876102d72304ad2337")
    receipt = m.bind_receipt(
        experiment_id="ARK-002B",
        plan_commit_sha="90818c165be5db7375502b876102d72304ad2337",
        code_paths={
            "runner": str(Path(__file__)),
            "harness": str(REPO / "experiments/ARK-001/run_ark001.py"),
            "metrics": str(REPO / "experiments/lib/ark_metrics.py"),
            "tasks": str(REPO / "experiments/lib/ark_tasks.py"),
        },
        config={
            "manifest_split_sha256": manifest["split_sha256"],
            "model": "Micro 4L/128w/4H compact-vocab",
            "batch": args.batch, "lr": args.lr, "wall_box_s": args.box,
            "init_seed": args.init_seed, "order_seed": args.order_seed,
            "device": str(device), "torch": torch.__version__,
        },
        results=result,
    )
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "status": result["status"],
                      "lift_off": result["lift_off_step_train09"],
                      "G90": result["sustained"]["G90"]["status"],
                      "G90_step": result["sustained"]["G90"]["step"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
