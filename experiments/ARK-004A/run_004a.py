"""ARK-004A runner: dense developmental mapping + minimal internal probes.

Training is byte-identical to ARK-002B's (frozen manifest, Micro model, same
optimizer); every eval additionally records margins and the frozen-counterfactual
column-selectivity probes. Probes never affect training.
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
import torch.nn.functional as F  # noqa: E402

from experiments.lib import ark_metrics as m  # noqa: E402
from experiments.lib import ark_tasks as t  # noqa: E402
from run_ark001 import CompactVocab, Micro, greedy_exact, loss_and_positions  # noqa: E402

PLAN_SHA = "9e2fe8520e6c38168d99ec2d24bc4995e54773ad"
SEEDS = (101, 202, 303, 404)


def parse_ab(prompt: str) -> tuple[int, int]:
    left, right = prompt.split("=")[0].split("+")
    return int(left), int(right)


def build_probe_sets(test_rows: list[tuple[str, str]]) -> dict:
    """Frozen counterfactual probe sets (2-digit answers only, uniform layout)."""
    import random

    rng = random.Random(4242)
    p_ones, p_tens = [], []
    for prompt, answer in test_rows:
        a, b = parse_ab(prompt)
        if a + b < 10:
            continue  # 1-digit answers have a different position layout
        ta, ua, tb, ub = a // 10, a % 10, b // 10, b % 10
        ones_alts = [u for u in range(0, 10 - ua) if u != ub]
        tens_alts = [x for x in range(6, 8) if x != ta and tb <= 9 - x]
        if ones_alts:
            ub2 = rng.choice(ones_alts)
            p_ones.append({
                "clean": (prompt, answer),
                "pert": (f"{a} + {tb * 10 + ub2} = ", f"{a + tb * 10 + ub2}"),
                "type": "ONES",
            })
        if tens_alts:
            ta2 = rng.choice(tens_alts)
            p_tens.append({
                "clean": (prompt, answer),
                "pert": (f"{ta2 * 10 + ua} + {b} = ", f"{ta2 * 10 + ua + b}"),
                "type": "TENS",
            })
    return {"P_ONES": p_ones[:80], "P_TENS": p_tens[:80]}


@torch.no_grad()
def probe_model(model, vocab, probe_sets: dict, device) -> dict:
    """Per-layer representation deltas + column selectivity + margins."""
    model.eval()
    layers = [f"block{i}" for i in range(len(model.blocks))] + ["final"]
    sums = {f"{k}_{side}": 0.0 for k in ("P_ONES", "P_TENS") for side in ("tens", "ones")}
    counts = {f"{k}_{side}": 0 for k in ("P_ONES", "P_TENS") for side in ("tens", "ones")}
    hidden_delta = {layer: 0.0 for layer in layers}
    hidden_n = {layer: 0 for layer in layers}
    margin_sums = {"tens": 0.0, "ones": 0.0}
    margin_n = 0

    activations: dict[str, torch.Tensor] = {}

    def hook(name):
        def grab(_module, _inp, out):
            activations[name] = out.detach()
        return grab

    handles = [block.register_forward_hook(hook(f"block{i}"))
               for i, block in enumerate(model.blocks)]
    handles.append(model.norm.register_forward_hook(hook("final")))

    for set_key in ("P_ONES", "P_TENS"):
        for case in probe_sets[set_key]:
            clean_ids = vocab.encode(case["clean"][0])
            pert_ids = vocab.encode(case["pert"][0])
            plen = len(clean_ids)
            if len(pert_ids) != plen:
                continue
            clean_answer = case["clean"][1]
            pert_answer = case["pert"][1]
            if len(clean_answer) != 2 or len(pert_answer) != 2:
                continue
            tens_tok_clean = vocab.encode(clean_answer[0])[-1]
            ones_tok_clean = vocab.encode(clean_answer[1])[-1]
            tens_tok_pert = vocab.encode(pert_answer[0])[-1]
            ones_tok_pert = vocab.encode(pert_answer[1])[-1]
            # feed prompt + the row's own tens answer digit so the ones-position
            # logits exist (the model reads the tens digit before predicting ones)
            batch = torch.tensor(
                [clean_ids + [tens_tok_clean], pert_ids + [tens_tok_pert]], device=device
            )
            activations.clear()
            logits = model(batch)  # [2, T, V]
            tens_pos = plen - 1   # input position whose logits predict the tens digit
            ones_pos = plen       # predicts the ones digit
            d_tens = abs(float(logits[1, tens_pos, tens_tok_pert] - logits[0, tens_pos, tens_tok_clean]))
            d_ones = abs(float(logits[1, ones_pos, ones_tok_pert] - logits[0, ones_pos, ones_tok_clean]))
            total = d_tens + d_ones
            if total > 1e-8:
                side = "ones" if set_key == "P_ONES" else "tens"
                targeted = d_ones if set_key == "P_ONES" else d_tens
                sums[f"{set_key}_{side}"] += targeted / total
                counts[f"{set_key}_{side}"] += 1
            for layer in layers:
                if layer not in activations:
                    continue
                h = activations[layer]
                delta = float((h[1] - h[0]).float().norm().item())
                hidden_delta[layer] += delta
                hidden_n[layer] += 1
            # correct-token margins on the CLEAN row (logit[correct] - best other)
            for pos, tok, side in ((tens_pos, tens_tok_clean, "tens"),
                                   (ones_pos, ones_tok_clean, "ones")):
                row = logits[0, pos]
                others = row.clone()
                others[tok] = -float("inf")
                margin_sums[side] += float(row[tok] - others.max())
            margin_n += 1
    for handle in handles:
        handle.remove()
    model.train()
    selectivity = {
        "ONES_ones_share": sums["P_ONES_ones"] / max(1, counts["P_ONES_ones"]),
        "TENS_tens_share": sums["P_TENS_tens"] / max(1, counts["P_TENS_tens"]),
        "n_ones": counts["P_ONES_ones"],
        "n_tens": counts["P_TENS_tens"],
    }
    hidden = {layer: hidden_delta[layer] / max(1, hidden_n[layer]) for layer in layers}
    margins = {"tens_margin": margin_sums["tens"] / max(1, margin_n),
               "ones_margin": margin_sums["ones"] / max(1, margin_n)}
    return {"selectivity": selectivity, "hidden_delta": hidden, "margins": margins}


def run_seed(seed: int, *, manifest: dict, box_s: float, steps: int, batch: int,
             lr: float, device, plan_sha: str) -> dict:
    torch.manual_seed(seed)
    vocab = CompactVocab()
    model = Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    probe_sets = build_probe_sets(test)
    rng = torch.Generator().manual_seed(seed)
    trajectory = []
    started = time.perf_counter()
    tokens = 0
    step = 0
    aborted = False
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        batch_rows = [train[i] for i in idx]
        loss, count = loss_and_positions(model, vocab, batch_rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += int(count.item())
        if step % 200 == 0 or step == 1:
            train_exact, _ = greedy_exact(model, vocab, train[:100], device)
            ood_exact, epp = greedy_exact(model, vocab, test, device)
            probe_readout = probe_model(model, vocab, probe_sets, device)
            trajectory.append({
                "step": step, "tokens": tokens, "exposures": step * batch / len(train),
                "loss": float(loss.detach()), "train_exact": train_exact,
                "test_exact": ood_exact, "test_per_position": epp,
                **probe_readout,
            })
        if time.perf_counter() - started > box_s:
            aborted = True
            break
    summary = m.sustained_summary(trajectory)
    return {
        "seed": seed,
        "status": "ABORTED_WALL_BOX" if aborted else "COMPLETED",
        "steps_run": step,
        "wall_seconds": time.perf_counter() - started,
        "supervised_tokens": tokens,
        "sustained": summary,
        "trajectory": trajectory,
        "plan_commit_sha256": plan_sha,
        "manifest_split_sha256": manifest["split_sha256"],
        "device": str(device),
        "torch": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--box", type=float, default=2400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-004A/TASK_MANIFEST.json"))
    device = torch.device(args.device if (args.device != "cuda" and torch.cuda.is_available() or args.device == "cpu") else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    result = run_seed(args.seed, manifest=manifest, box_s=args.box, steps=args.steps,
                      batch=args.batch, lr=1e-3, device=device, plan_sha=PLAN_SHA)
    receipt = m.bind_receipt(
        experiment_id="ARK-004A",
        plan_commit_sha=PLAN_SHA,
        code_paths={
            "runner": str(Path(__file__)),
            "harness": str(REPO / "experiments/ARK-001/run_ark001.py"),
            "metrics": str(REPO / "experiments/lib/ark_metrics.py"),
            "tasks": str(REPO / "experiments/lib/ark_tasks.py"),
        },
        config={"seed": args.seed, "manifest_split_sha256": manifest["split_sha256"],
                "batch": args.batch, "lr": 1e-3, "wall_box_s": args.box,
                "device": str(device), "torch": torch.__version__},
        results=result,
    )
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    su = result["sustained"]
    print(json.dumps({"seed": args.seed, "status": result["status"],
                      "steps": result["steps_run"], "M99": su["M99"]["step"],
                      "G50": su["G50"]["step"], "G90": su["G90"],
                      "final_ood": result["trajectory"][-1]["test_exact"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
