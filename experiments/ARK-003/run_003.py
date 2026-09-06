"""ARK-003 runner: four matched arms on the frozen T2 manifest.

Arms differ ONLY in the training stream (mission section 6). Everything else —
model, vocab (20 tokens incl. ';'), optimizer, steps, box, eval cadence — is
shared. Receipts bind the ARK-003 plan commit.
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

PLAN_SHA = "9d06a4920dabeca5802bd5cba8a9c39d417079ee"
TEACHER_RATE = 0.40
CURRICULUM_FRACTION = 0.25


class VocabWithSeparator(CompactVocab):
    """Compact vocab + ';' (same size for every arm; controls never see it)."""

    def __init__(self) -> None:
        super().__init__()
        self.table[";"] = len(self.table)
        self.size = len(self.table)


def parse_answer(prompt: str) -> tuple[int, int]:
    left, right = prompt.split("=")[0].split("+")
    return int(left), int(right)


def teacher_suffix(prompt: str, answer: str) -> str:
    """Aligned decomposition: ones rule, then tens rule (no-carry tier)."""
    a, b = parse_answer(prompt)
    ua, ub = a % 10, b % 10
    ta, tb = a // 10, b // 10
    return f" ; {ua}+{ub}={ua + ub} ; {ta}+{tb}={ta + tb}"


def encode_row(vocab, prompt: str, answer: str, suffix: str = "") -> tuple[list[int], int]:
    ids = vocab.encode(prompt)
    answer_ids = vocab.encode(answer) + [vocab.EOS]
    suffix_ids = vocab.encode(suffix) if suffix else []
    return ids + answer_ids + suffix_ids, len(ids)


def batch_loss(model, vocab, rows_with_suffix, device):
    tokens_list, prompt_lens, supervised = [], [], []
    for prompt, answer, suffix in rows_with_suffix:
        ids, plen = encode_row(vocab, prompt, answer, suffix)
        tokens_list.append(ids)
        prompt_lens.append(plen)
        supervised.append(len(ids) - plen)
    length = max(len(x) for x in tokens_list)
    pad = vocab.PAD
    tokens = torch.full((len(rows_with_suffix), length), pad, dtype=torch.long)
    for i, ids in enumerate(tokens_list):
        tokens[i, : len(ids)] = torch.tensor(ids)
    prompt_len = torch.tensor(prompt_lens, device=device)
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    positions = torch.arange(tokens.shape[1] - 1, device=device)[None, :]
    keep = positions >= (prompt_len - 1)[:, None]
    keep = keep & (targets != pad)
    if int(keep.sum().item()) == 0:
        raise ValueError("no supervised targets")
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    return (losses * keep).sum() / keep.sum(), int(keep.sum().item())


def counterfactual_set(vocab, rows: list[tuple[str, str]], n: int = 100) -> list[dict]:
    """S3: perturb only ub within the no-carry constraint; fixed set."""
    import random

    rng = random.Random(777)
    picked = rng.sample(rows, min(n, len(rows)))
    cases = []
    for prompt, answer in picked:
        a, b = parse_answer(prompt)
        ua, ub = a % 10, b % 10
        ta, tb = a // 10, b // 10
        candidates = [u for u in range(0, 10 - ua) if u != ub]
        if not candidates:
            continue
        ub2 = rng.choice(candidates)
        b2 = tb * 10 + ub2
        cases.append({
            "prompt": prompt, "answer": answer,
            "cf_prompt": f"{a} + {b2} = ", "cf_answer": f"{a + b2}",
            "cf_ones_correct_answer": f"{(ua + ub2) % 10}" if ua + ub2 < 10 else f"{ua + ub2}",
            "cf_tens_answer": f"{ta + tb}",
        })
    return cases


@torch.no_grad()
def counterfactual_locality(model, vocab, cases, device) -> float:
    """Fraction where the ones output adapts AND the tens output is invariant."""
    model.eval()
    ok = 0
    total = 0
    for case in cases:
        ids = vocab.encode(case["cf_prompt"])
        generated: list[int] = []
        for _ in range(4):
            logits = model(torch.tensor([ids + generated], device=device))[0][-1]
            nxt = int(torch.argmax(logits).item())
            if nxt in (vocab.EOS, vocab.PAD):
                break
            generated.append(nxt)
        text = vocab.decode(generated).strip()
        tens_ok = len(text) >= 2 and text[:-1] == case["cf_tens_answer"]
        ones_ok = len(text) >= 1 and text[-1] == case["cf_answer"][-1]
        ok += 1 if (tens_ok and ones_ok) else 0
        total += 1
    model.train()
    return ok / max(1, total)


def build_transfer_sets(manifest: dict) -> dict:
    train_ones = {(int(p.split("+")[0]) % 10, int(p.split("+")[1].split("=")[0].strip()) % 10)
                  for p, _ in manifest["train"]}
    s1 = [row for row in manifest["test"]
          if (int(row[0].split("+")[0]) % 10, int(row[0].split("+")[1].split("=")[0].strip()) % 10)
          not in train_ones]
    s2 = []
    for ha in range(1, 9):
        for hb in range(1, 10 - ha):
            for ma in range(0, 10):
                for mb in range(0, 10 - ma):
                    for ua in range(0, 10):
                        for ub in range(0, 10 - ua):
                            pass
    # 3-digit no-carry: a = ha*100 + ma*10 + ua, b = hb*100 + mb*10 + ub
    import random

    rng = random.Random(555)
    seen = set()
    while len(s2) < 200 and len(seen) < 500_000:
        ha = rng.randrange(1, 9)
        hb = rng.randrange(1, 10 - ha)
        ma, mb = rng.randrange(0, 10), rng.randrange(0, 10 - ma)
        ua, ub = rng.randrange(0, 10), rng.randrange(0, 10 - ua)
        a = ha * 100 + ma * 10 + ua
        b = hb * 100 + mb * 10 + ub
        if (a, b) in seen:
            continue
        seen.add((a, b))
        s2.append((f"{a} + {b} = ", f"{a + b}"))
    return {"S1_composition": s1, "S2_length_transfer": s2}


def run_arm(arm: str, *, manifest: dict, init_seed: int, order_seed: int,
            box_s: float, steps: int, batch: int, lr: float, device,
            checkpoint_dir: Path) -> dict:
    torch.manual_seed(init_seed)
    vocab = VocabWithSeparator()
    model = Micro(vocab.size, 128)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    train = [(p, a) for p, a in manifest["train"]]
    test = [(p, a) for p, a in manifest["test"]]
    t1 = t.t1_pool()
    cf_cases = counterfactual_set(vocab, test)
    transfer = build_transfer_sets(manifest)
    order_gen = torch.Generator().manual_seed(order_seed)
    trajectory: list[dict] = []
    saved_phases: dict[str, int | None] = {"M99": None, "G50": None, "G90": None}
    started = time.perf_counter()
    tokens = 0
    step = 0
    aborted = False
    model.train()
    for step in range(1, steps + 1):
        if arm == "B" and step <= int(steps * CURRICULUM_FRACTION):
            source = t1
        else:
            source = train
        idx = torch.randint(0, len(source), (batch,), generator=order_gen)
        batch_rows = [(source[i][0], source[i][1], "") for i in idx]
        if arm in ("C", "D") and source is train:
            for j in range(batch):
                if torch.rand(1, generator=order_gen).item() < TEACHER_RATE:
                    if arm == "C":
                        batch_rows[j] = (batch_rows[j][0], batch_rows[j][1],
                                         teacher_suffix(batch_rows[j][0], batch_rows[j][1]))
                    else:
                        k = int(torch.randint(0, len(train), (1,), generator=order_gen).item())
                        batch_rows[j] = (batch_rows[j][0], batch_rows[j][1],
                                         teacher_suffix(train[k][0], train[k][1]))
        loss, count = batch_loss(model, vocab, batch_rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += count
        if step % 200 == 0 or step == 1:
            train_exact, _ = greedy_exact(model, vocab, train[:100], device)
            s0, epp = greedy_exact(model, vocab, test, device)
            locality = counterfactual_locality(model, vocab, cf_cases, device)
            trajectory.append({
                "step": step, "tokens": tokens,
                "exposures": step * batch / len(train),
                "loss": float(loss.detach()),
                "train_exact": train_exact, "test_exact": s0,
                "counterfactual_locality": locality,
                "test_per_position": epp,
            })
            for phase, bar, key in (("M99", 0.99, "train_exact"), ("G50", 0.50, "test_exact"),
                                    ("G90", 0.90, "test_exact")):
                if saved_phases[phase] is None:
                    tail = [e for e in trajectory if e["step"] >= step - 400]
                    if len([e for e in tail if float(e[key]) >= bar]) >= 3:
                        saved_phases[phase] = step
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), checkpoint_dir / f"{arm}_{phase}.pt")
        if time.perf_counter() - started > box_s:
            aborted = True
            break
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / f"{arm}_final.pt")
    summary = m.sustained_summary(trajectory)
    phase_evals = {}
    for phase, path in (("M99", saved_phases["M99"]), ("G90", saved_phases["G90"]),
                        ("final", step)):
        if path is None:
            continue
        state = torch.load(checkpoint_dir / f"{arm}_{phase if phase != 'final' else 'final'}.pt",
                           map_location=device, weights_only=True)
        model.load_state_dict(state)
        s1, _ = greedy_exact(model, vocab, transfer["S1_composition"][:200], device)
        s2, _ = greedy_exact(model, vocab, transfer["S2_length_transfer"][:200], device)
        phase_evals[phase] = {"step": path, "S1_composition": s1, "S2_length_transfer": s2}
        # restore the final weights after intermediate phase evaluation
        if phase != "final":
            final_state = torch.load(checkpoint_dir / f"{arm}_final.pt",
                                     map_location=device, weights_only=True)
            model.load_state_dict(final_state)
    return {
        "arm": arm,
        "init_seed": init_seed,
        "order_seed": order_seed,
        "status": "ABORTED_WALL_BOX" if aborted else "COMPLETED",
        "steps_run": step,
        "wall_seconds": time.perf_counter() - started,
        "parameters": params,
        "supervised_tokens": tokens,
        "saved_phase_steps": saved_phases,
        "sustained": summary,
        "phase_transfer_evals": phase_evals,
        "trajectory_tail": trajectory[-8:],
        "trajectory_full": trajectory,
        "counterfactual_cases": len(cf_cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("A", "B", "C", "D"))
    parser.add_argument("--init-seed", type=int, default=29)
    parser.add_argument("--order-seed", type=int, default=29)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--box", type=float, default=1800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    manifest = t.load_or_build_manifest(str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"))
    result = run_arm(
        args.arm, manifest=manifest, init_seed=args.init_seed,
        order_seed=args.order_seed, box_s=args.box, steps=args.steps,
        batch=64, lr=1e-3, device=torch.device(args.device),
        checkpoint_dir=REPO / "experiments/ARK-003/checkpoints",
    )
    receipt = m.bind_receipt(
        experiment_id="ARK-003",
        plan_commit_sha=PLAN_SHA,
        code_paths={
            "runner": str(Path(__file__)),
            "harness": str(REPO / "experiments/ARK-001/run_ark001.py"),
            "metrics": str(REPO / "experiments/lib/ark_metrics.py"),
            "tasks": str(REPO / "experiments/lib/ark_tasks.py"),
            "manifest": str(REPO / "experiments/ARK-002B/TASK_MANIFEST.json"),
        },
        config={
            "arm": args.arm, "manifest_split_sha256": manifest["split_sha256"],
            "teacher_rate": TEACHER_RATE, "curriculum_fraction": CURRICULUM_FRACTION,
            "init_seed": args.init_seed, "order_seed": args.order_seed,
            "batch": 64, "lr": 1e-3, "wall_box_s": args.box,
            "device": args.device, "torch": torch.__version__,
            "screening": True,
        },
        results=result,
    )
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "status": result["status"],
                      "G90": result["sustained"]["G90"],
                      "final_ood": result["trajectory_tail"][-1]["test_exact"] if result["trajectory_tail"] else None,
                      "locality": result["trajectory_tail"][-1]["counterfactual_locality"] if result["trajectory_tail"] else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
