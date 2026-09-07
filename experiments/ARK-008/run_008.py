"""ARK-008: transfer the LR-retention protection law to binding-v2.

If the LR-threshold retention law (discovered on arithmetic) transfers to
binding/query-conditioned selection, it becomes a general cognition principle.
If it doesn't, it's task-specific. Either result is important.
"""

from __future__ import annotations

import argparse
import hashlib
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
from run_ark001 import Micro, CompactVocab  # noqa: E402


def build_binding_tasks(n_train=200, n_test=50, seed=42):
    """Generate paired binding worlds: facts + query -> answer.

    Simple synthetic binding: entities have colors, query asks about one.
    Train and test use disjoint entity names (structural holdout).
    """
    import random
    rng = random.Random(seed)

    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    train_entities = [f"obj_{i}" for i in range(50)]
    test_entities = [f"tst_{i}" for i in range(50)]

    def make_row(entities, rng):
        # 3-5 facts
        n_facts = rng.randint(3, 5)
        facts = []
        assignments = {}
        used_colors = set()
        for _ in range(n_facts):
            ent = rng.choice(entities)
            if ent in assignments:
                continue
            col = rng.choice([c for c in colors if c not in used_colors])
            used_colors.add(col)
            assignments[ent] = col
            facts.append((ent, col))
        query_ent = rng.choice(list(assignments.keys()))
        query = f"What color is {query_ent}?"
        answer = assignments[query_ent]
        prompt = ". ".join(f"The {e} is {c}" for e, c in facts) + f". {query} "
        return (prompt, answer)

    train = [make_row(train_entities, rng) for _ in range(n_train)]
    test = [make_row(test_entities, rng) for _ in range(n_test)]
    return train, test


def encode_binding(vocab, prompt, answer):
    """Encode using the compact vocab; unknown chars map to pad."""
    ids = [vocab.BOS]
    for ch in prompt:
        ids.append(vocab.table.get(ch, vocab.PAD))
    ans_ids = []
    for ch in answer:
        ans_ids.append(vocab.table.get(ch, vocab.EOS))
    return ids, ans_ids


def binding_batch(model, vocab, rows, device):
    """Compute binding-task loss with answer+EOS supervision."""
    all_tokens, all_plens = [], []
    for prompt, answer in rows:
        p_ids = vocab.encode(prompt)
        a_ids = [vocab.table.get(c, vocab.PAD) for c in answer] + [vocab.EOS]
        all_tokens.append(p_ids + a_ids)
        all_plens.append(len(p_ids))
    length = max(len(t) for t in all_tokens)
    tokens = torch.full((len(rows), length), vocab.PAD, dtype=torch.long)
    plen = torch.zeros(len(rows), dtype=torch.long)
    for i, t in enumerate(all_tokens):
        tokens[i, :len(t)] = torch.tensor(t)
        plen[i] = all_plens[i]
    tokens = tokens.to(device)
    plen = plen.to(device)
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    pos = torch.arange(tokens.shape[1] - 1, device=device)[None, :]
    keep = (pos >= (plen - 1)[:, None]) & (targets != vocab.PAD)
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        reduction="none").view(targets.shape)
    return (losses * keep).sum() / keep.sum(), int(keep.sum().item())


@torch.no_grad()
def binding_exact(model, vocab, rows, device, max_answer=8):
    model.eval()
    correct = 0
    for prompt, answer in rows:
        ids = vocab.encode(prompt)
        gen = []
        for _ in range(max_answer):
            logits = model(torch.tensor([ids + gen], device=device))[0][-1]
            nxt = int(torch.argmax(logits).item())
            if nxt in (vocab.EOS, vocab.PAD):
                break
            gen.append(nxt)
        text = vocab.decode(gen).strip()
        if text == answer:
            correct += 1
    model.train()
    return correct / len(rows)


def run_binding_retention(*, seed: int, lr_mult: float, device,
                          max_steps: int = 6000, trigger_threshold: float = 0.8,
                          post_steps: int = 4000, box_s: float = 1200) -> dict:
    """Train on binding-v2-style task, then fork at peak into LR arms."""
    torch.manual_seed(seed)
    train_rows, test_rows = build_binding_tasks(n_train=200, n_test=50, seed=seed)
    # build extended vocab from actual task text (binding uses letters, not just digits)
    all_text = set()
    for prompt, answer in train_rows + test_rows:
        all_text.update(prompt)
        all_text.update(answer)
    vocab = CompactVocab()
    for ch in sorted(all_text):
        if ch not in vocab.table:
            vocab.table[ch] = vocab.size
            vocab.size += 1
    model = Micro(vocab.size, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(seed)
    trajectory = []
    eval_steps, eval_ood = [], []
    trigger = None
    started = time.perf_counter()
    model.train()

    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train_rows), (32,), generator=rng)
        rows = [train_rows[i] for i in idx]
        loss, _ = binding_batch(model, vocab, rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == 1:
            train_acc = binding_exact(model, vocab, train_rows[:50], device)
            test_acc = binding_exact(model, vocab, test_rows, device)
            trajectory.append({"step": step, "train_exact": train_acc,
                               "test_exact": test_acc})
            eval_steps.append(step)
            eval_ood.append(test_acc)
            if trigger is None and test_acc >= trigger_threshold:
                trigger = step
        if time.perf_counter() - started > box_s:
            break

    if trigger is None:
        return {"seed": seed, "lr_mult": lr_mult, "status": "NO_TRIGGER",
                "final_test": trajectory[-1]["test_exact"] if trajectory else 0,
                "peak_test": max((e["test_exact"] for e in trajectory), default=0)}

    # fork at trigger: snapshot and continue with either high or low LR
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_state = {k: v for k, v in optimizer.state_dict().items()}
    torch_rng = torch.get_rng_state()
    post_traj = []

    new_lr = 1e-3 * lr_mult
    for group in optimizer.param_groups:
        group["lr"] = new_lr

    peak_test = max(e["test_exact"] for e in trajectory)
    model.train()
    for step in range(1, post_steps + 1):
        idx = torch.randint(0, len(train_rows), (32,), generator=rng)
        rows = [train_rows[i] for i in idx]
        loss, _ = binding_batch(model, vocab, rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0:
            test_acc = binding_exact(model, vocab, test_rows, device)
            post_traj.append({"step": max_steps + step, "test_exact": test_acc})

    final_test = post_traj[-1]["test_exact"] if post_traj else peak_test
    peak_overall = max(peak_test, max((e["test_exact"] for e in post_traj), default=0))
    return {
        "seed": seed, "lr_mult": lr_mult, "actual_lr": new_lr,
        "trigger_step": trigger, "post_steps_run": len(post_traj) * 200,
        "peak_test": round(peak_test, 4),
        "final_test": round(final_test, 4),
        "retention_ratio": round(final_test / max(peak_test, 1e-8), 4),
        "trajectory_tail": post_traj[-5:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="experiments/ARK-008/RESULT.json")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results = []
    for seed in (505, 606):
        for mult, label in ((1.0, "HIGH"), (0.01, "LOW")):
            print(f"=== seed {seed} LR x{mult} ({label}) ===", flush=True)
            result = run_binding_retention(seed=seed, lr_mult=mult, device=device)
            result["label"] = label
            results.append(result)
            print(json.dumps(result, indent=1), flush=True)

    # paired analysis
    high_finals = [r["final_test"] for r in results if r["label"] == "HIGH"]
    low_finals = [r["final_test"] for r in results if r["label"] == "LOW"]
    high_ret = [r["retention_ratio"] for r in results if r["label"] == "HIGH"]
    low_ret = [r["retention_ratio"] for r in results if r["label"] == "LOW"]

    receipt = {
        "schema": "arkenstone-ark008/v1",
        "question": "Does LR-retention protection transfer from arithmetic to binding?",
        "device": str(device), "torch": torch.__version__,
        "results": results,
        "summary": {
            "high_lr_final": [round(f, 3) for f in high_finals],
            "low_lr_final": [round(f, 3) for f in low_finals],
            "high_lr_retention": [round(r, 3) for r in high_ret],
            "low_lr_retention": [round(r, 3) for r in low_ret],
            "transfers": all(lr > h for lr, h in zip(low_ret, high_ret)),
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    main()
