"""CYR-GPU-004: P35-proxy LR-retention test using the REAL Cymek V5 architecture.

Uses v5_contracts.model_spec.ModelSpec + v5_model.core.initialize().
Production 24,576 tokenizer. Paired continuation with pre-generated indices.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize

# ---- tokenizer ----
def load_tokenizer(repo_path):
    import gzip
    from tokenizers import Tokenizer
    path = Path(repo_path) / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return Tokenizer.from_str(f.read())

# ---- data ----
def build_t2_data(seed=13):
    import random
    rng = random.Random(seed)
    tens = range(1, 6)
    train, seen = [], set()
    while len(train) < 500:
        ta = rng.choice([1,2,3,4,5]); ua = rng.randrange(10)
        tb = rng.randrange(1, 10-ta); ub = rng.randrange(0, 10-ua)
        a, b = ta*10+ua, tb*10+ub
        if (a,b) in seen: continue
        seen.add((a,b)); train.append((f"{a} + {b} = ", f"{a+b}"))
    tens_test = range(6, 8)
    test, seen_t = [], set()
    while len(test) < 197:
        ta = rng.choice([6,7]); ua = rng.randrange(10)
        tb = rng.randrange(1, 10-ta); ub = rng.randrange(0, 10-ua)
        a, b = ta*10+ua, tb*10+ub
        if (a,b) in seen_t or (a,b) in seen: continue
        seen_t.add((a,b)); test.append((f"{a} + {b} = ", f"{a+b}"))
    return train, test

# ---- eval ----
@torch.no_grad()
def greedy_exact(model, tokenizer, rows, device, max_answer=6):
    model.eval()
    correct = 0
    for prompt, answer in rows:
        ids = tokenizer.encode(prompt).ids
        gen = []
        for _ in range(max_answer):
            input_t = torch.tensor([ids + gen], device=device)
            seq_len = input_t.shape[1]
            positions = torch.arange(seq_len, device=device)[None, :]
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))[None, None, :, :]
            logits = model(input_t, positions, mask)[0, -1]
            nxt = int(torch.argmax(logits).item())
            if nxt in (0, 3): break
            gen.append(nxt)
        text = tokenizer.decode(gen)
        if text == answer: correct += 1
    model.train()
    return correct / len(rows)

def loss_and_sup(model, tokenizer, rows, device):
    prompts = [tokenizer.encode(p).ids for p, _ in rows]
    answers = [tokenizer.encode(a).ids + [3] for _, a in rows]
    length = max(len(p)+len(a) for p, a in zip(prompts, answers))
    tokens = torch.full((len(rows), length), 0, dtype=torch.long)
    plen = torch.zeros(len(rows), dtype=torch.long)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        tokens[i, :len(p)] = torch.tensor(p)
        tokens[i, len(p):len(p)+len(a)] = torch.tensor(a)
        plen[i] = len(p)
    tokens = tokens.to(device); plen = plen.to(device)
    seq_len = tokens.shape[1] - 1
    positions = torch.arange(seq_len, device=device)[None, :]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))[None, None, :, :]
    logits = model(tokens[:, :-1], positions, mask)
    targets = tokens[:, 1:]
    pos = torch.arange(tokens.shape[1]-1, device=device)[None, :]
    keep = (pos >= (plen-1)[:, None]) & (targets != 0)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), reduction="none").view(targets.shape)
    return (losses * keep).sum() / keep.sum(), int(keep.sum().item())

# ---- G90 ----
def detect_g90(eval_steps, eval_ood, bar=0.90, consec=3):
    streak, onset = 0, None
    for step, value in zip(eval_steps, eval_ood):
        if value >= bar:
            if streak == 0: onset = step
            streak += 1
            if streak >= consec: return onset, step
        else: streak, onset = 0, None
    return None, None

# ---- retention metrics ----
def retention_metrics(traj):
    if not traj: return {"status": "EMPTY"}
    ood = [e["test_exact"] for e in traj]
    ret90 = sum(1 for v in ood if v >= 0.90) / len(ood)
    ret50 = sum(1 for v in ood if v >= 0.50) / len(ood)
    area = sum(ood) / len(ood)
    c90, streak = None, 0
    for e in traj:
        if e["test_exact"] < 0.90:
            streak += 1
            if streak >= 3: c90 = e["step"]; break
        else: streak = 0
    peak, final = max(ood), ood[-1]
    return {"RET90": round(ret90,4), "RET50": round(ret50,4),
            "AREA": round(area,4), "T_COLLAPSE_90": c90,
            "PEAK_G": round(peak,4), "GAP": round(peak-final,4),
            "FINAL": round(final,4), "collapsed": c90 is not None}

# ---- parameter displacement ----
def _flat(model): return torch.cat([p.detach().data.reshape(-1) for p in model.parameters()])
@torch.no_grad()
def displacement(model, g90_flat):
    current = _flat(model)
    diff = (current - g90_flat).norm().item()
    return {"l2": round(diff, 3), "rel": round(diff / max(g90_flat.norm().item(), 1e-8), 5)}

# ---- main experiment ----
def run_arm(*, seed, lr, train, test, tokenizer, spec, device,
            acq_steps=16000, post_steps=8000, batch=32, eval_every=200):
    torch.manual_seed(seed)
    model = initialize(spec, seed=seed).to(device)
    pcount = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(seed)
    eval_steps, eval_ood, traj = [], [], []
    onset, confirm = None, None
    acq_tokens = 0
    model.train()
    t0 = time.time()
    for step in range(1, acq_steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        rows = [train[i] for i in idx]
        loss, sup = loss_and_sup(model, tokenizer, rows, device)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        acq_tokens += sup
        if step % eval_every == 0 or step == 1:
            tr = greedy_exact(model, tokenizer, train[:50], device)
            te = greedy_exact(model, tokenizer, test, device)
            traj.append({"step": step, "train_exact": tr, "test_exact": te})
            eval_steps.append(step); eval_ood.append(te)
            onset, confirm = detect_g90(eval_steps, eval_ood)
            if confirm is not None:
                print(f"  G90 onset={onset} confirm={confirm} step={step}", flush=True)
                break

    if confirm is None:
        return {"seed": seed, "lr": lr, "status": "NO_G90",
                "steps": step, "params": pcount,
                "peak_test": max((e["test_exact"] for e in traj), default=0)}

    # snapshot at G90 confirmation
    snap_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    snap_opt = {k: v for k, v in optimizer.state_dict().items()}
    g90_flat = _flat(model)

    # fork: continue with same LR for post_steps
    post_traj = []
    model.train()
    for step in range(1, post_steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        rows = [train[i] for i in idx]
        loss, sup = loss_and_sup(model, tokenizer, rows, device)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % eval_every == 0:
            te = greedy_exact(model, tokenizer, test, device)
            disp = displacement(model, g90_flat)
            post_traj.append({"step": step, "test_exact": te, **disp})

    ret = retention_metrics(post_traj)
    return {"seed": seed, "lr": lr, "status": "COMPLETE",
            "acq_steps": confirm or step, "post_steps": len(post_traj),
            "params": pcount, **ret,
            "post_traj": post_traj, "acq_traj": traj}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--acq-steps", type=int, default=16000)
    parser.add_argument("--post-steps", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(REPO)
    train, test = build_t2_data()
    spec = ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=24576, width=256, layers=6,
        query_heads=4, kv_heads=2, head_dimension=64,
        ffn_width=768, context_length=512,
        rope_base=10000.0, norm_epsilon=1e-5,
        tied_embeddings=True, qk_norm=True, qk_norm_affine=True,
        linear_bias=False, dropout=0.0)

    result = run_arm(seed=args.seed, lr=args.lr, train=train, test=test,
                     tokenizer=tokenizer, spec=spec, device=device,
                     acq_steps=args.acq_steps, post_steps=args.post_steps)

    receipt = {"experiment": "CYR-GPU-004", "plan_commit": "c34cc07",
               "seed": args.seed, "lr": args.lr,
               "manifest_split_sha256": MANIFEST_SHA,
               "spec_sha256": spec.sha256(), "params": result["params"],
               "device": str(device), "torch": torch.__version__,
               **result}
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "lr": args.lr, "status": result["status"],
                      "RET90": result.get("RET90"), "FINAL": result.get("FINAL")}))


MANIFEST_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"

if __name__ == "__main__":
    main()
