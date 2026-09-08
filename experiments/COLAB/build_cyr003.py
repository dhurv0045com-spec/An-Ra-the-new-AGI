"""Builds the CYR-GPU-003 Colab notebook: P35-scale LR-retention replication.

Self-contained. Uses the production 24,576 tokenizer (downloaded from the repo).
Model: Cymek P35 architecture scaled to fit T4 session (8 layers, width 256).
Runs: 2 seeds → acquisition to G90 → fork HIGH/LOW → 8k post-confirmation → retention.
"""
import json

def build():
    cells = []

    def md(src):
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": [l + "\n" for l in src.rstrip("\n").split("\n")]})

    def code(src):
        cells.append({"cell_type": "code", "metadata": {},
                      "source": src.splitlines(keepends=True),
                      "outputs": [], "execution_count": None})

    md("""# CYR-GPU-003 — P35-scale LR-retention protection replication

**Runtime: T4 GPU (NOT TPU). Runtime → Change runtime type → T4 GPU → Run all.**

| Phase | What | Steps |
|-------|------|-------|
| Setup | Clone repo, load production 24,576 tokenizer, verify manifest | — |
| Acquisition | Train 2 seeds to sustained G90 on T2 arithmetic | ~32k |
| Fork | At G90 confirmation → HIGH (1e-3) and LOW (1e-5) | — |
| Retention | 8k post-confirmation steps per arm | 32k |
| **Total** | | **~64k steps ≈ 60–90 min on T4** |

**Question:** Does dropping LR from 1e-3 to 1e-5 at the generalization
transition prevent the post-generalization collapse discovered at micro scale?

**Model:** Cymek P35 architecture scaled to 8 layers / 256 width (≈8M params)
with the production 24,576-token vocabulary. Architecture is identical to
the P35 spec except depth/width (evidence: vocab/width are NOT first-order
variables at micro scale per ARK-001).

**Data:** Frozen T2 manifest (500 train / 197 test, structural tens-band
holdout, zero commutation overlap).

**Results auto-download as CYR-GPU-003_RESULTS.zip at the end.**""")

    code(r'''# ======== CELL 0: SETUP ========
import subprocess, sys, os, json, hashlib, math, time

# Clone the repo at the plan commit
REPO = "/content/An-Ra-the-new-AGI"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--branch", "cymek-500m-readiness",
                    "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git", REPO],
                   check=True)
os.chdir(REPO)

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)

# Verify tokenizer artifact
TOK_PATH = os.path.join(REPO, "artifacts/e1/local_tournament/tokenizer-24576.json.gz")
assert os.path.exists(TOK_PATH), f"tokenizer not found at {TOK_PATH}"
tok_sha = hashlib.sha256(open(TOK_PATH, "rb").read()).hexdigest()
print(f"tokenizer sha256: {tok_sha[:16]}...")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random, gzip, time
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE} | torch: {torch.__version__}")

# Load the production 24,576 tokenizer
from tokenizers import Tokenizer
with gzip.open(TOK_PATH, "rt", encoding="utf-8") as f:
    TOKENIZER = Tokenizer.from_str(f.read())
assert TOKENIZER.get_vocab_size() == 24576
print(f"tokenizer loaded: {TOKENIZER.get_vocab_size()} tokens")

# Frozen T2 manifest (rebuilt deterministically, asserted against committed hash)
import random as _random
def _rows(split, n, ds=13):
    rng = _random.Random(ds)
    tens = range(1,6) if split=="train" else range(6,8)
    rows, seen, guard = [], set(), 0
    while len(rows) < n and guard < 2000000:
        guard += 1
        ta = rng.choice(list(tens)); ua = rng.randrange(10)
        tb = rng.randrange(1, 10-ta); ub = rng.randrange(0, 10-ua)
        a, b = ta*10+ua, tb*10+ub
        if (a,b) in seen: continue
        seen.add((a,b)); rows.append((f"{a} + {b} = ", f"{a+b}"))
    assert len(rows) == n; return rows

train_rows = _rows("train", 500)
raw_test = _rows("test", 260)
tp = {tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0])))) for p,_ in train_rows}
test_rows = []
for p, a in raw_test:
    pair = tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0]))))
    if pair not in tp:
        test_rows.append((p, a))
        if len(test_rows) == 200: break
MANIFEST_SHA = hashlib.sha256(json.dumps(
    {"train": [[p, a] for p, a in train_rows], "test": [[p, a] for p, a in test_rows]},
    sort_keys=True).encode()).hexdigest()
print(f"manifest sha: {MANIFEST_SHA[:12]} (500 train / {len(test_rows)} test)")

def encode_rows(vocab_tokenizer, rows, device):
    """Tokenize arithmetic rows using the production tokenizer."""
    prompts = [vocab_tokenizer.encode(p).ids for p, _ in rows]
    answers = [vocab_tokenizer.encode(a).ids + [3] for _, a in rows]  # 3=EOS
    length = max(len(p)+len(a) for p, a in zip(prompts, answers))
    tokens = torch.full((len(rows), length), 0, dtype=torch.long)  # 0=PAD
    plen = torch.zeros(len(rows), dtype=torch.long)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        tokens[i, :len(p)] = torch.tensor(p)
        tokens[i, len(p):len(p)+len(a)] = torch.tensor(a)
        plen[i] = len(p)
    return tokens.to(device), plen.to(device)

def loss_and_sup(model, tok, rows, device):
    tokens, plen = encode_rows(tok, rows, device)
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    pos = torch.arange(tokens.shape[1]-1, device=device)[None, :]
    keep = (pos >= (plen-1)[:, None]) & (targets != 0)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), reduction="none").view(targets.shape)
    return (losses * keep).sum() / keep.sum(), int(keep.sum().item())

@torch.no_grad()
def greedy_exact(model, tok, rows, device, max_answer=6):
    model.eval()
    correct = 0
    for prompt, answer in rows:
        ids = tok.encode(prompt).ids
        gen = []
        for _ in range(max_answer):
            logits = model(torch.tensor([ids + gen], device=device))[0][-1]
            nxt = int(torch.argmax(logits).item())
            if nxt in (0, 3): break  # PAD or EOS
            gen.append(nxt)
        # decode: strip special tokens, get text
        text = tok.decode(gen)
        if text == answer: correct += 1
    model.train()
    return correct / len(rows)

# P35-scale model: same architecture, reduced depth/width for T4 session
class RMSNorm(nn.Module):
    def __init__(self, w, eps=1e-5):
        super().__init__(); self.weight = nn.Parameter(torch.ones(w)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)

class Block(nn.Module):
    def __init__(self, w, heads, kv_heads, ffn):
        super().__init__()
        self.h = heads; self.kv = kv_heads; self.hd = w // heads
        self.n1, self.n2 = RMSNorm(w), RMSNorm(w)
        self.q = nn.Linear(w, w, bias=False)
        self.k = nn.Linear(w, w // heads * kv_heads // heads, bias=False)
        self.v = nn.Linear(w, w // heads * kv_heads // heads, bias=False)
        self.proj = nn.Linear(w, w, bias=False)
        self.gate = nn.Linear(w, ffn, bias=False)
        self.up = nn.Linear(w, ffn, bias=False)
        self.down = nn.Linear(ffn, w, bias=False)
    def forward(self, x):
        b, t, w = x.shape
        h = self.n1(x)
        q = self.q(h).view(b, t, self.h, self.hd).transpose(1, 2)
        k = self.k(h).view(b, t, self.kv, self.hd).transpose(1, 2)
        v = self.v(h).view(b, t, self.kv, self.hd).transpose(1, 2)
        # GQA: repeat KV to match Q heads
        k = k.repeat_interleave(self.h // self.kv, dim=1)
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        pos = torch.arange(t, device=x.device)
        inv = 10000.0 ** (-torch.arange(0, self.hd, 2, device=x.device).float() / self.hd)
        ph = pos.float()[:, None] * inv[None, :]
        cos, sin = ph.cos(), ph.sin()
        def rope(z):
            ze, zo = z[..., 0::2], z[..., 1::2]
            ce = torch.repeat_interleave(cos, 2, -1)[None, None]
            se = torch.repeat_interleave(sin, 2, -1)[None, None]
            return torch.cat((ze*ce[..., :ze.shape[-1]] - zo*se[..., :ze.shape[-1]],
                              ze*se[..., :ze.shape[-1]] + zo*ce[..., :ze.shape[-1]]), -1)
        q, k = rope(q), rope(k)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf"))
        out = torch.softmax(att, -1) @ v
        out = out.transpose(1, 2).contiguous().view(b, t, w)
        x = x + self.proj(out)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

class P35Scaled(nn.Module):
    """Cymek P35 architecture at 8L/256w for T4 session. Production 24,576 tokenizer."""
    def __init__(self, vocab_size=24576, width=256, layers=8, heads=4, kv_heads=2, ffn=768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList(Block(width, heads, kv_heads, ffn) for _ in range(layers))
        self.norm = RMSNorm(width)
    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks: x = blk(x)
        return self.norm(x) @ self.embed.weight.T

print("setup complete")
''')

    code(r'''# ======== CELL 1: ACQUISITION + FORK + RETENTION (seeds 707, 808) ========
import time as _time

results = []
for seed in (707, 808):
    if minutes_left() < 20:
        print(f"seed {seed}: SKIPPED_BUDGET"); continue
    print(f"\n=== seed {seed} ===", flush=True)
    torch.manual_seed(seed)
    model = P35Scaled(vocab_size=24576).to(DEVICE)
    pcount = sum(p.numel() for p in model.parameters())
    print(f"  params: {pcount:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(seed)
    eval_steps, eval_ood, traj = [], [], []
    onset, confirm = None, None
    model.train()
    t0 = _time.time()
    for step in range(1, 16001):
        idx = torch.randint(0, len(train_rows), (64,), generator=rng)
        rows = [train_rows[i] for i in idx]
        loss, _ = loss_and_sup(model, TOKENIZER, rows, DEVICE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0 or step == 1:
            tr = greedy_exact(model, TOKENIZER, train_rows[:100], DEVICE)
            te = greedy_exact(model, TOKENIZER, test_rows, DEVICE)
            traj.append({"step": step, "train_exact": tr, "test_exact": te})
            eval_steps.append(step); eval_ood.append(te)
            onset, confirm = detect_g90_oc(eval_steps, eval_ood)
            if confirm is not None:
                print(f"  G90 onset={onset} confirm={confirm} step={step}", flush=True)
                break
    if confirm is None:
        print(f"  seed {seed}: G90 not reached in 16k steps")
        results.append({"seed": seed, "status": "NO_TRIGGER"}); continue

    # snapshot at confirmation
    snap = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    snap_opt = {k: v for k, v in opt.state_dict().items()}
    g90_flat = torch.cat([p.detach().data.reshape(-1) for p in model.parameters()])

    # fork into HIGH and LOW
    for arm_name, arm_lr in (("HIGH", 1e-3), ("LOW", 1e-5)):
        if minutes_left() < 8:
            print(f"  {arm_name}: SKIPPED_BUDGET"); break
        print(f"  {arm_name} (lr={arm_lr})", flush=True)
        model.load_state_dict(snap)
        opt.load_state_dict(snap_opt)
        for g in opt.param_groups: g["lr"] = arm_lr
        post_traj = []
        model.train()
        for step in range(1, 8001):
            idx = torch.randint(0, len(train_rows), (64,), generator=rng)
            rows = [train_rows[i] for i in idx]
            loss, _ = loss_and_sup(model, TOKENIZER, rows, DEVICE)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 200 == 0:
                te = greedy_exact(model, TOKENIZER, test_rows, DEVICE)
                post_traj.append({"step": step, "test_exact": te})
        ret90 = sum(1 for e in post_traj if e["test_exact"] >= 0.90) / max(1, len(post_traj))
        area = sum(e["test_exact"] for e in post_traj) / max(1, len(post_traj))
        final = post_traj[-1]["test_exact"] if post_traj else 0
        results.append({"seed": seed, "arm": arm_name, "lr": arm_lr,
                        "trigger_step": confirm, "RET90": round(ret90, 4),
                        "FINAL_OOD": round(final, 4),
                        "post_traj": post_traj})
        print(f"  {arm_name}: RET90={ret90:.3f} final={final:.3f}", flush=True)

save_result("CYR-GPU-003_RETENTION.json", {"results": results,
    "plan_sha": "c34cc072c9bc6fe0c666a0f5c9beca13cfd7b838",
    "manifest_sha": MANIFEST_SHA,
    "model_params": pcount,
    "tokenizer_vocab": 24576})
''')

    code(r'''# ======== CELL 2: SUMMARY + DOWNLOAD ========
summary = {"device": DEVICE, "torch": torch.__version__,
           "manifest_sha": MANIFEST_SHA, "results": results}
print(json.dumps(summary, indent=1, default=str))
try:
    from google.colab import files
    import shutil
    shutil.make_archive("/content/CYR-GPU-003_RESULTS", "zip", RESULTS_DIR)
    files.download("/content/CYR-GPU-003_RESULTS.zip")
except Exception as exc:
    print("manual download from", RESULTS_DIR, ":", exc)
''')

    return cells


def build():
    cells = build()
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"provenance": [], "name": "CYR-GPU-003.ipynb"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU"},
                "cells": cells}
    out = Path(__file__).parent / "CYR-GPU-003.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    for cell in parsed["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cell", "exec")
    print(f"notebook: {out} | cells: {len(parsed['cells'])} | all compile")

if __name__ == "__main__":
    build()
