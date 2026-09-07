"""Builds the complete Arkenstone Colab notebook."""
import json
from pathlib import Path

MANIFEST_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"

def make_harness():
    return r'''# ======== ARKENSTONE SHARED HARNESS ========
import os, sys, json, math, time, hashlib, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
RESULTS_DIR = Path("/content/arkenstone_results"); RESULTS_DIR.mkdir(exist_ok=True)
BUDGET_MINUTES = 110
START_TIME = time.time()
def minutes_left(): return BUDGET_MINUTES - (time.time() - START_TIME) / 60
DEVICE_KIND, DEVICE = "cpu", torch.device("cpu")
try:
    import torch_xla; import torch_xla.core.xla_model as xm
    if xm.xla_device() is not None: DEVICE_KIND, DEVICE = "tpu", xm.xla_model()
except Exception: pass
if DEVICE_KIND == "cpu" and torch.cuda.is_available():
    DEVICE, DEVICE_KIND = torch.device("cuda"), "cuda"
print("DEVICE:", DEVICE_KIND, DEVICE)
def xla_sync():
    if DEVICE_KIND == "tpu": xm.mark_step()
COMPACT = ["<pad>","<bos>","<eos>","0","1","2","3","4","5","6","7","8","9","+","-","*","/","="," "]
class CompactVocab:
    PAD, BOS, EOS = 0, 1, 2
    def __init__(self): self.table = {t:i for i,t in enumerate(COMPACT)}; self.size = len(COMPACT)
    def encode(self, text): return [self.BOS] + [self.table[c] for c in text]
    def decode(self, ids):
        inv = {i:t for t,i in self.table.items()}
        return "".join(inv.get(i,"") for i in ids if i not in (0,1,2))
def _rows(split, n, ds=13):
    rng = random.Random(ds)
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
def build_manifest():
    train = _rows("train", 500); raw_test = _rows("test", 260)
    tp = {tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0])))) for p,_ in train}
    test = []
    for p, a in raw_test:
        pair = tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0]))))
        if pair in tp: continue
        test.append((p, a))
        if len(test) == 200: break
    t1 = [(f"{a} + {b} = ", f"{a+b}") for a in range(10) for b in range(10)]
    man = {"train": train, "test": test, "t1": t1}
    man["sha"] = hashlib.sha256(json.dumps({"train": man["train"], "test": man["test"]}, sort_keys=True).encode()).hexdigest()
    return man
MANIFEST = build_manifest()
EXPECTED = "MANIFEST_SHA_HERE"
assert MANIFEST["sha"] == EXPECTED, "manifest drifted!"
print("manifest OK:", MANIFEST["sha"][:12])
class RMSNorm(nn.Module):
    def __init__(self, w, eps=1e-5):
        super().__init__(); self.weight = nn.Parameter(torch.ones(w)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)
class Block(nn.Module):
    def __init__(self, w, heads, ffn):
        super().__init__(); self.h = heads
        self.n1, self.n2 = RMSNorm(w), RMSNorm(w)
        self.qkv = nn.Linear(w, 3*w, bias=False); self.proj = nn.Linear(w, w, bias=False)
        self.gate = nn.Linear(w, ffn, bias=False); self.up = nn.Linear(w, ffn, bias=False)
        self.down = nn.Linear(ffn, w, bias=False)
    def forward(self, x):
        b, t, w = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        hd = w // self.h
        q = q.view(b, t, self.h, hd).transpose(1,2)
        k = k.view(b, t, self.h, hd).transpose(1,2)
        v = v.view(b, t, self.h, hd).transpose(1,2)
        pos = torch.arange(t, device=x.device)
        inv = 10000.0 ** (-torch.arange(0, hd, 2, device=x.device).float() / hd)
        ph = pos.float()[:, None] * inv[None, :]
        cos, sin = ph.cos(), ph.sin()
        def rope(z):
            ze, zo = z[..., 0::2], z[..., 1::2]
            ce = torch.repeat_interleave(cos, 2, -1)[None, None]
            se = torch.repeat_interleave(sin, 2, -1)[None, None]
            return torch.cat((ze*ce[..., :ze.shape[-1]] - zo*se[..., :ze.shape[-1]], ze*se[..., :ze.shape[-1]] + zo*ce[..., :ze.shape[-1]]), -1)
        q, k = rope(q), rope(k)
        att = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(hd)
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf"))
        out = torch.softmax(att, -1) @ v
        out = out.transpose(1,2).contiguous().view(b, t, w)
        x = x + self.proj(out)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))
class Micro(nn.Module):
    def __init__(self, vocab, width=128, layers=4, ffn=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width, 4, ffn) for _ in range(layers))
        self.norm = RMSNorm(width)
    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks: x = blk(x)
        return self.norm(x) @ self.embed.weight.T
VOCAB = CompactVocab()
def encode_batch(vocab, rows, device):
    prompts = [vocab.encode(p) for p, _ in rows]
    answers = [vocab.encode(a) + [vocab.EOS] for _, a in rows]
    length = max(len(p)+len(a) for p, a in zip(prompts, answers))
    tokens = torch.full((len(rows), length), vocab.PAD, dtype=torch.long)
    plen = torch.zeros(len(rows), dtype=torch.long)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        tokens[i, :len(p)] = torch.tensor(p); tokens[i, len(p):len(p)+len(a)] = torch.tensor(a); plen[i] = len(p)
    return tokens.to(device), plen.to(device)
def loss_fn(model, vocab, rows, device):
    tokens, plen = encode_batch(vocab, rows, device)
    logits = model(tokens[:, :-1]); targets = tokens[:, 1:]
    pos = torch.arange(tokens.shape[1]-1, device=device)[None, :]
    keep = (pos >= (plen-1)[:, None]) & (targets != vocab.PAD)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view(targets.shape)
    return (losses * keep).sum() / keep.sum(), int(keep.sum().item())
@torch.no_grad()
def greedy_exact(model, vocab, rows, device, max_answer=6):
    model.eval()
    groups = {}
    for i, (p, _) in enumerate(rows): groups.setdefault(len(vocab.encode(p)), []).append(i)
    text_of = {}
    for idxs in groups.values():
        brows = [rows[i] for i in idxs]
        tokens = torch.tensor([vocab.encode(p) for p, _ in brows], device=device)
        finished = torch.zeros(len(brows), dtype=torch.bool)
        gen = [[] for _ in brows]
        for _ in range(max_answer):
            logits = model(tokens)[:, -1]; nxt = torch.argmax(logits, -1)
            tokens = torch.cat([tokens, torch.full((len(brows),1), vocab.PAD, dtype=torch.long, device=device)], 1)
            done = True
            for i in range(len(brows)):
                if finished[i]: continue
                t = int(nxt[i].item())
                if t in (vocab.EOS, vocab.PAD): finished[i] = True
                else: gen[i].append(t); tokens[i,-1] = t; done = False
            if done: break
        for i, (p, a) in enumerate(brows): text_of[idxs[i]] = vocab.decode(gen[i]).strip()
    correct = sum(1 for i, (_, a) in enumerate(rows) if text_of[i] == a)
    model.train()
    return correct / len(rows)
def sustained(traj, key, bar, consec=3):
    streak, start = 0, None
    for e in traj:
        if e[key] >= bar:
            if streak == 0: start = e["step"]
            streak += 1
            if streak >= consec: return start
        else: streak, start = 0, None
    return None
def detect_g90_onset_confirm(eval_steps, eval_ood, bar=0.90, consec=3):
    streak, onset = 0, None
    for step, value in zip(eval_steps, eval_ood):
        if value >= bar:
            if streak == 0: onset = step
            streak += 1
            if streak >= consec: return onset, step
        else: streak, onset = 0, None
    return None, None
def sha_of(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
def save_result(name, payload):
    payload["device"] = DEVICE_KIND
    p = RESULTS_DIR / name
    p.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("saved:", p)
def retention(traj):
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
    return {"RET90": round(ret90,3), "AREA": round(area,3), "T_COLLAPSE_90": c90,
            "FINAL": round(final,3), "collapsed": c90 is not None}
def displacement(model, g90_flat):
    current = torch.cat([p.detach().data.reshape(-1) for p in model.parameters()])
    diff = (current - g90_flat).norm().item()
    return {"l2": round(diff, 3), "rel": round(diff / max(g90_flat.norm().item(), 1e-8), 5)}
'''

def make_acq():
    return r'''# ======== PHASE 1: ACQUISITION (seeds 707, 808 -> G90) ========
ACQ_SEEDS = [707, 808]
acquired = {}
for seed in ACQ_SEEDS:
    torch.manual_seed(seed)
    model = Micro(VOCAB.size, 128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(seed)
    eval_steps, eval_ood, traj = [], [], []
    trigger_onset, trigger_confirm = None, None
    model.train()
    for step in range(1, 28001):
        idx = torch.randint(0, len(MANIFEST["train"]), (64,), generator=rng)
        rows = [MANIFEST["train"][i] for i in idx]
        loss, sup = loss_fn(model, VOCAB, rows, DEVICE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); xla_sync()
        if step % 200 == 0 or step == 1:
            tr = greedy_exact(model, VOCAB, MANIFEST["train"][:100], DEVICE)
            te = greedy_exact(model, VOCAB, MANIFEST["test"], DEVICE)
            traj.append({"step": step, "train_exact": tr, "test_exact": te})
            eval_steps.append(step); eval_ood.append(te)
            onset, confirm = detect_g90_onset_confirm(eval_steps, eval_ood)
            if confirm is not None:
                trigger_onset, trigger_confirm = onset, confirm
                print(f"G90 onset={onset} confirm={confirm}", flush=True)
                break
    snapshot = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    g90_flat = torch.cat([p.detach().data.reshape(-1) for p in model.parameters()])
    opt_state = opt.state_dict()
    acquired[seed] = {"snapshot": snapshot, "opt": opt_state, "g90_flat": g90_flat,
                      "onset": trigger_onset, "confirm": trigger_confirm, "rng": rng}
    print(f"seed {seed} acquired: onset={trigger_onset} confirm={trigger_confirm}")
'''

def make_ark007():
    return r'''# ======== PHASE 2: ARK-007 PAIRED CONTINUATION (PRIMARY) ========
CONT_SEEDS = list(range(1701, 1709))
POST_STEPS = 8000
ark007 = []
for seed in ACQ_SEEDS:
    if seed not in acquired: continue
    entry = acquired[seed]
    print(f"=== ARK-007 seed {seed} (confirm {entry['confirm']}) ===")
    for cs in CONT_SEEDS:
        rng_cont = torch.Generator().manual_seed(cs)
        indices = [torch.randint(0, len(MANIFEST["train"]), (64,), generator=rng_cont).tolist() for _ in range(POST_STEPS)]
        for arm_name, arm_lr in (("HIGH", 1e-3), ("LOW", 1e-5)):
            if minutes_left() < 5: print("BUDGET stop"); break
            model = Micro(VOCAB.size, 128).to(DEVICE)
            model.load_state_dict({k: v.to(DEVICE) for k, v in entry["snapshot"].items()})
            opt = torch.optim.AdamW(model.parameters(), lr=arm_lr, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
            opt.load_state_dict(entry["opt"])
            for g in opt.param_groups: g["lr"] = arm_lr
            traj = []
            model.train()
            for step in range(1, POST_STEPS + 1):
                batch_idx = indices[step - 1]
                rows = [MANIFEST["train"][i] for i in batch_idx]
                loss, _ = loss_fn(model, VOCAB, rows, DEVICE)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); xla_sync()
                if step % 200 == 0:
                    te = greedy_exact(model, VOCAB, MANIFEST["test"], DEVICE)
                    traj.append({"step": step, "test_exact": te})
            ret = retention(traj)
            ark007.append({"seed": seed, "order": cs, "arm": arm_name, **ret})
            print(f"  s{seed} ord{cs} {arm_name}: RET90={ret['RET90']} final={ret['FINAL']}", flush=True)
    if minutes_left() < 5: break
save_result("ARK-007_RESULT.json", {"forks": ark007})
'''

def make_ark006():
    return r'''# ======== PHASE 3: ARK-006 LR DOSE-RESPONSE ========
dose_results = []
for seed in ACQ_SEEDS:
    if seed not in acquired: continue
    entry = acquired[seed]
    for mult in (0.001, 0.01, 0.1):
        if minutes_left() < 5: print("BUDGET stop"); break
        model = Micro(VOCAB.size, 128).to(DEVICE)
        model.load_state_dict({k: v.to(DEVICE) for k, v in entry["snapshot"].items()})
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3*mult, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
        opt.load_state_dict(entry["opt"])
        for g in opt.param_groups: g["lr"] = 1e-3*mult
        rng = torch.Generator().manual_seed(seed)
        traj = []
        model.train()
        for step in range(1, 8001):
            idx = torch.randint(0, len(MANIFEST["train"]), (64,), generator=rng)
            rows = [MANIFEST["train"][i] for i in idx]
            loss, _ = loss_fn(model, VOCAB, rows, DEVICE)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); xla_sync()
            if step % 200 == 0:
                te = greedy_exact(model, VOCAB, MANIFEST["test"], DEVICE)
                traj.append({"step": step, "test_exact": te})
        ret = retention(traj)
        dose_results.append({"seed": seed, "lr_mult": mult, **ret})
        print(f"  s{seed} x{mult}: RET90={ret['RET90']} final={ret['FINAL']}", flush=True)
save_result("ARK-006_RESULT.json", {"dose_response": dose_results})
'''

def make_summary():
    return r'''# ======== FINAL: summary + download ========
all_results = {}
for p in sorted(RESULTS_DIR.glob("*.json")):
    try: all_results[p.name] = json.loads(p.read_text(encoding="utf-8"))
    except: pass
program = {"schema": "arkenstone-colab-program/v3", "device": DEVICE_KIND,
           "elapsed_minutes": round((time.time() - START_TIME) / 60, 1),
           "manifest": MANIFEST["sha"], "results": all_results}
program["sha"] = sha_of(program)
(RESULTS_DIR / "PROGRAM_SUMMARY.json").write_text(json.dumps(program, indent=1) + "\n", encoding="utf-8")
print("experiments completed:", len(all_results))
try:
    from google.colab import files
    import shutil
    shutil.make_archive("/content/arkenstone_results", "zip", RESULTS_DIR)
    files.download("/content/arkenstone_results.zip")
except Exception as exc:
    print("manual download from /content/arkenstone_results/:", exc)
'''

harness = make_harness().replace("MANIFEST_SHA_HERE", MANIFEST_SHA)
cells = [
    {"cell_type": "markdown", "metadata": {},
     "source": ["# ARKENSTONE — ALL EXPERIMENTS\n\n**Runtime: T4 GPU → Run all → ~90 min → auto-download results**"]},
    {"cell_type": "code", "metadata": {}, "source": harness.splitlines(keepends=True), "outputs": [], "execution_count": None},
    {"cell_type": "code", "metadata": {}, "source": make_acq().splitlines(keepends=True), "outputs": [], "execution_count": None},
    {"cell_type": "code", "metadata": {}, "source": make_ark007().splitlines(keepends=True), "outputs": [], "execution_count": None},
    {"cell_type": "code", "metadata": {}, "source": make_ark006().splitlines(keepends=True), "outputs": [], "execution_count": None},
    {"cell_type": "code", "metadata": {}, "source": make_summary().splitlines(keepends=True), "outputs": [], "execution_count": None},
]

notebook = {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"provenance": [], "name": "arkenstone_all.ipynb"},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "cells": cells}

out = Path(__file__).parent / "arkenstone_all.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
parsed = json.loads(out.read_text(encoding="utf-8"))
for cell in parsed["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), "cell", "exec")
print(f"notebook written: {out} | cells: {len(parsed['cells'])} | all compile | manifest: {MANIFEST_SHA[:12]}")
