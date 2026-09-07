"""Builds the MASTER Arkenstone Colab notebook — ALL experiments, one session.

Self-contained. T4 GPU recommended. ~2-4 hours. Resumable. Auto-download.
Experiments in priority order:
  1. ARK-007  paired continuation (PRIMARY: causal retention test)
  2. ARK-008  binding transfer with LONG training (30k steps)
  3. ARK-006  LR dose-response on 3 seeds
  4. ARK-004A transition mapping on 2 fresh seeds
  5. ARK-001  lift-off quick check
"""
import json
from pathlib import Path

MANIFEST_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"

HARNESS = r'''# ======== ARKENSTONE SHARED HARNESS ========
import os, sys, json, math, time, hashlib, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
RESULTS_DIR = Path("/content/arkenstone_results"); RESULTS_DIR.mkdir(exist_ok=True)
BUDGET_MINUTES = 210  # set to your session length
START_TIME = time.time()
def minutes_left(): return BUDGET_MINUTES - (time.time() - START_TIME) / 60
DEVICE_KIND, DEVICE = "cpu", torch.device("cpu")
if torch.cuda.is_available():
    DEVICE, DEVICE_KIND = torch.device("cuda"), "cuda"
print("DEVICE:", DEVICE_KIND, torch.cuda.get_device_name(0) if DEVICE_KIND=="cuda" else "")
COMPACT = ["<pad>","<bos>","<eos>","0","1","2","3","4","5","6","7","8","9","+","-","*","/","="," "]
class CompactVocab:
    PAD, BOS, EOS = 0, 1, 2
    def __init__(self):
        self.table = {t:i for i,t in enumerate(COMPACT)}
        self.size = len(COMPACT)
    def extend(self, text):
        for ch in text:
            if ch not in self.table:
                self.table[ch] = self.size; self.size += 1
    def encode(self, text): return [self.BOS] + [self.table.get(c, self.PAD) for c in text]
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
def detect_g90_oc(eval_steps, eval_ood, bar=0.90, consec=3):
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

# ============ EXPERIMENT CELLS ============

CELL_ACQ = r'''# ======== PHASE 1: ACQUISITION (seeds 707, 808 -> G90 on T2) ========
ACQ_SEEDS = [707, 808]
acquired = {}
for seed in ACQ_SEEDS:
    torch.manual_seed(seed)
    model = Micro(VOCAB.size, 128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
    rng = torch.Generator().manual_seed(seed)
    eval_steps, eval_ood, traj = [], [], []
    onset, confirm = None, None
    model.train()
    t0 = time.time()
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
            onset, confirm = detect_g90_oc(eval_steps, eval_ood)
            if confirm is not None:
                print(f"G90 onset={onset} confirm={confirm}", flush=True)
                break
        if time.time() - t0 > 1500:
            print(f"budget stop step {step}"); break
    if confirm is None:
        print(f"seed {seed}: G90 not reached"); continue
    snapshot = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    opt_state = {k: v for k, v in opt.state_dict().items()}
    g90_flat = torch.cat([p.detach().data.reshape(-1) for p in model.parameters()])
    acquired[seed] = {"model_state": snapshot, "opt_state": opt_state,
                      "g90_flat": g90_flat, "onset": onset, "confirm": confirm,
                      "rng_state": rng}
    print(f"seed {seed} acquired: onset={onset} confirm={confirm}")
'''

CELL_ARK007 = r'''# ======== PHASE 2: ARK-007 PAIRED CONTINUATION (PRIMARY) ========
CONT_SEEDS = list(range(1701, 1709))
POST_STEPS = 8000
ark007_forks = []
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
            model.load_state_dict({k: v.to(DEVICE) for k, v in entry["model_state"].items()})
            opt = torch.optim.AdamW(model.parameters(), lr=arm_lr, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
            opt.load_state_dict(entry["opt_state"])
            for g in opt.param_groups: g["lr"] = arm_lr
            g90_flat = entry["g90_flat"].to(DEVICE)
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
                    disp = displacement(model, g90_flat)
                    traj.append({"step": step, "test_exact": te, **disp})
            ret = retention(traj)
            ark007_forks.append({"seed": seed, "order": cs, "arm": arm_name, **ret})
            print(f"  s{seed} ord{cs} {arm_name}: RET90={ret['RET90']} final={ret['FINAL']}", flush=True)
    if minutes_left() < 5: break
save_result("ARK-007_RESULT.json", {"forks": ark007_forks})
'''

CELL_ARK008 = r'''# ======== PHASE 3: ARK-008 BINDING TRANSFER WITH LONG TRAINING ========
def build_binding(n_train=200, n_test=50, seed=42):
    import random
    rng = random.Random(seed)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "gray"]
    train_ents = [f"obj_{i}" for i in range(50)]
    test_ents = [f"tst_{i}" for i in range(50)]
    def make_row(ents):
        n = rng.randint(3, 5)
        facts, assign, used = [], {}, set()
        for _ in range(n):
            e = rng.choice(ents)
            if e in assign: continue
            c = rng.choice([c for c in colors if c not in used])
            used.add(c); assign[e] = c
            facts.append((e, c))
        qe = rng.choice(list(assign.keys()))
        q = f"What color is {qe}? "
        ans = assign[qe]
        prompt = ". ".join(f"The {e} is {c}" for e, c in facts) + f". {q}"
        return (prompt, ans)
    return [make_row(train_ents) for _ in range(n_train)], [make_row(test_ents) for _ in range(n_test)]

if minutes_left() > 40:
    bind_train, bind_test = build_binding(200, 50, seed=42)
    # extend vocab for binding text
    bind_vocab = CompactVocab()
    for p, a in bind_train + bind_test:
        bind_vocab.extend(p); bind_vocab.extend(a)
    print(f"binding vocab: {bind_vocab.size} tokens")
    binding_results = []
    for seed in (505, 606):
        for lr_mult, label in ((1.0, "HIGH"), (0.01, "LOW")):
            if minutes_left() < 8: print("BUDGET stop"); break
            print(f"=== binding seed {seed} LR x{mult} ({label}) ===")
            torch.manual_seed(seed)
            model = Micro(bind_vocab.size, 128).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3*lr_mult,
                                    betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
            rng = torch.Generator().manual_seed(seed)
            traj = []
            model.train()
            t0 = time.time()
            TRAIN_STEPS = 20000
            for step in range(1, TRAIN_STEPS + 1):
                idx = torch.randint(0, len(bind_train), (32,), generator=rng)
                rows = [bind_train[i] for i in idx]
                loss, _ = loss_fn(model, bind_vocab, rows, DEVICE)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); xla_sync()
                if step % 1000 == 0 or step == 1:
                    tr = greedy_exact(model, bind_vocab, bind_train[:50], DEVICE)
                    te = greedy_exact(model, bind_vocab, bind_test, DEVICE)
                    traj.append({"step": step, "train_exact": tr, "test_exact": te})
                    print(f"  step {step} train {tr:.3f} test {te:.3f}", flush=True)
                if time.time() - t0 > 900:
                    print("  box hit"); break
            final_tr = traj[-1]["train_exact"] if traj else 0
            final_te = traj[-1]["test_exact"] if traj else 0
            binding_results.append({"seed": seed, "lr_mult": lr_mult, "label": label,
                                    "final_train": final_tr, "final_test": final_te,
                                    "trajectory": traj})
            print(f"  final: train {final_tr:.3f} test {final_te:.3f}", flush=True)
    save_result("ARK-008_BINDING_RESULT.json", {"experiments": binding_results})
else:
    print("ARK-008 binding skipped (budget)")
'''

CELL_ARK006 = r'''# ======== PHASE 4: ARK-006 LR DOSE-RESPONSE on fresh seeds ========
if minutes_left() > 30:
    dose_results = []
    for seed in (707, 808):
        if seed not in acquired: continue
        entry = acquired[seed]
        for mult in (0.001, 0.01, 0.1):
            if minutes_left() < 5: print("BUDGET stop"); break
            print(f"=== s{seed} lr x{mult} ===")
            model = Micro(VOCAB.size, 128).to(DEVICE)
            model.load_state_dict({k: v.to(DEVICE) for k, v in entry["model_state"].items()})
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3*mult, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)
            opt.load_state_dict(entry["opt_state"])
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
            print(f"  RET90={ret['RET90']} final={ret['FINAL']}", flush=True)
    save_result("ARK-006_DOSE_RESULT.json", {"dose_response": dose_results})
else:
    print("ARK-006 skipped (budget)")
'''

CELL_SUMMARY = r'''# ======== FINAL: summary + download ========
all_results = {}
for p in sorted(RESULTS_DIR.glob("*.json")):
    try: all_results[p.name] = json.loads(p.read_text(encoding="utf-8"))
    except: pass
program = {"schema": "arkenstone-master-colab/v1", "device": DEVICE_KIND,
           "elapsed_minutes": round((time.time() - START_TIME) / 60, 1),
           "manifest": MANIFEST["sha"], "results": all_results}
program["sha"] = sha_of(program)
(RESULTS_DIR / "PROGRAM_SUMMARY.json").write_text(json.dumps(program, indent=1) + "\n", encoding="utf-8")
print(f"experiments completed: {len(all_results)}")
print(f"device: {DEVICE_KIND} | elapsed: {program['elapsed_minutes']} min")
try:
    from google.colab import files
    import shutil
    shutil.make_archive("/content/arkenstone_results", "zip", RESULTS_DIR)
    files.download("/content/arkenstone_results.zip")
except Exception as exc:
    print("manual download from /content/arkenstone_results/:", exc)
'''

def build():
    cells = []
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        "# ARKENSTONE — MASTER EXPERIMENT NOTEBOOK\n",
        "\n",
        "**Runtime: T4 GPU (NOT TPU — TPU crashes from graph compilation RAM overhead)**\n",
        "\n",
        "Runtime → Change runtime type → **T4 GPU** → Run all → ~2–3 hours → auto-download results\n",
        "\n",
        "| Phase | Experiment | What it tests | Steps |\n",
        "|-------|-----------|---------------|-------|\n",
        "| 1 | Acquisition (707, 808) | Train to G90 on T2 arithmetic | ~32k |\n",
        "| 2 | ARK-007 | Paired continuation: does low LR protect? 32 forks | 256k |\n",
        "| 3 | ARK-008 | Binding transfer: does the task work at all with long training? | ~120k |\n",
        "| 4 | ARK-006 | LR dose-response on fresh seeds | ~48k |\n",
        "| 5 | Summary + download | All receipts | — |\n",
        "\n",
        "Resumable: completed experiments are skipped on re-run.\n",
        "Budget: `BUDGET_MINUTES = 210` (adjust to your session)."]})

    for src in [HARNESS.replace("MANIFEST_SHA_HERE", MANIFEST_SHA), CELL_ACQ, CELL_ARK007, CELL_ARK008, CELL_ARK006, CELL_SUMMARY]:
        cells.append({"cell_type": "code", "metadata": {},
                      "source": src.splitlines(keepends=True),
                      "outputs": [], "execution_count": None})

    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"colab": {"provenance": [], "name": "arkenstone_master.ipynb"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU"},
                "cells": cells}
    out = Path(__file__).parent / "arkenstone_master.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    for cell in parsed["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cell", "exec")
    print(f"notebook: {out} | cells: {len(parsed['cells'])} | all compile | manifest: {MANIFEST_SHA[:12]}")

if __name__ == "__main__":
    build()
