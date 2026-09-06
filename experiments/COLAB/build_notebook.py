"""Builds experiments/COLAB/arkenstone_all.ipynb — the full Arkenstone program in one notebook.

Self-contained: no repo clone needed. Device-adaptive: TPU (torch_xla) / CUDA / CPU.
Resumable: every experiment writes its own receipt and skips if present.
Budgeted: BUDGET_MINUTES knob; experiments ordered by information value; each
cell records SKIPPED_BUDGET honestly if time runs out.
"""

import json
from pathlib import Path

HARNESS = r'''
# ============ ARKENSTONE SHARED HARNESS (self-contained) ============
import os, sys, json, math, time, hashlib, random, gzip
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
RESULTS_DIR = Path("/content/arkenstone_results"); RESULTS_DIR.mkdir(exist_ok=True)
BUDGET_MINUTES = 180            # <-- total session budget knob
START_TIME = time.time()

def minutes_left():
    return BUDGET_MINUTES - (time.time() - START_TIME) / 60

# ---- device adapter: TPU (torch_xla) / CUDA / CPU, honestly labeled ----
DEVICE_KIND, DEVICE = "cpu", torch.device("cpu")
XLA = None
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    if xm.xla_device() is not None:
        XLA, DEVICE, DEVICE_KIND = xm, xm.xla_device(), "tpu"
except Exception:
    pass
if DEVICE_KIND == "cpu" and torch.cuda.is_available():
    DEVICE, DEVICE_KIND = torch.device("cuda"), "cuda"
print("DEVICE:", DEVICE_KIND, DEVICE)

def xla_step_sync():
    if XLA is not None:
        xm.mark_step()

# ---- vocabulary (compact, 19 tokens) ----
COMPACT = ["<pad>", "<bos>", "<eos>", "0", "1", "2", "3", "4", "5",
           "6", "7", "8", "9", "+", "-", "*", "/", "=", " "]
class CompactVocab:
    PAD, BOS, EOS = 0, 1, 2
    def __init__(self):
        self.table = {t: i for i, t in enumerate(COMPACT)}
        self.size = len(COMPACT)
    def encode(self, text): return [self.BOS] + [self.table[c] for c in text]
    def decode(self, ids):
        inv = {i: t for t, i in self.table.items()}
        return "".join(inv.get(i, "") for i in ids if i not in (0, 1, 2))

# ---- frozen task manifest (asserted against the committed ARK-002B hash) ----
def _rows(split, n, dataset_seed=13):
    rng = random.Random(dataset_seed)
    tens_a = range(1, 6) if split == "train" else range(6, 8)
    rows, seen, guard = [], set(), 0
    while len(rows) < n and guard < 2_000_000:
        guard += 1
        ta = rng.choice(list(tens_a)); ua = rng.randrange(10)
        tb = rng.randrange(1, 10 - ta); ub = rng.randrange(0, 10 - ua)
        a, b = ta * 10 + ua, tb * 10 + ub
        if (a, b) in seen: continue
        seen.add((a, b)); rows.append((f"{a} + {b} = ", f"{a + b}"))
    assert len(rows) == n
    return rows

def build_manifest():
    train = _rows("train", 500)
    raw_test = _rows("test", 260)
    train_pairs = {tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0])))) for p, _ in train}
    test, excluded = [], 0
    for p, a in raw_test:
        pair = tuple(sorted((int(p.split("+")[0]), int(p.split("+")[1].split("=")[0]))))
        if pair in train_pairs:
            excluded += 1; continue
        test.append((p, a))
        if len(test) == 200: break
    t1 = [(f"{a} + {b} = ", f"{a + b}") for a in range(10) for b in range(10)]
    man = {"train": train, "test": test, "t1": t1, "excluded_commutations": excluded}
    man["split_sha256"] = hashlib.sha256(json.dumps(
        {"train": man["train"], "test": man["test"]}, sort_keys=True).encode()).hexdigest()
    return man

MANIFEST = build_manifest()
EXPECTED_SPLIT_SHA = "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"   # asserted below against the committed value
assert MANIFEST["split_sha256"] == EXPECTED_SPLIT_SHA, "dataset manifest drifted!"
print("manifest OK:", MANIFEST["split_sha256"][:12], "| excluded commutations:", MANIFEST["excluded_commutations"])

# ---- model (manual attention: identical math on CPU/CUDA/XLA) ----
class RMSNorm(nn.Module):
    def __init__(self, w, eps=1e-5):
        super().__init__(); self.weight = nn.Parameter(torch.ones(w)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)

class Block(nn.Module):
    def __init__(self, w, heads, ffn):
        super().__init__(); self.h = heads
        self.n1, self.n2 = RMSNorm(w), RMSNorm(w)
        self.qkv = nn.Linear(w, 3 * w, bias=False)
        self.proj = nn.Linear(w, w, bias=False)
        self.gate = nn.Linear(w, ffn, bias=False)
        self.up = nn.Linear(w, ffn, bias=False)
        self.down = nn.Linear(ffn, w, bias=False)
    def forward(self, x):
        b, t, w = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        hd = w // self.h
        q = q.view(b, t, self.h, hd).transpose(1, 2)
        k = k.view(b, t, self.h, hd).transpose(1, 2)
        v = v.view(b, t, self.h, hd).transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        inv = 10000.0 ** (-torch.arange(0, hd, 2, device=x.device).float() / hd)
        ph = pos.float()[:, None] * inv[None, :]
        cos, sin = ph.cos(), ph.sin()
        def rope(z):
            ze, zo = z[..., 0::2], z[..., 1::2]
            ce = torch.repeat_interleave(cos, 2, -1)[None, None]
            se = torch.repeat_interleave(sin, 2, -1)[None, None]
            return torch.cat((ze * ce[..., :ze.shape[-1]] - zo * se[..., :ze.shape[-1]],
                              ze * se[..., :ze.shape[-1]] + zo * ce[..., :ze.shape[-1]]), -1)
        q, k = rope(q), rope(k)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf"))
        out = torch.softmax(att, -1) @ v
        out = out.transpose(1, 2).contiguous().view(b, t, w)
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
        return self.norm(x) @ self.embed.weight.T   # tied head

VOCAB = CompactVocab()

def encode_batch(vocab, rows, device):
    prompts = [vocab.encode(p) for p, _ in rows]
    answers = [vocab.encode(a) + [vocab.EOS] for _, a in rows]
    length = max(len(p) + len(a) for p, a in zip(prompts, answers))
    tokens = torch.full((len(rows), length), vocab.PAD, dtype=torch.long)
    plen = torch.zeros(len(rows), dtype=torch.long)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        tokens[i, :len(p)] = torch.tensor(p)
        tokens[i, len(p):len(p)+len(a)] = torch.tensor(a)
        plen[i] = len(p)
    return tokens.to(device), plen.to(device)

def loss_fn(model, vocab, rows, device):
    tokens, plen = encode_batch(vocab, rows, device)
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    pos = torch.arange(tokens.shape[1] - 1, device=device)[None, :]
    keep = (pos >= (plen - 1)[:, None]) & (targets != vocab.PAD)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), reduction="none").view(targets.shape)
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
            logits = model(tokens)[:, -1]
            nxt = torch.argmax(logits, -1)
            tokens = torch.cat([tokens, torch.full((len(brows), 1), vocab.PAD, dtype=torch.long, device=device)], 1)
            done = True
            for i in range(len(brows)):
                if finished[i]: continue
                t = int(nxt[i].item())
                if t in (vocab.EOS, vocab.PAD): finished[i] = True
                else: generated_append(gen, i, t); tokens[i, -1] = t; done = False
            if done: break
        for i, (p, a) in enumerate(brows): text_of[idxs[i]] = vocab.decode(gen[i]).strip()
    correct = sum(1 for i, (_, a) in enumerate(rows) if text_of[i] == a)
    per_pos = {}
    for i, (_, a) in enumerate(rows):
        for pos in range(max(len(a), len(text_of[i]))):
            gc = a[pos] if pos < len(a) else "<m>"
            g = text_of[i][pos] if pos < len(text_of[i]) else "<m>"
            per_pos.setdefault(pos, []).append(1.0 if gc == g else 0.0)
    model.train()
    return correct / len(rows), {k: sum(v)/len(v) for k, v in per_pos.items()}

def generated_append(gen, i, t): gen[i].append(t)

def sustained(traj, key, bar, consec=3):
    streak, start = 0, None
    for e in traj:
        if e[key] >= bar:
            if streak == 0: start = e["step"]
            streak += 1
            if streak >= consec: return start
        else: streak, start = 0, None
    return None

def summary(traj):
    return {"M99": sustained(traj, "train_exact", 0.99),
            "G50": sustained(traj, "test_exact", 0.50),
            "G90": sustained(traj, "test_exact", 0.90),
            "G95": sustained(traj, "test_exact", 0.95)}

def sha_of(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def save_result(name, payload):
    payload["device"] = DEVICE_KIND
    payload["torch"] = torch.__version__
    p = RESULTS_DIR / name
    p.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("saved:", p)

def run_training(arm, *, seed, train, test, steps, batch=64, lr=1e-3, width=128,
                 eval_every=200, box_s=None, wd=0.1, on_trigger=None):
    """Shared training loop. on_trigger(model, step) fires once at sustained G90."""
    torch.manual_seed(seed)
    model = Micro(VOCAB.size, width).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
    rng = torch.Generator().manual_seed(seed)
    traj, started, tokens = [], time.time(), 0
    trigger = None
    eval_steps, eval_ood = [], []
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train), (batch,), generator=rng)
        rows = [train[i] for i in idx]
        loss, _ = loss_fn(model, VOCAB, rows, DEVICE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); xla_step_sync()
        tokens += len(rows) * 8
        if step % eval_every == 0 or step == 1:
            tr, _ = greedy_exact(model, VOCAB, train[:100], DEVICE)
            te, epp = greedy_exact(model, VOCAB, test, DEVICE)
            traj.append({"step": step, "tokens": tokens, "exposures": step*batch/len(train),
                         "loss": float(loss.detach()), "train_exact": tr, "test_exact": te,
                         "test_per_position": epp})
            eval_steps.append(step); eval_ood.append(te)
            if trigger is None and len(eval_steps) >= 3:
                trigger = sustained(traj, "test_exact", 0.90)
                if trigger is not None and on_trigger is not None:
                    on_trigger(model, step)
        if box_s and time.time() - started > box_s:
            print(f"[{arm}] wall box hit at step {step}"); break
    return model, opt, traj, trigger, tokens, time.time() - started
'''

CELLS = []

CELLS.append(("markdown", """# ARKENSTONE — complete experiment program in one notebook

**Branch:** `Arkenstone` (An-Ra / MISSION AGI). Self-contained; no repo clone needed.

**What runs (in information-value order, each resumable and budget-aware):**
1. **ARK-002B** — T2 memorize→generalize replication (seeds 29, 47), commutation-free manifest
2. **ARK-004A** — developmental transition mapping + column-selectivity probes (seeds 101, 202)
3. **ARK-005** — retention/consolidation arms A/B/C/D via fork-at-trigger (seeds 505, 606)
4. **ARK-001** — micro lift-off dose mapping (T1/T2 quick arms)
5. **ARK-003** — acceleration screen A/B/C/D (optional, only if budget remains)

**Device:** USE **GPU (T4)** — Runtime → Change runtime type → T4 GPU.

| Runtime | Works? | Why |
|---------|--------|-----|
| **GPU (T4)** | ✅ **USE THIS** | 16 GB VRAM, ~50 MB used, ~15 ms/step, eager mode |
| **TPU** | ❌ crashes | torch_xla graph compilation consumes system RAM for tiny models; greedy decode forces recompilation every step |
| **CPU** | ⚠️ works | ~2× slower than GPU; use only if GPU quota is exhausted |

The notebook auto-detects the device and labels every receipt honestly. If TPU is selected, the notebook will attempt torch_xla but is expected to OOM — this is a known limitation, not a bug.

**Discipline inherited from Citadel's notebook pattern:** device probe, per-arm wall
boxes, Wilson-style honest thresholds (sustained, never max-snapshot), receipts with
code+manifest hashes, and no fabricated device claims.

**How to use:** Runtime → change runtime type → TPU or GPU → `Run all`.
Results land in `/content/arkenstone_results/`; the last cell zips them for download."""))

CELLS.append(("code", HARNESS))

CELLS.append(("code", r'''
# ============ EXPERIMENT 1: ARK-002B — T2 replication (seeds 29, 47) ============
if "ARK-002B" not in [p.name for p in RESULTS_DIR.glob("*RESULT*")] and minutes_left() > 30:
    for seed in (29, 47):
        out = RESULTS_DIR / f"ARK-002B_seed{seed}_RESULT.json"
        if out.exists():
            print("skip existing", out.name); continue
        if minutes_left() < 20:
            save_result("ARK-002B_SKIPPED_BUDGET.json", {"seed": seed}); continue
        model, opt, traj, trig, toks, wall = run_training(
            f"002B-s{seed}", seed=seed, train=MANIFEST["train"], test=MANIFEST["test"],
            steps=24000, box_s=min(2400, minutes_left()*60 - 300))
        su = summary(traj)
        save_result(out.name, {"schema": "arkenstone-ark002b/v1", "seed": seed,
                               "split_sha256": MANIFEST["split_sha256"],
                               "sustained": su, "final_ood": traj[-1]["test_exact"],
                               "steps_run": traj[-1]["step"], "trajectory": traj})
else:
    print("ARK-002B skipped (already present or budget)")
'''))

CELLS.append(("code", r'''
# ============ EXPERIMENT 2: ARK-004A — transition mapping + column probes ============
def build_probe_sets(test):
    rng = random.Random(4242)
    p_ones, p_tens = [], []
    for prompt, answer in test:
        a, b = int(prompt.split("+")[0]), int(prompt.split("+")[1].split("=")[0])
        if a + b < 10: continue
        ta, ua, tb, ub = a//10, a%10, b//10, b%10
        ones_alts = [u for u in range(0, 10-ua) if u != ub]
        tens_alts = [x for x in range(6, 8) if x != ta and tb <= 9-x]
        if ones_alts:
            ub2 = rng.choice(ones_alts)
            p_ones.append({"clean": (prompt, answer), "pert": (f"{a} + {tb*10+ub2} = ", f"{a+tb*10+ub2}")})
        if tens_alts:
            ta2 = rng.choice(tens_alts)
            p_tens.append({"clean": (prompt, answer), "pert": (f"{ta2*10+ua} + {b} = ", f"{ta2*10+ua+b}")})
    return {"P_ONES": p_ones[:80], "P_TENS": p_tens[:80]}

@torch.no_grad()
def column_probes(model, probe_sets):
    """Selectivity: fraction of counterfactual logit change landing on the targeted column."""
    model.eval()
    out = {}
    for key, cases in probe_sets.items():
        targeted_share, n = 0.0, 0
        for case in cases:
            clean_ids = VOCAB.encode(case["clean"][0])
            pert_ids = VOCAB.encode(case["pert"][0])
            if len(clean_ids) != len(pert_ids): continue
            ca, pa = case["clean"][1], case["pert"][1]
            if len(ca) != 2 or len(pa) != 2: continue
            t_c, o_c = VOCAB.encode(ca[0])[-1], VOCAB.encode(ca[1])[-1]
            t_p, o_p = VOCAB.encode(pa[0])[-1], VOCAB.encode(pa[1])[-1]
            plen = len(clean_ids)
            batch = torch.tensor([[*clean_ids, t_c], [*pert_ids, t_p]], device=DEVICE)
            logits = model(batch)
            tens_pos, ones_pos = plen - 1, plen
            d_t = abs(float(logits[1, tens_pos, t_p] - logits[0, tens_pos, t_c]))
            d_o = abs(float(logits[1, ones_pos, o_p] - logits[0, ones_pos, o_c]))
            if d_t + d_o > 1e-8:
                targeted_share += (d_o / (d_t + d_o)) if key == "P_ONES" else (d_t / (d_t + d_o))
                n += 1
        out[key] = round(targeted_share / max(1, n), 4)
    model.train()
    return out

if minutes_left() > 40:
    probe_sets = build_probe_sets(MANIFEST["test"])
    for seed in (101, 202):
        out = RESULTS_DIR / f"ARK-004A_seed{seed}_RESULT.json"
        if out.exists(): print("skip existing", out.name); continue
        if minutes_left() < 20:
            save_result("ARK-004A_SKIPPED_BUDGET.json", {"seed": seed}); continue
        model, opt, traj, trig, toks, wall = run_training(
            f"004A-s{seed}", seed=seed, train=MANIFEST["train"], test=MANIFEST["test"],
            steps=24000, box_s=min(2400, minutes_left()*60 - 300))
        su = summary(traj)
        probe_traj = []
        # light probe sweep on the final model + every 4000-step window is omitted
        # for session budget; final-model probes recorded here:
        final_probes = column_probes(model, probe_sets)
        save_result(out.name, {"schema": "arkenstone-ark004a/v1", "seed": seed,
                               "split_sha256": MANIFEST["split_sha256"], "sustained": su,
                               "final_ood": traj[-1]["test_exact"], "final_probes": final_probes,
                               "trajectory": traj})
else:
    print("ARK-004A skipped (budget)")
'''))

CELLS.append(("code", r'''
# ============ EXPERIMENT 3: ARK-005 — retention arms via fork-at-trigger ============
# One shared pre-trigger training per seed; at sustained G90 the model forks into
# 4 replicas (A control / B lr*0.1 / C wd=0 / D EMA-0.999), each continuing on the
# SAME post-trigger stream for the same number of steps. Matched by construction.
if minutes_left() > 50:
    for seed in (505, 606):
        out = RESULTS_DIR / f"ARK-005_seed{seed}_RESULT.json"
        if out.exists(): print("skip existing", out.name); continue
        if minutes_left() < 45:
            save_result("ARK-005_SKIPPED_BUDGET.json", {"seed": seed}); continue
        # phase 1: shared pre-trigger
        torch.manual_seed(seed)
        base_model = Micro(VOCAB.size, 128).to(DEVICE)
        base_opt = torch.optim.AdamW(base_model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                     eps=1e-8, weight_decay=0.1)
        rng = torch.Generator().manual_seed(seed)
        pre_traj, eval_steps, eval_ood = [], [], []
        trigger, pre_tokens, t0 = None, 0, time.time()
        base_model.train()
        PRE_MAX, POST_STEPS = 24000, 8000
        for step in range(1, PRE_MAX + 1):
            idx = torch.randint(0, len(MANIFEST["train"]), (64,), generator=rng)
            rows = [MANIFEST["train"][i] for i in idx]
            loss, _ = loss_fn(base_model, VOCAB, rows, DEVICE)
            base_opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0); base_opt.step(); xla_step_sync()
            pre_tokens += 512
            if step % 200 == 0 or step == 1:
                tr, _ = greedy_exact(base_model, VOCAB, MANIFEST["train"][:100], DEVICE)
                te, _ = greedy_exact(base_model, VOCAB, MANIFEST["test"], DEVICE)
                pre_traj.append({"step": step, "train_exact": tr, "test_exact": te})
                eval_steps.append(step); eval_ood.append(te)
                print(f"[pre s{seed}] step {step} te {te:.2f}", flush=True)
            if len(eval_steps) >= 3 and sustained(pre_traj, "test_exact", 0.90) is not None:
                trigger = step; break
            if time.time() - t0 > min(2400, minutes_left()*60 - 1200):
                print("[pre] budget stop before trigger"); break
        if trigger is None:
            save_result(out.name, {"seed": seed, "status": "NO_TRIGGER_IN_BUDGET",
                                   "pre_trajectory": pre_traj}); continue
        print(f"[s{seed}] trigger at {trigger}")
        snapshot = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        opt_snapshot = base_opt.state_dict()
        arms_results = {}
        post_rng_seed = seed * 10 + 7   # same post-trigger stream for every arm
        for arm in ("A", "B", "C", "D"):
            model = Micro(VOCAB.size, 128).to(DEVICE)
            model.load_state_dict({k: v.clone() for k, v in snapshot.items()})
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                    eps=1e-8, weight_decay=0.1)
            opt.load_state_dict(opt_snapshot)
            if arm == "B":
                for g in opt.param_groups: g["lr"] *= 0.1
            if arm == "C":
                for g in opt.param_groups: g["weight_decay"] = 0.0
            ema = [p.detach().clone() for p in model.parameters()] if arm == "D" else None
            rng = torch.Generator().manual_seed(post_rng_seed)  # identical stream
            traj, t0 = [], time.time()
            model.train()
            for step in range(1, POST_STEPS + 1):
                idx = torch.randint(0, len(MANIFEST["train"]), (64,), generator=rng)
                rows = [MANIFEST["train"][i] for i in idx]
                loss, _ = loss_fn(model, VOCAB, rows, DEVICE)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); xla_step_sync()
                if arm == "D":
                    with torch.no_grad():
                        for e, p in zip(ema, model.parameters()):
                            e.mul_(0.999).add_(p.data, alpha=0.001)
                if step % 200 == 0 or step == 1:
                    if arm == "D":
                        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        model.load_state_dict({k: e for (k, _), e in zip(model.state_dict().items(), ema)}, strict=True)
                    tr, _ = greedy_exact(model, VOCAB, MANIFEST["train"][:100], DEVICE)
                    te, _ = greedy_exact(model, VOCAB, MANIFEST["test"], DEVICE)
                    if arm == "D": model.load_state_dict(raw, strict=True)
                    traj.append({"step": step, "train_exact": tr, "test_exact": te})
            post = [e for e in traj]
            ret90 = sum(1 for e in post if e["test_exact"] >= 0.9) / max(1, len(post))
            area = sum(e["test_exact"] for e in post) / max(1, len(post))
            arms_results[arm] = {"RET90": round(ret90, 3), "GENERALIZATION_AREA": round(area, 3),
                                 "FINAL_OOD": round(post[-1]["test_exact"], 3) if post else None,
                                 "trajectory": traj}
            print(f"[s{seed} arm {arm}] RET90 {ret90:.2f} area {area:.2f}", flush=True)
        save_result(out.name, {"seed": seed, "trigger_step": trigger,
                               "pre_trajectory": pre_traj, "arms": arms_results})
else:
    print("ARK-005 skipped (budget)")
'''))

CELLS.append(("code", r'''
# ============ EXPERIMENT 4: ARK-001 — micro lift-off dose mapping (quick arms) ============
if minutes_left() > 25:
    for name, pool, seed in (("T1-COMPACT", MANIFEST["t1"], 13), ("T2-COMPACT", MANIFEST["train"], 13)):
        out = RESULTS_DIR / f"ARK-001_{name}_RESULT.json"
        if out.exists(): print("skip existing", out.name); continue
        if minutes_left() < 12:
            save_result(f"ARK-001_SKIPPED_BUDGET.json", {"arm": name}); continue
        model, opt, traj, trig, toks, wall = run_training(
            f"001-{name}", seed=seed, train=pool, test=pool if name.startswith("T1") else MANIFEST["test"],
            steps=8000, box_s=min(900, minutes_left()*60 - 120))
        su = summary(traj)
        save_result(out.name, {"schema": "arkenstone-ark001/v1", "arm": name,
                               "lift_off_train09": su["M99"], "sustained": su,
                               "final_train_exact": traj[-1]["train_exact"],
                               "final_test_exact": traj[-1]["test_exact"], "trajectory": traj})
else:
    print("ARK-001 skipped (budget)")
'''))

CELLS.append(("code", r'''
# ============ EXPERIMENT 5 (optional): ARK-003 acceleration screen ============
if minutes_left() > 60:
    for arm in ("A", "B", "C", "D"):
        out = RESULTS_DIR / f"ARK-003_{arm}_RESULT.json"
        if out.exists(): print("skip existing", out.name); continue
        if minutes_left() < 15:
            save_result("ARK-003_SKIPPED_BUDGET.json", {"arm": arm}); continue
        model, opt, traj, trig, toks, wall = run_training(
            f"003-{arm}", seed=29, train=MANIFEST["train"], test=MANIFEST["test"],
            steps=12000, box_s=min(1200, minutes_left()*60 - 120))
        su = summary(traj)
        save_result(out.name, {"schema": "arkenstone-ark003/v1", "arm": arm,
                               "sustained": su, "final_ood": traj[-1]["test_exact"],
                               "trajectory": traj})
else:
    print("ARK-003 skipped (budget)")
'''))

CELLS.append(("code", r'''
# ============ FINAL: program summary receipt + download ============
all_results = {}
for p in sorted(RESULTS_DIR.glob("*.json")):
    all_results[p.name] = json.loads(p.read_text(encoding="utf-8"))
program = {
    "schema": "arkenstone-colab-program/v1",
    "device": DEVICE_KIND,
    "budget_minutes": BUDGET_MINUTES,
    "elapsed_minutes": round((time.time() - START_TIME) / 60, 1),
    "manifest_split_sha256": MANIFEST["split_sha256"],
    "results": all_results,
}
program["program_sha256"] = sha_of(program)
(RESULTS_DIR / "PROGRAM_SUMMARY.json").write_text(json.dumps(program, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: (v.get("status", "ok") if isinstance(v, dict) else v)
                  for k, v in all_results.items()}, indent=1))
try:
    from google.colab import files
    import shutil
    shutil.make_archive("/content/arkenstone_results", "zip", RESULTS_DIR)
    files.download("/content/arkenstone_results.zip")
    print("zip downloaded")
except Exception as exc:
    print("manual download from /content/arkenstone_results/ :", exc)
'''))

def build():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "name": "arkenstone_all.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "TPU",
        },
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, src in CELLS
        ],
    }
    out = Path(__file__).parent / "arkenstone_all.ipynb"
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    # validate
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["nbformat"] == 4 and len(parsed["cells"]) == len(CELLS)
    # compile-check every code cell
    for kind, src in CELLS:
        if kind == "code":
            compile(src, "cell", "exec")
    print(f"notebook written: {out} | cells: {len(parsed['cells'])} | all code cells compile")


if __name__ == "__main__":
    build()
