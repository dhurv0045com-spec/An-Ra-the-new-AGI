from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ark018_v3_common import Ark018GPT


def select_binding_tokens(tok, counts: np.ndarray) -> list[int]:
    for threshold in [256, 128, 64]:
        candidates = []
        for tid in range(min(len(counts), tok.get_vocab_size())):
            if counts[tid] < threshold:
                continue
            text = tok.decode([tid])
            stripped = text.strip()
            if not (4 <= len(stripped) <= 10 and stripped.isascii() and stripped.isalpha() and stripped.islower()):
                continue
            if tok.encode(text).ids != [tid]:
                continue
            if int(hashlib.sha256(stripped.encode()).hexdigest(), 16) % 4 != 0:
                continue
            candidates.append((-int(counts[tid]), tid, stripped))
        candidates.sort()
        if len(candidates) >= 24:
            return [x[1] for x in candidates[:24]]
    raise RuntimeError("insufficient eligible single-token words for ARK-018 binding probe")


def make_factsets(keys, vals):
    factsets = []
    for kt in itertools.combinations(keys, 3):
        for vt in itertools.permutations(vals, 3):
            factsets.append(tuple(zip(kt, vt)))
    rng = random.Random(424218)
    rng.shuffle(factsets)
    return factsets[:400], factsets[400:450], factsets[450:500]


def semantic_examples(factsets):
    out = []
    for facts in factsets:
        for q, ans in facts:
            out.append((facts, q, ans))
    return out


def template(tok):
    return {
        "prefix": tok.encode("Facts: ").ids,
        "means": tok.encode(" means ").ids,
        "sep": tok.encode("; ").ids,
        "query": tok.encode("Query: ").ids,
        "tail": tok.encode(" means").ids,
    }


def render(tpl, facts, query, order):
    ff = [facts[i] for i in order]
    ids = list(tpl["prefix"])
    for k, v in ff:
        ids += [int(k)] + tpl["means"] + [int(v)] + tpl["sep"]
    ids += tpl["query"] + [int(query)] + tpl["tail"]
    return ids[-255:]


def perm_for(seed: int, step: int, semantic_idx: int):
    perms = list(itertools.permutations(range(3)))
    h = hashlib.sha256(f"ark018-bind:{seed}:{step}:{semantic_idx}".encode()).digest()
    return perms[int.from_bytes(h[:8], "big") % len(perms)]


def make_batch_examples(tpl, semantics, indices, mode, seed, step):
    rows = []
    for idx in indices:
        facts, q, ans = semantics[int(idx)]
        if mode == "canonical": order = (0, 1, 2)
        elif mode == "reversed": order = (2, 1, 0)
        else: order = perm_for(seed, step, int(idx))
        rows.append((render(tpl, facts, q, order), int(ans)))
    return rows


def logits_answers(model, examples, device):
    max_len = max(len(p) for p, _ in examples)
    x = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    ans = torch.tensor([a for _, a in examples], dtype=torch.long, device=device)
    for i, (p, _) in enumerate(examples):
        x[i, -len(p):] = torch.tensor(p, dtype=torch.long, device=device)
    return model(x)[:, -1, :], ans


@torch.no_grad()
def evaluate(model, tpl, semantics, device, mode):
    hit = 0
    n = 0
    for i in range(0, len(semantics), 64):
        idx = list(range(i, min(i + 64, len(semantics))))
        rows = make_batch_examples(tpl, semantics, idx, mode, 0, 0)
        logits, ans = logits_answers(model, rows, device)
        hit += int((logits.argmax(-1) == ans).sum().item())
        n += len(ans)
    return hit / max(1, n)


def metrics(model, tpl, semantics, device):
    canonical = evaluate(model, tpl, semantics, device, "canonical")
    order_only = evaluate(model, tpl, semantics, device, "reversed")
    q = canonical >= .90 and order_only >= .85
    return {"canonical": canonical, "order_only": order_only, "query_order": order_only, "qualified": q}


def run_binding_probe_from_checkpoint(base, seed: int, arm: str, prepared, tok, device) -> dict:
    out_path = base.result_dir() / f"ARK-018_SEED_{seed}_{arm}_BINDING_PROBE.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    ckpt = torch.load(base.arm_checkpoint_path(seed, arm), map_location="cpu", weights_only=False)
    if int(ckpt["step"]) < int(prepared["horizon_updates"]):
        raise RuntimeError("binding probe requested before pretraining complete")

    counts = np.load(base.prepared_dir() / "token_counts.npy")
    chosen = select_binding_tokens(tok, counts)
    keys, vals = chosen[:6], chosen[6:12]
    train_fs, control_fs, sealed_fs = make_factsets(keys, vals)
    train_sem = semantic_examples(train_fs)
    control_sem = semantic_examples(control_fs)
    sealed_sem = semantic_examples(sealed_fs)
    tpl = template(tok)

    model = Ark018GPT().to(device)
    model.load_state_dict(ckpt["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), eps=1e-8, weight_decay=.1)
    rng = random.Random(700000 + seed)
    trajectory = []
    streak = 0
    qualification_step = None
    qualified_snapshot = None

    for step in range(1, 1501):
        idx = [rng.randrange(len(train_sem)) for _ in range(64)]
        rows = make_batch_examples(tpl, train_sem, idx, "augmented", seed, step)
        logits, ans = logits_answers(model, rows, device)
        loss = F.cross_entropy(logits, ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            m = metrics(model, tpl, control_sem, device)
            trajectory.append({"step": step, **m})
            streak = streak + 1 if m["qualified"] else 0
            if streak >= 3:
                qualification_step = step
                qualified_snapshot = {
                    "model": copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}),
                    "optimizer": copy.deepcopy(opt.state_dict()),
                }
                break

    sealed_at_qualification = metrics(model, tpl, sealed_sem, device) if qualification_step else None
    retention = None
    if qualified_snapshot:
        stream_rng = random.Random(810000 + seed)
        stream = [stream_rng.randrange(len(train_sem)) for _ in range(600 * 64)]
        retention = {}
        for label, lr in [("HIGH",3e-4),("LOW",3e-6)]:
            m = Ark018GPT().to(device)
            m.load_state_dict(qualified_snapshot["model"])
            o = torch.optim.AdamW(m.parameters(), lr=lr, betas=(.9,.95), eps=1e-8, weight_decay=.1)
            o.load_state_dict(qualified_snapshot["optimizer"])
            for g in o.param_groups: g["lr"] = lr
            curve = []
            for step in range(1, 601):
                idx = stream[(step-1)*64:step*64]
                rows = make_batch_examples(tpl, train_sem, idx, "canonical", seed, step)
                logits, ans = logits_answers(m, rows, device)
                loss = F.cross_entropy(logits, ans)
                o.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); o.step()
                if step % 100 == 0:
                    curve.append({"step":step, **metrics(m,tpl,sealed_sem,device)})
            retention[label] = curve
            del m, o
            torch.cuda.empty_cache()

    payload = {
        "schema":"arkenstone-ark018-binding-probe/v3-fast",
        "seed":seed,
        "arm":arm,
        "selected_token_ids":chosen,
        "train_factsets":len(train_fs),
        "control_factsets":len(control_fs),
        "sealed_factsets":len(sealed_fs),
        "qualification_step":qualification_step,
        "control_trajectory":trajectory,
        "sealed_at_qualification":sealed_at_qualification,
        "retention_high_vs_low":retention,
        "claim_boundary":"controlled one-token temporary-binding acquisition/retention diagnostic",
    }
    base.safe_save_json(out_path,payload)
    del model,opt
    torch.cuda.empty_cache()
    return payload
