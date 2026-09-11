from __future__ import annotations

import argparse, copy, hashlib, itertools, json, math, random, sys, time, traceback, zipfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'ARK-018'))
import ark019_v4_core as C
import run_ark019_v3 as V3

ARK018_ROOT = Path('/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1')
OUT = Path('/content/drive/MyDrive/genisis-arkenstone/ARK019_GUARDIAN_V4')
EXPECTED_HORIZON = 8000


class SessionTimebox(RuntimeError):
    pass


def savej(p: Path, x: Mapping[str, Any]) -> None:
    V3.savej(p, x)


def hjson(x: Any) -> str:
    return V3.hjson(x)


def hfile(p: Path) -> str:
    return V3.hfile(p)


def setup() -> None:
    V3.setup()


def device() -> torch.device:
    if not torch.cuda.is_available(): raise RuntimeError('ARK-019 V4 requires Colab CUDA/T4')
    return torch.device('cuda')


def opt_for(m, lr): return torch.optim.AdamW(m.parameters(), lr=lr, betas=(.9, .95), eps=1e-8, weight_decay=.1)

def opt_to(o, d): V3.opt_to(o, d)

def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def snapshot(m, o, sc):
    return {'model': cpu_state(m), 'optimizer': copy.deepcopy(o.state_dict()), 'scaler': copy.deepcopy(sc.state_dict()),
            'cpu_rng': torch.get_rng_state().cpu(), 'cuda_rng': [x.cpu() for x in torch.cuda.get_rng_state_all()]}


def restore(s, d, lr=C.HIGH_LR):
    m = V3.Ark018GPT().to(d); m.load_state_dict(s['model'])
    o = opt_for(m, lr); o.load_state_dict(s['optimizer']); opt_to(o, d)
    for g in o.param_groups: g['lr'] = lr
    sc = V3.make_scaler(d); sc.load_state_dict(s.get('scaler', {}))
    torch.set_rng_state(s['cpu_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
    return m, o, sc


def model_hash(m) -> str: return V3.model_state_hash(m)


def optimizer_hash(o) -> str:
    sd = o.state_dict(); h = hashlib.sha256()
    for gi, g in enumerate(sd['param_groups']):
        h.update(f'g{gi}'.encode())
        for k, v in sorted((k, v) for k, v in g.items() if k != 'params'):
            h.update(str(k).encode()); h.update(repr(v).encode())
        h.update(','.join(map(str, g['params'])).encode())
    for pid, state in sorted(sd['state'].items(), key=lambda kv: int(kv[0])):
        h.update(f'p{pid}'.encode())
        for k, v in sorted(state.items()):
            h.update(str(k).encode())
            if torch.is_tensor(v):
                a = v.detach().cpu().contiguous(); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
            else: h.update(repr(v).encode())
    return h.hexdigest()


def scaler_hash(sc) -> str: return hjson(sc.state_dict())

def load_substrate(): return V3.load_substrate()


def source_checkpoint(seed, prep):
    p = V3.ckpt_path(seed)
    if not p.exists(): raise FileNotFoundError(p)
    x = torch.load(p, map_location='cpu', weights_only=False)
    if int(x.get('step', -1)) != int(prep['horizon_updates']) or x.get('arm') != 'SCIENCE_ONLY':
        raise RuntimeError(f'ARK-018 source checkpoint identity mismatch seed={seed}')
    if x.get('science_sha256') != V3.EXPECTED_SCIENCE_SHA or x.get('tokenizer_sha256') != prep['tokenizer_sha256']:
        raise RuntimeError(f'ARK-018 source content identity mismatch seed={seed}')
    return p, x


def split_factsets(keys, vals, seed):
    x = [tuple(zip(kt, vt)) for kt in itertools.combinations(keys, 3) for vt in itertools.permutations(vals, 3)]
    random.Random(seed).shuffle(x)
    if len(x) < 600: raise RuntimeError('insufficient factsets')
    return {'train': x[:400], 'parent_control': x[400:450], 'main_control': x[450:500],
            'validation': x[500:550], 'sealed': x[550:600]}


def sem(fs): return [(f, q, a) for f in fs for q, a in f]

def task_semantics(keys, vals, seed): return {k: sem(v) for k, v in split_factsets(keys, vals, seed).items()}


def templates(tok):
    return {'A': {'p': tok.encode('Facts: ').ids, 'm': tok.encode(' means ').ids, 's': tok.encode('; ').ids,
                  'q': tok.encode('Query: ').ids, 't': tok.encode(' means').ids},
            'B': {'p': tok.encode('Map: ').ids, 'm': tok.encode(' -> ').ids, 's': tok.encode(' | ').ids,
                  'q': tok.encode('Requested ').ids, 't': tok.encode(' =>').ids}}


def render(t, f, q, order):
    ids = list(t['p'])
    for i in order:
        k, v = f[i]; ids += [int(k)] + list(t['m']) + [int(v)] + list(t['s'])
    ids += list(t['q']) + [int(q)] + list(t['t'])
    return ids[-(V3.CONTEXT - 1):]


def rows(t, semantics, idx, mode, seed, step):
    out = []
    for i in idx:
        f, q, a = semantics[int(i)]
        if mode == 'canonical': order = (0, 1, 2)
        elif mode == 'reversed': order = (2, 1, 0)
        elif mode == 'query_order': order = C.query_order_perm(f, q)
        elif mode == 'augmented': order = C.augmented_perm(seed, step, int(i))
        elif mode == 'nonidentity': order = C.nonidentity_perm(seed, step, int(i))
        else: raise ValueError(mode)
        out.append((render(t, f, q, order), int(a)))
    return out


def logits_answers(m, rs, d):
    w = max(len(p) for p, _ in rs); x = torch.zeros((len(rs), w), dtype=torch.long, device=d)
    a = torch.tensor([a for _, a in rs], dtype=torch.long, device=d)
    for i, (p, _) in enumerate(rs): x[i, -len(p):] = torch.tensor(p, dtype=torch.long, device=d)
    return m(x)[:, -1, :], a


def binding_loss(m, rs, d):
    z, a = logits_answers(m, rs, d); return F.cross_entropy(z.float(), a, reduction='none')


@torch.no_grad()
def mode_acc(m, t, semantics, d, mode):
    hit = n = 0
    for st in range(0, len(semantics), 64):
        idx = list(range(st, min(st + 64, len(semantics))))
        z, a = logits_answers(m, rows(t, semantics, idx, mode, 0, 0), d)
        hit += int((z.argmax(-1) == a).sum()); n += len(a)
    return hit / max(1, n)


@torch.no_grad()
def bmetrics(m, t, semantics, d):
    rec = {'canonical': mode_acc(m, t, semantics, d, 'canonical'),
           'order_only': mode_acc(m, t, semantics, d, 'reversed'),
           'query_order': mode_acc(m, t, semantics, d, 'query_order')}
    rec['qualified'] = C.qualified(rec); rec['healthy_margin'] = C.healthy(rec); return rec


def science(m, bufs, d, sealed=True):
    out = {'control': V3.eval_buffer(m, bufs['control'], d, 71901, sequences=24)}
    if sealed: out['sealed'] = V3.eval_buffer(m, bufs['sealed'], d, 71902, sequences=24)
    return out


def real_batch(buf, n, seed, step, d, tag='main'):
    xs, ys, starts = [], [], []; need = V3.CONTEXT + 1
    for slot in range(n):
        z = int.from_bytes(hashlib.sha256(f'ark019v4-real:{tag}:{seed}:{step}:{slot}'.encode()).digest()[:8], 'big') % (len(buf) - need + 1)
        r = np.asarray(buf[z:z + need], dtype=np.int64); xs.append(r[:-1]); ys.append(r[1:]); starts.append(z)
    return torch.tensor(np.stack(xs), dtype=torch.long, device=d), torch.tensor(np.stack(ys), dtype=torch.long, device=d), starts


def pnames(m):
    wanted = {'tok.weight', 'blocks.0.attn.qkv.weight', 'blocks.4.mlp.2.weight', 'blocks.9.attn.qkv.weight', 'ln_f.weight'}
    return [n for n, _ in m.named_parameters() if n in wanted]


def psnap(m, names):
    p = dict(m.named_parameters()); return {n: p[n].detach().float().clone() for n in names}


def pdelta(x, m):
    p = dict(m.named_parameters()); return math.sqrt(sum(float(((p[n].detach().float() - v) ** 2).sum()) for n, v in x.items()))


def backup(m): return [p.detach().clone() for p in m.parameters()]

def fulldelta(b, m): return math.sqrt(sum(float(((p.detach().float() - x.float()) ** 2).sum()) for x, p in zip(b, m.parameters())))


def cap_project(b, m, raw, cap):
    if raw <= cap or raw <= 0: return raw
    scale = cap / raw
    with torch.no_grad():
        for x, p in zip(b, m.parameters()): p.copy_(x + (p - x) * scale)
    return cap


def displacement(m, parent):
    return math.sqrt(sum(float(((v.detach().float().cpu() - parent[k].float()) ** 2).sum()) for k, v in m.state_dict().items()))


def mixed_update(m, o, sc, train_buf, task_t, task_sem, stream_seed, step, task_slots, replay_t, replay_sem,
                 replay_level, cap, d, names, tag='main', full_delta=False):
    do_replay = replay_level == 32 or (replay_level == 64 and step % 2 == 0)
    replay_slots = int(do_replay); real_slots = C.BATCH_SLOTS - int(task_slots) - replay_slots
    if real_slots <= 0: raise RuntimeError('non-positive real-text slots')
    x, y, starts = real_batch(train_buf, real_slots, stream_seed, step, d, tag=tag)
    ti = C.deterministic_indices(stream_seed, step, task_slots, len(task_sem), f'{tag}-task')
    tr = rows(task_t, task_sem, ti, 'augmented', stream_seed, step)
    o.zero_grad(set_to_none=True)
    with V3.autocast_ctx(d):
        z = m(x); tl = F.cross_entropy(z.float().reshape(-1, z.size(-1)), y.reshape(-1), reduction='none').view(real_slots, -1).mean(1)
        pieces = [tl, binding_loss(m, tr, d)]; ri = None
        if do_replay:
            ri = C.deterministic_indices(stream_seed, step, 1, len(replay_sem), f'{tag}-replay-{replay_level}')[0]
            pieces.append(binding_loss(m, rows(replay_t, replay_sem, [ri], 'nonidentity', stream_seed + replay_level, step), d))
        loss = torch.cat(pieces).sum() / C.BATCH_SLOTS
    sc.scale(loss).backward(); sc.unscale_(o); grad = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    pb = psnap(m, names); fb = backup(m) if cap is not None or full_delta else None
    sc.step(o); sc.update(); pd = pdelta(pb, m); raw = applied = None; capped = False
    if fb is not None:
        raw = fulldelta(fb, m); applied = raw
        if cap is not None and raw > cap: applied = cap_project(fb, m, raw, cap); capped = True
    return {'loss': float(loss.detach()), 'grad': grad, 'projected_delta': pd, 'raw_full_delta': raw,
            'applied_full_delta': applied, 'capped': capped, 'replay': do_replay,
            'replay_level': replay_level if do_replay else 0, 'replay_index': ri,
            'real_slots': real_slots, 'task_slots': int(task_slots), 'real_starts_sha256': hjson(starts)}


def save_checkpoint(path: Path, payload: Mapping[str, Any], m, o, sc) -> None:
    body = dict(payload); body.update(snapshot(m, o, sc)); path.parent.mkdir(parents=True, exist_ok=True)
    q = path.with_suffix(path.suffix + '.tmp'); torch.save(body, q); q.replace(path)


def load_checkpoint(path: Path, expected: Mapping[str, Any], d, lr=C.HIGH_LR):
    x = torch.load(path, map_location='cpu', weights_only=False)
    for k, v in expected.items():
        if x.get(k) != v: raise RuntimeError(f'checkpoint identity mismatch {k}: {x.get(k)!r} != {v!r}')
    m, o, sc = restore(x, d, lr); return x, m, o, sc


def select_tokens(tok, counts, n=24):
    for threshold in [256, 128, 64, 32]:
        candidates = []
        for tid in range(min(len(counts), tok.get_vocab_size())):
            if counts[tid] < threshold: continue
            text = tok.decode([tid]); stripped = text.strip()
            if not (4 <= len(stripped) <= 10 and stripped.isascii() and stripped.isalpha() and stripped.islower()): continue
            if tok.encode(text).ids != [tid]: continue
            if int(hashlib.sha256(stripped.encode()).hexdigest(), 16) % 4 != 0: continue
            candidates.append((-int(counts[tid]), tid, stripped))
        candidates.sort()
        if len(candidates) >= n: return [x[1] for x in candidates[:n]]
    raise RuntimeError(f'insufficient eligible single-token words; need {n}')


def build_world(tok, counts):
    ids = select_tokens(tok, counts, 24); A = task_semantics(ids[:6], ids[6:12], 524218); B = task_semantics(ids[12:18], ids[18:24], 524219)
    return ids, templates(tok), A, B


def source_science(seed, base, bufs, d):
    m = V3.Ark018GPT().to(d); m.load_state_dict(base['model']); s = science(m, bufs, d, sealed=True); del m; torch.cuda.empty_cache(); return s


def parent_paths(seed):
    p = OUT / 'parents' / f'seed_{seed}'; return p, p / 'PARENT_V4.json', p / 'PARENT_V4.pt', p / 'PARENT_V4_RESUME.pt'


def acquire_parent(seed, prep, bufs, tA, A, d, deadline):
    pd, rp, sp, cp = parent_paths(seed); pd.mkdir(parents=True, exist_ok=True)
    source_path, base = source_checkpoint(seed, prep); source_sha = hfile(source_path); task_hash = hjson(A)
    expected = {'schema': 'arkenstone-ark019-v4-parent-ckpt/v1', 'seed': seed, 'source_checkpoint_sha256': source_sha, 'task_hash': task_hash}
    if rp.exists() and sp.exists():
        r = json.loads(rp.read_text())
        if r.get('status') == 'QUALIFIED' and r.get('source_checkpoint_sha256') == source_sha and r.get('task_hash') == task_hash: return r
        raise RuntimeError(f'incompatible V4 parent cache seed={seed}')
    baseline = source_science(seed, base, bufs, d)
    if cp.exists():
        x, m, o, sc = load_checkpoint(cp, expected, d); start = int(x['step']); traj = list(x['trajectory']); streak = int(x['streak'])
    else:
        m = V3.Ark018GPT().to(d); m.load_state_dict(base['model']); o = opt_for(m, C.HIGH_LR); sc = V3.make_scaler(d); start = 0; traj = []; streak = 0
    names = pnames(m)
    for step in range(start + 1, C.PARENT_MAX_UPDATES + 1):
        if time.monotonic() >= deadline:
            save_checkpoint(cp, {**expected, 'step': step - 1, 'trajectory': traj, 'streak': streak}, m, o, sc)
            raise SessionTimebox(f'parent seed {seed} paused at {step-1}')
        rec = mixed_update(m, o, sc, bufs['train'], tA, A['train'], seed, step, C.PARENT_A_SLOTS, tA, A['train'], 0, None, d, names, tag='parent')
        if step % C.PARENT_EVAL_EVERY == 0:
            am = bmetrics(m, tA, A['parent_control'], d); ss = science(m, bufs, d, sealed=False)
            rel = (float(ss['control']['nll']) - float(baseline['control']['nll'])) / max(float(baseline['control']['nll']), 1e-12)
            good = C.qualified(am) and rel <= C.PARENT_SCIENCE_CONTROL_REL_MAX; streak = streak + 1 if good else 0
            row = {'step': step, 'loss': rec['loss'], 'a_parent_control': am, 'science_control_nll': ss['control']['nll'],
                   'science_control_relative': rel, 'joint_gate': good, 'streak': streak}
            traj.append(row); savej(pd / 'PARENT_PROGRESS.json', {'status': 'RUNNING', 'seed': seed, 'trajectory': traj})
            print('V4 PARENT', seed, step, 'A', round(C.robust_min(am), 4), 'science_rel', round(rel, 4), 'streak', streak, flush=True)
            if streak >= C.PARENT_STREAK:
                val = bmetrics(m, tA, A['validation'], d)
                if C.qualified(val):
                    final_science = science(m, bufs, d, sealed=True); torch.save(snapshot(m, o, sc), sp)
                    r = {'schema': 'arkenstone-ark019-v4-parent/v1', 'status': 'QUALIFIED', 'seed': seed,
                         'confirmation_step': step, 'validation': val, 'source_science': baseline, 'parent_science': final_science,
                         'science_control_relative': rel, 'source_checkpoint_sha256': source_sha, 'task_hash': task_hash,
                         'parent_model_sha256': model_hash(m), 'parent_state_sha256': hfile(sp), 'trajectory': traj,
                         'parent_mix': {'real_slots': C.PARENT_REAL_SLOTS, 'skill_a_slots': C.PARENT_A_SLOTS}}
                    savej(rp, r); cp.unlink(missing_ok=True); del m, o; torch.cuda.empty_cache(); return r
                streak = 0
        if step % C.CHECKPOINT_EVERY == 0: save_checkpoint(cp, {**expected, 'step': step, 'trajectory': traj, 'streak': streak}, m, o, sc)
    fail = {'schema': 'arkenstone-ark019-v4-parent/v1', 'status': 'FAILED_TO_QUALIFY_JOINT_GATE', 'seed': seed,
            'trajectory': traj, 'source_checkpoint_sha256': source_sha, 'task_hash': task_hash}
    savej(rp, fail); del m, o; torch.cuda.empty_cache(); return fail


def load_parent(seed): return torch.load(parent_paths(seed)[2], map_location='cpu', weights_only=False)


def pilot_path(seed, slots):
    p = OUT / 'dose_pilot' / f'p{seed}_slots{slots}'; return p, p / 'RESULT.json', p / 'RESUME.pt'


def run_dose_pilot(parent_seed, slots, order_seed, parent, bufs, tB, B, d, deadline):
    pd, rp, cp = pilot_path(parent_seed, slots); pd.mkdir(parents=True, exist_ok=True)
    parent_sha = V3.state_hash(parent['model']); task_hash = hjson(B)
    expected = {'schema': 'arkenstone-ark019-v4-pilot-ckpt/v1', 'parent_seed': parent_seed, 'slots': slots,
                'order_seed': order_seed, 'parent_sha': parent_sha, 'task_hash': task_hash}
    if rp.exists():
        r = json.loads(rp.read_text())
        if r.get('parent_sha') == parent_sha and r.get('slots') == slots and r.get('order_seed') == order_seed: return r
        raise RuntimeError(f'incompatible dose pilot {pd}')
    if cp.exists():
        x, m, o, sc = load_checkpoint(cp, expected, d); start = int(x['step']); traj = list(x['trajectory']); streak = int(x['streak'])
    else:
        m, o, sc = restore(parent, d, C.HIGH_LR); start = 0; traj = []; streak = 0
    names = pnames(m); confirmation = None; val = None
    for step in range(start + 1, C.PILOT_MAX_UPDATES + 1):
        if time.monotonic() >= deadline:
            save_checkpoint(cp, {**expected, 'step': step - 1, 'trajectory': traj, 'streak': streak}, m, o, sc)
            raise SessionTimebox(f'dose pilot p{parent_seed}/slots{slots} paused at {step-1}')
        rec = mixed_update(m, o, sc, bufs['train'], tB, B['train'], order_seed, step, slots, tB, B['train'], 0, None, d, names, tag=f'pilot-{slots}')
        if step % C.PILOT_EVAL_EVERY == 0:
            bm = bmetrics(m, tB, B['parent_control'], d); streak = streak + 1 if C.qualified(bm) else 0
            traj.append({'step': step, 'loss': rec['loss'], 'b_pilot_control': bm, 'streak': streak})
            print('V4 B-DOSE', parent_seed, slots, step, round(C.robust_min(bm), 4), 'streak', streak, flush=True)
            if streak >= C.PILOT_STREAK:
                val = bmetrics(m, tB, B['validation'], d)
                if C.qualified(val): confirmation = step; break
                streak = 0
        if step % C.CHECKPOINT_EVERY == 0: save_checkpoint(cp, {**expected, 'step': step, 'trajectory': traj, 'streak': streak}, m, o, sc)
    sci = science(m, bufs, d, sealed=False); status = 'QUALIFIED' if confirmation is not None and confirmation <= C.PILOT_CONFIRM_DEADLINE else 'FAILED'
    r = {'schema': 'arkenstone-ark019-v4-dose-pilot/v1', 'status': status, 'parent_seed': parent_seed, 'slots': slots,
         'order_seed': order_seed, 'confirmation_step': confirmation, 'validation': val,
         'validation_qualified': bool(val and C.qualified(val)), 'science_control_nll': sci['control']['nll'],
         'parent_sha': parent_sha, 'task_hash': task_hash, 'trajectory': traj}
    savej(rp, r); cp.unlink(missing_ok=True); del m, o; torch.cuda.empty_cache(); return r


def select_dose(parents, bufs, tB, B, d, deadline):
    p = OUT / 'DOSE_SELECTION.json'
    if p.exists():
        r = json.loads(p.read_text())
        if r.get('status') == 'PASS' and int(r['selected_b_slots']) in C.PILOT_B_SLOT_CANDIDATES: return r
        raise RuntimeError('prior V4 dose selection did not pass')
    results = {}
    for slots in C.PILOT_B_SLOT_CANDIDATES:
        rows_out = []
        for seed, order_seed in zip(C.PRETRAIN_SEEDS, C.PILOT_ORDER_SEEDS):
            rows_out.append(run_dose_pilot(seed, slots, order_seed, parents[seed], bufs, tB, B, d, deadline))
        results[slots] = rows_out; choice = C.choose_b_slots(results)
        if choice['status'] == 'PASS': savej(p, choice); return choice
    choice = C.choose_b_slots(results); savej(p, choice); return choice


def capcal(parent_seed, order_seed, selected_slots, parent, bufs, tA, A, tB, B, d, deadline):
    p = OUT / 'matched_sets' / f'p{parent_seed}_b{order_seed}' / 'CAP_CALIBRATION.json'
    if p.exists(): return json.loads(p.read_text())
    if time.monotonic() >= deadline: raise SessionTimebox('before cap calibration')
    m, o, sc = restore(parent, d, C.LOW_LR); names = pnames(m); ds = []
    for step in range(1, C.CAP_SHADOW_STEPS + 1):
        rec = mixed_update(m, o, sc, bufs['train'], tB, B['train'], order_seed, step, selected_slots, tA, A['train'], 0, None, d, names, tag='cap-shadow', full_delta=True)
        ds.append(float(rec['applied_full_delta']))
    med = float(np.median(ds)); r = {'schema': 'arkenstone-ark019-v4-capcal/v1', 'parent_seed': parent_seed,
        'order_seed': order_seed, 'selected_b_slots': selected_slots, 'shadow_lr': C.LOW_LR, 'steps': C.CAP_SHADOW_STEPS,
        'full_delta_norms': ds, 'median_low_delta': med, 'cap16x': 16 * med, 'parent_model_sha256': V3.state_hash(parent['model'])}
    savej(p, r); del m, o; torch.cuda.empty_cache(); return r


def arm_paths(ps, bs, arm):
    p = OUT / 'matched_sets' / f'p{ps}_b{bs}' / arm; return p, p / 'RESULT.json', p / 'RESUME.pt', p / 'PARTIAL.json'


def run_arm(ps, bs, arm, selected_slots, parent, cap16, bufs, tA, A, tB, B, d, deadline):
    od, rp, cp, pp = arm_paths(ps, bs, arm); od.mkdir(parents=True, exist_ok=True)
    parent_sha = V3.state_hash(parent['model']); a_hash = hjson(A); b_hash = hjson(B)
    expected = {'schema': 'arkenstone-ark019-v4-arm-ckpt/v1', 'parent_seed': ps, 'order_seed': bs, 'arm': arm,
                'selected_b_slots': selected_slots, 'parent_sha': parent_sha, 'a_hash': a_hash, 'b_hash': b_hash, 'cap16x': float(cap16)}
    if rp.exists():
        r = json.loads(rp.read_text())
        if r.get('status') == 'COMPLETE' and int(r.get('step', -1)) == C.HORIZON and r.get('parent_sha') == parent_sha: return r
        raise RuntimeError(f'incompatible V4 arm result {rp}')
    if cp.exists():
        x, m, o, sc = load_checkpoint(cp, expected, d); start = int(x['step']); ctrl = x['controller']; cnt = x['counters']
        control_trace = C.dedupe_trajectory(list(x.get('control_trace', [])), start); measurement_trace = C.dedupe_trajectory(list(x.get('measurement_trace', [])), start)
    else:
        m, o, sc = restore(parent, d, C.HIGH_LR); start = 0; ctrl = C.initial_controller(arm)
        cnt = {'replay_slots': 0, 'real_slots': 0, 'skill_b_slots': 0, 'capped_steps': 0, 'protection_updates': 0, 'projected_path': 0.0}
        control_trace = []; measurement_trace = []
    names = pnames(m)
    if start == 0:
        control_trace.append({'step': 0, 'a_control': bmetrics(m, tA, A['main_control'], d), 'b_control': bmetrics(m, tB, B['main_control'], d), 'controller': copy.deepcopy(ctrl)})
        s0 = science(m, bufs, d, sealed=True); measurement_trace.append({'step': 0, 'a_sealed': bmetrics(m, tA, A['sealed'], d),
            'b_sealed': bmetrics(m, tB, B['sealed'], d), 'science': s0, 'full_displacement': 0.0})
    for step in range(start + 1, C.HORIZON + 1):
        if time.monotonic() >= deadline:
            payload = {**expected, 'step': step - 1, 'controller': ctrl, 'counters': cnt, 'control_trace': control_trace, 'measurement_trace': measurement_trace}
            save_checkpoint(cp, payload, m, o, sc)
            savej(pp, {'schema': 'arkenstone-ark019-v4-partial/v1', 'status': 'PARTIAL_SESSION', 'step': step - 1,
                       'parent_seed': ps, 'order_seed': bs, 'arm': arm, 'control_trace': control_trace, 'measurement_trace': measurement_trace,
                       'controller': ctrl, 'counters': cnt})
            del m, o; torch.cuda.empty_cache(); raise SessionTimebox(f'arm {ps}/{bs}/{arm} paused at {step-1}')
        replay, cap = C.treatment(arm, ctrl, cap16, step); cnt['protection_updates'] += int(bool(replay or cap is not None))
        rec = mixed_update(m, o, sc, bufs['train'], tB, B['train'], bs, step, selected_slots, tA, A['train'], replay, cap, d, names, tag=f'main-{arm}')
        cnt['replay_slots'] += int(rec['replay']); cnt['real_slots'] += int(rec['real_slots']); cnt['skill_b_slots'] += selected_slots
        cnt['capped_steps'] += int(rec['capped']); cnt['projected_path'] += float(rec['projected_delta'])
        if step % C.CONTROL_EVERY == 0:
            ac = bmetrics(m, tA, A['main_control'], d); bc = None
            if step % C.B_CONTROL_EVERY == 0: bc = bmetrics(m, tB, B['main_control'], d); C.update_b_confirmation(ctrl, step, bc)
            C.update_controller(arm, ctrl, step, ac); control_trace.append({'step': step, 'a_control': ac, 'b_control': bc, 'controller': copy.deepcopy(ctrl), 'loss': rec['loss']})
            print(f'V4 [{ps}/{bs}/{arm}] {step}/{C.HORIZON} A={C.robust_min(ac):.3f} B={None if bc is None else round(C.robust_min(bc),3)} state={ctrl["state"]}', flush=True)
        if step % C.MEASURE_EVERY == 0:
            measurement_trace.append({'step': step, 'a_sealed': bmetrics(m, tA, A['sealed'], d), 'b_sealed': bmetrics(m, tB, B['sealed'], d),
                'science': science(m, bufs, d, sealed=True), 'full_displacement': displacement(m, parent['model']),
                'last_projected_delta': rec['projected_delta'], 'last_raw_full_delta': rec['raw_full_delta'], 'last_applied_full_delta': rec['applied_full_delta']})
            savej(pp, {'schema': 'arkenstone-ark019-v4-partial/v1', 'status': 'RUNNING', 'step': step, 'parent_seed': ps, 'order_seed': bs,
                       'arm': arm, 'control_trace': control_trace, 'measurement_trace': measurement_trace, 'controller': ctrl, 'counters': cnt})
        if step % C.CHECKPOINT_EVERY == 0:
            save_checkpoint(cp, {**expected, 'step': step, 'controller': ctrl, 'counters': cnt, 'control_trace': control_trace, 'measurement_trace': measurement_trace}, m, o, sc)
    control_trace = C.dedupe_trajectory(control_trace, C.HORIZON); measurement_trace = C.dedupe_trajectory(measurement_trace, C.HORIZON); f = measurement_trace[-1]
    r = {'schema': 'arkenstone-ark019-v4-arm/v1', 'status': 'COMPLETE', 'step': C.HORIZON, 'parent_seed': ps, 'order_seed': bs, 'arm': arm,
         'selected_b_slots': selected_slots, 'cap16x': cap16, 'parent_sha': parent_sha, 'control_trace': control_trace, 'measurement_trace': measurement_trace,
         'controller': ctrl, 'counters': cnt, 'b_qualification_step': ctrl.get('b_qualification_step'), 'b_confirmation_step': ctrl.get('b_confirmation_step'),
         'final_a_sealed': f['a_sealed'], 'final_b_sealed': f['b_sealed'], 'final_science_sealed_nll': float(f['science']['sealed']['nll']),
         'final_model_sha256': model_hash(m), 'final_optimizer_sha256': optimizer_hash(o)}
    savej(rp, r); cp.unlink(missing_ok=True); del m, o; torch.cuda.empty_cache(); return r


def exact_resume_smoke(parent, selected_slots, bufs, tA, A, tB, B, d):
    p = OUT / 'EXACT_RESUME_SMOKE_V4.json'
    if p.exists(): return json.loads(p.read_text())
    def advance(m, o, sc, start, stop):
        names = pnames(m); tr = []
        for step in range(start, stop + 1):
            r = mixed_update(m, o, sc, bufs['train'], tB, B['train'], 439991, step, selected_slots, tA, A['train'], 64, None, d, names, tag='resume-smoke')
            tr.append((step, r['replay'], r['real_starts_sha256'], round(r['loss'], 10)))
        return tr
    a, ao, asc = restore(parent, d); t1 = advance(a, ao, asc, 1, 10); h1 = (model_hash(a), optimizer_hash(ao), scaler_hash(asc))
    b, bo, bsc = restore(parent, d); t2a = advance(b, bo, bsc, 1, 5); ss = snapshot(b, bo, bsc); del b, bo; torch.cuda.empty_cache()
    c, co, csc = restore(ss, d); t2b = advance(c, co, csc, 6, 10); h2 = (model_hash(c), optimizer_hash(co), scaler_hash(csc))
    ok = h1 == h2 and t1 == t2a + t2b
    r = {'schema': 'arkenstone-ark019-v4-exact-resume/v1', 'status': 'PASS' if ok else 'FAIL', 'model_optimizer_scaler_identical': h1 == h2,
         'telemetry_identical': t1 == t2a + t2b, 'uninterrupted': h1, 'resumed': h2}; savej(p, r)
    if not ok: raise RuntimeError('ARK-019 V4 exact-resume smoke failed')
    del a, ao, c, co; torch.cuda.empty_cache(); return r


def calibrate(parent, selected_slots, bufs, tA, A, tB, B, d):
    p = OUT / 'RUNTIME_CALIBRATION_V4.json'
    if p.exists(): return json.loads(p.read_text())
    rec = {}
    for label, replay, capflag in [('PLASTIC', 0, False), ('REPLAY32', 32, False), ('CAP16X_PROXY', 0, True)]:
        m, o, sc = restore(parent, d); names = pnames(m); cap = 1e9 if capflag else None
        for step in (1, 2): mixed_update(m, o, sc, bufs['train'], tB, B['train'], 449991, step, selected_slots, tA, A['train'], replay, cap, d, names, tag='runtime', full_delta=capflag)
        torch.cuda.synchronize(); st = time.monotonic(); n = 5
        for step in range(3, 3 + n): mixed_update(m, o, sc, bufs['train'], tB, B['train'], 449991, step, selected_slots, tA, A['train'], replay, cap, d, names, tag='runtime', full_delta=capflag)
        torch.cuda.synchronize(); rec[label] = {'update_seconds': (time.monotonic() - st) / n}; del m, o; torch.cuda.empty_cache()
    m, o, sc = restore(parent, d); torch.cuda.synchronize(); st = time.monotonic(); bmetrics(m, tA, A['main_control'], d); bmetrics(m, tB, B['main_control'], d); torch.cuda.synchronize(); control_sec = time.monotonic() - st
    torch.cuda.synchronize(); st = time.monotonic(); bmetrics(m, tA, A['sealed'], d); bmetrics(m, tB, B['sealed'], d); science(m, bufs, d, sealed=True); torch.cuda.synchronize(); measure_sec = time.monotonic() - st
    del m, o; torch.cuda.empty_cache(); worst = max(v['update_seconds'] for v in rec.values())
    main_updates = len(C.PRETRAIN_SEEDS) * len(C.MAIN_ORDER_SEEDS) * len(C.ARMS) * C.HORIZON
    control_events = len(C.PRETRAIN_SEEDS) * len(C.MAIN_ORDER_SEEDS) * len(C.ARMS) * (C.HORIZON // C.CONTROL_EVERY)
    measure_events = len(C.PRETRAIN_SEEDS) * len(C.MAIN_ORDER_SEEDS) * len(C.ARMS) * (C.HORIZON // C.MEASURE_EVERY)
    est = (main_updates * worst + control_events * control_sec + measure_events * measure_sec) * C.RUNTIME_SAFETY_FACTOR
    sessions = max(1, math.ceil(est / ((C.SESSION_WALL_MINUTES - C.PACKAGING_RESERVE_MINUTES) * 60)))
    r = {'schema': 'arkenstone-ark019-v4-runtime/v1', 'status': 'PASS', 'selected_b_slots': selected_slots, 'arms': rec,
         'control_eval_seconds': control_sec, 'measurement_seconds': measure_sec, 'estimated_main_seconds': est,
         'estimated_main_sessions': sessions, 'session_wall_minutes': C.SESSION_WALL_MINUTES, 'safety_factor': C.RUNTIME_SAFETY_FACTOR,
         'protocol_changes_from_runtime': False}; savej(p, r); return r


def collect_arm_results():
    out = {}
    for p in OUT.glob('matched_sets/p*_b*/*/RESULT.json'):
        r = json.loads(p.read_text()); out[f'{r["parent_seed"]}:{r["order_seed"]}:{r["arm"]}'] = r
    return out


def package(final=False):
    manifests = {}
    for p in sorted(OUT.rglob('*.json')):
        if p.name == 'ZIP_MANIFEST_V4.json': continue
        manifests[str(p.relative_to(OUT))] = hfile(p)
    savej(OUT / 'ZIP_MANIFEST_V4.json', {'schema': 'arkenstone-ark019-v4-manifest/v1', 'members': manifests})
    name = 'ARKENSTONE_ARK019_V4_GUARDIAN_RESULTS.zip' if final else 'ARKENSTONE_ARK019_V4_GUARDIAN_PARTIAL.zip'; z = OUT / name
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as f:
        for p in sorted(OUT.rglob('*.json')): f.write(p, str(p.relative_to(OUT)))
    (OUT / (name + '.sha256')).write_text(hfile(z) + '  ' + name + '\n'); return {'path': str(z), 'sha256': hfile(z)}


def run_all():
    setup(); d = device(); OUT.mkdir(parents=True, exist_ok=True); started = time.monotonic(); deadline = started + (C.SESSION_WALL_MINUTES - C.PACKAGING_RESERVE_MINUTES) * 60
    prep, tok, bufs, counts = load_substrate(); ids, tt, A, B = build_world(tok, counts); sources = []
    for seed in C.PRETRAIN_SEEDS:
        p, x = source_checkpoint(seed, prep); sources.append({'seed': seed, 'sha256': hfile(p), 'step': int(x['step']), 'model_sha256': V3.state_hash(x['model'])})
    savej(OUT / 'ENTRY_RECEIPT_V4.json', {'schema': 'arkenstone-ark019-v4-entry/v1', 'r3_bundle_sha256': C.R3_BUNDLE_SHA256,
          'r3_official_verdict': C.R3_OFFICIAL_VERDICT, 'science_sha256': V3.EXPECTED_SCIENCE_SHA, 'tokenizer_sha256': prep['tokenizer_sha256'],
          'sources': sources, 'selected_token_ids': ids, 'skill_a_ids': ids[:12], 'skill_b_ids': ids[12:24],
          'overlap': sorted(set(ids[:12]) & set(ids[12:24])), 'a_task_hash': hjson(A), 'b_task_hash': hjson(B)})
    if set(ids[:12]) & set(ids[12:24]): raise RuntimeError('A/B token overlap')
    try:
        parents = {}
        for seed in C.PRETRAIN_SEEDS:
            r = acquire_parent(seed, prep, bufs, tt['A'], A, d, deadline)
            if r.get('status') != 'QUALIFIED': raise RuntimeError(f'V4 joint parent gate failed seed={seed}')
            parents[seed] = load_parent(seed)
        dose = select_dose(parents, bufs, tt['B'], B, d, deadline)
        if dose.get('status') != 'PASS':
            result = {'schema': 'arkenstone-ark019-v4-result/v1', 'status': 'BLOCKED_BEFORE_MAIN', 'verdict': 'INCONCLUSIVE_NO_VIABLE_SKILL_B_DOSE', 'dose_selection': dose, 'authorized': False}
            savej(OUT / 'ARK-019_V4_RESULT.json', result); result['bundle'] = package(final=True); return result
        selected_slots = int(dose['selected_b_slots']); smoke = exact_resume_smoke(parents[C.PRETRAIN_SEEDS[0]], selected_slots, bufs, tt['A'], A, tt['B'], B, d)
        runtime = calibrate(parents[C.PRETRAIN_SEEDS[0]], selected_slots, bufs, tt['A'], A, tt['B'], B, d)
        savej(OUT / 'PREEXECUTION_GATE_V4.json', {'schema': 'arkenstone-ark019-v4-preexecution/v1', 'status': 'PASS', 'dose_selection': dose,
              'exact_resume': smoke, 'runtime': runtime, 'note': 'runtime estimates sessions only; protocol is not reduced'})
        for ps in C.PRETRAIN_SEEDS:
            for bs in C.MAIN_ORDER_SEEDS:
                cr = capcal(ps, bs, selected_slots, parents[ps], bufs, tt['A'], A, tt['B'], B, d, deadline)
                for arm in C.ARMS: run_arm(ps, bs, arm, selected_slots, parents[ps], float(cr['cap16x']), bufs, tt['A'], A, tt['B'], B, d, deadline)
        arms = collect_arm_results(); dec = C.decide(arms)
        result = {'schema': 'arkenstone-ark019-v4-result/v1', 'status': 'COMPLETE', 'decision': dec, 'selected_b_slots': selected_slots,
                  'arm_count': len(arms), 'wall_seconds_last_session': time.monotonic() - started,
                  'claim_boundary': 'real-text proxy continual-learning controller only'}
        savej(OUT / 'ARK-019_V4_RESULT.json', result); result['bundle'] = package(final=True); return result
    except SessionTimebox as e:
        arms = collect_arm_results(); state = {'schema': 'arkenstone-ark019-v4-session/v1', 'status': 'PARTIAL_SESSION', 'message': str(e),
                 'completed_arm_count': len(arms), 'wall_seconds': time.monotonic() - started,
                 'instruction': 'rerun the same frozen notebook; exact Drive checkpoints resume work'}
        savej(OUT / 'SESSION_STATE_V4.json', state); state['bundle'] = package(final=False); return state


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['all'], default='all'); _ = ap.parse_args()
    try:
        r = run_all(); print('ARK-019 V4 STATUS', r.get('status')); print('VERDICT', r.get('decision', {}).get('verdict', r.get('verdict'))); print('BUNDLE', r.get('bundle')); return 0
    except Exception as e:
        OUT.mkdir(parents=True, exist_ok=True); savej(OUT / 'ARK-019_V4_FAILURE.json', {'schema': 'arkenstone-ark019-v4-failure/v1', 'status': 'FAILED',
              'exception': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()}); traceback.print_exc()
        try: package(final=False)
        except Exception: pass
        return 1


if __name__ == '__main__': raise SystemExit(main())
