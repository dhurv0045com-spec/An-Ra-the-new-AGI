"""Frozen Y4 composition baseline, corrected donor training, and Y1 recovery."""
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import random
import torch
import torch.nn.functional as F

from experiments.avendi_y4.models import Y4TypedCompositionModel, encode_y4_typed_batch, ENT_TO_IDX, p4_y4_loss
from experiments.avendi_y4.tasks import ENTITY_64, build_state_swap_pairs
from experiments.avendi_y4.plan import Y1_EXPECTED_CKPT_SHAS
from tools.avendi_y4_frozen_bridge import load_model

SEEDS = (7, 19, 41)

def jobs():
    rows = []
    for seed in SEEDS:
        baseline = f'C-baseline-s{seed}'
        scout = f'C-scout-s{seed}'
        parent_y4 = f'C-categorical_endpoint_only-s{seed}'
        rows += [
            dict(
                id=baseline, stage='C', arm='baseline', seed=seed, depends=[], updates=0,
                role='diagnostic', profile_group='composition_diagnostic', objective='eval_only',
                checkpoint_parent=parent_y4, split_counts={'train': 0, 'dev': 512, 'confirm': 0, 'transfer': 0},
                allowed_exclusions=['diagnostic_dev_only'],
            ),
            dict(
                id=scout, stage='C', arm='scout', seed=seed, depends=[baseline], updates=0,
                role='diagnostic', profile_group='composition_diagnostic', objective='eval_only',
                checkpoint_parent=parent_y4, split_counts={'train': 0, 'dev': 512, 'confirm': 0, 'transfer': 0},
                allowed_exclusions=['diagnostic_dev_only'],
            ),
        ]
        rows += [
            dict(
                id=f'C-{arm}-s{seed}', stage='C', arm=arm, seed=seed, depends=[scout], updates=1024,
                role='primary', profile_group=f'composition_{arm}', objective=arm,
                checkpoint_parent=parent_y4, split_counts={'train': 1024, 'dev': 512, 'confirm': 512, 'transfer': 512},
                allowed_exclusions=['C_insufficient_headroom', 'C_baseline_unqualified'],
            )
            for arm in ('ordinary', 'iit', 'sham')
        ]
    for lin in Y1_EXPECTED_CKPT_SHAS:
        rows.append(dict(
            id=f'G-{lin}', stage='G', arm=lin, seed=int(lin.rsplit('s', 1)[1]), depends=[], updates=0,
            role='diagnostic', profile_group='historical_y1', objective='parity_verify',
            checkpoint_parent=lin, split_counts={'dev': 0},
            allowed_exclusions=['G_optional_y1_missing'],
        ))
    return rows


def canonical_panel_signature(item):
    edges = tuple(sorted((str(a).strip(), str(b).strip()) for a, b in item['rows']))
    return (edges, str(item['query']).strip(), int(item['depth']))

def walk(rows, start, steps):
    # Independent graph validation; duplicate keys are invalid even if values agree.
    edges = {}
    for k, v in rows:
        if k in edges:
            raise ValueError('duplicate graph key')
        edges[k] = v
    for _ in range(steps):
        if start not in edges:
            raise ValueError('missing graph edge')
        start = edges[start]
    return start

def validate_pair(p):
    d, s = p['depth'], p['patch_step']
    checks = [(p['donor_rows'], p['donor_query'], s, p['donor_step']),
              (p['donor_rows'], p['donor_query'], d, p['donor_own_answer']),
              (p['recipient_rows'], p['query'], d, p['recipient_answer']),
              (p['recipient_rows'], p['donor_step'], d-s, p['donor_on_recipient_answer']),
              (p['rewired_rows'], p['donor_step'], d-s, p['rewired_donor_on_recipient_answer']),
              (p['original_donor_rows'], p['donor_query'], s, p['wrong_donor_state']),
              (p['wrong_donor_rows'], p['wrong_donor_state'], d-s, p['wrong_donor_actual_answer'])]
    if any(walk(rows, q, n) != target for rows, q, n, target in checks):
        raise ValueError('invalid intervention target')
    if len({p['recipient_answer'], p['donor_on_recipient_answer'], p['donor_own_answer']}) != 3 or p['wrong_donor_actual_answer'] == p['donor_on_recipient_answer']:
        raise ValueError('indistinguishable intervention')
    return True

def pairs(n, seed, depth=2):
    result = build_state_swap_pairs(n, seed, depth=depth, patch_step=1)
    for p in result:
        p['original_donor_rows'] = list(p['donor_rows'])
        old, new = p['wrong_donor_state'], p['donor_step']
        p['donor_rows'] = [(new if k == old else k, new if v == old else v) for k, v in p['donor_rows']]
        validate_pair(p)
    return result

def panel(n, seed, depth):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        nodes = rng.sample(ENTITY_64, min(64, depth + 25))
        rows = list(zip(nodes[:depth], nodes[1:depth+1]))
        # Distractors are distinct keys and cannot alter the intended path.
        rows += [(nodes[j], nodes[(j+1) % len(nodes)]) for j in range(depth+1, len(nodes)-1)]
        rng.shuffle(rows)
        gold = walk(rows, nodes[0], depth)
        out.append(dict(case_id=f'd{depth}-{i}', rows=rows, query=nodes[0], gold=gold, depth=depth))
    return out

def measure(model, cells, device):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cells), 32):
            batch = cells[start:start+32]
            rk, rv, q, vm, gold = encode_y4_typed_batch(batch, torch.device(device))
            pred = model.forward_trace(rk, rv, q, vm, steps=batch[0]['depth'])['final_logits'].argmax(-1)
            result += [dict(case_id=c['case_id'], query=c['query'], rows=c['rows'], depth=c['depth'], gold=ENTITY_64[b], prediction=ENTITY_64[a], correct=bool(a == b)) for c, a, b in zip(batch, pred.tolist(), gold.tolist())]
    return dict(correct=sum(r['correct'] for r in result), total=len(result), cases=result)

def measure_interventions(model, fixtures, device):
    """Evaluate actual soft donors, oracle, identity, rewire and wrong-donor controls."""
    records = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(fixtures), 32):
            batch = fixtures[start:start+32]
            for p in batch:
                validate_pair(p)
            def encode(rows, query, gold):
                return encode_y4_typed_batch([dict(rows=p[rows], query=p[query], gold=p[gold]) for p in batch], torch.device(device))
            dr, dv, dq, dm, _ = encode('donor_rows', 'donor_query', 'donor_own_answer')
            odr, odv, odq, odm, _ = encode('original_donor_rows', 'donor_query', 'donor_own_answer')
            rr, rv, rq, rm, _ = encode('recipient_rows', 'query', 'recipient_answer')
            wr, wv, wq, wm, _ = encode('rewired_rows', 'query', 'rewired_donor_on_recipient_answer')
            xr, xv, xq, xm, _ = encode('wrong_donor_rows', 'query', 'wrong_donor_actual_answer')
            depth = batch[0]['depth']
            native = model.forward_trace(rr, rv, rq, rm, steps=depth)
            donor = model.forward_trace(dr, dv, dq, dm, steps=1)['states'][0]
            wrong = model.forward_trace(odr, odv, odq, odm, steps=1)['states'][0]
            oracle = F.one_hot(torch.tensor([ENT_TO_IDX[p['donor_step']] for p in batch], device=device), num_classes=len(ENTITY_64)).float()
            def predict(r, v, q, mask, state):
                return model.forward_trace(r, v, q, mask, steps=depth, patch_at=1, patch_tensor=state)['final_logits'].argmax(-1).tolist()
            results = {
                'unpatched': native['final_logits'].argmax(-1).tolist(),
                'identity': predict(rr, rv, rq, rm, native['states'][0]),
                'actual_donor': predict(rr, rv, rq, rm, donor),
                'oracle_donor': predict(rr, rv, rq, rm, oracle),
                'rewired': predict(wr, wv, wq, wm, donor),
                'oracle_rewired': predict(wr, wv, wq, wm, oracle),
                'wrong_donor': predict(xr, xv, xq, xm, wrong),
            }
            targets = {'unpatched':'recipient_answer', 'identity':'recipient_answer', 'actual_donor':'donor_on_recipient_answer', 'oracle_donor':'donor_on_recipient_answer', 'rewired':'rewired_donor_on_recipient_answer', 'oracle_rewired':'rewired_donor_on_recipient_answer', 'wrong_donor':'wrong_donor_actual_answer'}
            for i, p in enumerate(batch):
                outcomes = {k: {'prediction':ENTITY_64[v[i]], 'gold':p[targets[k]], 'correct':ENTITY_64[v[i]] == p[targets[k]]} for k,v in results.items()}
                records.append({'pair_id':p['pair_id'], 'fixture':p, 'outcomes':outcomes, 'identity_same_prediction':results['identity'][i] == results['unpatched'][i], 'donor_state_prediction':ENTITY_64[int(donor[i].argmax())], 'wrong_donor_state_prediction':ENTITY_64[int(wrong[i].argmax())]})
    return {'total':len(records), 'correct':{k:sum(r['outcomes'][k]['correct'] for r in records) for k in targets}, 'identity_same_prediction':sum(r['identity_same_prediction'] for r in records), 'cases':records}

def intervention_loss(model, batch, arm):
    cells = [dict(rows=p['recipient_rows'], query=p['query'], gold=p['recipient_answer'], depth=p['depth']) for p in batch]
    loss = p4_y4_loss(model, cells, supervision='endpoint_only')
    device = next(model.parameters()).device
    if arm in ('ordinary', 'iit'):
        donor_rows, target = 'donor_rows', 'donor_on_recipient_answer'
    else:
        # Real sham: actual wrong-donor state and its independently solved recipient suffix.
        donor_rows, target = 'original_donor_rows', 'wrong_donor_actual_answer'
    # Identical recipient graph, including a valid alternate suffix, for all arms.
    recipient_rows = 'wrong_donor_rows'
    donor = [dict(rows=p[donor_rows], query=p['donor_query'], gold=p['donor_own_answer']) for p in batch]
    recipient = [dict(rows=p[recipient_rows], query=p['query'], gold=p[target]) for p in batch]
    dr, dv, dq, dm, _ = encode_y4_typed_batch(donor, device)
    rr, rv, rq, rm, g = encode_y4_typed_batch(recipient, device)
    state = model.forward_trace(dr, dv, dq, dm, steps=1)['states'][0]
    logits = model.forward_trace(rr, rv, rq, rm, steps=batch[0]['depth'], patch_at=1, patch_tensor=state)['final_logits']
    return loss + (0.0 if arm == 'ordinary' else .25) * F.cross_entropy(logits, g)

def preflight():
    fixtures = pairs(16, 123, 2) + pairs(16, 124, 4)
    broken = dict(fixtures[0], donor_step=fixtures[0]['wrong_donor_state'])
    try:
        validate_pair(broken)
    except ValueError:
        caught = True
    else:
        caught = False
    dev_sigs = {canonical_panel_signature(c) for d in (5, 6) for c in panel(32, 9000 + d, d)}
    conf_sigs = {canonical_panel_signature(c) for d in (5, 6) for c in panel(32, 10000 + d, d)}
    trans_sigs = {canonical_panel_signature(c) for d in (5, 6) for c in panel(32, 12000 + d, d)}
    disjoint = not (dev_sigs & conf_sigs or dev_sigs & trans_sigs or conf_sigs & trans_sigs)
    sample = panel(2, 9005, 5)[0]
    reordered = dict(sample, case_id='confirm-reordered', rows=list(reversed(sample['rows'])))
    renamed_caught = canonical_panel_signature(reordered) == canonical_panel_signature(sample)
    return dict(
        passed=bool(caught and disjoint and renamed_caught),
        fixture_count=len(fixtures),
        deliberately_corrupt_donor_detected=caught,
        disjoint_eval_panels=disjoint,
        renamed_duplicate_caught=renamed_caught,
    )

def _historical(job, ctx):
    from experiments.avendi_y4.verify import stage_historical_checkpoints_to_writable
    from experiments.avendi_y4.module import _find_y1_evidence, _y1_ref_parity
    from anra_v5.planthir import load_legacy_byte_checkpoint
    from anra_v5.elendril.avendi.frozen import _read_evidence_member_bytes, load_y1_suite
    import json
    lin = job['arm']
    roots = [str(p) for p in ctx.inputs]
    stage = Path(ctx.root) / 'historical' / job['id']
    info = stage_historical_checkpoints_to_writable(roots, [lin], stage)['staged'][lin]
    if not info.get('manifest_verified'):
        return dict(status='BLOCKED', verification='MISSING_OR_INVALID_CHECKPOINT', staging=info)
    evidence = _find_y1_evidence(roots)
    if evidence is None:
        return dict(status='BLOCKED', verification='MISSING_RAW_REFERENCE_EVIDENCE', staging=info)
    raw = _read_evidence_member_bytes(evidence, 'prepared/development_suite.json')
    if raw is None:
        return dict(status='BLOCKED', verification='MISSING_REFERENCE_SUITE', staging=info)
    cases = [c for c in load_y1_suite(json.loads(raw.decode()))['cases'] if c.panel == 'main-base']
    try:
        loaded = load_legacy_byte_checkpoint(stage, lineage_id=lin, checkpoint_sha256=info['checkpoint_sha256'], device=str(ctx.device))
        model = loaded['model'] if 'model' in loaded else loaded['policy'].model
        parity = _y1_ref_parity(model, lin, info['checkpoint_sha256'], evidence, cases, {c.case_id:c.public_prompt for c in cases}, str(ctx.device))
        full_parity = (parity.get('parity_status') == 'PASS' and parity.get('reference_cases') == len(cases) and parity.get('matched_reference_cases') == len(cases) and len(cases) > 0)
        return dict(status='COMPLETED' if full_parity else 'BLOCKED', verification='PARITY_VERIFIED' if full_parity else 'PARITY_INCOMPLETE_OR_FAILED', staging=info, parity=parity, qualification='NOT_CLAIMED')
    except Exception as exc:
        return dict(status='BLOCKED', verification='RESTORE_OR_PARITY_FAILED', error=f'{type(exc).__name__}: {exc}', staging=info)

def run(job, ctx):
    if job['stage'] == 'G':
        return _historical(job, ctx)
    seed, arm = job['seed'], job['arm']
    if arm in ('baseline', 'scout') and getattr(ctx, 'split', 'dev') != 'dev':
        return dict(status='BLOCKED', verification='BASELINE_SCOUT_DEV_ONLY')
    model = Y4TypedCompositionModel('categorical', 128, seed=seed)
    if ctx.full is None or not Path(ctx.full).is_file():
        return dict(status='BLOCKED', verification='Y4_FULL_MISSING', qualification='NOT_TESTED')
    try:
        with ZipFile(ctx.full) as archive:
            digest = load_model(archive, f'C-categorical_endpoint_only-s{seed}', model)
            reference_model = None
            if arm == 'baseline':
                reference_model = Y4TypedCompositionModel('categorical', 128, seed=seed)
                reference_digest = load_model(archive, f'C-categorical_true_step-s{seed}', reference_model)
    except Exception as exc:
        return dict(status='BLOCKED', verification='Y4_CHECKPOINT_INVALID', error=f'{type(exc).__name__}: {exc}')
    model.to(ctx.device)
    if arm == 'baseline':
        data = {str(d): measure(model, panel(32 if ctx.canary else 256, 5000+d, d), ctx.device) for d in (3,4)}
        qualified = all(v['correct'] / v['total'] >= .90 for v in data.values())
        reference_model.to(ctx.device)
        reference = {str(d): measure(reference_model, panel(32 if ctx.canary else 256, 5000+d, d), ctx.device) for d in (3, 4)}
        return dict(status='COMPLETED', verification='CHECKPOINT_AND_PANEL_MEASURED', qualification='QUALIFIED' if qualified else 'UNQUALIFIED', checkpoint_sha256=digest, parent_job=f'C-categorical_endpoint_only-s{seed}', panels=data, true_step_reference={'job_id':f'C-categorical_true_step-s{seed}', 'checkpoint_sha256':reference_digest, 'panels':reference})
    if arm == 'scout':
        baseline = ctx.previous(f'C-baseline-s{seed}')
        if not baseline or baseline.get('status') != 'COMPLETED':
            return dict(status='BLOCKED', verification='BASELINE_MISSING')
        if baseline.get('qualification') != 'QUALIFIED':
            return dict(status='GATED', verification='BASELINE_UNQUALIFIED')
        if baseline.get('checkpoint_sha256') != digest:
            return dict(status='BLOCKED', verification='BASELINE_PARENT_MISMATCH')
        data = {str(d): measure(model, panel(32 if ctx.canary else 256, 6000+d, d), ctx.device) for d in (5,6)}
        n = sum(v['total'] for v in data.values()); correct = sum(v['correct'] for v in data.values())
        return dict(status='COMPLETED', verification='CHECKPOINT_AND_PANEL_MEASURED', checkpoint_sha256=digest, panels=data, headroom=(n-correct)/n, has_headroom=(n-correct)/n >= .10)
    prior = ctx.previous(f'C-scout-s{seed}')
    if not prior or prior.get('status') != 'COMPLETED':
        return dict(status='BLOCKED', verification='SCOUT_MISSING')
    if prior.get('checkpoint_sha256') != digest:
        return dict(status='BLOCKED', verification='SCOUT_PARENT_MISMATCH')
    if not prior.get('has_headroom'):
        return dict(status='GATED', verification='INSUFFICIENT_HEADROOM', scout_headroom=prior.get('headroom'))
    train = pairs(64 if ctx.canary else 1024, 7000+seed, depth=2)
    if ctx.split == 'confirm':
        ctx.load_parent(job['id'], model)
        stats = {'restored': True}
    else:
        rng = random.Random(8000+seed)
        count = 1 if ctx.canary else job['updates']
        batches = [[rng.randrange(len(train)) for _ in range(16)] for _ in range(count)]
        stats = ctx.train(model, train, batches, lambda m, b: intervention_loss(m, b, arm), 1e-4)
    eval_seed = 9000 if ctx.split == 'dev' else 10000
    data = {str(d): measure(model, panel(32 if ctx.canary else 256, eval_seed+d, d), ctx.device) for d in (5,6)}
    transfer_data = {str(d): measure(model, panel(32 if ctx.canary else 256, 12000+d, d), ctx.device) for d in (5,6)}
    swaps = {str(d): measure_interventions(model, pairs(16 if ctx.canary else 256, eval_seed+100+d, depth=d), ctx.device) for d in (2, 3, 4)}
    return dict(status='COMPLETED', verification='TRAINED_AND_MEASURED', qualification='PENDING_MATCHED_COMPARISON', initial_checkpoint_sha256=digest, parent_job=f'C-categorical_endpoint_only-s{seed}', estimand='additional corrected donor objective versus endpoint-only continuation; sham uses alternate valid donor', train=stats, panels=data, transfer_panels=transfer_data, interventions=swaps)
