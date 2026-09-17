"""HORM-002: competence-gated, feedback-isolating hormonal comparison.

See PLAN.md (preregistered). Phase 0 calibration (HAL_OFF ladder, labeled
`calibration`), then three causal arms (HAL_OFF / HAL_CONST / HAL_ON) x two
seeds. CPU-only, 2 threads, sequential, HORM-001 receipt discipline. The
measured feedback signal is training-batch answer accuracy (>= 0.875 ->
success appraisal); held-out/evaluation data never feeds appraisal.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from v5_contracts.model_spec import V5A_250M
from v5_contracts.model_spec_hormonal import V5A_250M_HORMONAL_V1
from v5_identity.hormonal_model import initialize_hormonal
from v5_identity.hormonal_state import HORMONE_NAMES, HALState, VerifiedOutcome
from v5_model.core import initialize, packed_layout
from v5_objectives.causal_lm import causal_lm_loss

MOD_DEFAULT = 97
VOCAB = 106  # PAD=0, BOS=1, SEP=2, values v -> v+3 (v < 97), EOS=100
BOS, PAD, SEP, EOS, DIGIT_OFFSET = 1, 0, 2, 100, 3
BATCH = 16
EVAL_PAIRS = 512
LR = 3e-3
SPIKE_RATIO, SPIKE_WINDOW, SPIKE_COOLDOWN = 2.0, 25, 50
SUCCESS_THRESHOLD = 0.875
ARMS = ('HAL_OFF', 'HAL_CONST', 'HAL_ON')
SEEDS = (424242, 424243)
CALIBRATION_LADDER = ((97, 12000, 250), (23, 6000, 250), (23, 12000, 250))
COMPETENCE_PASS = 0.50
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / 'experiments/HORM-002/PLAN.md',
             ROOT / 'v5_contracts/model_spec_hormonal.py']
    for folder in ('v5_identity', 'v5_model', 'v5_objectives'):
        paths.extend(sorted((ROOT / folder).glob('*.py')))
    paths.extend(sorted((ROOT / 'tests').glob('test_hormonal*.py')))
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def protected_hashes():
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    paths = [ROOT / p for p in tracked
             if p.startswith('blueprint/') or p == 'artifacts/v5/launch_readiness.json']
    paths += [ROOT / 'v5_contracts/model_spec.py', *sorted((ROOT / 'v5_model').glob('*.py'))]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def tensor_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode_pair(a, b, mod, with_answer):
    row = [BOS, DIGIT_OFFSET + a, SEP, DIGIT_OFFSET + b, SEP]
    return row + [DIGIT_OFFSET + (a + b) % mod, EOS] if with_answer else row


def make_batch(pairs, mod, with_answer, torch_module=torch):
    tokens = torch_module.tensor([encode_pair(a, b, mod, with_answer) for a, b in pairs],
                                 dtype=torch_module.int64)
    positions, mask = packed_layout(torch_module.zeros_like(tokens), torch_module=torch_module)
    return tokens, positions, mask


def build_datasets(mod):
    generator = torch.Generator().manual_seed(SEEDS[0])
    pairs = [(a, b) for a in range(mod) for b in range(mod)]
    shuffled = [pairs[i] for i in torch.randperm(len(pairs), generator=generator).tolist()]
    probe_count = min(EVAL_PAIRS, mod * mod // 5)
    probe, train = shuffled[:probe_count], shuffled[probe_count:]
    assert not set(probe) & set(train)
    return probe, train


def score_predictions(predictions, pairs, mod):
    if len(predictions) != len(pairs) or not pairs:
        raise ValueError('prediction coverage mismatch or empty probe')
    return sum(list(row) == [DIGIT_OFFSET + (a + b) % mod, EOS]
               for row, (a, b) in zip(predictions, pairs)) / len(pairs)


def predict(model, probe, mod, torch_module=torch):
    was_training = model.training
    model.eval()
    predictions = []
    try:
        with torch_module.no_grad():
            for start in range(0, len(probe), 64):
                tokens, _, _ = make_batch(probe[start:start + 64], mod, False, torch_module)
                generated = []
                for _ in range(2):
                    positions, mask = packed_layout(torch_module.zeros_like(tokens),
                                                    torch_module=torch_module)
                    logits = model(tokens, positions, mask)
                    if not torch_module.isfinite(logits).all():
                        raise FloatingPointError('nonfinite evaluation logits')
                    next_token = logits[:, -1].argmax(-1)
                    generated.append(next_token)
                    tokens = torch_module.cat([tokens, next_token[:, None]], dim=1)
                predictions.extend(torch_module.stack(generated, dim=1).tolist())
    finally:
        model.train(was_training)
    return predictions


def evaluate(model, probe, mod, torch_module=torch):
    return score_predictions(predict(model, probe, mod, torch_module), probe, mod)


def const_frozen_ok(state_receipts):
    return len({json.dumps(s, sort_keys=True) for s in state_receipts}) == 1


def run_lane(seed, arm, results_dir, torch_module=torch, *, mod=MOD_DEFAULT,
             updates=12000, eval_every=250, probe_limit=None,
             success_threshold=SUCCESS_THRESHOLD, purpose="engineering_smoke_only"):
    if seed not in SEEDS or arm not in ARMS or updates < 1 or eval_every < 1:
        raise ValueError('invalid lane')
    if purpose not in ('engineering_smoke_only', 'calibration', 'registered'):
        raise ValueError('invalid purpose')
    if purpose != 'engineering_smoke_only':
        if ((mod, updates, eval_every) not in CALIBRATION_LADDER
                or probe_limit is not None or success_threshold != SUCCESS_THRESHOLD):
            raise ValueError('non-protocol measured lane')
        if purpose == 'calibration' and (seed != SEEDS[0] or arm != 'HAL_OFF'):
            raise ValueError('calibration must use the first seed and OFF arm')
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    lane_path = results_dir / f'lane_{arm}_{seed}.json'
    checkpoint_path = results_dir / f'lane_{arm}_{seed}.pt'
    if lane_path.exists() or checkpoint_path.exists():
        raise FileExistsError('refusing to overwrite prior evidence')
    torch_module.set_num_threads(2)
    torch_module.manual_seed(seed)
    torch_module.use_deterministic_algorithms(True)
    probe, train = build_datasets(mod)
    if probe_limit is not None:
        probe = probe[:probe_limit]
    tiny = dataclasses.replace(V5A_250M, vocabulary_size=VOCAB, width=64, layers=2,
                               query_heads=2, kv_heads=1, head_dimension=32,
                               ffn_width=128, context_length=16)
    spec = dataclasses.replace(V5A_250M_HORMONAL_V1, base=tiny) if arm != 'HAL_OFF' else tiny

    def construct():
        return (initialize_hormonal(spec, seed, torch_module=torch_module)
                if arm != 'HAL_OFF' else initialize(tiny, seed, torch_module=torch_module))

    model = construct()
    core = model.core if arm != 'HAL_OFF' else model
    initial_hash = tensor_hash(core)
    hal = None if arm == 'HAL_OFF' else HALState()
    if hal is not None:
        model.set_hormones(hal)
        x, pos, mask = make_batch(train[:BATCH], mod, False)
        with torch_module.no_grad():
            assert torch_module.equal(core(x, pos, mask), model(x, pos, mask))
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=LR, betas=(.9, .999),
                                         eps=1e-8, weight_decay=.01)
    model.train()
    trace, evaluations = [], []
    events = {'success': 0, 'surprise': 0}
    history, last_spike = [], -SPIKE_COOLDOWN
    started = time.monotonic()
    initial_accuracy = evaluate(model, probe, mod)
    for update in range(1, updates + 1):
        if time.monotonic() - started > 600:
            raise TimeoutError('10-minute lane budget exceeded')
        pairs = [train[((update - 1) * BATCH + i) % len(train)] for i in range(BATCH)]
        tokens, positions, mask = make_batch(pairs, mod, True)
        if hal is not None:
            model.set_hormones(hal)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, positions, mask)
        # Training-batch feedback from the loss forward; never from the probe.
        batch_answer_accuracy = (logits[:, -3].argmax(-1) == tokens[:, -2]).float().mean().item()
        loss, target_count = causal_lm_loss(logits, tokens, torch_module.ones_like(tokens),
                                            bos_id=BOS, pad_id=PAD, torch_module=torch_module)
        if not torch_module.isfinite(loss):
            raise FloatingPointError('nonfinite train loss')
        loss.backward()
        grad_norm = torch_module.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                          error_if_nonfinite=True)
        optimizer.step()
        if any(not torch_module.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite parameter after optimizer')
        value = loss.detach().item()
        median = (statistics.median(history[-SPIKE_WINDOW:])
                  if len(history) >= SPIKE_WINDOW else None)
        spike = (median is not None and value > SPIKE_RATIO * median
                 and update - last_spike >= SPIKE_COOLDOWN)
        fired = {'success': False, 'surprise': False}
        if hal is not None and arm == 'HAL_ON':
            hal = hal.decay()
            if batch_answer_accuracy >= success_threshold:
                hal = hal.appraise(VerifiedOutcome('success', f'train-batch/update-{update}'))
                events['success'] += 1
                fired['success'] = True
            if spike:
                hal = hal.appraise(VerifiedOutcome('surprise', f'train-loss/update-{update}'))
                events['surprise'] += 1
                fired['surprise'] = True
                last_spike = update
            model.set_hormones(hal)
        elif hal is not None:  # HAL_CONST: state frozen at baselines forever
            pass
        trace.append({'update': update, 'loss': value, 'grad_norm': grad_norm.item(),
                      'batch_answer_accuracy': batch_answer_accuracy,
                      'eligible_targets': target_count, 'prior_median': median,
                      'appraisal_success': fired['success'], 'appraisal_surprise': fired['surprise'],
                      'state': hal.as_receipt() if hal is not None else None,
                      'log_temperature': (model.log_temperature().detach().tolist()
                                          if hal is not None else [0.0, 0.0])})
        history.append(value)
        if update % eval_every == 0 or update == updates:
            accuracy = evaluate(model, probe, mod)
            evaluations.append({'update': update, 'accuracy': accuracy})
            print(f'{arm} seed={seed} update={update} exact_stop={accuracy:.4f}', flush=True)
    predictions = predict(model, probe, mod)
    torch_module.save({'spec_sha256': spec.sha256(), 'model': model.state_dict()}, checkpoint_path)
    restored = construct()
    payload = torch_module.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert payload['spec_sha256'] == spec.sha256()
    restored.load_state_dict(payload['model'])
    x, pos, mask = make_batch(probe[:4], mod, False)
    with torch_module.no_grad():
        reload_exact = torch_module.equal(model(x, pos, mask), restored(x, pos, mask))
    assert reload_exact and predictions == predict(restored, probe, mod)
    window = [e for e in evaluations if e['update'] >= 0.45 * updates]
    span = window[-1]['update'] - window[0]['update'] if len(window) > 1 else 0
    auc = (sum((a['accuracy'] + b['accuracy']) / 2 * (b['update'] - a['update'])
               for a, b in zip(window, window[1:])) / span) if span else None
    state_receipts = [r['state'] for r in trace if r['state'] is not None]
    result = {
        'seed': seed, 'arm': arm, 'mod': mod, 'updates': updates,
        'purpose': purpose,
        'final_accuracy': score_predictions(predictions, probe, mod),
        'initial_accuracy': initial_accuracy,
        'train_final_accuracy': evaluate(model, train[:512], mod),
        'formation_auc': auc,
        'first_acquisition_update': next((e['update'] for e in evaluations
                                          if e['accuracy'] >= 0.10), None),
        'appraisal_events': events,
        'const_state_frozen': (const_frozen_ok(state_receipts) if arm == 'HAL_CONST' else None),
        'nan_count': 0,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'initial_core_sha256': initial_hash, 'final_core_sha256': tensor_hash(core),
        'data_sha256': digest({'probe': probe, 'train': train, 'mod': mod}),
        'spec_sha256': spec.sha256(), 'spec': spec.canonical(),
        'checkpoint_sha256': file_hash(checkpoint_path),
        'checkpoint_reload_exact': reload_exact, 'source_sha256': source_hashes(),
        'elapsed_seconds': time.monotonic() - started,
        'evaluations': evaluations, 'training_trace': trace,
        'probe_predictions': [{'a': a, 'b': b, 'generated': p}
                              for (a, b), p in zip(probe, predictions)],
    }
    lane_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def run_calibration(results_dir, torch_module=torch):
    calibration = []
    for mod, updates, eval_every in CALIBRATION_LADDER:
        print(f'calibration mod={mod} updates={updates}', flush=True)
        lane_dir = Path(results_dir) / 'calibration' / f'mod{mod}_updates{updates}'
        lane = run_lane(SEEDS[0], 'HAL_OFF', lane_dir, torch_module, mod=mod,
                        updates=updates, eval_every=eval_every, purpose='calibration')
        calibration.append({'mod': mod, 'updates': updates,
                            'final': lane['final_accuracy'],
                            'train': lane['train_final_accuracy'],
                            'passed': (lane['final_accuracy'] >= COMPETENCE_PASS
                                       and lane['train_final_accuracy'] >= COMPETENCE_PASS)})
        if calibration[-1]['passed']:
            break
    receipt = {'ladder': [list(step) for step in CALIBRATION_LADDER],
               'competence_threshold': COMPETENCE_PASS, 'calibration': calibration,
               'frozen_config': (dict(zip(('mod', 'updates', 'eval_every'),
                                          CALIBRATION_LADDER[len(calibration) - 1]))
                                 if calibration[-1]['passed'] else None),
               'calibration_result_sha256': digest(calibration)}
    (results_dir / 'CALIBRATION.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt


def decide(rows):
    try:
        expected = {(s, a) for s in SEEDS for a in ARMS}
        if len(rows) != 6 or {(r['seed'], r['arm']) for r in rows} != expected:
            return 'ENGINEERING_FAILURE'
        for r in rows:
            if (r['nan_count'] or not math.isfinite(r['final_accuracy'])
                    or not 0 <= r['final_accuracy'] <= 1
                    or r.get('purpose') != 'registered'
                    or not r.get('checkpoint_reload_exact')):
                return 'ENGINEERING_FAILURE'
            if r['arm'] == 'HAL_CONST' and (sum(r['appraisal_events'].values())
                                            or r.get('const_state_frozen') is not True):
                return 'ENGINEERING_FAILURE'
            if r['arm'] == 'HAL_OFF' and sum(r['appraisal_events'].values()):
                return 'ENGINEERING_FAILURE'
        off = {r['seed']: r['final_accuracy'] for r in rows if r['arm'] == 'HAL_OFF'}
        const = {r['seed']: r['final_accuracy'] for r in rows if r['arm'] == 'HAL_CONST'}
        on = {r['seed']: r['final_accuracy'] for r in rows if r['arm'] == 'HAL_ON'}
        if any(v < 0.5 for v in off.values()):
            return 'COMPETENCE_GATE_FAILED'
        if any(sum(r['appraisal_events'].values()) == 0 for r in rows if r['arm'] == 'HAL_ON'):
            return 'ENGINEERING_FAILURE'
        d_feedback = [on[s] - const[s] for s in SEEDS]
        if all(d >= .10 for d in d_feedback):
            return 'FEEDBACK_EFFECT_SUPPORTED_AT_DEV_SCALE'
        if all(abs(d) <= .05 for d in d_feedback):
            return 'NO_FEEDBACK_EFFECT_AT_THIS_SCALE'
        if all(d <= -.10 for d in d_feedback):
            return 'FEEDBACK_EFFECT_HARMFUL_AT_DEV_SCALE'
        if max(d_feedback) >= .10 and min(d_feedback) <= -.10:
            return 'FEEDBACK_SIGN_CONFLICT'
        if any(abs(d) >= .10 for d in d_feedback):
            return 'FEEDBACK_MIXED_OR_SEED_SENSITIVE'
        return 'FEEDBACK_INCONCLUSIVE_SMALL_EFFECT'
    except (KeyError, TypeError, ValueError):
        return 'ENGINEERING_FAILURE'


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'experiments/HORM-002/results'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('use an empty output directory')
    protected, sources = protected_hashes(), source_hashes()
    (out / 'PRERUN.json').write_text(json.dumps(
        {'protected_before': protected, 'source_before': sources, 'torch': torch.__version__,
         'python': sys.version, 'cpu_threads': 2, 'device': 'cpu'}, indent=2), encoding='utf-8')
    rows, error = [], None
    try:
        receipt = run_calibration(out)
        if not receipt['frozen_config']:
            verdict = 'COMPETENCE_GATE_FAILED'
        else:
            config = receipt['frozen_config']
            for seed in SEEDS:
                for arm in ARMS:
                    rows.append(run_lane(seed, arm, out / 'causal', mod=config['mod'],
                                         updates=config['updates'],
                                         eval_every=config['eval_every'], purpose='registered'))
            verdict = decide(rows)
        if protected != protected_hashes() or sources != source_hashes():
            raise RuntimeError('protected or source hashes changed')
    except Exception as exc:  # noqa: BLE001 - fail closed into the receipt
        verdict, error = 'ENGINEERING_FAILURE', repr(exc)
        receipt = None
    result = {'verdict': verdict, 'error': error, 'calibration': receipt,
              'protected_unchanged': protected == protected_hashes(),
              'sources_unchanged': sources == source_hashes(),
              'lanes': [{k: r[k] for k in ('seed', 'arm', 'final_accuracy',
                                           'train_final_accuracy', 'formation_auc',
                                           'appraisal_events', 'elapsed_seconds')}
                        for r in rows],
              'artifact_sha256': {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob('*'))
                                  if p.is_file()}}
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False),
                                     encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
