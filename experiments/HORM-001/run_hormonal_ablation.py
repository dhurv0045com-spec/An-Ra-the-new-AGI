"""Bounded HORM-001 experiment. See PLAN.md for the pre-run protocol."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from v5_contracts.model_spec import V5A_250M
from v5_contracts.model_spec_hormonal import V5A_250M_HORMONAL_V1
from v5_identity.hormonal_model import initialize_hormonal
from v5_identity.hormonal_state import HALState, VerifiedOutcome
from v5_model.core import initialize, packed_layout
from v5_objectives.causal_lm import causal_lm_loss

MOD, VOCAB = 97, 106
BOS, PAD, SEP, EOS, DIGIT_OFFSET = 1, 0, 2, 100, 3
TRAIN_UPDATES, BATCH, EVAL_EVERY, EVAL_PAIRS = 1500, 16, 50, 512
LR = 3e-3
SPIKE_RATIO, SPIKE_WINDOW, SPIKE_COOLDOWN = 2.0, 25, 50
ARMS = ('HAL_OFF', 'HAL_ON')
SEEDS = (424242, 424243)
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / 'experiments/HORM-001/PLAN.md',
             ROOT / 'v5_contracts/model_spec_hormonal.py']
    for folder in ('v5_identity', 'v5_model', 'v5_objectives'):
        paths.extend(sorted((ROOT / folder).glob('*.py')))
    paths.extend(sorted((ROOT / 'tests').glob('test_hormonal*.py')))
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def protected_hashes():
    paths = [ROOT / 'v5_contracts/model_spec.py', *sorted((ROOT / 'v5_model').glob('*.py'))]
    # Use tracked paths only; do not traverse checkpoints or credential directories.
    import subprocess
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    paths.extend(ROOT / p for p in tracked if p.endswith('launch_readiness.json') or p.startswith('blueprint/'))
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def tensor_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode_pair(a, b, with_answer):
    row = [BOS, DIGIT_OFFSET + a, SEP, DIGIT_OFFSET + b, SEP]
    return row + [DIGIT_OFFSET + (a + b) % MOD, EOS] if with_answer else row


def make_batch(pairs, with_answer, torch_module=torch):
    tokens = torch_module.tensor([encode_pair(a, b, with_answer) for a, b in pairs], dtype=torch_module.int64)
    segments = torch_module.zeros_like(tokens)
    positions, mask = packed_layout(segments, torch_module=torch_module)
    return tokens, segments, positions, mask


def build_datasets(seed):
    # Dataset seed is fixed independently of initialization seed.
    generator = torch.Generator().manual_seed(SEEDS[0])
    pairs = [(a, b) for a in range(MOD) for b in range(MOD)]
    shuffled = [pairs[i] for i in torch.randperm(len(pairs), generator=generator).tolist()]
    probe, train = shuffled[:EVAL_PAIRS], shuffled[EVAL_PAIRS:]
    assert not set(probe) & set(train)
    return probe, train


def score_predictions(predictions, pairs):
    if len(predictions) != len(pairs) or not pairs:
        raise ValueError('prediction coverage mismatch or empty probe')
    return sum(list(row) == [DIGIT_OFFSET + (a + b) % MOD, EOS]
               for row, (a, b) in zip(predictions, pairs)) / len(pairs)


def predict(model, probe, torch_module=torch):
    was_training = model.training
    model.eval()
    predictions = []
    try:
        with torch_module.no_grad():
            for start in range(0, len(probe), 64):
                tokens, _, _, _ = make_batch(probe[start:start + 64], False, torch_module)
                generated = []
                for _ in range(2):
                    positions, mask = packed_layout(torch_module.zeros_like(tokens), torch_module=torch_module)
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


def evaluate(model, probe, torch_module=torch):
    return score_predictions(predict(model, probe, torch_module), probe)


def decide(rows):
    expected = {(s, a) for s in SEEDS for a in ARMS}
    try:
        if len(rows) != 4 or {(r['seed'], r['arm']) for r in rows} != expected:
            return 'ENGINEERING_FAILURE'
        if any(r['nan_count'] or not math.isfinite(r['final_accuracy']) or
               not 0 <= r['final_accuracy'] <= 1 or
               r.get('purpose', 'registered') != 'registered' for r in rows):
            return 'ENGINEERING_FAILURE'
        scores = {(r['seed'], r['arm']): r['final_accuracy'] for r in rows}
        ds = [scores[s, 'HAL_ON'] - scores[s, 'HAL_OFF'] for s in SEEDS]
        if all(d >= .10 for d in ds):
            return 'HORMONAL_EFFECT_SUPPORTED_AT_DEV_SCALE'
        if all(abs(d) <= .05 for d in ds):
            return 'NO_MEASURABLE_EFFECT_AT_THIS_SCALE'
        if all(d <= -.10 for d in ds):
            return 'HORMONAL_EFFECT_HARMFUL_AT_DEV_SCALE'
        if max(ds) >= .10 and min(ds) <= -.10:
            return 'SEED_SIGN_CONFLICT'
        if any(abs(d) >= .10 for d in ds):
            return 'MIXED_OR_SEED_SENSITIVE'
        return 'INCONCLUSIVE_SMALL_EFFECT'
    except (KeyError, TypeError, ValueError):
        return 'ENGINEERING_FAILURE'


def run_lane(seed, arm, results_dir, torch_module=torch, *, updates=TRAIN_UPDATES,
             eval_every=EVAL_EVERY, probe_limit=None):
    if seed not in SEEDS or arm not in ARMS or updates < 1 or eval_every < 1:
        raise ValueError('invalid lane')
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    lane_path = results_dir / f'lane_{arm}_{seed}.json'
    checkpoint_path = results_dir / f'lane_{arm}_{seed}.pt'
    if lane_path.exists() or checkpoint_path.exists():
        raise FileExistsError('refusing to overwrite prior evidence')
    torch_module.set_num_threads(2)
    torch_module.manual_seed(seed)
    torch_module.use_deterministic_algorithms(True)
    probe, train = build_datasets(seed)
    if probe_limit is not None:
        probe = probe[:probe_limit]
    tiny = dataclasses.replace(V5A_250M, vocabulary_size=VOCAB, width=64, layers=2,
        query_heads=2, kv_heads=1, head_dimension=32, ffn_width=128, context_length=16)
    spec = dataclasses.replace(V5A_250M_HORMONAL_V1, base=tiny) if arm == 'HAL_ON' else tiny
    def construct():
        return initialize_hormonal(spec, seed, torch_module=torch_module) if arm == 'HAL_ON' else initialize(spec, seed, torch_module=torch_module)
    model = construct()
    core = model.core if arm == 'HAL_ON' else model
    initial_hash = tensor_hash(core)
    hal = HALState() if arm == 'HAL_ON' else None
    if hal is not None:
        model.set_hormones(hal)
        x, _, pos, mask = make_batch(train[:BATCH], False)
        with torch_module.no_grad():
            assert torch_module.equal(core(x, pos, mask), model(x, pos, mask))
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    model.train()
    history, trace, evaluations = [], [], []
    last_spike = -SPIKE_COOLDOWN
    started = time.monotonic()
    initial_accuracy = evaluate(model, probe)
    for update in range(1, updates + 1):
        if time.monotonic() - started > 600:
            raise TimeoutError('10-minute lane budget exceeded')
        pairs = [train[((update - 1) * BATCH + i) % len(train)] for i in range(BATCH)]
        tokens, segments, positions, mask = make_batch(pairs, True)
        if hal is not None:
            model.set_hormones(hal)
        optimizer.zero_grad(set_to_none=True)
        loss, target_count = causal_lm_loss(model(tokens, positions, mask), tokens, segments,
            bos_id=BOS, pad_id=PAD, torch_module=torch_module)
        if not torch_module.isfinite(loss):
            raise FloatingPointError('nonfinite train loss')
        loss.backward()
        grad_norm = torch_module.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch_module.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite parameter after optimizer')
        value = loss.detach().item()
        median = statistics.median(history[-SPIKE_WINDOW:]) if len(history) >= SPIKE_WINDOW else None
        spike = median is not None and value > SPIKE_RATIO * median and update - last_spike >= SPIKE_COOLDOWN
        if hal is not None:
            hal = hal.decay()
            if spike:
                hal = hal.appraise(VerifiedOutcome('surprise', f'train-loss/update-{update}'))
                last_spike = update
            model.set_hormones(hal)
        trace.append({'update': update, 'loss': value, 'grad_norm': grad_norm.item(),
            'eligible_targets': target_count, 'prior_median': median,
            'appraisal': bool(spike and hal is not None),
            'state': hal.as_receipt() if hal is not None else None,
            'log_temperature': model.log_temperature().detach().tolist() if hal is not None else [0.0, 0.0]})
        history.append(value)
        if update % eval_every == 0 or update == updates:
            accuracy = evaluate(model, probe)
            evaluations.append({'update': update, 'accuracy': accuracy})
            print(f'{arm} seed={seed} update={update} exact_stop={accuracy:.4f}', flush=True)
    predictions = predict(model, probe)
    torch_module.save({'spec_sha256': spec.sha256(), 'model': model.state_dict()}, checkpoint_path)
    restored = construct()
    payload = torch_module.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert payload['spec_sha256'] == spec.sha256()
    restored.load_state_dict(payload['model'])
    x, _, pos, mask = make_batch(probe[:4], False)
    with torch_module.no_grad():
        reload_exact = torch_module.equal(model(x, pos, mask), restored(x, pos, mask))
    assert reload_exact and predictions == predict(restored, probe)
    window = [e for e in evaluations if 450 <= e['update'] <= 1500]
    auc = sum((a['accuracy'] + b['accuracy']) / 2 * (b['update'] - a['update'])
              for a, b in zip(window, window[1:])) / 1050 if len(window) > 1 else None
    result = {'seed': seed, 'arm': arm, 'updates': updates, 'purpose': 'registered' if
        (updates, eval_every, probe_limit) == (TRAIN_UPDATES, EVAL_EVERY, None) else 'engineering_smoke_only',
        'final_accuracy': score_predictions(predictions, probe), 'initial_accuracy': initial_accuracy,
        'train_final_accuracy': evaluate(model, train[:512]), 'formation_auc_450_1500': auc,
        'first_acquisition_update': next((e['update'] for e in evaluations if e['accuracy'] >= .10), None),
        'spike_events': sum(r['appraisal'] for r in trace), 'nan_count': 0,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'initial_core_sha256': initial_hash,
        'final_core_sha256': tensor_hash(core), 'data_sha256': digest({'probe': probe, 'train': train}),
        'spec_sha256': spec.sha256(), 'spec': spec.canonical(), 'checkpoint_sha256': file_hash(checkpoint_path),
        'checkpoint_reload_exact': reload_exact, 'source_sha256': source_hashes(),
        'elapsed_seconds': time.monotonic() - started, 'evaluations': evaluations, 'training_trace': trace,
        'probe_predictions': [{'a': a, 'b': b, 'generated': p} for (a, b), p in zip(probe, predictions)]}
    lane_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'experiments/HORM-001/results'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('use an empty output directory')
    protected, sources = protected_hashes(), source_hashes()
    receipt = {'protected_before': protected, 'source_before': sources, 'torch': torch.__version__,
               'python': sys.version, 'cpu_threads': 2, 'device': 'cpu'}
    (out / 'PRERUN.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    rows = []
    try:
        for seed in SEEDS:
            for arm in ARMS:
                rows.append(run_lane(seed, arm, out))
        for seed in SEEDS:
            pair = [r for r in rows if r['seed'] == seed]
            assert pair[0]['initial_core_sha256'] == pair[1]['initial_core_sha256']
            assert pair[0]['data_sha256'] == pair[1]['data_sha256']
        assert protected == protected_hashes() and sources == source_hashes()
        verdict = decide(rows)
        error = None
    except Exception as exc:
        verdict, error = 'ENGINEERING_FAILURE', repr(exc)
    result = {'verdict': verdict, 'error': error, 'protected_unchanged': protected == protected_hashes(),
              'sources_unchanged': sources == source_hashes(), 'lanes': [
                  {k: r[k] for k in ('seed', 'arm', 'final_accuracy', 'train_final_accuracy',
                      'formation_auc_450_1500', 'spike_events', 'elapsed_seconds')} for r in rows],
              'artifact_sha256': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
