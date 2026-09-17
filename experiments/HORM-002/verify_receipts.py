"""Read-only independent arithmetic and hash audit of saved HORM-002 evidence."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = load(OUT / 'RESULT.json')
    before = load(OUT / 'PRERUN.json')
    assert result['error'] is None, result['error']
    assert result['protected_unchanged'] and result['sources_unchanged']
    for group in ('protected_before', 'source_before'):
        for name, expected in before[group].items():
            assert sha(ROOT / name) == expected, name
    for name, expected in result['artifact_sha256'].items():
        assert sha(OUT / name) == expected, name
    rows = []
    for path in sorted(OUT.rglob('lane_*.json')):
        row = load(path)
        assert row['checkpoint_reload_exact'] and row['nan_count'] == 0
        assert sha(path.with_suffix('.pt')) == row['checkpoint_sha256']
        assert row['source_sha256'] == before['source_before']
        predictions = row['probe_predictions']
        pairs = {(p['a'], p['b']) for p in predictions}
        assert len(predictions) == len(pairs) == min(512, row['mod'] ** 2 // 5)
        correct = sum(p['generated'] == [3 + (p['a'] + p['b']) % row['mod'], 100]
                      for p in predictions)
        assert correct / len(predictions) == row['final_accuracy']
        trace = row['training_trace']
        assert len(trace) == row['updates']
        assert [t['update'] for t in trace] == list(range(1, row['updates'] + 1))
        assert all(math.isfinite(t['loss']) and math.isfinite(t['grad_norm']) for t in trace)
        assert all(t['eligible_targets'] == 96 for t in trace)
        for event in ('success', 'surprise'):
            assert sum(t['appraisal_' + event] for t in trace) == row['appraisal_events'][event]
        if row['arm'] == 'HAL_ON':
            assert all(t['appraisal_success'] == (t['batch_answer_accuracy'] >= .875) for t in trace)
        else:
            assert sum(row['appraisal_events'].values()) == 0
        if row['arm'] == 'HAL_CONST':
            assert row['const_state_frozen'] is True
            assert len({json.dumps(t['state'], sort_keys=True) for t in trace}) == 1
        if row['arm'] == 'HAL_OFF':
            assert all(t['state'] is None for t in trace)
        expected_purpose = 'calibration' if 'calibration' in path.parts else 'registered'
        assert row['purpose'] == expected_purpose
        rows.append(row)
        print(f"{path.relative_to(OUT)}: {correct}/{len(predictions)}; train={row['train_final_accuracy']:.6f}; events={row['appraisal_events']}")
    calibration = load(OUT / 'CALIBRATION.json')
    ladder = [(97, 12000, 250), (23, 6000, 250), (23, 12000, 250)]
    calrows = {(r['mod'], r['updates']): r for r in rows if r['purpose'] == 'calibration'}
    for i, entry in enumerate(calibration['calibration']):
        assert (entry['mod'], entry['updates']) == ladder[i][:2]
        lane = calrows[(entry['mod'], entry['updates'])]
        assert entry['final'] == lane['final_accuracy'] and entry['train'] == lane['train_final_accuracy']
        assert entry['passed'] == (entry['final'] >= .5 and entry['train'] >= .5)
        assert not entry['passed'] or i == len(calibration['calibration']) - 1
    causal = [r for r in rows if r['purpose'] == 'registered']
    if calibration['frozen_config'] is None:
        assert len(calrows) == 3 and not causal
        assert result['verdict'] == 'COMPETENCE_GATE_FAILED'
    else:
        assert len(causal) == 6
        for seed in (424242, 424243):
            group = [r for r in causal if r['seed'] == seed]
            assert {r['arm'] for r in group} == {'HAL_OFF', 'HAL_CONST', 'HAL_ON'}
            for field in ('data_sha256', 'initial_core_sha256'):
                assert len({r[field] for r in group}) == 1
            assert all((r['mod'], r['updates']) == (calibration['frozen_config']['mod'], calibration['frozen_config']['updates']) for r in group)
        by_seed = {s: {r['arm']: r for r in causal if r['seed'] == s} for s in (424242, 424243)}
        delta = [g['HAL_ON']['final_accuracy'] - g['HAL_CONST']['final_accuracy'] for g in by_seed.values()]
        if any(g['HAL_OFF']['final_accuracy'] < .5 for g in by_seed.values()):
            verdict = 'COMPETENCE_GATE_FAILED'
        elif any(sum(g['HAL_ON']['appraisal_events'].values()) == 0 for g in by_seed.values()):
            verdict = 'ENGINEERING_FAILURE'
        elif min(delta) >= .10:
            verdict = 'FEEDBACK_EFFECT_SUPPORTED_AT_DEV_SCALE'
        elif max(map(abs, delta)) <= .05:
            verdict = 'NO_FEEDBACK_EFFECT_AT_THIS_SCALE'
        elif max(delta) <= -.10:
            verdict = 'FEEDBACK_EFFECT_HARMFUL_AT_DEV_SCALE'
        elif min(delta) <= -.10 and max(delta) >= .10:
            verdict = 'FEEDBACK_SIGN_CONFLICT'
        elif max(map(abs, delta)) >= .10:
            verdict = 'FEEDBACK_MIXED_OR_SEED_SENSITIVE'
        else:
            verdict = 'FEEDBACK_INCONCLUSIVE_SMALL_EFFECT'
        assert result['verdict'] == verdict
    print(json.dumps({'verified_lanes': len(rows), 'protected_files': len(before['protected_before']),
                      'source_files': len(before['source_before']), 'artifact_hashes': len(result['artifact_sha256']),
                      'verdict': result['verdict'], 'status': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
