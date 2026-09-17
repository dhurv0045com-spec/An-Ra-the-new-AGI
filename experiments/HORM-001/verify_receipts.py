"""Independent read-only verification of HORM-001 saved evidence."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((OUT / 'RESULT.json').read_text())
    pre = json.loads((OUT / 'PRERUN.json').read_text())
    assert result['error'] is None
    for name, expected in result['artifact_sha256'].items():
        assert sha(OUT / name) == expected, name
    for group in ('protected_before', 'source_before'):
        for name, expected in pre[group].items():
            assert sha(ROOT / name) == expected, name
    rows = []
    for seed in (424242, 424243):
        pair = []
        for arm in ('HAL_OFF', 'HAL_ON'):
            lane = json.loads((OUT / f'lane_{arm}_{seed}.json').read_text())
            assert lane['purpose'] == 'registered' and lane['updates'] == 1500
            assert lane['nan_count'] == 0 and lane['checkpoint_reload_exact']
            assert len(lane['training_trace']) == 1500
            assert [e['update'] for e in lane['evaluations']] == list(range(50, 1501, 50))
            predictions = lane['probe_predictions']
            assert len(predictions) == 512
            assert len({(p['a'], p['b']) for p in predictions}) == 512
            correct = sum(p['generated'] == [(p['a'] + p['b']) % 97 + 3, 100] for p in predictions)
            assert correct / 512 == lane['final_accuracy'] == lane['evaluations'][-1]['accuracy']
            assert all(math.isfinite(t['loss']) and math.isfinite(t['grad_norm']) and t['eligible_targets'] == 96 for t in lane['training_trace'])
            assert all(all(abs(v) <= 1.5 for v in t['log_temperature']) for t in lane['training_trace'])
            assert lane['source_sha256'] == pre['source_before']
            assert sha(OUT / f'lane_{arm}_{seed}.pt') == lane['checkpoint_sha256']
            pair.append(lane)
            rows.append({'seed': seed, 'arm': arm, 'correct': correct, 'total': 512,
                         'final_accuracy': lane['final_accuracy'], 'train_accuracy': lane['train_final_accuracy'],
                         'parameters': lane['parameter_count'], 'events': lane['spike_events'],
                         'final_loss': lane['training_trace'][-1]['loss'],
                         'initial_loss': lane['training_trace'][0]['loss'],
                         'final_log_temperature': lane['training_trace'][-1]['log_temperature'],
                         'first_acquisition': lane['first_acquisition_update']})
        assert pair[0]['initial_core_sha256'] == pair[1]['initial_core_sha256']
        assert pair[0]['data_sha256'] == pair[1]['data_sha256']
        assert pair[0]['initial_accuracy'] == pair[1]['initial_accuracy']
        assert pair[1]['parameter_count'] - pair[0]['parameter_count'] == 14
    print(json.dumps({'verification': 'PASS', 'verdict': result['verdict'],
                      'protected_files': len(pre['protected_before']), 'lanes': rows}, indent=2))


if __name__ == '__main__':
    main()
