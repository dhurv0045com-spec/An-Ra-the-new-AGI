from pathlib import Path
import importlib

from v5_experiments import cyr_gpu011 as v11

ROOT = Path(__file__).resolve().parents[1]


def test_corrected_pairs_preserve_tens_band_and_are_disjoint():
    import importlib.util
    assert importlib.util.find_spec('v5_experiments.cyr_gpu012_closure') is not None, 'corrected probe module missing'
    core = importlib.import_module('v5_experiments.cyr_gpu012_closure')
    data = v11.load_ark002b_manifest(ROOT / 'docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json')
    battery = core.make_battery(data)
    excluded = {tuple(r['canonical_pair']) for k in ('train', 'dev_controller', 'dev_measurement', 'sealed_reserved') for r in data[k]}
    assert len(battery['COMMUTATION_MATCHED_BAND']) == 104
    for name in ('COMMUTATION_MATCHED_BAND', 'COMMUTATION_OOD_BAND'):
        rows = battery[name]
        assert rows and len(rows) % 2 == 0
        for left, right in zip(rows[::2], rows[1::2]):
            assert left['a'] // 10 == left['b'] // 10 == right['a'] // 10 == right['b'] // 10
            assert (left['a'], left['b']) == (right['b'], right['a'])
            assert left['pair_id'] == right['pair_id']
            assert tuple(sorted((left['a'], left['b']))) not in excluded
            assert left['answer'] == right['answer'] == str(left['a'] + left['b'])
    old = v11.make_reasoning_battery(data)
    for name in ('STANDARD', 'LOCALITY', 'CARRY', 'TRIPLE_ADD', 'THREE_DIGIT'):
        assert battery[name] == old[name]
    assert battery == core.make_battery(data)


def test_constant_wrong_predictions_do_not_prove_commutation():
    from v5_experiments.cyr_gpu012_closure import score_pairs
    rows = [dict(pair_id='p', role=role, generated='00', expected='24', stop='EOS')
            for role in ('base', 'reversed')]
    assert score_pairs(rows)['content_consistency'] == 1
    assert score_pairs(rows)['both_exact_with_eos'] == 0
    for row in rows:
        row['generated'] = '24'
    assert score_pairs(rows)['both_exact_with_eos'] == 1
    rows[0]['stop'] = 'MAX_TOKENS'
    assert score_pairs(rows)['both_exact_with_eos'] == 0
    import pytest
    with pytest.raises(ValueError):
        score_pairs(rows[:1])


def test_exhaustive_decisions():
    from v5_experiments.cyr_gpu012_closure import decide
    def record(controller, measurement, transient=None, updates=18000):
        return dict(updates=updates, row_presentations=updates * 64,
                    trace=[dict(dev_controller=dict(complete_exact_with_valid_stop=x)) for x in controller],
                    reasoning_battery_final=dict(STANDARD=dict(complete_exact_with_valid_stop=measurement)),
                    g90_confirm_update=transient)
    assert decide(record([.95]*3, .95)) == 'COMPACT_G90_FULL_EXPOSURE'
    assert decide(record([.7]*3, .7)) == 'NO_G90_AT_FULL_EXPOSURE'
    for c, m, t in [([.95]*3, .7, None), ([.7]*3, .95, None), ([.7]*3, .7, 5000)]:
        assert decide(record(c, m, t)) == 'AMBIGUOUS_PARTIAL_OR_TRANSIENT_G90'
    assert decide(record([1]*3, 1, updates=17999)) == 'INCOMPLETE_EXPOSURE'


def test_corrected_evaluator_scores_actual_pair_outputs():
    from anra_v5.cyr_gpu012_closure_run import corrected_battery
    from anra_v5 import cyr_gpu011_run as runner
    from unittest.mock import patch
    from v5_experiments.cyr_gpu012_closure import make_battery
    data = v11.load_ark002b_manifest(ROOT / 'docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json')
    battery = make_battery(data)
    def generated(_model, _tok, rows, **kwargs):
        return [dict(world_id=r['world_id'], pair_id=r['pair_id'], role=r['role'],
                     expected=r['answer'], generated=r['answer'], stop='EOS', exact=True) for r in rows]
    def original(*args, **kwargs):
        return dict(structural_flags={'COMMUTATION_INVARIANCE': True}, candidate_free_predictions={})
    with patch.object(runner, '_generate_texts', generated):
        result = corrected_battery(original, None, None, battery, torch=None, device=None,
                                   special={}, include_predictions=True)
    assert 'COMMUTATION_INVARIANCE' not in result['structural_flags']
    assert result['COMMUTATION_MATCHED_BAND']['paired']['both_exact_with_eos'] == 1
    assert result['candidate_free_predictions']['COMMUTATION_MATCHED_BAND']['count'] == 104
