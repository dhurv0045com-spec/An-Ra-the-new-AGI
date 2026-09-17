"""HORM-002 protocol tests: competence gate, control isolation, feedback wiring."""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path('experiments/HORM-002/run_hormonal_feedback.py')
spec = importlib.util.spec_from_file_location('horm_feedback', PATH)
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)


def _rows(d_feedback, d_mech=(0.0, 0.0), off=(0.6, 0.6), purpose='registered',
          const_events=0, on_events=1, const_frozen=True):
    rows = []
    for i, seed in enumerate(fb.SEEDS):
        for arm, score, events in (
                ('HAL_OFF', off[i], 0),
                ('HAL_CONST', off[i] + d_mech[i], const_events),
                ('HAL_ON', off[i] + d_mech[i] + d_feedback[i], on_events)):
            rows.append({'seed': seed, 'arm': arm, 'final_accuracy': score,
                         'nan_count': 0, 'purpose': purpose, 'checkpoint_reload_exact': True,
                         'appraisal_events': {'success': events, 'surprise': 0},
                         'const_state_frozen': const_frozen if arm == 'HAL_CONST' else None})
    return rows


def test_decide_verdict_table():
    assert fb.decide(_rows([0.2, 0.2])) == 'FEEDBACK_EFFECT_SUPPORTED_AT_DEV_SCALE'
    assert fb.decide(_rows([0.0, 0.02])) == 'NO_FEEDBACK_EFFECT_AT_THIS_SCALE'
    assert fb.decide(_rows([-0.2, -0.2])) == 'FEEDBACK_EFFECT_HARMFUL_AT_DEV_SCALE'
    assert fb.decide(_rows([0.2, -0.2])) == 'FEEDBACK_SIGN_CONFLICT'
    assert fb.decide(_rows([0.2, 0.0])) == 'FEEDBACK_MIXED_OR_SEED_SENSITIVE'
    assert fb.decide(_rows([0.07, 0.07])) == 'FEEDBACK_INCONCLUSIVE_SMALL_EFFECT'
    assert fb.decide(_rows([0.2, 0.2], off=(0.4, 0.6))) == 'COMPETENCE_GATE_FAILED'
    assert fb.decide(_rows([0.2, 0.2], off=(0.4, 0.6), on_events=0)) == 'COMPETENCE_GATE_FAILED'
    assert fb.decide(_rows([0.2, 0.2], purpose='engineering_smoke_only')) == 'ENGINEERING_FAILURE'
    assert fb.decide(_rows([0.2, 0.2], const_events=1)) == 'ENGINEERING_FAILURE'
    assert fb.decide(_rows([0.2, 0.2], const_frozen=False)) == 'ENGINEERING_FAILURE'
    assert fb.decide(_rows([0.2, 0.2], on_events=0)) == 'ENGINEERING_FAILURE'
    assert fb.decide(_rows([float('nan'), 0.2])) == 'ENGINEERING_FAILURE'
    assert fb.decide(_rows([0.2, 0.2])[:-1]) == 'ENGINEERING_FAILURE'
    duplicate = _rows([0.2, 0.2])
    duplicate[-1] = duplicate[0]
    assert fb.decide(duplicate) == 'ENGINEERING_FAILURE'


def test_calibration_ladder_frozen():
    assert fb.CALIBRATION_LADDER == ((97, 12000, 250), (23, 6000, 250), (23, 12000, 250))


def test_probe_disjoint_and_train_majority_for_both_moduli():
    for mod in (97, 23):
        probe, train = fb.build_datasets(mod)
        assert len(probe) == min(512, mod * mod // 5)
        assert len(train) >= 0.8 * mod * mod
        assert not set(probe) & set(train)
        assert set(probe) | set(train) == {(a, b) for a in range(mod) for b in range(mod)}


def test_const_arm_state_frozen_and_on_arm_events_fire(tmp_path):
    for arm, expect_events in (('HAL_CONST', 0), ('HAL_ON', 2)):
        lane = fb.run_lane(424242, arm, tmp_path, mod=23, updates=2, eval_every=1,
                           probe_limit=4, success_threshold=0.0)
        assert lane['purpose'] == 'engineering_smoke_only'
        assert sum(lane['appraisal_events'].values()) == expect_events
        states = [tuple(row['state'][name] for name in fb.HORMONE_NAMES)
                  for row in lane['training_trace']]
        if arm == 'HAL_CONST':
            assert len(set(states)) == 1
        else:
            assert len(set(states)) > 1


def test_calibration_paths_and_stop_rule(monkeypatch, tmp_path):
    calls = []
    def fake(seed, arm, path, *args, **kwargs):
        path = Path(path)
        assert path not in calls
        calls.append(path)
        assert kwargs['purpose'] == 'calibration'
        return {'final_accuracy': 0.6 if len(calls) == 2 else 0.1,
                'train_final_accuracy': 0.8}
    monkeypatch.setattr(fb, 'run_lane', fake)
    result = fb.run_calibration(tmp_path)
    assert len(calls) == 2
    assert result['frozen_config'] == {'mod': 23, 'updates': 6000, 'eval_every': 250}


def test_no_gradient_accumulation(monkeypatch, tmp_path):
    calls = []
    original = torch.optim.AdamW.step
    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(1)
        # Poison grads after step: the next update must clear them.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.fill_(float('nan'))
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    fb.run_lane(424242, 'HAL_OFF', tmp_path, updates=2, eval_every=2, probe_limit=4)
    assert len(calls) == 2


def test_smoke_receipts_and_reload(tmp_path):
    rows = [fb.run_lane(424242, a, tmp_path, mod=23, updates=2, eval_every=1,
                        probe_limit=4, success_threshold=0.0) for a in fb.ARMS]
    assert rows[0]['initial_core_sha256'] == rows[1]['initial_core_sha256'] == rows[2]['initial_core_sha256']
    assert rows[0]['data_sha256'] == rows[1]['data_sha256'] == rows[2]['data_sha256']
    assert rows[1]['parameter_count'] == rows[2]['parameter_count']
    assert rows[1]['parameter_count'] - rows[0]['parameter_count'] == 14
    assert all(r['checkpoint_reload_exact'] for r in rows)
    assert fb.decide(rows) == 'ENGINEERING_FAILURE'
