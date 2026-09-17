"""Evaluator and split integrity for the bounded HORM-001 experiment."""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path('experiments/HORM-001/run_hormonal_ablation.py')
spec = importlib.util.spec_from_file_location('horm_experiment', PATH)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_probe_is_disjoint_and_fixed_across_model_seeds():
    probe, train = run.build_datasets(424242)
    probe2, train2 = run.build_datasets(424243)
    assert not set(probe) & set(train)
    assert len(set(probe) | set(train)) == 97 ** 2
    assert probe == probe2 and train == train2


def test_exact_score_requires_correct_answer_and_stop():
    gold = [(2, 3), (96, 2)]
    answer = [[8, run.EOS], [4, run.EOS]]
    assert run.score_predictions(answer, gold) == 1.0
    assert run.score_predictions([[8, 0], [4, run.EOS]], gold) == 0.5
    assert run.score_predictions([[9, run.EOS], [5, run.EOS]], gold) == 0.0
    with pytest.raises(ValueError):
        run.score_predictions(answer[:1], gold)


def test_evaluator_generates_without_gold_and_restores_mode():
    class Scripted(torch.nn.Module):
        def forward(self, tokens, positions, mask):
            assert tokens.shape[1] in (5, 6)
            expected = ((tokens[:, 1] - 3 + tokens[:, 3] - 3) % 97) + 3
            if tokens.shape[1] == 6:
                assert torch.equal(tokens[:, -1], expected)
                expected = torch.full_like(expected, run.EOS)
            logits = torch.full((*tokens.shape, run.VOCAB), -10.0)
            logits[torch.arange(len(tokens)), -1, expected] = 10.0
            return logits
    model = Scripted().eval()
    assert run.evaluate(model, [(2, 3), (96, 2)]) == 1.0
    assert not model.training


def test_decision_exhaustive_and_fail_closed():
    def rows(deltas):
        return [{'seed': s, 'arm': a, 'final_accuracy': 0.5 + (d if a == 'HAL_ON' else 0), 'nan_count': 0}
                for s, d in zip(run.SEEDS, deltas) for a in run.ARMS]
    assert run.decide(rows([0.2, 0.2])) == 'HORMONAL_EFFECT_SUPPORTED_AT_DEV_SCALE'
    assert run.decide(rows([-0.2, -0.2])) == 'HORMONAL_EFFECT_HARMFUL_AT_DEV_SCALE'
    assert run.decide(rows([0.0, 0.02])) == 'NO_MEASURABLE_EFFECT_AT_THIS_SCALE'
    assert run.decide(rows([0.2, -0.2])) == 'SEED_SIGN_CONFLICT'
    assert run.decide(rows([0.07, 0.07])) == 'INCONCLUSIVE_SMALL_EFFECT'
    assert run.decide(rows([float('nan'), 0])) == 'ENGINEERING_FAILURE'
    assert run.decide(rows([0, 0])[:-1]) == 'ENGINEERING_FAILURE'
    duplicate = rows([0, 0]); duplicate[-1] = duplicate[0]
    assert run.decide(duplicate) == 'ENGINEERING_FAILURE'


def test_two_update_pipeline_receipts_and_reload(tmp_path):
    rows = [run.run_lane(424242, a, tmp_path, updates=2, eval_every=1, probe_limit=4) for a in run.ARMS]
    assert rows[0]['initial_core_sha256'] == rows[1]['initial_core_sha256']
    assert rows[0]['data_sha256'] == rows[1]['data_sha256']
    assert rows[1]['parameter_count'] - rows[0]['parameter_count'] == 14
    assert all(r['checkpoint_reload_exact'] for r in rows)
    assert all(r['updates'] == 2 and r['nan_count'] == 0 for r in rows)
    assert all(r['purpose'] == 'engineering_smoke_only' for r in rows)
    assert run.decide(rows) == 'ENGINEERING_FAILURE'
