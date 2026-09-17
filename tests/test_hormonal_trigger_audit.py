"""Post-run engineering audit; injected losses are NOT scientific results."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('horm_trigger', Path('experiments/HORM-001/run_hormonal_ablation.py'))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_actual_runner_spike_updates_live_state(monkeypatch, tmp_path):
    original = run.causal_lm_loss
    calls = 0
    def injected_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        loss, count = original(*args, **kwargs)
        target = 3.0 if calls == 26 else 1.0
        return loss - loss.detach() + target, count
    monkeypatch.setattr(run, 'causal_lm_loss', injected_loss)
    row = run.run_lane(424242, 'HAL_ON', tmp_path, updates=27, eval_every=27, probe_limit=4)
    assert row['purpose'] == 'engineering_smoke_only'
    assert row['spike_events'] == 1
    before, spike, after = row['training_trace'][24:27]
    assert not before['appraisal'] and spike['appraisal'] and not after['appraisal']
    assert spike['state']['adrenaline'] == pytest.approx(0.5)
    assert after['state']['adrenaline'] == pytest.approx(0.41)
    assert after['state']['cortisol'] > spike['state']['cortisol']
    assert row['checkpoint_reload_exact']
