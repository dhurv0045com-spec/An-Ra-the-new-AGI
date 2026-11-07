"""CPU-only worker contract tests; scripted raw outputs are test doubles."""
from copy import deepcopy
import json
import time

import pytest

from anra_v5.elendril.cognition import OperationCandidate
from anra_v5.elendril.codec import canonical
from anra_v5.elendril.v6_data import make_dataset, render_public
from anra_v5.elendril.v6_evaluation import ControlPolicy, action_tokens, evaluate
from tools.elendril_v6_worker import run_supplemental
from tools.verify_elendril_v6 import verify_supplements


class ScriptedRawTestPolicy:
    origin = 'model'
    model_identity = 'scripted-test-double-only'

    def __init__(self, surface=None, memory=False, forbidden=False):
        self.surface, self.memory, self.forbidden = surface, memory, forbidden
        self.prompts = []

    def for_case(self, case, *, memory_available=None):
        assert not hasattr(case, 'truth')
        return ScriptedRawTestPolicy(case.surface, memory_available, self.forbidden)

    def propose(self, state, legal, channels, positions):
        assert not self.forbidden, 'cached conditions performed another model call'
        self.prompts.append(render_public(state, channels, self.surface, bool(self.memory) and 'RETRIEVE' in legal))
        return OperationCandidate('ANSWER', 'A', .5, 2, 1, action_tokens('ANSWER', 'A'))


@pytest.fixture
def receipts():
    data = make_dataset('validation', worlds=28)
    base = evaluate(data, ScriptedRawTestPolicy())
    controls = {k: evaluate(data, ControlPolicy(k)) for k in ('fixed', 'always-query', 'no-query')}
    return data, base, controls


def test_condition_saving_resume_and_independent_verification(tmp_path, receipts):
    data, base, controls = receipts
    bundle = run_supplemental(data, ScriptedRawTestPolicy(), base, controls, tmp_path,
                              deadline=time.monotonic() + 120)
    assert all(r['status'] == 'COMPLETE' for r in bundle['experiments'].values())
    assert len(list(tmp_path.glob('supplement-*.json'))) == 8
    assert verify_supplements(data, bundle, controls, base['model_identity'])
    resumed = run_supplemental(data, ScriptedRawTestPolicy(forbidden=True), base, controls, tmp_path,
                               deadline=time.monotonic() - 1)
    assert len(resumed['reused_conditions']) == 7
    assert canonical(resumed['experiments']) == canonical(bundle['experiments'])
    cached = tmp_path / 'supplement-S16-1.json'
    changed = json.loads(cached.read_bytes())
    changed['episodes'][0]['outcome']['correct'] ^= True
    cached.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='replay'):
        run_supplemental(data, ScriptedRawTestPolicy(forbidden=True), base, controls, tmp_path,
                         deadline=time.monotonic() - 1)


def test_deadline_skips_and_partial_raw_results_are_still_replayed(tmp_path, receipts):
    data, base, controls = receipts
    skipped = run_supplemental(data, ScriptedRawTestPolicy(forbidden=True), base, controls, tmp_path,
                               deadline=time.monotonic() - 1)
    assert skipped['experiments']['S19']['status'] == 'COMPLETE'
    assert all(skipped['experiments'][s]['status'] == 'SKIPPED' for s in ('S16', 'S17', 'S18'))
    assert verify_supplements(data, skipped, controls, base['model_identity'])
    forecast = run_supplemental(data, ScriptedRawTestPolicy(forbidden=True), base, controls, tmp_path / 'forecast',
                                deadline=time.monotonic() + 120, seconds_per_episode=100.)
    assert all(forecast['experiments'][s]['status'] == 'SKIPPED' for s in ('S16','S17','S18'))
    with pytest.raises(ValueError, match='timing forecast'):
        run_supplemental(data, ScriptedRawTestPolicy(), base, controls, tmp_path / 'invalid',
                         deadline=time.monotonic() + 120, seconds_per_episode=float('nan'))
    full = run_supplemental(data, ScriptedRawTestPolicy(), base, controls, tmp_path,
                           deadline=time.monotonic() + 120)
    partial = deepcopy(skipped)
    raw = full['experiments']['S16']['records']['1']
    partial['experiments']['S16'].update(status='PARTIAL', records={'1': raw}, missing_conditions=['2', '4'])
    assert verify_supplements(data, partial, controls, base['model_identity'])
    partial['experiments']['S16']['records']['1']['budget']['decisions'] = 4
    with pytest.raises(ValueError, match='budget'):
        verify_supplements(data, partial, controls, base['model_identity'])
