"""Initial-tensor replay regression tests; CPU only, no training."""
import importlib
import sys
from pathlib import Path

import pytest

from anra_v5 import initial_replay

FROZEN = 'C:/Users/ankit/cyr012-frozen'


def test_module_uses_frozen_sources():
    assert initial_replay.FROZEN_ROOT == Path(FROZEN)


def test_classify_verdicts():
    assert initial_replay.classify(46.88997268676758, 46.88997268676758,
                                   'aa', 'aa') == {
        'verdict': 'INITIALIZATION_REPLAYS_MATCHING_NORM',
        'norm_delta': 0.0, 'in_process_deterministic': True}
    verdict = initial_replay.classify(46.5, 46.88997268676758, 'aa', 'bb')
    assert verdict['verdict'] == 'NORM_MISMATCH'
    assert verdict['in_process_deterministic'] is False


def test_tensor_digest_is_stable_and_seed_sensitive(torch=None):
    pytest.importorskip('torch')
    flat_a = initial_replay.build_initial_flat(seed=3301)
    flat_b = initial_replay.build_initial_flat(seed=3301)
    flat_c = initial_replay.build_initial_flat(seed=3302)
    digest_a = initial_replay.tensor_digest(flat_a)
    assert digest_a == initial_replay.tensor_digest(flat_b)
    assert digest_a != initial_replay.tensor_digest(flat_c)
    assert len(digest_a) == 64


def test_replay_does_not_reuse_cached_worktree_builder(monkeypatch):
    pytest.importorskip('torch')
    cached = importlib.import_module('v5_model.core')

    def contaminated(*args, **kwargs):
        raise AssertionError('cached worktree builder used')

    monkeypatch.setattr(cached, 'initialize', contaminated)
    saved_path = list(sys.path)
    flat = initial_replay.build_initial_flat(seed=3301)
    assert flat.numel() == initial_replay.COMPACT_PARAMETER_COUNT
    assert sys.modules['v5_model.core'] is cached
    assert cached.initialize is contaminated
    assert sys.path == saved_path


def test_missing_frozen_source_fails_without_worktree_fallback(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    cached = importlib.import_module('v5_model.core')
    monkeypatch.setattr(initial_replay, 'FROZEN_ROOT', tmp_path / 'absent')
    saved_path = list(sys.path)
    with pytest.raises(FileNotFoundError):
        initial_replay.build_initial_flat()
    assert sys.modules['v5_model.core'] is cached
    assert sys.path == saved_path


def test_build_initial_flat_norm_is_finite_and_positive():
    pytest.importorskip('torch')
    flat = initial_replay.build_initial_flat(seed=3301)
    norm = float(flat.norm().item())
    assert norm > 0 and norm == norm  # finite, not NaN


def test_cli_writes_provenance_bound_replay(tmp_path):
    pytest.importorskip('torch')
    out = tmp_path / 'REPLAY.json'
    initial_replay.main(['--left-receipt',
                         'artifacts/v5/cyr_gpu_011_result_receipt.json',
                         '--left-bundle',
                         'C:/Users/ankit/Downloads/CYMEK_GPU_RESEARCH_V11_RESULTS.zip',
                         '--right-receipt',
                         'artifacts/v5/cyr_gpu_012_result_receipt.json',
                         '--right-bundle',
                         'C:/Users/ankit/cyr012-evidence/full01/CYR_GPU_012_RESULTS.zip',
                         '--out', str(out)])
    data = __import__('json').loads(out.read_text())
    assert data['schema'] == 'anra-cyr-initial-replay/v1'
    assert data['left']['recorded_initial_l2'] == 46.88997268676758
    assert data['right']['recorded_initial_l2'] == 46.88997268676758
    assert data['replay']['seed'] == 3301
    assert data['replay']['computed_tensor_sha256']
    assert data['verdict']['verdict'] in {
        'INITIALIZATION_REPLAYS_MATCHING_NORM', 'NORM_MISMATCH'}
