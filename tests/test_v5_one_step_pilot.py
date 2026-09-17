from pathlib import Path


def test_fixture_is_pair_disjoint_and_deterministic():
    from v5_experiments.one_step_pilot import make_fixture
    from v5_experiments.cyr_gpu011 import load_ark002b_manifest
    data = load_ark002b_manifest(Path('docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json'))
    fixture = make_fixture(data)
    assert fixture == make_fixture(data)
    assert len(fixture['train']) == 64
    assert len(fixture['holdout']) == 100
    original = {tuple(sorted(r['canonical_pair'])) for k in ('train', 'dev_controller', 'dev_measurement', 'sealed_reserved') for r in data[k]}
    train = {tuple(r['canonical_pair']) for r in fixture['train']}
    holdout = {tuple(r['canonical_pair']) for r in fixture['holdout']}
    assert len(train) == 64 and len(holdout) == 100
    assert not train & holdout and not (train | holdout) & original
    for rows in fixture.values():
        assert {r['a'] % 10 + r['b'] % 10 >= 10 for r in rows} == {True, False}
        for r in rows:
            assert r['answer'] == str(r['a'] + r['b'])
            assert r['prompt'] == f"{r['a']} + {r['b']} = "


def test_exact_requires_all_answer_tokens_and_eos():
    from v5_experiments.one_step_pilot import exact_flags
    assert exact_flags([[3, 4, 2], [3, 4, 0]], [[3, 4, 2], [3, 4, 2]]) == [True, False]
