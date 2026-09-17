def test_escalation_requires_complete_exact_flat_trace():
    from v5_experiments.dose_pilot import should_escalate
    flat = [{'step': i, 'correct': 27} for i in (0, 50, 100, 150, 200)]
    assert should_escalate(flat, complete=True)
    assert not should_escalate(flat, complete=False)
    assert not should_escalate(flat[:-1], complete=True)
    assert not should_escalate(flat + [flat[-1]], complete=True)
    for value in (26, 28):
        varied = [dict(r) for r in flat]
        varied[2]['correct'] = value
        assert not should_escalate(varied, complete=True)
