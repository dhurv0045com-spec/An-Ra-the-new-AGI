from experiments.avendi_y5.analysis import summarize


def test_missing_is_incomplete_not_negative():
    s = summarize({'dev': {}, 'confirm': {}})
    assert s['matrix']['dev']['E_relevant_vs_sham']['gate'] == 'INCOMPLETE'


def test_matched_seed_differences_and_denominators():
    rows = {}
    for seed in (7, 19, 41):
        for arm, acc in [('relevant', .8), ('sham', .4)]:
            raw = [dict(gold=0, truth=0, public_laws={'query': [0, 1]}, relevant=int(i >= int(20*acc)), no_feedback=0, irrelevant=0, erased=0, edited=0) for i in range(20)]
            change = 1-acc
            rows[f'E-{arm}-s{seed}'] = {'metrics': {'samples': 20, 'responses': raw, 'accuracy': {'relevant': acc, 'none': 1, 'irrelevant': 1, 'incorrect': 1, 'erased': 1, 'edited': 0}, 'behavior_change': {'write': change, 'erase': change, 'edit': change}}}
    c = summarize({'dev': rows})['matrix']['dev']['E_relevant_vs_sham']
    assert c['gate'] == 'PASS' and abs(c['mean_difference']-.4) < 1e-9
    assert all(p['left_denominator'] == p['right_denominator'] == 20 for p in c['pairs'])
    del rows['E-sham-s41']
    assert summarize({'dev': rows})['matrix']['dev']['E_relevant_vs_sham']['gate'] == 'INCOMPLETE'


def test_optimization_and_rendering_failures_are_distinct():
    def row(a,b): return {'familiar_metrics': {'action_accuracy': a}, 'metrics': {'action_accuracy': b}}
    rows = {'D-informative-s7': row(.4,.3), 'D-informative-s19': row(.9,.3)}
    d = summarize({'dev':rows})['matrix']['dev']['rendering_diagnoses']
    assert d['D-informative-s7'] == 'OPTIMIZATION_OR_TASK_FAILURE'
    assert d['D-informative-s19'] == 'HELD_OUT_RENDERING_FAILURE'


def test_canary_and_forged_raw_metrics_never_qualify():
    from experiments.avendi_y5.information import P6LearnedMemoryPolicy, _eval_d
    rows = {}
    for seed in (7, 19, 41):
        for arm in ('informative', 'noninformative'):
            metrics = _eval_d(P6LearnedMemoryPolicy(width=8, seed=seed), arm, 'dev')
            rows[f'D-{arm}-s{seed}'] = {'metrics': metrics}
    summary = summarize({'dev': rows})['matrix']['dev']['D_informative_vs_noninformative']
    assert summary['gate'] == 'DESCRIPTIVE'
    assert summary['seed_difference_95pct_t_interval'] is not None
    row = rows['D-informative-s7']['metrics']['responses'][0]
    row['terminal_prediction'] = (1-row['truth']) if row['terminal_prediction'] == row['truth'] else row['truth']
    assert summarize({'dev': rows})['matrix']['dev']['D_informative_vs_noninformative']['gate'] == 'INCOMPLETE'
    rows['D-informative-s7']['engineering_canary'] = True
    assert summarize({'dev': rows})['matrix']['dev']['D_informative_vs_noninformative']['gate'] == 'INCOMPLETE'
