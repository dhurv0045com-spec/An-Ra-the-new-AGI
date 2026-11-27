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


def test_stage_a_b_f_gates_and_rescore_evidence(tmp_path):
    from experiments.avendi_y5.analysis import rescore_evidence
    from experiments.avendi_y5.runtime import publish
    rows = {}
    for seed in (7, 19, 41):
        # Stage A: varied gains >=0.15 on unseen while retaining familiar within <=0.05
        rows[f'AB-reader-varied-ce-s{seed}'] = {
            'unseen_format': {'family_rate': 0.80, 'families': 16},
            'formats': {'0': {'family_rate': 0.90, 'families': 16}},
        }
        rows[f'AB-reader-fixed-ce-s{seed}'] = {
            'unseen_format': {'family_rate': 0.40, 'families': 16},
            'formats': {'0': {'family_rate': 0.92, 'families': 16}},
        }
        # Stage B: pointer beats BOTH sham and ce across varied & fixed
        rows[f'AB-answer-varied-pointer-s{seed}'] = {'primary': {'family_rate': 0.85, 'families': 16}}
        rows[f'AB-answer-varied-sham-s{seed}'] = {'primary': {'family_rate': 0.50, 'families': 16}}
        rows[f'AB-answer-varied-ce-s{seed}'] = {'primary': {'family_rate': 0.55, 'families': 16}}
        rows[f'AB-answer-fixed-pointer-s{seed}'] = {'primary': {'family_rate': 0.85, 'families': 16}}
        rows[f'AB-answer-fixed-sham-s{seed}'] = {'primary': {'family_rate': 0.50, 'families': 16}}
        rows[f'AB-answer-fixed-ce-s{seed}'] = {'primary': {'family_rate': 0.55, 'families': 16}}
        # Stage F stratified parent sources
        rows[f'F-new-s{seed}'] = {
            'seed': seed, 'arm': 'new', 'old_source': {'kind': 'AB_answer' if seed == 7 else 'Y4_pointer_reader'},
            'old_endpoint': {'family_rate': 0.85, 'families': 16}, 'new_endpoint': {'family_rate': 0.85, 'families': 16},
            'parent_new_endpoint': {'family_rate': 0.1, 'families': 16}, 'parent_new_dev': {'family_rate': 0.1, 'families': 16},
        }
    mat = summarize({'confirm': rows})['matrix']['confirm']
    assert mat['A_reader_varied_vs_fixed_unseen']['gate'] == 'PASS'
    assert mat['A_reader_varied_familiar_retention']['gate'] == 'PASS'
    assert mat['B_varied_pointer_vs_ce']['gate'] == 'PASS'
    assert mat['B_all_seed_dominance'] == {'varied': 'PASS', 'fixed': 'PASS'}
    assert set(mat['F_parent_stratification']) == {'AB_answer', 'Y4_pointer_reader'}

    # Independent rescorer works strictly from EVIDENCE without weights/ or optimizer.bin, and catches corruption.
    from experiments.avendi_y5.information import P6LearnedMemoryPolicy, _eval_d
    ev_dir = tmp_path / 'evidence'
    d_met = _eval_d(P6LearnedMemoryPolicy(width=8, seed=7), 'informative', 'dev')
    publish(ev_dir / 'CONFIRM_LOCK.json', {'eligible': {'D-informative-s7': {'dev_sha256': 'x'}}})
    publish(ev_dir / 'dev' / 'D-informative-s7.json', {
        'job': {'id': 'D-informative-s7', 'stage': 'D'}, 'split': 'dev', 'status': 'COMPLETE',
        'result': {'metrics': d_met},
    })
    assert rescore_evidence(ev_dir)['verified']
    # Tamper response accuracy -> strict rescore must raise ValueError.
    r0 = d_met['responses'][0]
    flipped = (1 - int(r0['truth'])) if int(r0['terminal_prediction']) == int(r0['truth']) else int(r0['truth'])
    d_bad = dict(d_met, responses=[dict(r0, terminal_prediction=flipped), *d_met['responses'][1:]])
    publish(ev_dir / 'dev' / 'D-informative-s19.json', {
        'job': {'id': 'D-informative-s19', 'stage': 'D'}, 'split': 'dev', 'status': 'COMPLETE',
        'result': {'metrics': d_bad},
    })
    import pytest
    with pytest.raises(ValueError, match='EVIDENCE_RESCORE_FAILED'):
        rescore_evidence(ev_dir, strict=True)

