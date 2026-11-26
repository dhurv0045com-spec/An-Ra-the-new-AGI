"""Compact, evidence-limited post-run matrix for Y5.

Thresholds below are operational analysis gates, not evidence of a registered
scientific hypothesis. Per-seed contrasts remain visible; no significance or
qualification is inferred from a missing arm, a canary, or dev alone.
"""
from statistics import mean, stdev
from math import sqrt

SEEDS = (7, 19, 41)


def _rate(result, path):
    cur = result
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else None


def _panel(result):
    panels = result.get('panels', {})
    if not panels:
        return None
    n = sum(p.get('total', 0) for p in panels.values())
    return sum(p.get('correct', 0) for p in panels.values()) / n if n else None


def _denominator(result, path):
    cur = result
    for key in path:
        if not isinstance(cur, dict): return None
        cur = cur.get(key, {})
    if not isinstance(cur, dict): return None
    return cur.get('families', cur.get('denominator', cur.get('samples')))


def _contrast(records, left, right, metric, denominator, threshold=0.0):
    pairs = []
    for seed in SEEDS:
        a = records.get(f'{left}-s{seed}', {})
        b = records.get(f'{right}-s{seed}', {})
        usable = lambda x: x.get('status', 'COMPLETE') in ('COMPLETE', 'COMPLETED') and not x.get('engineering_canary', False)
        av = metric(a) if usable(a) and (denominator(a) or 0) > 0 else None
        bv = metric(b) if usable(b) and (denominator(b) or 0) > 0 else None
        pairs.append({'seed': seed, 'left': av, 'right': bv,
                      'difference': av-bv if av is not None and bv is not None else None,
                      'left_denominator': denominator(a), 'right_denominator': denominator(b),
                      'left_status': a.get('status', 'COMPLETE' if av is not None else 'MISSING'),
                      'right_status': b.get('status', 'COMPLETE' if bv is not None else 'MISSING')})
    vals = [p['difference'] for p in pairs if p['difference'] is not None]
    complete = len(vals) == len(SEEDS)
    ci = None
    if complete:
        # Student t with two degrees of freedom, across independent seed differences.
        half = 4.3026527299 * stdev(vals) / sqrt(len(vals))
        ci = [mean(vals)-half, mean(vals)+half]
    return {'left': left, 'right': right, 'pairs': pairs,
            'seed_difference_95pct_t_interval': ci,
            'seed_signs': {'positive': sum(v > 0 for v in vals), 'zero': sum(v == 0 for v in vals), 'negative': sum(v < 0 for v in vals)},
            'mean_difference': mean(vals) if vals else None,
            'gate': ('PASS' if all(v > threshold for v in vals) else 'FAIL') if complete else 'INCOMPLETE',
            'threshold': threshold, 'rule': 'all three matched seed differences strictly exceed threshold; exploratory, not a preregistered significance test'}


def _family(path):
    return lambda r: _rate(r, (*path, 'family_rate'))


def _raw(r):
    metrics = r.get('metrics', {})
    rows = metrics.get('responses', [])
    if not rows or len(rows) != metrics.get('samples'): return []
    return rows


def _dmetric(r):
    rows = _raw(r)
    if not rows: return None
    try:
        if any(x['action'] != x['menu'][x['action_position']] for x in rows): return None
        total = sum(float(x['terminal_prediction'] == x['truth']) - x['action'] * x['cost'] for x in rows)
        value = total / len(rows)
        if abs(value - r['metrics']['realized_utility']) > 1e-9: return None
        return value
    except (KeyError, IndexError, TypeError): return None


def _e_verified(r):
    rows = _raw(r)
    if not rows: return False
    try:
        m = r['metrics']; n = len(rows)
        rates = {name: sum(x[field] == x['gold'] for x in rows)/n for name, field in
                 [('relevant','relevant'), ('none','no_feedback'), ('irrelevant','irrelevant'),
                  ('incorrect','edited'), ('erased','erased')]}
        rates['edited'] = sum(x['edited'] == x['public_laws']['query'][1-x['truth']] for x in rows)/n
        changes = {'write': sum(x['relevant'] != x['no_feedback'] for x in rows)/n,
                   'erase': sum(x['erased'] != x['relevant'] for x in rows)/n,
                   'edit': sum(x['edited'] != x['relevant'] for x in rows)/n}
        return (all(abs(m['accuracy'][k]-v) < 1e-9 for k,v in rates.items()) and
                all(abs(m['behavior_change'][k]-v) < 1e-9 for k,v in changes.items()))
    except (KeyError, TypeError): return False


def _emetric(r):
    return r['metrics']['accuracy']['relevant'] if _e_verified(r) else None


def _sample(r): return len(_raw(r)) or None

def _panel_n(r): return sum(x.get('total', 0) for x in r.get('panels', {}).values()) or None


def _chain_den(r): return _rate(r, ('unseen_format', 'chain_family_denominator'))
def _chain_rate(r): return _rate(r, ('unseen_format', 'chain_family_rate'))


def _intervention_n(r):
    return sum(v.get('total', 0) for v in r.get('interventions', {}).values()) or None


def _actual_donor(r):
    panels = r.get('interventions', {})
    n = _intervention_n(r)
    if not n or any('actual_donor' not in p.get('correct', {}) for p in panels.values()): return None
    return sum(p['correct']['actual_donor'] for p in panels.values()) / n


def _d_gate(r, arm):
    utility = _dmetric(r)
    if utility is None: return 'INCOMPLETE'
    rows = _raw(r)
    try:
        expected = sum(max(.5, ({'informative':.75, 'noninformative':.5}[arm] - x['cost'])) for x in rows)/len(rows)
        if abs(expected-r['metrics']['oracle_expected_utility']) > 1e-9: return 'INCOMPLETE'
        if not r['metrics']['action_invariant_to_hidden_observation']: return 'FAIL'
        if expected-utility > .05: return 'FAIL'
        if arm == 'noninformative':
            return 'PASS' if sum(x['action'] == 0 for x in rows)/len(rows) >= .9 else 'FAIL'
        low = [x for x in rows if x['cost'] == .05]
        high = [x for x in rows if x['cost'] == .35]
        if not low or not high: return 'INCOMPLETE'
        groups = {}
        for x in rows:
            groups.setdefault((tuple(x['public']), x['truth']), {})[x['observation']] = x['post_tool_prediction']
        if not groups or any(set(g) != {0, 1} for g in groups.values()): return 'INCOMPLETE'
        responsive = sum(g[0] != g[1] for g in groups.values())/len(groups)
        accuracy = sum(x['post_tool_prediction'] == x['truth'] for x in rows)/len(rows)
        return 'PASS' if (sum(x['action'] == 1 for x in low)/len(low) >= .8 and
                          sum(x['action'] == 0 for x in high)/len(high) >= .8 and
                          responsive >= .8 and accuracy >= .70) else 'FAIL'
    except (KeyError, TypeError, ZeroDivisionError): return 'INCOMPLETE'


def _diagnosis(result):
    # Compare the same measured capability on trained and held-out renderings.
    familiar = _rate(result, ('familiar_metrics', 'action_accuracy'))
    heldout = _rate(result, ('metrics', 'action_accuracy'))
    if familiar is None:
        familiar = _rate(result, ('familiar_metrics', 'accuracy', 'relevant'))
        heldout = _rate(result, ('metrics', 'accuracy', 'relevant'))
    if familiar is None or heldout is None: return 'INCOMPLETE'
    if familiar < .8: return 'OPTIMIZATION_OR_TASK_FAILURE'
    if heldout < .8: return 'HELD_OUT_RENDERING_FAILURE'
    return 'BOTH_MEET_OPERATIONAL_THRESHOLD'


def _within_job_gates(job_id, r):
    if r.get('engineering_canary') or r.get('status', 'COMPLETE') not in ('COMPLETE', 'COMPLETED'): return 'INCOMPLETE'
    if job_id.startswith('D-'):
        return _d_gate(r, 'noninformative' if job_id.startswith('D-noninformative-') else 'informative')
    if job_id.startswith('E-'):
        if _emetric(r) is None: return 'INCOMPLETE'
        acc = r.get('metrics', {}).get('accuracy', {})
        change = r.get('metrics', {}).get('behavior_change', {})
        vals = [acc.get('relevant'), acc.get('none'), change.get('erase'), change.get('edit'), acc.get('incorrect'), acc.get('irrelevant'), acc.get('edited')]
        if any(v is None for v in vals): return 'INCOMPLETE'
        return 'PASS' if vals[0] >= .8 and vals[0]-vals[1] >= .15 and vals[2] >= .1 and vals[3] >= .1 and vals[0]-vals[4] >= .15 and vals[0]-vals[5] >= .15 and vals[6] >= .8 else 'FAIL'
    if job_id.startswith('F-'):
        old = _rate(r, ('old_endpoint', 'family_rate'))
        new = _rate(r, ('new_endpoint', 'family_rate'))
        parent = _rate(r, ('parent_new_endpoint', 'family_rate'))
        parent_dev = _rate(r, ('parent_new_dev', 'family_rate'))
        if any(v is None for v in (old, new, parent, parent_dev)) or any((_denominator(r, (k,)) or 0) <= 0 for k in ('old_endpoint', 'new_endpoint', 'parent_new_endpoint', 'parent_new_dev')): return 'INCOMPLETE'
        return 'PASS' if old >= .8 and new >= .8 and parent_dev < .8 and new-parent >= .15 else 'FAIL'
    return None


def summarize(rows):
    matrix = {}
    for split in ('dev', 'confirm'):
        r = rows.get(split, {})
        specs = {
            'A_reader_varied_vs_fixed_unseen': ('AB-reader-varied-ce', 'AB-reader-fixed-ce', _family(('unseen_format',)), lambda x:_denominator(x, ('unseen_format',))),
            'A_connected_varied_vs_fixed_unseen': ('AB-reader-varied-ce', 'AB-reader-fixed-ce', _chain_rate, _chain_den),
            'B_varied_pointer_vs_sham': ('AB-answer-varied-pointer', 'AB-answer-varied-sham', _family(('primary',)), lambda x:_denominator(x, ('primary',))),
            'B_pointer_vs_sham': ('AB-answer-fixed-pointer', 'AB-answer-fixed-sham', _family(('primary',)), lambda x:_denominator(x, ('primary',))),
            'B_pointer_vs_ce': ('AB-answer-fixed-pointer', 'AB-answer-fixed-ce', _family(('primary',)), lambda x:_denominator(x, ('primary',))),
            'C_iit_vs_ordinary': ('C-iit', 'C-ordinary', _panel, _panel_n),
            'C_iit_vs_sham': ('C-iit', 'C-sham', _panel, _panel_n),
            'C_actual_donor_iit_vs_ordinary': ('C-iit', 'C-ordinary', _actual_donor, _intervention_n),
            'C_actual_donor_iit_vs_sham': ('C-iit', 'C-sham', _actual_donor, _intervention_n),
            'D_informative_vs_noninformative': ('D-informative', 'D-noninformative', _dmetric, _sample),
            'E_relevant_vs_sham': ('E-relevant', 'E-sham', _emetric, _sample),
            'F_new_vs_control_old': ('F-new', 'F-old_control', _family(('old_endpoint',)), lambda x:_denominator(x, ('old_endpoint',))),
            'F_cap_vs_new_old': ('F-new_cap', 'F-new', _family(('old_endpoint',)), lambda x:_denominator(x, ('old_endpoint',))),
            'F_cap_vs_new_new': ('F-new_cap', 'F-new', _family(('new_endpoint',)), lambda x:_denominator(x, ('new_endpoint',))),
        }
        out = {k:_contrast(r,*v) for k,v in specs.items()}
        # Different channels have different oracle values: this contrast is descriptive.
        if out['D_informative_vs_noninformative']['gate'] != 'INCOMPLETE':
            out['D_informative_vs_noninformative']['gate'] = 'DESCRIPTIVE'
        out['within_job_gates'] = {k:_within_job_gates(k,v) for k,v in r.items() if k.startswith(('D-', 'E-', 'F-'))}
        out['all_seed_within_gates'] = {}
        for prefix in ('D-informative', 'D-noninformative', 'E-relevant', 'E-sham', 'F-new', 'F-new_cap', 'F-old_control'):
            gates = [out['within_job_gates'].get(f'{prefix}-s{seed}', 'INCOMPLETE') for seed in SEEDS]
            out['all_seed_within_gates'][prefix] = 'INCOMPLETE' if 'INCOMPLETE' in gates else 'PASS' if all(g == 'PASS' for g in gates) else 'FAIL'
        out['rendering_diagnoses'] = {k:_diagnosis(v) for k,v in r.items() if k.startswith(('D-', 'E-'))}
        out['status_counts'] = {status:sum(v.get('status', 'COMPLETE') == status for v in r.values())
                                for status in ('COMPLETE', 'COMPLETED', 'GATED', 'BLOCKED', 'FAILED')}
        matrix[split] = out
    return {'matrix':matrix, 'interpretation': 'PASS/FAIL are operational contrasts, not causal qualification; INCOMPLETE includes missing or gated jobs. Dev does not substitute for independent confirmation. Compare denominators and inspect raw outputs.',
            'uncertainty_note': '95% t intervals use the three independent seed-level paired differences, not individual cases or members of a family. With three seeds they are highly uncertain and conditional on the fixed evaluation panels; no case-level independence is assumed.', 'canary_not_qualification': True, 'gate_definitions': {'E': 'relevant and edited-target >= .80, relevant exceeds none, incorrect, and irrelevant by >= .15, erase and edit each change >= .10', 'F': 'old and new complete-family rates each >= .80, parent dev new < .80 and matched endpoint gain >= .15', 'D': 'oracle expected paid utility regret <= .05; informative low/high cost decisions >= .80, observation response >= .80, post-tool accuracy >= .70; noninformative declines tool >= .90; action invariant to hidden observation', 'D_note': 'Utility contrasts across information channels describe different environments, and are not by themselves a causal advantage gate.'}}
