"""Compact, evidence-limited post-run matrix for Y5.

Thresholds below are operational analysis gates, not evidence of a registered
scientific hypothesis. Per-seed contrasts remain visible; no significance or
qualification is inferred from a missing arm, a canary, or dev alone.
"""
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZipFile

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


def _familiar_retention(r):
    # Style 0 on familiar prompt structure; fallback to primary if formats absent.
    v = _rate(r, ('formats', '0', 'family_rate'))
    return v if v is not None else _rate(r, ('primary', 'family_rate'))


def _familiar_retention_den(r):
    v = _denominator(r, ('formats', '0'))
    return v if v is not None else _denominator(r, ('primary',))


def summarize(rows):
    matrix = {}
    for split in ('dev', 'confirm'):
        r = rows.get(split, {})
        specs = {
            'A_reader_varied_vs_fixed_unseen': ('AB-reader-varied-ce', 'AB-reader-fixed-ce', _family(('unseen_format',)), lambda x:_denominator(x, ('unseen_format',)), 0.15),
            'A_reader_varied_familiar_retention': ('AB-reader-varied-ce', 'AB-reader-fixed-ce', _familiar_retention, _familiar_retention_den, -0.05),
            'A_connected_varied_vs_fixed_unseen': ('AB-reader-varied-ce', 'AB-reader-fixed-ce', _chain_rate, _chain_den, 0.0),
            'B_varied_pointer_vs_sham': ('AB-answer-varied-pointer', 'AB-answer-varied-sham', _family(('primary',)), lambda x:_denominator(x, ('primary',)), 0.0),
            'B_varied_pointer_vs_ce': ('AB-answer-varied-pointer', 'AB-answer-varied-ce', _family(('primary',)), lambda x:_denominator(x, ('primary',)), 0.0),
            'B_pointer_vs_sham': ('AB-answer-fixed-pointer', 'AB-answer-fixed-sham', _family(('primary',)), lambda x:_denominator(x, ('primary',)), 0.0),
            'B_pointer_vs_ce': ('AB-answer-fixed-pointer', 'AB-answer-fixed-ce', _family(('primary',)), lambda x:_denominator(x, ('primary',)), 0.0),
            'C_iit_vs_ordinary': ('C-iit', 'C-ordinary', _panel, _panel_n, 0.0),
            'C_iit_vs_sham': ('C-iit', 'C-sham', _panel, _panel_n, 0.0),
            'C_actual_donor_iit_vs_ordinary': ('C-iit', 'C-ordinary', _actual_donor, _intervention_n, 0.0),
            'C_actual_donor_iit_vs_sham': ('C-iit', 'C-sham', _actual_donor, _intervention_n, 0.0),
            'D_informative_vs_noninformative': ('D-informative', 'D-noninformative', _dmetric, _sample, 0.0),
            'E_relevant_vs_sham': ('E-relevant', 'E-sham', _emetric, _sample, 0.0),
            'F_new_vs_control_old': ('F-new', 'F-old_control', _family(('old_endpoint',)), lambda x:_denominator(x, ('old_endpoint',)), 0.0),
            'F_cap_vs_new_old': ('F-new_cap', 'F-new', _family(('old_endpoint',)), lambda x:_denominator(x, ('old_endpoint',)), 0.0),
            'F_cap_vs_new_new': ('F-new_cap', 'F-new', _family(('new_endpoint',)), lambda x:_denominator(x, ('new_endpoint',)), 0.0),
        }
        out = {k: _contrast(r, left, right, met, den, threshold=thr) for k, (left, right, met, den, thr) in specs.items()}
        # Retain >= -0.05 inclusive check for Stage A familiar retention (varied >= fixed - 0.05).
        ret = out['A_reader_varied_familiar_retention']
        ret_vals = [p['difference'] for p in ret['pairs'] if p['difference'] is not None]
        if len(ret_vals) == len(SEEDS):
            ret['gate'] = 'PASS' if all(v >= -0.05 for v in ret_vals) else 'FAIL'
            ret['rule'] = 'all three matched seed differences satisfy varied - fixed >= -0.05 (max degradation 0.05)'
        # Stage B requires pointer to beat both sham and CE at all 3 seeds for varied & fixed.
        out['B_all_seed_dominance'] = {
            'varied': 'PASS' if (out['B_varied_pointer_vs_sham']['gate'] == 'PASS' and out['B_varied_pointer_vs_ce']['gate'] == 'PASS') else ('INCOMPLETE' if 'INCOMPLETE' in (out['B_varied_pointer_vs_sham']['gate'], out['B_varied_pointer_vs_ce']['gate']) else 'FAIL'),
            'fixed': 'PASS' if (out['B_pointer_vs_sham']['gate'] == 'PASS' and out['B_pointer_vs_ce']['gate'] == 'PASS') else ('INCOMPLETE' if 'INCOMPLETE' in (out['B_pointer_vs_sham']['gate'], out['B_pointer_vs_ce']['gate']) else 'FAIL'),
        }
        # Different channels have different oracle values: this contrast is descriptive.
        if out['D_informative_vs_noninformative']['gate'] != 'INCOMPLETE':
            out['D_informative_vs_noninformative']['gate'] = 'DESCRIPTIVE'
        out['within_job_gates'] = {k:_within_job_gates(k,v) for k,v in r.items() if k.startswith(('D-', 'E-', 'F-'))}
        out['all_seed_within_gates'] = {}
        for prefix in ('D-informative', 'D-noninformative', 'E-relevant', 'E-sham', 'F-new', 'F-new_cap', 'F-old_control'):
            gates = [out['within_job_gates'].get(f'{prefix}-s{seed}', 'INCOMPLETE') for seed in SEEDS]
            out['all_seed_within_gates'][prefix] = 'INCOMPLETE' if 'INCOMPLETE' in gates else 'PASS' if all(g == 'PASS' for g in gates) else 'FAIL'
        out['rendering_diagnoses'] = {k:_diagnosis(v) for k,v in r.items() if k.startswith(('D-', 'E-'))}
        # Stratify Stage F by parent lineage kind (AB_answer vs Y4_pointer_reader).
        strat = {}
        for jid, row in sorted(r.items()):
            if not jid.startswith('F-'):
                continue
            kind = (row.get('old_source') or {}).get('kind', 'UNKNOWN')
            strat.setdefault(kind, {})[jid] = {
                'seed': row.get('seed'),
                'arm': row.get('arm'),
                'old_family_rate': _rate(row, ('old_endpoint', 'family_rate')),
                'new_family_rate': _rate(row, ('new_endpoint', 'family_rate')),
                'new_family_gain': row.get('new_family_gain'),
                'gate': out['within_job_gates'].get(jid),
            }
        out['F_parent_stratification'] = strat
        out['blind_baselines'] = {
            jid: (row.get('primary') or {}).get('blind_baselines')
            for jid, row in sorted(r.items())
            if jid.startswith(('AB-', 'F-')) and (row.get('primary') or {}).get('blind_baselines')
        }
        out['status_counts'] = {status:sum(v.get('status', 'COMPLETE') == status for v in r.values())
                                for status in ('COMPLETE', 'COMPLETED', 'GATED', 'BLOCKED', 'FAILED')}
        matrix[split] = out
    # Separate transfer and diagnostic roles explicitly so confirm/transfer/diagnostic never conflate.
    transfer_summary = {}
    confirm_rows = rows.get('confirm', {}) or rows.get('dev', {})
    for jid, row in sorted(confirm_rows.items()):
        if 'transfer_format' in row or 'unseen_format' in row:
            tf = row.get('transfer_format') or row.get('unseen_format')
            transfer_summary[jid] = {'type': 'reader_transfer', 'family_rate': tf.get('family_rate'), 'denominator': tf.get('families')}
        elif 'transfer_panels' in row:
            tp = row['transfer_panels']
            n = sum(p.get('total', 0) for p in tp.values())
            transfer_summary[jid] = {'type': 'composition_transfer', 'accuracy': sum(p.get('correct', 0) for p in tp.values()) / max(1, n), 'denominator': n}
        elif 'transfer_metrics' in row:
            tm = row['transfer_metrics']
            transfer_summary[jid] = {'type': 'info_memory_transfer', 'samples': tm.get('samples')}
    diagnostic_roles = sorted(jid for jid in set(rows.get('dev', {})) | set(rows.get('confirm', {}))
                              if jid.startswith(('G-', 'C-baseline-', 'C-scout-')))
    return {'matrix':matrix, 'transfer_results': transfer_summary, 'diagnostic_roles': diagnostic_roles,
            'interpretation': 'PASS/FAIL are operational contrasts, not causal qualification; INCOMPLETE includes missing or gated jobs. Dev does not substitute for independent confirmation. Compare denominators and inspect raw outputs.',
            'uncertainty_note': '95% t intervals use the three independent seed-level paired differences, not individual cases or members of a family. With three seeds they are highly uncertain and conditional on the fixed evaluation panels; no case-level independence is assumed.', 'canary_not_qualification': True, 'gate_definitions': {'A': 'unseen connected-family gain >= .15 and familiar retention degradation <= .05 at all 3 seeds', 'B': 'true spans beat both sham and CE at all 3 seeds for varied and fixed formatting', 'E': 'relevant and edited-target >= .80, relevant exceeds none, incorrect, and irrelevant by >= .15, erase and edit each change >= .10', 'F': 'old and new complete-family rates each >= .80, parent dev new < .80 and matched endpoint gain >= .15 (stratified by parent source kind)', 'D': 'oracle expected paid utility regret <= .05; informative low/high cost decisions >= .80, observation response >= .80, post-tool accuracy >= .70; noninformative declines tool >= .90; action invariant to hidden observation', 'D_note': 'Utility contrasts across information channels describe different environments, and are not by themselves a causal advantage gate.'}}


def _read_evidence_files(evidence_path):
    """Load strict EVIDENCE receipts from directory or ZIP without touching weights/ or optimizer.bin."""
    p = Path(evidence_path)
    if not p.exists():
        raise FileNotFoundError(f'missing evidence path: {p}')
    docs = {}
    if p.is_file():
        with ZipFile(p) as zf:
            for name in zf.namelist():
                if name.endswith('/') or name.startswith('weights/') or name.endswith('optimizer.bin'):
                    continue
                if name.endswith('.json'):
                    docs[name] = json.loads(zf.read(name).decode('utf-8'))
    else:
        for f in sorted(p.rglob('*.json')):
            rel = f.relative_to(p).as_posix()
            if rel.startswith('weights/') or rel.endswith('optimizer.bin'):
                continue
            docs[rel] = json.loads(f.read_text(encoding='utf-8'))
    return docs


def _recompute_reader_eval(block, tag, defects):
    preds = block.get('predictions')
    if not isinstance(preds, list):
        defects.append(f'{tag}:missing_predictions')
        return
    if len(preds) != block.get('denominator'):
        defects.append(f'{tag}:denominator_mismatch')
        return
    seen_ids = set()
    fams = {}
    ok = 0
    for row in preds:
        wid = str(row.get('id'))
        if wid in seen_ids:
            defects.append(f'{tag}:duplicate_id:{wid}')
            return
        seen_ids.add(wid)
        if row.get('eos') is None:
            pred = row.get('prediction')
            row_ok = isinstance(pred, dict) and pred.get('rows') == row.get('gold_rows') and pred.get('query') == row.get('gold_query') and pred.get('depth') == row.get('gold_depth')
        else:
            row_ok = bool(row.get('eos')) and (row.get('prediction') == row.get('gold'))
        if bool(row.get('correct')) != bool(row_ok):
            defects.append(f'{tag}:corrupted_correct_flag:{wid}')
            return
        ok += int(row_ok)
        fams.setdefault(row.get('family_id'), []).append(bool(row_ok))
    if ok != block.get('correct'):
        defects.append(f'{tag}:correct_count_mismatch')
    fam_ok = sum(all(v) for v in fams.values())
    if fam_ok != block.get('family_correct') or len(fams) != block.get('families'):
        defects.append(f'{tag}:family_count_mismatch')
    elif abs((fam_ok / max(1, len(fams))) - float(block.get('family_rate', -1))) > 1e-9:
        defects.append(f'{tag}:family_rate_mismatch')


def rescore_evidence(evidence_path, *, strict=True):
    """Independently rescore all stage primary metrics strictly from EVIDENCE receipts without weights/."""
    docs = _read_evidence_files(evidence_path)
    defects = []
    lock_doc = docs.get('CONFIRM_LOCK.json', {}).get('content') if 'content' in docs.get('CONFIRM_LOCK.json', {}) else docs.get('CONFIRM_LOCK.json')
    rescored = {}
    for rel, raw in sorted(docs.items()):
        if not (rel.startswith(('dev/', 'confirm/')) and rel.endswith('.json')):
            continue
        rec = raw['content'] if isinstance(raw, dict) and 'content' in raw and 'sha256' in raw else raw
        if isinstance(raw, dict) and 'sha256' in raw:
            encoded = json.dumps(raw['content'], sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()
            if hashlib.sha256(encoded).hexdigest() != raw['sha256']:
                defects.append(f'{rel}:sha256_tampered')
                continue
        split = rec.get('split')
        jid = (rec.get('job') or {}).get('id') or rel.rsplit('/', 1)[-1].removesuffix('.json')
        res = rec.get('result') or {}
        if split == 'confirm' and lock_doc is not None:
            eligible = lock_doc.get('eligible', {})
            if jid not in eligible:
                defects.append(f'{rel}:confirm_outside_lock:{jid}')
        stage = (rec.get('job') or {}).get('stage', jid.split('-', 1)[0])
        if res.get('status') in ('GATED', 'BLOCKED') or rec.get('status') in ('GATED', 'BLOCKED'):
            rescored[rel] = {'status': rec.get('status', res.get('status'))}
            continue
        if stage in ('AB', 'F') and 'primary' in res:
            _recompute_reader_eval(res['primary'], f'{rel}:primary', defects)
            for k in ('old_endpoint', 'new_endpoint', 'unseen_format'):
                if isinstance(res.get(k), dict) and 'predictions' in res[k]:
                    _recompute_reader_eval(res[k], f'{rel}:{k}', defects)
        elif stage == 'C' and 'panels' in res:
            for d, pb in res['panels'].items():
                cases = pb.get('cases', [])
                if len(cases) != pb.get('total') or len({c.get('case_id') for c in cases}) != len(cases):
                    defects.append(f'{rel}:panel_{d}_case_mismatch')
                elif sum(int(c.get('prediction') == c.get('gold') and bool(c.get('correct'))) for c in cases) != pb.get('correct'):
                    defects.append(f'{rel}:panel_{d}_correct_mismatch')
            for d, ib in (res.get('interventions') or {}).items():
                cases = ib.get('cases', [])
                if len(cases) != ib.get('total'):
                    defects.append(f'{rel}:intervention_{d}_total_mismatch')
                else:
                    expected_actual = sum(int((c.get('outcomes') or {}).get('actual_donor', {}).get('correct', False)) for c in cases)
                    if expected_actual != (ib.get('correct') or {}).get('actual_donor'):
                        defects.append(f'{rel}:intervention_{d}_actual_donor_mismatch')
        elif stage == 'D' and 'metrics' in res:
            if _dmetric(res) is None:
                defects.append(f'{rel}:stage_d_raw_mismatch')
        elif stage == 'E' and 'metrics' in res:
            if not _e_verified(res):
                defects.append(f'{rel}:stage_e_raw_mismatch')
        rescored[rel] = {'status': 'RESCORED'}
    if defects and strict:
        raise ValueError('EVIDENCE_RESCORE_FAILED: ' + '; '.join(defects))
    return {
        'verified': len(defects) == 0,
        'status': 'VERIFIED_FROM_EVIDENCE_ONLY' if not defects else 'FAILED',
        'evidence_only': True,
        'rescored_count': len(rescored),
        'defects': defects,
    }
