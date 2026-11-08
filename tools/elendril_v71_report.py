"""Compact, source-agnostic summaries of saved V7.1 worker evaluations."""
import csv
from collections import defaultdict
import io
import json
from pathlib import Path
import re

from tools.elendril_v6_export import _write_derived, canonical

_COLUMNS = ('job_id', 'profile', 'method', 'mechanism', 'seed', 'epoch', 'update', 'checkpoint',
            'mode', 'cohort', 'episodes', 'correct', 'accuracy', 'nonnull_correct', 'nonnull_total',
            'nonnull_accuracy', 'query_needed', 'needed_queried', 'query_precision', 'query_recall',
            'invalid', 'utility_numerator', 'utility_denominator', 'answer_entropy_bits',
            'cost_0_utility', 'cost_01_utility', 'cost_05_utility', 'cost_1_utility', 'confidence_count', 'brier')
_PATTERN = re.compile(r'^(S3|S8|S20)-(uniform|blocked|b-only|replay|fresh-b)-(ce|intermediate)-s(\d+)$')


def _identity(directory):
    for path in sorted(directory.glob('*-request.json')):
        try:
            job = json.loads(path.read_bytes()).get('job', {})
            if job.get('job_id') == directory.name:
                return {k: job.get(k) for k in ('profile', 'method', 'mechanism', 'seed')}
        except (OSError, ValueError, AttributeError):
            continue
    match = _PATTERN.fullmatch(directory.name)
    if not match:
        frozen = re.fullmatch(r'(?:original)?(uniform|blocked)-s(\d+)', directory.name)
        if frozen:
            return {'profile': 'S3', 'method': frozen.group(1), 'mechanism': 'ce', 'seed': int(frozen.group(2))}
        return {'profile': None, 'method': None, 'mechanism': None, 'seed': None}
    profile, method, mechanism, seed = match.groups()
    return {'profile': profile, 'method': method, 'mechanism': mechanism, 'seed': int(seed)}


def _matched(free, grammar):
    if free['checkpoint'] != grammar['checkpoint'] or free['update'] != grammar['update']:
        return []
    output = []
    for cohort in sorted(set(free['evaluation']['records']) & set(grammar['evaluation']['records'])):
        a, b = free['evaluation']['records'][cohort], grammar['evaluation']['records'][cohort]
        if a is None or b is None: continue
        a, b = a['raw'], b['raw']
        if a['cases_sha256'] != b['cases_sha256']:
            continue
        left = {r['case_id']: r for r in a['episodes']}
        right = {r['case_id']: r for r in b['episodes']}
        if len(left) != len(a['episodes']) or len(right) != len(b['episodes']):
            continue
        shared = sorted(set(left) & set(right))
        pairs = [(left[k], right[k]) for k in shared if left[k]['world_id'] == right[k]['world_id']]
        output.append({'cohort': cohort, 'paired_denominator': len(pairs),
            'free_only': sum(x['outcome']['correct'] and not y['outcome']['correct'] for x,y in pairs),
            'grammar_only': sum(y['outcome']['correct'] and not x['outcome']['correct'] for x,y in pairs),
            'both_correct': sum(x['outcome']['correct'] and y['outcome']['correct'] for x,y in pairs),
            'neither_correct': sum(not x['outcome']['correct'] and not y['outcome']['correct'] for x,y in pairs)})
    return output


def write_report(output, *, campaign=None):
    """Summarize saved results; raw receipts remain separate and authoritative."""
    root = Path(output); jobs = root / 'analysis' / 'jobs'
    rows, missing, errors, pairings = [], [], [], []
    if jobs.is_dir():
        for directory in sorted(jobs.iterdir()):
            if directory.is_symlink() or not directory.is_dir(): continue
            identity = _identity(directory)
            if any(value is None for value in identity.values()):
                missing.append({'job_id': directory.name, 'reason': 'job identity unavailable'})
            files = sorted(directory.glob('epoch-*.json'))
            grammar_file = directory / 'grammar-final.json'
            free, grammar = [], None
            if not files: missing.append({'job_id': directory.name, 'reason': 'epoch evaluations unavailable'})
            if not grammar_file.exists(): missing.append({'job_id': directory.name, 'reason': 'final grammar evaluation unavailable'})
            for path in files + ([grammar_file] if grammar_file.exists() else []):
                try:
                    if path.is_symlink() or path.stat().st_size > 512 << 20:
                        raise ValueError('linked or oversized report')
                    data = json.loads(path.read_bytes())
                    evaluation = data['evaluation']; grammar_mode = path == grammar_file
                    if data.get('grammar') is not grammar_mode: raise ValueError('grammar flag disagrees with filename')
                    if evaluation['cause_status'] != 'CAUSE_UNRESOLVED': raise ValueError('invalid cause status')
                    match = re.fullmatch(r'epoch-(\d+)\.json', path.name)
                    if not grammar_mode and not match: raise ValueError('invalid epoch filename')
                    epoch = int(match.group(1)) if match else None
                    target = grammar if grammar_mode else free
                    if grammar_mode: grammar = data
                    else: target.append(data)
                    for cohort, metrics in evaluation['metrics'].items():
                        if metrics is None:
                            missing.append({'job_id': directory.name, 'cohort': cohort, 'reason': 'cohort unavailable'})
                            continue
                        row = {'job_id': directory.name, **identity, 'epoch': epoch, 'update': data['update'],
                               'checkpoint': data['checkpoint'], 'mode': 'grammar' if grammar_mode else 'free', 'cohort': cohort}
                        values = metrics['all']
                        row.update({k: values.get(k) for k in _COLUMNS if k not in row})
                        repriced = values.get('fixed_policy_repricing', {}).get('values', [])
                        for column, item in zip(('cost_0_utility', 'cost_01_utility', 'cost_05_utility', 'cost_1_utility'), repriced):
                            denominator = item.get('utility_denominator')
                            if denominator:
                                row[column] = item['utility_numerator'] / denominator
                        calibration = values.get('calibration', {})
                        row['confidence_count'], row['brier'] = calibration.get('count'), calibration.get('brier')
                        rows.append(row)
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    errors.append({'path': str(path), 'message': str(exc)})
            if grammar:
                matches = [data for data in free if data['checkpoint'] == grammar['checkpoint'] and data['update'] == grammar['update']]
                if len(matches) == 1:
                    try:
                        pairs = _matched(matches[0], grammar)
                        if not pairs: missing.append({'job_id': directory.name, 'reason': 'matched final cohorts unavailable'})
                        pairings.extend({'job_id': directory.name, **pair} for pair in pairs)
                    except (ValueError, KeyError, TypeError, AttributeError) as exc:
                        errors.append({'path': str(grammar_file), 'message': str(exc)})
                else: missing.append({'job_id': directory.name, 'reason': 'matching free final checkpoint unavailable or ambiguous'})
    else:
        missing.append({'reason': 'jobs directory unavailable'})
    groups = defaultdict(set)
    for row in rows:
        if row['mode'] == 'free' and row['mechanism'] == 'ce' and row['method'] in ('uniform', 'blocked'):
            key = (row['profile'], row['seed'], row['epoch'], row['cohort'])
            groups[key].add(row['method'])
    blocks = [{'profile': k[0], 'seed': k[1], 'epoch': k[2], 'cohort': k[3],
               'methods_present': sorted(v), 'methods_missing': sorted({'uniform','blocked'}-v),
               'complete': v == {'uniform','blocked'}} for k,v in sorted(groups.items(), key=lambda x: str(x[0]))]
    report = {'status': 'PARTIAL' if missing or errors else 'AVAILABLE' if rows else 'NOT_RUN',
              'cause_status': 'CAUSE_UNRESOLVED', 'rows': rows, 'paired_seed_blocks': blocks,
              'free_grammar': pairings, 'missing_inputs': missing, 'errors': errors,
              'retention': 'NOT_QUALIFIED: phase evidence not evaluated in this summary',
              'forward_transfer': 'NOT_QUALIFIED: matched fresh baseline required',
              'interpretation': 'Descriptive cohort results and matched controls only; no model promotion.'}
    if campaign and campaign.get('phase')=='continual':
        report['retention']=[{'job_id':r.get('job',{}).get('job_id'),'status':r['status'],
            'B_before':r.get('B_before'),'B_after':r.get('B_after'),'novel_B_acquisition':r.get('novel_B_acquisition'),
            'retention':r.get('retention',{'status':'NOT_QUALIFIED'})} for r in campaign.get('results',[])]
    analysis = root / 'analysis'; analysis.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    _write_derived(analysis / 'report.csv', buffer.getvalue().encode('utf8'))
    _write_derived(analysis / 'summary.json', canonical(report))
    lines = ['# V7.1 evidence summary', '', 'CAUSE UNRESOLVED', '', f"Status: {report['status']}",
             f"Cohort rows: {len(rows)}. Matched free/grammar cohorts: {len(pairings)}.",
             f"Missing inputs: {len(missing)}. Report errors: {len(errors)}.", '', report['interpretation'],
             json.dumps(report['retention']) if isinstance(report['retention'],list) else report['retention'], report['forward_transfer'], '',
             'See report.csv for per-cohort measurements and summary.json for paired comparisons and missing inputs.']
    _write_derived(analysis / 'report.md', ('\n'.join(lines)+'\n').encode('utf8'))
    return report
