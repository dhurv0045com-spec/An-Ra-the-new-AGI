"""Compact, source-agnostic summaries of saved V7.16 worker evaluations.

Raw receipts remain separate and authoritative. The summary never promotes a
model: it reports cohort measurements, per-arm aggregates against the
preregistered decision rules, and the exact set of missing inputs. Causal
status stays CAUSE_UNRESOLVED until a controlled replicated intervention
isolates a mechanism.
"""
import csv
from collections import defaultdict
import io
import json
from pathlib import Path
import re

from anra_v5.elendril.codec import canonical
from tools.elendril_v6_export import _write_derived

_COLUMNS = ('job_id', 'arm', 'profile', 'mechanism', 'seed', 'epoch', 'update', 'checkpoint',
            'mode', 'cohort', 'episodes', 'correct', 'accuracy', 'invalid', 'utility',
            'verified_queries', 'answer_entropy_bits', 'confidence_count', 'brier')
# Cohorts carried into the per-arm aggregate table (E01, E03, E05, E11 proxies).
_AGGREGATE_COHORTS = ('base', 'memory-manipulations', 'hop-1', 'hop-2', 'hop-3', 'hop-4',
                      'balanced-payload', 'robustness-row-order')


def _identity(directory):
    for path in sorted(directory.glob('*-request.json')):
        try:
            job = json.loads(path.read_bytes()).get('job', {})
            if job.get('job_id') == directory.name:
                return {key: job.get(key) for key in ('arm', 'profile', 'mechanism', 'seed')}
        except (OSError, ValueError, AttributeError):
            continue
    match = re.fullmatch(r'(S3|S8)-([a-z0-9-]+)-s(\d+)', directory.name)
    if match:
        return {'arm': None, 'profile': match.group(1), 'mechanism': match.group(2),
                'seed': int(match.group(3))}
    return {'arm': None, 'profile': None, 'mechanism': None, 'seed': None}


def write_summary(output, *, campaign=None):
    """Summarize saved V7.16 results; per-arm aggregates use final evaluations."""
    root = Path(output)
    jobs_root = root / 'analysis' / 'jobs'
    registered = list(campaign.get('registered_job_ids', [])) if campaign else []
    rows, missing, errors, arms = [], [], [], defaultdict(lambda: defaultdict(list))
    if jobs_root.is_dir():
        for directory in sorted(jobs_root.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                continue
            identity = _identity(directory)
            if any(value is None for value in identity.values()):
                missing.append({'job_id': directory.name,
                                'reason': 'job identity unavailable'})
            final_epoch = None
            for path in sorted(directory.glob('epoch-*.json')):
                try:
                    if path.is_symlink() or path.stat().st_size > 512 << 20:
                        raise ValueError('linked or oversized report')
                    data = json.loads(path.read_bytes())
                    evaluation = data['evaluation']
                    if data.get('grammar') is not False:
                        raise ValueError('epoch report must be the free-decoding evaluation')
                    if evaluation['cause_status'] != 'CAUSE_UNRESOLVED':
                        raise ValueError('invalid cause status')
                    epoch = int(re.fullmatch(r'epoch-(\d+)\.json', path.name).group(1))
                    if final_epoch is None or epoch > final_epoch:
                        final_epoch = (epoch, data)
                    for cohort, metrics in evaluation['metrics'].items():
                        if metrics is None:
                            missing.append({'job_id': directory.name, 'cohort': cohort,
                                            'reason': 'cohort unavailable'})
                            continue
                        rows.append(_row(directory.name, identity, epoch, data, cohort,
                                         metrics, mode='free'))
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    errors.append({'path': str(path), 'message': str(exc)})
            grammar_file = directory / 'grammar-final.json'
            if not grammar_file.exists():
                if directory.name in registered:
                    missing.append({'job_id': directory.name,
                                    'reason': 'final grammar evaluation unavailable'})
                continue
            try:
                if grammar_file.is_symlink() or grammar_file.stat().st_size > 512 << 20:
                    raise ValueError('linked or oversized report')
                grammar = json.loads(grammar_file.read_bytes())
                if grammar.get('grammar') is not True:
                    raise ValueError('grammar flag disagrees with filename')
                if grammar['evaluation']['cause_status'] != 'CAUSE_UNRESOLVED':
                    raise ValueError('invalid cause status')
                for cohort, metrics in grammar['evaluation']['metrics'].items():
                    if metrics is None:
                        missing.append({'job_id': directory.name, 'cohort': cohort,
                                        'reason': 'cohort unavailable'})
                        continue
                    rows.append(_row(directory.name, identity, None, grammar, cohort,
                                     metrics, mode='grammar'))
                arms[identity['arm'] or identity['mechanism'] or directory.name]\
                    [directory.name] = grammar['evaluation']['metrics']
                if final_epoch is None and directory.name in registered:
                    missing.append({'job_id': directory.name,
                                    'reason': 'final epoch evaluation unavailable'})
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                errors.append({'path': str(grammar_file), 'message': str(exc)})
    else:
        missing.append({'reason': 'jobs directory unavailable'})
    aggregate = {}
    seen_finals = set()
    for arm, per_job in sorted(arms.items()):
        seen_finals.update(per_job)
        entry = {"jobs": {}}
        for job_id, metrics in sorted(per_job.items()):
            entry["jobs"][job_id] = {
                cohort: {"correct": metrics[cohort]['all']['correct'],
                         "episodes": metrics[cohort]['all']['episodes']}
                for cohort in _AGGREGATE_COHORTS
                if cohort in metrics and metrics[cohort] is not None
                and metrics[cohort].get('all')}
        for cohort in _AGGREGATE_COHORTS:
            correct = sum(row["correct"] for row in entry["jobs"].values()
                          if cohort in row)
            episodes = sum(row["episodes"] for row in entry["jobs"].values()
                           if cohort in row)
            if episodes:
                entry[cohort] = {'correct': correct, 'episodes': episodes,
                                 'accuracy': correct / episodes}
        base = entry.get('base')
        if base and base['episodes'] % 112 == 0:
            entry['confirmation_per_seed'] = base['correct'] // (base['episodes'] // 112)
            entry['confirmation_total'] = f"{base['correct']}/{base['episodes']}"
        aggregate[arm] = entry
    complete = (bool(registered) and all(job in seen_finals for job in registered)
                and not [row for row in missing if row.get('job_id') in registered]
                and not errors)
    report = {'status': 'AVAILABLE' if complete and rows else
              'PARTIAL' if rows or missing or errors else 'NOT_RUN',
              'cause_status': 'CAUSE_UNRESOLVED',
              'registered_jobs': registered, 'rows': rows, 'arm_aggregates': aggregate,
              'missing_inputs': missing, 'errors': errors,
              'interpretation': 'Descriptive cohort results against preregistered '
                                'thresholds; no model promotion and no causal claim.'}
    analysis = root / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    _write_derived(analysis / 'report.csv', buffer.getvalue().encode('utf8'))
    _write_derived(analysis / 'summary.json', canonical(report))
    _write_derived(analysis / 'report.md', (
        '# Elendril V7.16 evidence summary\n\nCAUSE UNRESOLVED\n\n'
        f"Status: {report['status']}. Cohort rows: {len(rows)}. "
        f"Missing inputs: {len(missing)}. Errors: {len(errors)}.\n\n"
        + json.dumps(aggregate, indent=2, sort_keys=True) + '\n\n'
        + report['interpretation'] + '\n').encode('utf8'))
    return report


def _row(job_id, identity, epoch, data, cohort, metrics, *, mode):
    values = metrics['all']
    row = {'job_id': job_id, **identity, 'epoch': epoch, 'update': data['update'],
           'checkpoint': data['checkpoint'], 'mode': mode, 'cohort': cohort}
    row.update({key: values.get(key) for key in _COLUMNS if key not in row})
    calibration = values.get('calibration', {})
    row['confidence_count'] = calibration.get('count')
    row['brier'] = calibration.get('brier')
    return row
