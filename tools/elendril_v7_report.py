"""Derive separate-cohort V7 trajectories from retained diagnostic receipts."""
from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path
import re

from anra_v5.elendril.v7_protocol import METHODS
from tools.elendril_v6_export import _write_derived, canonical

_FIELDS = ('job_id', 'seed', 'method', 'update', 'cohort', 'condition', 'data_identity',
           'checkpoint', 'nonnull_correct', 'nonnull_total', 'nonnull_accuracy',
           'query_needed', 'needed_queried', 'query_recall', 'answer_entropy_bits', 'receipt')
_MAX_REPORT = 64 << 20


def _fraction(numerator, denominator):
    if type(numerator) not in (int, float) or type(denominator) not in (int, float) or denominator <= 0:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / denominator


def summarize(output) -> dict:
    """Read reports without modifying raw data; retain errors and incomplete blocks."""
    root = Path(output)
    jobs = root / 'analysis' / 'jobs'
    rows, errors = [], []
    if not jobs.is_dir():
        return {'status': 'NOT_RUN', 'rows': [], 'paired_blocks': [], 'errors': []}
    for directory in sorted(jobs.iterdir()):
        match = re.fullmatch(r'(uniform|blocked|interleaved)-s(\d+)', directory.name)
        if not match or not directory.is_dir() or directory.is_symlink():
            continue
        method, seed = match.group(1), int(match.group(2))
        files = []
        for pattern, cohort in [('diagnostic-u*.json', 'monitor'), ('development-u*.json', 'development'),
                                ('training-u*.json', 'training'), ('confirmation.json', 'confirmation')]:
            files.extend((path, cohort) for path in sorted(directory.glob(pattern)))
        for path, cohort in files:
            try:
                if path.is_symlink() or path.stat().st_size > _MAX_REPORT:
                    raise ValueError('linked or oversized diagnostic report')
                report = json.loads(path.read_bytes())
                update = report['update']
                if type(update) is not int or update < 0:
                    raise ValueError('invalid update')
                filename_update = re.search(r'-u(\d+)\.json$', path.name)
                if filename_update and int(filename_update.group(1)) != update:
                    raise ValueError('filename and report update disagree')
                for condition in ('interactive', 'supplied'):
                    finding = report['diagnosis']['findings'].get(condition, {})
                    if not finding or 'nonnull_total' not in finding:
                        continue
                    n, total = finding.get('nonnull_correct'), finding.get('nonnull_total')
                    needed, queried = finding.get('query_needed'), finding.get('needed_queried')
                    rows.append({'job_id': directory.name, 'seed': seed, 'method': method,
                        'update': update, 'cohort': cohort, 'condition': condition,
                        'data_identity': report['data_identity'], 'checkpoint': report['checkpoint'],
                        'nonnull_correct': n, 'nonnull_total': total, 'nonnull_accuracy': _fraction(n, total),
                        'query_needed': needed, 'needed_queried': queried,
                        'query_recall': _fraction(queried, needed),
                        'answer_entropy_bits': finding.get('answer_entropy_bits'),
                        'receipt': path.relative_to(root).as_posix()})
            except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
                errors.append({'path': str(path), 'error_type': type(exc).__name__, 'message': str(exc)})
    rows.sort(key=lambda row: (row['cohort'], row['condition'], row['seed'], row['update'], row['method']))
    groups = defaultdict(set)
    for row in rows:
        key = (row['seed'], row['update'], row['cohort'], row['condition'], row['data_identity'])
        groups[key].add(row['method'])
    blocks = []
    for (seed, update, cohort, condition, data_identity), present in sorted(groups.items()):
        blocks.append({'seed': seed, 'update': update, 'cohort': cohort, 'condition': condition,
                       'data_identity': data_identity, 'methods_present': sorted(present),
                       'methods_missing': sorted(set(METHODS) - present),
                       'complete': present == set(METHODS)})
    return {'status': 'COMPLETE' if rows and not errors else 'PARTIAL' if rows or errors else 'NOT_RUN',
            'rows': rows, 'paired_blocks': blocks, 'errors': errors,
            'interpretation': 'Rows are separate by data cohort, condition, seed and update; no mixed-cohort pooling or causal inference.'}


def write_summary(output) -> dict:
    """Write text reports first; plots are optional and never gate raw CSV."""
    root = Path(output)
    report = summarize(root)
    analysis = root / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(report['rows'])
    _write_derived(analysis / 'trajectory.csv', stream.getvalue().encode('utf-8'))
    charts = {'status': 'NOT_RUN', 'paths': []}
    if report['rows']:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            directory = root / 'charts'
            directory.mkdir(exist_ok=True)
            for cohort, condition in sorted({(r['cohort'], r['condition']) for r in report['rows']}):
                subset = [r for r in report['rows'] if r['cohort'] == cohort and r['condition'] == condition]
                fig, ax = plt.subplots()
                for method, seed in sorted({(r['method'], r['seed']) for r in subset}):
                    line = sorted((r for r in subset if r['method'] == method and r['seed'] == seed
                                   and r['nonnull_accuracy'] is not None), key=lambda r: r['update'])
                    if line:
                        ax.plot([r['update'] for r in line], [r['nonnull_accuracy'] for r in line],
                                marker='o', label=f'{method} seed {seed}')
                ax.set(title=f'{cohort} / {condition}', xlabel='Update', ylabel='Non-null accuracy', ylim=(0, 1))
                if ax.lines: ax.legend(fontsize='small')
                fig.tight_layout()
                path = directory / f'{cohort}-{condition}.png'
                fig.savefig(path, dpi=120)
                plt.close(fig)
                charts['paths'].append(str(path))
            charts['status'] = 'CREATED'
        except Exception as exc:
            charts.update(status='NOT_QUALIFIED', error_type=type(exc).__name__, message=str(exc))
    report['charts'] = charts
    _write_derived(analysis / 'summary.json', canonical(report))
    return report
