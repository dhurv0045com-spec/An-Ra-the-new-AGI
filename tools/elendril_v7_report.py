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


def _reference_datasets(root):
    from anra_v5.elendril.v6_data import read_dataset
    from anra_v5.elendril.v7_diagnostics import _cohorts
    available, errors = {}, []
    for name in ('monitor', 'validation', 'confirmation', 'train-monitor'):
        path = Path(root) / 'data' / (name + '.json')
        if not path.is_file():
            continue
        try:
            data = read_dataset(path)
            available[data.identity] = {label: {c.case_id: c for c in cases}
                                        for label, cases in _cohorts(data).items()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'reference_dataset': name, 'message': str(exc)})
    return available, errors


def _reference_traces(diagnostic, cohorts):
    """Compare actual public states to a reference algorithm, not an optimality proof."""
    from dataclasses import replace
    from anra_v5.elendril.cognition import CognitiveState
    from anra_v5.elendril.v6_data import public_action
    result = {}
    for label in ('interactive', 'supplied', 'interactive-swap', 'supplied-swap'):
        record = diagnostic.get('records', {}).get(label)
        if record is None:
            continue
        raw = record['raw']
        for episode in raw['episodes']:
            case = cohorts.get(label, {}).get(episode['case_id'])
            if case is None:
                continue
            state = CognitiveState(case.goal).observe(case.environment.initial)
            events = list(case.environment.initial)
            events.extend(e for values in case.environment.channels.values() for e in values)
            if case.memory is not None:
                events.extend(case.memory.events)
            by_id = {event.identity: event for event in events}
            channels = list(case.environment.channels)
            memory_available = case.memory is not None and raw.get('memory_mode', 'on') != 'off'
            requests, released, post_release = 0, False, False
            mismatch = None
            decisions = episode['trace']['decisions']
            for index, decision in enumerate(decisions):
                if released:
                    post_release = True
                usable_channels = tuple(channels) if requests < raw['budget']['requests'] else ()
                usable_memory = memory_available and requests < raw['budget']['requests']
                expected = public_action(state, usable_channels, case.surface, usable_memory)
                actual = (decision['operation'], decision['argument'])
                if mismatch is None and expected != actual:
                    mismatch = {'decision_index': index, 'index_base': 0,
                                'expected': {'operation': expected[0], 'argument': expected[1]},
                                'actual': {'operation': actual[0], 'argument': actual[1]}}
                evidence = decision['evidence']
                if evidence:
                    state = state.observe(tuple(by_id[identity] for identity in evidence))
                    released = True
                if actual[0] == 'QUERY' and actual[1] in channels:
                    channels.remove(actual[1]); requests += 1
                elif actual[0] == 'RETRIEVE':
                    memory_available = False; requests += 1
                elif actual[0] in ('DELIBERATE', 'REVISE'):
                    state = replace(state, hypothesis=actual[1])
            original_label = label.replace('supplied', 'interactive', 1)
            original = cohorts.get(original_label, {}).get(episode['case_id'])
            exogenous = bool(label.startswith('supplied') and original and original.environment.channels)
            result[(label, episode['case_id'])] = {'status': 'AVAILABLE', 'first_disagreement': mismatch,
                'exogenous_evidence_supplied': exogenous,
                'evidence_release_observed': released, 'post_release_step_observed': post_release,
                'invalid_action_observed': episode['trace']['status'] == 'invalid_action'}
    return result


def scientific_report(output, summary=None, *, reference_root=None) -> dict:
    """Describe paired raw observations; these comparisons do not identify a cause."""
    root = Path(output)
    summary = summarize(root) if summary is None else summary
    # One source receipt appears twice in summary rows, once per condition.
    receipts = {row['receipt']: row for row in summary['rows']}
    cases, comparisons, errors, probes = [], [], [], []
    datasets, reference_errors = _reference_datasets(reference_root if reference_root is not None else root)
    errors.extend(reference_errors)
    for name, meta in sorted(receipts.items()):
        try:
            path = root / name
            if path.is_symlink() or path.stat().st_size > _MAX_REPORT:
                raise ValueError('linked or oversized diagnostic report')
            diagnostic = json.loads(path.read_bytes())['diagnosis']
            probes.extend({'job_id':meta['job_id'],'seed':meta['seed'],'method':meta['method'],
                           'cohort':meta['cohort'],'update':meta['update'],'receipt':name,**probe}
                          for probe in diagnostic.get('token_probes',[]))
            cohorts = datasets.get(meta['data_identity'])
            reference = _reference_traces(diagnostic, cohorts) if cohorts is not None else {}
            episodes = diagnostic.get('episodes', [])
            if not isinstance(episodes, list):
                raise ValueError('diagnosis episodes must be a list')
            grouped = defaultdict(list)
            seen = set()
            for item in episodes:
                intervention, case_id = item['cohort'], item['case_id']
                if not isinstance(intervention, str) or not isinstance(case_id, str):
                    raise ValueError('invalid case identity or intervention')
                if (intervention, case_id) in seen:
                    raise ValueError('duplicate case in intervention')
                seen.add((intervention, case_id))
                a, b = item.get('interactive_correct'), item.get('supplied_correct')
                if a is not None and type(a) is not bool or b is not None and type(b) is not bool:
                    raise ValueError('case outcome must be boolean or null')
                prompt = item.get('queried_then_prompt_identical')
                if prompt is not None and type(prompt) is not bool:
                    raise ValueError('prompt identity must be boolean or null')
                raw_case=next((r for r in diagnostic.get('records',{}).get(intervention,{}).get('raw',{}).get('episodes',[])
                               if r['case_id']==case_id),{})
                row = {'job_id': meta['job_id'], 'seed': meta['seed'], 'method': meta['method'],
                       'update': meta['update'], 'cohort': meta['cohort'],
                       'data_identity': meta['data_identity'], 'checkpoint': meta['checkpoint'],
                       'intervention': intervention, 'case_id': case_id,
                       'kind':raw_case.get('kind'),'family':raw_case.get('family'),
                       'symptom': item.get('symptom'),
                       'first_observed_divergence': item.get('first_observable_divergence'),
                       'queried_then_prompt_identical': prompt,
                       'interactive_correct': a, 'supplied_correct': b, 'receipt': name,
                       'engineered_reference': {label: reference.get((label, case_id), {'status': 'UNAVAILABLE'})
                           for label in (intervention, intervention.replace('interactive', 'supplied', 1))}}
                grouped[intervention].append(row)
            # Publish a receipt's rows only after validating all of them.
            for intervention, entries in sorted(grouped.items()):
                cases.extend(entries)
                observed = [e for e in entries if e['interactive_correct'] is not None and e['supplied_correct'] is not None]
                counts = {'both_correct': 0, 'interactive_only': 0, 'supplied_only': 0, 'neither_correct': 0}
                for e in observed:
                    key = ('both_correct' if e['interactive_correct'] and e['supplied_correct'] else
                           'interactive_only' if e['interactive_correct'] else
                           'supplied_only' if e['supplied_correct'] else 'neither_correct')
                    counts[key] += 1
                comparisons.append({'job_id': meta['job_id'], 'seed': meta['seed'], 'method': meta['method'],
                    'update': meta['update'], 'cohort': meta['cohort'], 'data_identity': meta['data_identity'],
                    'intervention': intervention, 'paired_denominator': len(observed),
                    'unavailable_outcome_pairs': len(entries) - len(observed), **counts,
                    'interactive_correct': counts['both_correct'] + counts['interactive_only'],
                    'supplied_correct': counts['both_correct'] + counts['supplied_only'], 'receipt': name})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append({'receipt': name, 'error_type': type(exc).__name__, 'message': str(exc)})
    # Follow the same case in the same dataset, job, cohort and intervention only.
    trajectories = defaultdict(list)
    for row in cases:
        if row['cohort'] not in ('monitor', 'training'):
            continue
        key = (row['job_id'], row['seed'], row['method'], row['cohort'], row['data_identity'],
               row['intervention'], row['case_id'])
        trajectories[key].append(row)
    transitions = []
    for key, points in sorted(trajectories.items()):
        points.sort(key=lambda row: row['update'])
        for before, after in zip(points, points[1:]):
            if before['update'] == after['update']:
                errors.append({'receipt': after['receipt'], 'message': 'duplicate case/update; transition omitted'})
                continue
            for condition in ('interactive', 'supplied'):
                a, b = before[condition + '_correct'], after[condition + '_correct']
                if a is None or b is None:
                    label = 'UNAVAILABLE'
                elif not a and b:
                    label = 'ACQUISITION_OBSERVED'
                elif a and not b:
                    label = 'LOSS_OBSERVED'
                elif a:
                    label = 'CORRECT_AT_BOTH'
                else:
                    label = 'INCORRECT_AT_BOTH'
                transitions.append({'job_id': key[0], 'seed': key[1], 'method': key[2], 'cohort': key[3],
                    'data_identity': key[4], 'intervention': key[5], 'case_id': key[6], 'condition': condition,
                    'from_update': before['update'], 'to_update': after['update'],
                    'from_correct': a, 'to_correct': b, 'observation': label})
    return {'status': 'PARTIAL' if errors else 'AVAILABLE' if cases else 'NOT_RUN',
        'cause_status': 'UNRESOLVED',
        'symptom_scope': 'Observed outcomes, first recorded action divergence, and prompt identity are descriptive symptoms.',
        'competing_hypotheses': ['query selection or evidence access', 'use of supplied evidence',
                                 'representation or answer decoding', 'case or sampling effects'],
        'demonstrated_cause': None,
        'diagnostic_scope': {
            'query_selection':'Needed-query flags and observed QUERY decisions; an omitted query is a symptom, not its cause.',
            'observation_incorporation':'Released event IDs, subsequent model steps and prompt identity show engine delivery, not neural understanding.',
            'evidence_extraction':'Supplied-payload outcomes and changed-payload pairs test behavioral use; internal extraction remains unresolved.',
            'answer_generation':'Raw actions, invalid-action flags and value/EOS probes separate observations; teacher forcing is not successful free decoding.',
            'causal_updating':'Revision cases and engineered-reference disagreements are observable; no independently demonstrated causal-update mechanism.',
            'training_retention':'Same-case correctness transitions show acquisition/loss observations; controlled retention and new-skill transfer remain unqualified.'},
        'reference_limit': 'The engineered public-action reference is not a uniquely optimal path, a learned policy, or a root cause; correctness can coexist with disagreement.',
        'cause_limit': 'These observations do not establish a controlled, replicated mechanism correction.',
        'intervention_limit': 'Supplying evidence changes the public environment; paired outcome differences alone do not establish why.',
        'causal_updating_and_retention': 'UNAVAILABLE: case observations alone do not demonstrate causal updating or retention.',
        'next_discriminating_experiments': [
            'If needed queries are omitted, compare actual-payload supply against an equal-budget sham release on the same frozen checkpoints and held-out paired worlds across seeds.',
            'If supplied cases fail, inspect the measured first-value and teacher-forced EOS probes; next compare ordinary decoding with gold-free action-grammar-constrained decoding at matched work across seeds.',
            'If training fits but held-out evidence swaps fail, preregister a conventional pointer/copy competitor with matched parameters, input exposure and measured compute.',
            'If correct cases become incorrect, branch from the last successful checkpoint into matched replay/no-replay continuations with independently measured new-skill acquisition before asserting retention.'],
        'answer_generation_probes':probes,
        'case_observations': cases, 'paired_interventions': comparisons, 'case_transitions': transitions,
        'errors': errors}


def _scientific_markdown(report):
    lines = ['# V7 scientific observations', '', f"CAUSE {report['cause_status']}", '',
             report['symptom_scope'], report['cause_limit'], report['reference_limit'], report['intervention_limit'],
             report['causal_updating_and_retention'], '', '## Paired outcomes', '']
    if not report['paired_interventions']:
        lines.append('No paired case outcomes available.')
    for p in report['paired_interventions']:
        lines.append(f"- {p['job_id']} {p['cohort']} update {p['update']} {p['intervention']}: "
                     f"interactive {p['interactive_correct']}/{p['paired_denominator']}; "
                     f"supplied {p['supplied_correct']}/{p['paired_denominator']}; "
                     f"unavailable pairs {p['unavailable_outcome_pairs']}.")
    lines.extend(['', 'Case-level divergence, prompt identity and matched trajectories are in scientific_report.json.',
                  '', '## Competing hypotheses', ''])
    lines.extend('- ' + item for item in report['competing_hypotheses'])
    lines.extend(['', '## Conditional next experiments', ''])
    lines.extend('- ' + item for item in report['next_discriminating_experiments'])
    return '\n'.join(lines) + '\n'


def write_summary(output, *, reference_root=None) -> dict:
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
    science = scientific_report(root, report, reference_root=reference_root)
    _write_derived(analysis / 'scientific_report.json', canonical(science))
    _write_derived(analysis / 'scientific_report.md', _scientific_markdown(science).encode('utf-8'))
    return report
