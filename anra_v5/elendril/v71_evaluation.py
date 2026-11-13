"""Replay-verifiable V7.1 cohort metrics; observational, not a mechanism claim."""
from collections import Counter, defaultdict
from itertools import combinations
import math

from .v7_measurement import evaluate, verify_record

SCHEMA = 'elendril-v71-evaluation-v1'
LIMITATIONS = 'Engineered interventions and paired outcomes are observational; no causal mechanism or promotion established.'


def _ratio(n, d):
    return n / d if d else None


def _metrics(cases, rows):
    case_by_id = {c.case_id: c for c in cases}
    def group(subset):
        nonnull = [r for r in subset if case_by_id[r['case_id']].truth.expected is not None]
        null = [r for r in subset if case_by_id[r['case_id']].truth.expected is None]
        queried = [r for r in subset if any(d['operation'] == 'QUERY' for d in r['trace']['decisions'])]
        needed = [r for r in subset if r['query_needed']]
        useful = sum(r['query_needed'] for r in queried)
        counts = Counter(r['trace']['answer'] for r in subset if r['trace']['status'] == 'answered')
        n = sum(counts.values())
        entropy = -math.fsum((v/n)*math.log2(v/n) for v in counts.values()) if n else None
        pairsets = defaultdict(list)
        for r in subset:
            if r['pair_id']:
                pairsets[r['pair_id']].append(r['outcome']['correct'])
        complete = [v for v in pairsets.values() if len(v) == 2]
        decisions = [d for r in subset for d in r['trace']['decisions']]
        # Mirror the independently replayed V6 cost rule: invalid terminal
        # QUERY attempts do not consume verified query cost.
        verified_queries = 0
        for row in subset:
            trace = row['trace']
            env = case_by_id[row['case_id']].environment
            for i, d in enumerate(trace['decisions']):
                if d['operation'] == 'QUERY' and d['argument'] in env.channels and not (
                        trace['status'] == 'invalid_action' and i == len(trace['decisions']) - 1):
                    verified_queries += 1
        correct = sum(bool(r['outcome']['correct']) for r in subset)
        confidence_rows = [(r['confidence'], int(r['outcome']['correct'])) for r in subset
                           if type(r['confidence']) in (int, float) and math.isfinite(r['confidence'])
                           and 0 <= r['confidence'] <= 1]
        bins = []
        for i in range(5):
            members = [(p, y) for p, y in confidence_rows if min(4, int(p * 5)) == i]
            bins.append({'index': i, 'lower': i / 5, 'upper': (i + 1) / 5,
                         'count': len(members),
                         'mean_confidence': math.fsum(p for p, _ in members) / len(members) if members else None,
                         'accuracy': _ratio(sum(y for _, y in members), len(members))})
        return {'episodes': len(subset), 'correct': correct, 'accuracy': _ratio(correct, len(subset)),
            'nonnull_total': len(nonnull), 'nonnull_correct': sum(bool(r['outcome']['correct']) for r in nonnull),
            'nonnull_accuracy': _ratio(sum(bool(r['outcome']['correct']) for r in nonnull), len(nonnull)),
            'null_total': len(null), 'correct_abstentions': sum(r['trace']['status'] == 'abstained' for r in null),
            'abstention_accuracy': _ratio(sum(r['trace']['status'] == 'abstained' for r in null), len(null)),
            'query_attempt_episodes': len(queried), 'query_needed': len(needed), 'needed_queried': useful,
            'query_precision': _ratio(useful, len(queried)), 'query_recall': _ratio(useful, len(needed)),
            'queries_attempted': sum(d['operation'] == 'QUERY' for d in decisions),
            'verified_queries': verified_queries, 'retrievals': sum(d['operation'] == 'RETRIEVE' for d in decisions),
            'model_calls': sum(d['model_calls'] for d in decisions),
            'model_positions': sum(d['model_positions'] for d in decisions),
            'invalid': sum(r['trace']['status'] == 'invalid_action' for r in subset),
            'utility_numerator': 10*correct - verified_queries, 'utility_denominator': 10,
            'fixed_policy_repricing': {'qualification': 'NOT cost adaptation; actions are unchanged',
                'values': [{'query_cost_numerator': n, 'query_cost_denominator': d,
                            'utility_numerator': d*correct - n*verified_queries, 'utility_denominator': d}
                           for n, d in ((0, 1), (1, 10), (1, 2), (1, 1))]},
            'calibration': {'status': 'AVAILABLE' if confidence_rows else 'UNAVAILABLE',
                'count': len(confidence_rows),
                'brier': math.fsum((p-y)**2 for p,y in confidence_rows) / len(confidence_rows) if confidence_rows else None,
                'bins': bins, 'qualification': 'Recorded action confidence; no fitting or oracle decisions.'},
            'answer_entropy_bits': entropy, 'both_correct_pairs': sum(all(v) for v in complete),
            'complete_pairs': len(complete), 'incomplete_pairs': len(pairsets)-len(complete)}
    return {'all': group(rows), 'by_kind': {k: group([r for r in rows if r['kind'] == k])
                for k in sorted({r['kind'] for r in rows})}}


def _comparisons(records):
    result = []
    for left, right in combinations(sorted(records), 2):
        if records[left] is None or records[right] is None:
            continue
        def indexed(record):
            grouped = defaultdict(list)
            for row in record['raw']['episodes']:
                grouped[row['world_id']].append(row)
            return grouped
        a, b = indexed(records[left]), indexed(records[right])
        common = sorted(set(a) & set(b))
        pairs = [(a[k][0], b[k][0]) for k in common if len(a[k]) == len(b[k]) == 1]
        result.append({'left': left, 'right': right, 'status': 'AVAILABLE' if pairs else 'UNAVAILABLE',
            'matching': 'unique shared world_id only; ambiguous repeated worlds omitted',
            'paired_denominator': len(pairs), 'ambiguous_worlds': len(common)-len(pairs),
            'left_only': sum(x['outcome']['correct'] and not y['outcome']['correct'] for x,y in pairs),
            'right_only': sum(y['outcome']['correct'] and not x['outcome']['correct'] for x,y in pairs),
            'both_correct': sum(x['outcome']['correct'] and y['outcome']['correct'] for x,y in pairs),
            'neither_correct': sum(not x['outcome']['correct'] and not y['outcome']['correct'] for x,y in pairs)})
    return result


def _within_cohort(records):
    """Compare explicitly paired panel variants without collapsing multiple variants."""
    output = []
    for name, record in records.items():
        if record is None:
            continue
        variants = defaultdict(lambda: defaultdict(list))
        for row in record['raw']['episodes']:
            panel, pair = row['panel'], row['pair_id']
            if not pair or not panel:
                continue
            if panel.startswith('payload-original-'): panel = 'payload-original'
            if panel.startswith('payload-alternative-'): panel = 'payload-alternative'
            variants[panel][pair].append(row)
        for left, right in combinations(sorted(variants), 2):
            common = set(variants[left]) & set(variants[right])
            pairs = [(variants[left][k][0], variants[right][k][0]) for k in sorted(common)
                     if len(variants[left][k]) == len(variants[right][k]) == 1]
            output.append({'cohort': name, 'left': left, 'right': right,
                'status': 'AVAILABLE' if pairs else 'UNAVAILABLE', 'paired_denominator': len(pairs),
                'ambiguous_pairs': len(common)-len(pairs),
                'left_only': sum(a['outcome']['correct'] and not b['outcome']['correct'] for a,b in pairs),
                'right_only': sum(b['outcome']['correct'] and not a['outcome']['correct'] for a,b in pairs),
                'both_correct': sum(a['outcome']['correct'] and b['outcome']['correct'] for a,b in pairs),
                'neither_correct': sum(not a['outcome']['correct'] and not b['outcome']['correct'] for a,b in pairs)})
    return output


def _references(suite, records):
    from tools.elendril_v7_report import _reference_traces
    result = {}
    for name, record in records.items():
        if record is None:
            result[name] = None
            continue
        cases = {'interactive': {case.case_id: case for case in suite[name]}}
        traces = _reference_traces({'records': {'interactive': record}}, cases)
        result[name] = {case_id: value for (_, case_id), value in sorted(traces.items())}
    return result


def _derive(suite, records):
    return {'metrics': {name: _metrics(suite[name], record['raw']['episodes']) if record is not None else None
                        for name, record in records.items()},
            'comparisons': _comparisons(records), 'within_cohort_comparisons': _within_cohort(records),
            'engineered_reference': _references(suite, records)}


def evaluate_suite(suite, policy, *, backend=None, conditions=None):
    """Evaluate supplied public cases; backend is reserved for optional future probes."""
    names = sorted(suite) if conditions is None else list(conditions)
    if len(names) != len(set(names)) or any(name not in suite for name in names):
        raise ValueError('unknown or duplicate suite condition')
    if getattr(policy, 'mode', None) == 'oracle':
        raise ValueError('truth-bearing oracle actors are not permitted')
    records = {name: evaluate(suite[name], policy) if suite[name] else None for name in names}
    return {'schema': SCHEMA, 'conditions': names, 'records': records, **_derive(suite, records),
            'cause_status': 'CAUSE_UNRESOLVED',
            'limitations': LIMITATIONS,
            'probe_status': 'NOT_RUN'}


def _equal(left, right, path=''):
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-14, abs_tol=1e-14) if path.endswith('answer_entropy_bits') else left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k], str(k)) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a,b,path) for a,b in zip(left,right))
    return type(left) is type(right) and left == right


def verify_suite(suite, report):
    if (set(report) != {'schema', 'conditions', 'records', 'metrics', 'comparisons', 'within_cohort_comparisons', 'engineered_reference', 'cause_status', 'limitations', 'probe_status'}
            or report.get('schema') != SCHEMA or report.get('cause_status') != 'CAUSE_UNRESOLVED'
            or report.get('probe_status') != 'NOT_RUN' or report.get('limitations') != LIMITATIONS):
        raise ValueError('unsupported evaluation or causal status')
    names = report['conditions']
    if len(names) != len(set(names)) or set(report['records']) != set(names) or any(n not in suite for n in names):
        raise ValueError('suite condition mismatch')
    for name in names:
        record = report['records'][name]
        if not suite[name]:
            if record is not None: raise ValueError('unexpected record for unavailable condition')
        else:
            if record is None: raise ValueError('missing condition record')
            verify_record(suite[name], record)
    derived = _derive(suite, report['records'])
    if any(not _equal(derived[key], report[key]) for key in ('metrics', 'comparisons', 'within_cohort_comparisons', 'engineered_reference')):
        raise ValueError('derived suite metrics or comparisons differ')
    return {'status': 'VERIFIED', 'cause_status': 'CAUSE_UNRESOLVED', 'conditions': len(names)}


def summarize_reports(reports):
    """Return individual measurements without pooling cohorts or model runs."""
    return {'status': 'AVAILABLE' if reports else 'NOT_RUN', 'cause_status': 'CAUSE_UNRESOLVED',
            'runs': [{'metrics': r['metrics'], 'comparisons': r['comparisons'],
                      'within_cohort_comparisons': r['within_cohort_comparisons']} for r in reports]}


def acquisition_gate(result, minimum=.8):
    if not 0 <= minimum <= 1: raise ValueError('invalid acquisition threshold')
    if not isinstance(result, dict): return {'status': 'NOT_QUALIFIED'}
    total, correct = result.get('nonnull_total'), result.get('nonnull_correct')
    if type(total) is not int or type(correct) is not int or total <= 0 or not 0 <= correct <= total:
        return {'status': 'NOT_QUALIFIED'}
    return {'status': 'ACQUIRED' if correct / total >= minimum else 'NOT_ACQUIRED',
            'correct': correct, 'total': total, 'threshold': minimum}


def retention(before, after):
    """Require old skill before and new skill after; no forward-transfer claim."""
    if not isinstance(after, dict): return {'status': 'NOT_QUALIFIED'}
    prior, new = acquisition_gate(before), acquisition_gate(after.get('new_skill'))
    current = after.get('old_skill')
    now = acquisition_gate(current)
    if prior['status'] != 'ACQUIRED' or new['status'] != 'ACQUIRED' or now['status'] == 'NOT_QUALIFIED':
        return {'status': 'NOT_QUALIFIED', 'old_before': prior, 'new_after': new, 'old_after': now}
    return {'status': 'OBSERVED', 'old_before': prior, 'new_after': new, 'old_after': now,
            'delta': current['nonnull_correct']/current['nonnull_total'] - before['nonnull_correct']/before['nonnull_total'],
            'forward_transfer': 'NOT_QUALIFIED: matched fresh baseline required'}
