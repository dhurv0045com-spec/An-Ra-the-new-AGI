"""Stable V7 measurement and independent replay of historical V6 receipts."""
import copy
import hashlib
import math
import sys
from pathlib import Path

from . import v6_evaluation as v6
from .codec import canonical
from .cognition import Budget, OperationCandidate

SCHEMA = 'elendril-v7-evaluation-v1'


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def stable_metrics(raw):
    """Return V6 metric shape with exact tenths represented explicitly."""
    result = copy.deepcopy(raw['metrics'])
    for group in [result['all'], *result['by_kind'].values(), *result['by_family'].values()]:
        numerator = 10 * group['correct'] - group['verified_queries']
        group['utility_numerator'] = numerator
        group['utility_denominator'] = 10
        group['utility'] = numerator / 10
    return result


def evaluate(data, policy, **kwargs):
    raw = v6.evaluate(data, policy, **kwargs)
    return {'schema': SCHEMA, 'source_sha256': source_identity(), 'raw': raw,
            'metrics': stable_metrics(raw)}


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('non-finite receipt number')
    if isinstance(value, dict):
        for item in value.values(): _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _finite(item)


def _strip_utilities(metrics):
    result = copy.deepcopy(metrics)
    for group in [result['all'], *result['by_kind'].values(), *result['by_family'].values()]:
        del group['utility']
    return result


def _check_utility(group):
    n, correct, queries, value = (group[k] for k in ('episodes', 'correct', 'verified_queries', 'utility'))
    if any(type(x) is not int or x < 0 for x in (n, correct, queries)) or type(value) not in (int, float):
        raise ValueError('invalid aggregate')
    # Each row is formed with two binary64 operations; reduction has at most
    # n additions. gamma_n times the sum of magnitudes bounds naive summation;
    # extra terms cover row construction and the final rational conversion.
    eps = sys.float_info.epsilon
    if n * eps >= .5: raise ValueError('unbounded reduction')
    bound = ((n * eps / (1 - n * eps)) + 4 * eps) * (n + queries / 10 + 1)
    exact = (10 * correct - queries) / 10
    if not math.isfinite(value) or abs(value - exact) > bound:
        raise ValueError('aggregate utility outside floating-point reduction bound')


def verify_legacy(data, raw):
    """Replay every recorded action and allow only bounded aggregate reduction drift."""
    _finite(raw)
    indexed = {row['case_id']: row for row in raw['episodes']}
    if len(indexed) != len(raw['episodes']): raise ValueError('duplicate cases')

    class ReplayFactory:
        mode = raw['policy']
        representation = raw['decoder_schema']
        def for_case(self, case, *, memory_available=None):
            steps = indexed[case.case_id]['trace']['decisions']
            if raw['memory_mode'] == 'prefetch' and case.memory is not None: steps = steps[1:]
            class Replay:
                model_identity = raw['model_identity']
                origin = 'privileged_oracle' if raw['policy'] == 'oracle' else 'model'
                def __init__(self): self.index, self.prompts = 0, []
                def propose(self, state, legal, channels, positions):
                    if self.index == len(steps): return None
                    step = steps[self.index]; self.index += 1
                    self.origin = step['origin']
                    if self.origin == 'model':
                        operation, argument = v6.parse_action(tuple(step['raw_tokens']), channels, raw['decoder_schema'])
                        if raw['decoder_schema'] == 'utf8-raw':
                            from .v6_data import render_public
                            self.prompts.append(render_public(state, channels, case.surface, bool(memory_available) and 'RETRIEVE' in legal))
                        if (operation, argument) != (step['operation'], step['argument']):
                            raise ValueError('raw action mismatch')
                    elif self.origin in ('fixed_baseline', 'privileged_oracle') and raw['model_identity'] is None:
                        operation, argument = step['operation'], step['argument']
                    else: raise ValueError('invalid action origin')
                    return OperationCandidate(operation, argument, step['predicted_success'], step['model_positions'], step['model_calls'], tuple(step['raw_tokens']))
            return Replay()

    try:
        checked = v6.evaluate(data, ReplayFactory(), split=raw['split'], budget=Budget(**raw['budget']), memory_mode=raw['memory_mode'])
        for receipt in (raw, checked):
            metrics = receipt['metrics']
            for group in [metrics['all'], *metrics['by_kind'].values(), *metrics['by_family'].values()]: _check_utility(group)
        left, right = copy.deepcopy(raw), copy.deepcopy(checked)
        left['metrics'], right['metrics'] = _strip_utilities(left['metrics']), _strip_utilities(right['metrics'])
        if canonical(left) != canonical(right): raise ValueError('raw replay mismatch')
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError('malformed receipt') from exc
    return stable_metrics(checked)


def verify_record(data, record):
    _finite(record)
    if set(record) != {'schema', 'source_sha256', 'raw', 'metrics'} or record['schema'] != SCHEMA or record['source_sha256'] != source_identity():
        raise ValueError('V7 schema or source mismatch')
    metrics = verify_legacy(data, record['raw'])
    if canonical(record['metrics']) != canonical(metrics): raise ValueError('stable metrics mismatch')
    return metrics
