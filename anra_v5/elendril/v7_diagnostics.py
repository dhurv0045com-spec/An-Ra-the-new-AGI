"""Matched public-evidence intervention, independent replay and bounded token probes.

This is a diagnostic, not a causal identification or a promotion gate. Truth is
joined only by evaluation and by the explicitly labelled post-hoc token probe.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import math
from pathlib import Path

from .codec import canonical, digest
from .cognition import Budget, CognitiveState, PublicEnvironment
from .representation import UTF8, SpecialToken
from .v6_data import panel_cases, render_public
from .v6_evaluation import DEFAULT_BUDGET

SCHEMA = 'elendril-v7-diagnostics-v1'


def _cases(data):
    return tuple(data.cases if hasattr(data, 'cases') else data)


def _cohorts(data):
    cases = _cases(data)
    # panel_cases accepts a dataset, not arbitrary case iterables.
    class Data:
        pass
    holder = Data()
    holder.cases = cases
    swaps = panel_cases(holder, 'payload-swap')
    def supply(case):
        payload = tuple(event for events in case.environment.channels.values() for event in events)
        return replace(case, environment=PublicEnvironment((*case.environment.initial, *payload), {}))
    return {'interactive': cases, 'supplied': tuple(map(supply, cases)),
            'interactive-swap': swaps, 'supplied-swap': tuple(map(supply, swaps)),
            'known': tuple(case for case in cases if case.kind == 'known')}


def _source():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _summary(cases, records):
    summaries = {}
    for name, record in records.items():
        if record is None:
            summaries[name] = {'episodes': 0}
            continue
        rows = record['raw']['episodes']
        lookup = {c.case_id: c for c in cases[name]}
        answers = Counter(r['trace']['answer'] for r in rows if r['trace']['status'] == 'answered')
        n = sum(answers.values())
        pairs = {}
        for row in rows:
            if row['pair_id']:
                pairs.setdefault(row['pair_id'], []).append(row['outcome']['correct'])
        needed = [r for r in rows if r['query_needed']]
        # In supplied cohorts, query_needed is no longer meaningful: it is
        # computed from the transformed public world, not the original decision.
        summaries[name] = {
            'episodes': len(rows),
            'nonnull_total': sum(lookup[r['case_id']].truth.expected is not None for r in rows),
            'nonnull_correct': sum(lookup[r['case_id']].truth.expected is not None and r['outcome']['correct'] for r in rows),
            'null_total': sum(lookup[r['case_id']].truth.expected is None for r in rows),
            'correct_abstentions': sum(lookup[r['case_id']].truth.expected is None and r['trace']['status'] == 'abstained' for r in rows),
            'query_needed': len(needed) if not name.startswith('supplied') else None,
            'needed_queried': sum(any(d['operation'] == 'QUERY' and d['evidence'] for d in r['trace']['decisions']) for r in needed) if not name.startswith('supplied') else None,
            'needless_query': sum(not r['query_needed'] and any(d['operation'] == 'QUERY' and d['evidence'] for d in r['trace']['decisions']) for r in rows) if not name.startswith('supplied') else None,
            'answer_counts': dict(sorted(answers.items())),
            'answer_entropy_bits': -sum((v/n)*math.log2(v/n) for v in answers.values()) if n else None,
            'dominant_answer': min(answers, key=lambda k: (-answers[k], k)) if answers else None,
            'both_correct_payload_pairs': sum(len(v) == 2 and all(v) for v in pairs.values()),
            'complete_payload_pairs': sum(len(v) == 2 for v in pairs.values()),
        }
    return summaries


def _comparisons(records):
    out = []
    for left, right in [('interactive', 'supplied'), ('interactive-swap', 'supplied-swap')]:
        if records[left] is None:
            continue
        supplied = {r['case_id']: r for r in records[right]['raw']['episodes']}
        for row in records[left]['raw']['episodes']:
            other = supplied[row['case_id']]
            decisions = row['trace']['decisions']
            query = next((i for i, d in enumerate(decisions) if d['operation'] == 'QUERY' and d['evidence']), None)
            prompts, supplied_prompts = row['prompts'], other['prompts']
            identity = (prompts[query + 1] == supplied_prompts[0]
                        if query is not None and len(prompts) > query + 1 and supplied_prompts else None)
            a, b = row['trace'], other['trace']
            if query is None and row['query_needed']:
                symptom, divergence = 'needed_query_not_observed', 'interactive_action_before_payload'
            else:
                # Compare actions after an actual release, or initial actions
                # where no release occurred; retain the first observed mismatch.
                start = query + 1 if query is not None else 0
                aa = [(d['operation'], d['argument']) for d in decisions[start:]]
                bb = [(d['operation'], d['argument']) for d in b['decisions']]
                mismatch = next((i for i in range(max(len(aa), len(bb)))
                                 if (aa[i] if i < len(aa) else None) != (bb[i] if i < len(bb) else None)), None)
                divergence = None if mismatch is None else f'action_{mismatch + 1}_after_release_alignment'
                symptom = 'outcomes_differ' if (a['status'], a['answer']) != (b['status'], b['answer']) else 'same_terminal_behavior'
            out.append({'cohort': left, 'case_id': row['case_id'], 'symptom': symptom,
                        'first_observable_divergence': divergence,
                        'queried_then_prompt_identical': identity,
                        'interactive_correct': row['outcome']['correct'], 'supplied_correct': other['outcome']['correct']})
    return out


def _probes(cases, backend):
    if backend is None:
        return []
    import torch
    selected = sorted((c for c in cases if c.truth.expected is not None), key=lambda c: (c.world_id, c.case_id))[:8]
    results = []
    for case in selected:
        state = CognitiveState(case.goal).observe(case.environment.initial)
        prompt = render_public(state, (), case.surface, case.memory is not None)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        gold = list(UTF8.encode(case.truth.expected))
        if len(gold) != 1:
            raise ValueError('probe requires one-token public value')
        if len(ids) + 1 > backend.binding.context_length:
            raise ValueError('probe context exceeds backend binding')
        with torch.no_grad():
            distribution = backend.next_distribution(ids)
            values = {symbol: float(distribution[UTF8.encode(symbol)[0]]) for symbol in 'ABCDEFGHIJKLMNOP'}
            first = float(distribution[gold[0]])
            eos = float(backend.next_distribution([*ids, gold[0]])[UTF8.eos_id])
        results.append({'case_id': case.case_id, 'prompt_sha256': digest(prompt), 'gold': case.truth.expected,
                        'first_value_probabilities': values, 'gold_first_probability': first,
                        'gold_first_nll': -math.log(first) if first > 0 else None,
                        'teacher_forced_eos_probability': eos,
                        'teacher_forced_value_to_eos_nll': -math.log(first * eos) if first * eos > 0 else None})
    return results


def diagnose(data, policy_factory, *, backend=None):
    from .v7_measurement import evaluate
    if getattr(policy_factory, 'mode', None) == 'oracle':
        raise ValueError('diagnostic actor cannot receive truth')
    cases = _cohorts(data)
    records = {}
    for name, cohort in cases.items():
        budget = replace(DEFAULT_BUDGET, decisions=3) if name.startswith('supplied') else DEFAULT_BUDGET
        records[name] = evaluate(cohort, policy_factory, budget=budget) if cohort else None
    identities = {r['raw']['model_identity'] for r in records.values() if r is not None}
    if len(identities) != 1:
        raise ValueError('diagnostic requires one frozen model identity')
    if backend is not None and next(iter(identities)) != (backend.binding.checkpoint_sha256 or backend.binding.identity):
        raise ValueError('probe backend does not match the evaluated checkpoint')
    probes = _probes(cases['supplied'], backend)
    return {'schema': SCHEMA, 'source_sha256': _source(), 'records': records,
            'supply_attribution': {'origin': 'engineered_supply', 'budget_decisions': 3,
                'exogenous_releases': {name: sum(bool(c.environment.channels) for c in cases[name.replace('supplied', 'interactive')])
                                       for name in ('supplied', 'supplied-swap')},
                'request_budget': 2, 'model_calls': 'recorded unchanged by evaluator'},
            'findings': _summary(cases, records), 'episodes': _comparisons(records),
            'candidate_causes': {'status': 'UNRESOLVED', 'note': 'Observed differences do not identify a unique mechanism.'},
            'known_cohort_note': 'Direct-known cases are separate, not matched to supplied cases; identical stateless prompts need not imply distinct mechanisms.',
            'token_probes': probes, 'token_probe_note': 'Post-hoc gold labels only; EOS is teacher-forced. Probe probabilities require the original backend to reproduce.'}


def verify_diagnosis(data, result, *, backend=None):
    from .v7_measurement import verify_record
    if result.get('schema') != SCHEMA or result.get('source_sha256') != _source():
        raise ValueError('diagnostic schema/source mismatch')
    cases = _cohorts(data)
    records = result['records']
    if set(records) != set(cases):
        raise ValueError('diagnostic cohort mismatch')
    for name, cohort in cases.items():
        if not cohort:
            if records[name] is not None:
                raise ValueError('nonempty record for empty cohort')
            continue
        verify_record(cohort, records[name])
        expected_budget = replace(DEFAULT_BUDGET, decisions=3) if name.startswith('supplied') else DEFAULT_BUDGET
        from dataclasses import asdict
        if records[name]['raw']['budget'] != asdict(expected_budget):
            raise ValueError('diagnostic budget mismatch')
    identities = {r['raw']['model_identity'] for r in records.values() if r is not None}
    if len(identities) != 1:
        raise ValueError('mixed identities')
    attribution = {'origin': 'engineered_supply', 'budget_decisions': 3,
        'exogenous_releases': {name: sum(bool(c.environment.channels) for c in cases[name.replace('supplied', 'interactive')]) for name in ('supplied', 'supplied-swap')},
        'request_budget': 2, 'model_calls': 'recorded unchanged by evaluator'}
    if result['supply_attribution'] != attribution or result['findings'] != _summary(cases, records) or result['episodes'] != _comparisons(records):
        raise ValueError('diagnostic attribution or summary mismatch')
    if result['candidate_causes'] != {'status': 'UNRESOLVED', 'note': 'Observed differences do not identify a unique mechanism.'}:
        raise ValueError('invalid causal attribution')
    probes = result['token_probes']
    if backend is not None:
        if next(iter(identities)) != (backend.binding.checkpoint_sha256 or backend.binding.identity):
            raise ValueError('probe backend does not match the evaluated checkpoint')
        if probes != _probes(cases['supplied'], backend):
            raise ValueError('probe outputs differ from backend recomputation')
    selected = sorted((c for c in cases['supplied'] if c.truth.expected is not None), key=lambda c: (c.world_id, c.case_id))[:8]
    if probes and len(probes) != len(selected):
        raise ValueError('probe selection mismatch')
    for case, probe in zip(selected, probes):
        prompt = render_public(CognitiveState(case.goal).observe(case.environment.initial), (), case.surface, case.memory is not None)
        probabilities = probe['first_value_probabilities']
        if (probe['case_id'] != case.case_id or probe['prompt_sha256'] != digest(prompt)
                or probe['gold'] != case.truth.expected or set(probabilities) != set('ABCDEFGHIJKLMNOP')
                or any(not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())):
            raise ValueError('invalid probe context or probability')
        first, eos = probabilities[case.truth.expected], probe['teacher_forced_eos_probability']
        if (not isinstance(eos, (float, int)) or not math.isfinite(eos) or not 0 <= eos <= 1
                or probe['gold_first_probability'] != first
                or probe['gold_first_nll'] != (-math.log(first) if first > 0 else None)
                or probe['teacher_forced_value_to_eos_nll'] != (-math.log(first * eos) if first * eos > 0 else None)):
            raise ValueError('invalid probe score')
    return result['findings']
