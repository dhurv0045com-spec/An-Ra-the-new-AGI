"""Frozen exploratory V6 development experiments; no training or promotion gate."""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from .cognition import Budget, ExperienceEvent, MemoryVersion, PublicEnvironment
from .v6_data import SYMBOLS

SPECS = (
    dict(id='S16', name='Effort-budget sensitivity', hypothesis='Added decisions improve terminal success on evidence-needing worlds without worsening others.', falsifier='No useful paired gains or regressions elsewhere.', next_decision='Inspect both gains and regressions before changing the budget.'),
    dict(id='S17', name='Symbol-permutation equivariance', hypothesis='Robust semantic binding survives one fixed global symbol bijection.', falsifier='A paired correctness change under a bijection.', next_decision='Inspect symbol-sensitive failures.'),
    dict(id='S18', name='Stale-memory interference', hypothesis='Older same-source evidence does not override current facts.', falsifier='Paired correctness falls with stale prefetched evidence.', next_decision='Inspect susceptibility to stale evidence.'),
    dict(id='S19', name='Query-cost robustness', hypothesis='The primary model advantage over no-query survives query-price variation.', falsifier='Candidate advantage disappears or reverses at a tested price.', next_decision='Review cost tradeoffs; behavior remains fixed.'),
)
_IDS = {s['id'] for s in SPECS}
_KINDS = ('known', 'missing', 'conflict', 'null', 'unavailable', 'route', 'memory')
_SCHEMA = 'elendril-v6-supplement-v1'


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _cohort(data):
    cases = tuple(data.cases if hasattr(data, 'cases') else data)
    selected = []
    for kind in _KINDS:
        group = sorted((c for c in cases if c.kind == kind), key=lambda c: (c.world_id, c.case_id))
        distinct = []
        for c in group:
            if c.world_id not in {x.world_id for x in distinct}:
                distinct.append(c)
        if len(distinct) < 4:
            return ()
        selected.extend(distinct[:4])
    return tuple(selected)


def _permuted(case):
    mapping = dict(zip(SYMBOLS, SYMBOLS[1:] + SYMBOLS[:1]))
    mapped = {}
    def event(e):
        if e.identity not in mapped:
            parents = tuple(mapped[p].identity for p in e.parents)
            mapped[e.identity] = replace(e, subject=mapping[e.subject], value=mapping[e.value] if e.value else None, parents=parents)
        return mapped[e.identity]
    # All parent IDs must already be available in the public/memory order.
    initial = tuple(event(e) for e in case.environment.initial)
    channels = {k: tuple(event(e) for e in v) for k, v in case.environment.channels.items()}
    memory = None if case.memory is None else replace(case.memory, events=tuple(event(e) for e in case.memory.events))
    goal = replace(case.goal, subject=mapping[case.goal.subject], slots=tuple(mapping[s] for s in case.goal.slots))
    truth = replace(case.truth, expected=mapping[case.truth.expected] if case.truth.expected else None,
                    admitted_public=tuple(mapped[x].identity for x in case.truth.admitted_public))
    return replace(case, goal=goal, environment=PublicEnvironment(initial, channels), memory=memory, truth=truth)


def _stale(case):
    relevant = [e for e in case.environment.initial if e.subject == case.goal.subject and e.value is not None]
    if not relevant or not case.environment.initial or min(e.tick for e in case.environment.initial) < 1:
        return None
    current = max(relevant, key=lambda e: e.tick)
    wrong = next(s for s in SYMBOLS if s != current.value)
    old = ExperienceEvent(case.goal.reusable_scope, current.subject, current.source,
                          min(e.tick for e in case.environment.initial) - 1, wrong)
    # The engineered observation is public evidence but does not change the original answer.
    truth = replace(case.truth, admitted_public=(*case.truth.admitted_public, old.identity))
    return replace(case, memory=MemoryVersion((old,)), truth=truth)


def condition_specs(data, experiment_id):
    if experiment_id not in _IDS:
        raise ValueError('unknown supplement experiment')
    if experiment_id == 'S19':
        return {}
    cohort = _cohort(data)
    if not cohort:
        return {}
    def spec(cases, decisions=4, memory_mode='on'):
        return {'cases': tuple(cases), 'budget': Budget(decisions=decisions, requests=2, model_positions=12000), 'memory_mode': memory_mode}
    if experiment_id == 'S16':
        return {str(n): spec(cohort, n) for n in (1, 2, 4)}
    if experiment_id == 'S17':
        return {'baseline': spec(cohort), 'permuted': spec(tuple(_permuted(c) for c in cohort))}
    known = tuple(c for c in cohort if c.kind == 'known')
    stale = tuple(_stale(c) for c in known)
    if len(known) != 4 or any(c is None for c in stale):
        return {}
    baseline = tuple(replace(c, memory=MemoryVersion()) for c in known)
    return {'baseline': spec(baseline, memory_mode='prefetch'), 'stale': spec(stale, memory_mode='prefetch')}


def _paired(a, b):
    if [r['case_id'] for r in a['episodes']] != [r['case_id'] for r in b['episodes']]:
        raise ValueError('paired cases/order mismatch')
    rows = list(zip(a['episodes'], b['episodes']))
    n = len(rows)
    def stats(record):
        totals = record['metrics']['all']
        return {'accuracy': totals['correct']/n, 'mean_utility': totals['utility']/n,
                'mean_model_positions': totals['model_positions']/n}
    return {'first': stats(a), 'second': stats(b),
            'gains': sum(not x['outcome']['correct'] and y['outcome']['correct'] for x,y in rows),
            'regressions': sum(x['outcome']['correct'] and not y['outcome']['correct'] for x,y in rows),
            'both_correct': sum(x['outcome']['correct'] and y['outcome']['correct'] for x,y in rows),
            'changed_outcome_rate': sum(x['outcome']['correct'] != y['outcome']['correct'] for x,y in rows) / n,
            'pairs': n}


def summarize(experiment_id, records, controls=None):
    if experiment_id not in _IDS:
        raise ValueError('unknown supplement experiment')
    records = dict(records)
    required = {'S16': {'1','2','4'}, 'S17': {'baseline','permuted'}, 'S18': {'baseline','stale'}, 'S19': {'model'}}[experiment_id]
    if not records:
        return {'schema': _SCHEMA, 'source_sha256': _source_sha256(), 'experiment_id': experiment_id, 'status': 'NOT_QUALIFIED', 'records': records, 'summary': {'reason': 'required development cohort or records unavailable', 'exploratory': True}}
    if set(records) != required:
        raise ValueError('incomplete experiment conditions')
    if experiment_id != 'S19':
        first = next(iter(records.values()))
        for rec in records.values():
            if any(rec[key] != first[key] for key in ('model_identity', 'policy', 'decoder_schema', 'registry_sha256', 'source_sha256', 'split')):
                raise ValueError('condition policy or source mismatch')
    if experiment_id == 'S19':
        if not isinstance(controls, dict) or not {'fixed','always-query','no-query'} <= set(controls):
            raise ValueError('S19 requires all matched controls')
        model = records['model']
        if model['policy'] != 'model' or not model['model_identity']:
            raise ValueError('S19 requires a bound model record')
        compared = {'model': model, **{k: controls[k] for k in ('fixed','always-query','no-query')}}
        fields = ('cases_sha256','dataset_sha256','budget','memory_mode','split','registry_sha256','source_sha256')
        for name, rec in compared.items():
            if any(rec[f] != model[f] for f in fields) or [r['case_id'] for r in rec['episodes']] != [r['case_id'] for r in model['episodes']]:
                raise ValueError('S19 control is not matched')
            if name != 'model' and (rec['policy'] != name or rec['model_identity'] is not None):
                raise ValueError('S19 mislabeled control')
        costs = {}
        for name, rec in compared.items():
            totals = rec['metrics']['all']; n = totals['episodes']
            if not n: raise ValueError('empty S19 record')
            costs[name] = {str(lam): (totals['correct'] - lam * totals['verified_queries']) / n for lam in (0, .1, .5, 1)}
        base = compared['no-query']['metrics']['all']; mod = model['metrics']['all']
        dc = mod['correct'] - base['correct']; dq = mod['verified_queries'] - base['verified_queries']
        rankings = {}
        for lam in ('0', '0.1', '0.5', '1'):
            values = {name: row[lam] for name, row in costs.items()}
            rankings[lam] = [sorted(name for name, value in values.items() if value == score) for score in sorted(set(values.values()), reverse=True)]
        summary = {'exploratory': True, 'behavior': 'fixed; no adaptive cost-awareness claim',
                   'mean_utility_by_query_cost': costs, 'rankings_with_ties': rankings,
                   'break_even_vs_no_query': None if dq == 0 else dc / dq,
                   'break_even_note': 'Equal query counts yield no unique break-even price.' if dq == 0 else 'May lie outside the nonnegative or tested price range.'}
    else:
        pairs = [('1','2'),('2','4'),('1','4')] if experiment_id == 'S16' else [('baseline', 'permuted' if experiment_id == 'S17' else 'stale')]
        summary = {'exploratory': True, 'paired': {a + '_vs_' + b: _paired(records[a], records[b]) for a,b in pairs}}
        if experiment_id == 'S17':
            rows = zip(records['baseline']['episodes'], records['permuted']['episodes'])
            mapping = dict(zip(SYMBOLS, SYMBOLS[1:] + SYMBOLS[:1]))
            summary['output_equivariance'] = sum((mapping.get(a['trace']['answer']) if a['trace']['answer'] else None) == b['trace']['answer'] and a['trace']['status'] == b['trace']['status'] for a,b in rows) / len(records['baseline']['episodes'])
        if experiment_id == 'S18':
            a, b = (records[name]['metrics']['all'] for name in ('baseline','stale'))
            summary['charged_costs'] = {name: {'requests': r['metrics']['all']['requests'], 'model_positions': r['metrics']['all']['model_positions']} for name, r in records.items()}
            summary['charged_cost_difference_stale_minus_baseline'] = {'requests': b['requests'] - a['requests'], 'model_positions': b['model_positions'] - a['model_positions']}
    summary['limitations'] = 'Exploratory development-only; no training or promotion. S16/S17 use 28 fixed worlds; S18 uses four known worlds. S17 tests one fixed bijection, not arbitrary symbol invariance.'
    return {'schema': _SCHEMA, 'source_sha256': _source_sha256(), 'experiment_id': experiment_id, 'status': 'COMPLETE', 'records': records, 'summary': summary}


def verify_experiment(data, experiment_id, report, controls=None):
    from .codec import canonical
    from .v6_evaluation import independent_rescore
    specs = condition_specs(data, experiment_id)
    records = report['records']
    if experiment_id != 'S19':
        if set(records) != set(specs): raise ValueError('condition cohort mismatch')
        for name, spec in specs.items():
            record = records[name]
            if (record['budget'] != asdict(spec['budget']) or record['memory_mode'] != spec['memory_mode']
                    or record['split'] != getattr(data, 'split', 'development')):
                raise ValueError('condition protocol mismatch')
            independent_rescore(spec['cases'], record)
    elif records:
        independent_rescore(data, records['model'])
        for name in ('fixed','always-query','no-query'):
            independent_rescore(data, controls[name])
    expected = summarize(experiment_id, records, controls)
    if canonical(expected) != canonical(report): raise ValueError('supplement summary mismatch')
    return True


def run_experiment(data, policy, experiment_id, controls=None, base_record=None):
    from .v6_evaluation import evaluate
    if experiment_id == 'S19':
        records = {} if base_record is None else {'model': base_record}
    else:
        records = {name: evaluate(spec['cases'], policy, split=getattr(data, 'split', 'development'), budget=spec['budget'], memory_mode=spec['memory_mode'])
                   for name, spec in condition_specs(data, experiment_id).items()}
    result = summarize(experiment_id, records, controls)
    verify_experiment(data, experiment_id, result, controls)
    return result
