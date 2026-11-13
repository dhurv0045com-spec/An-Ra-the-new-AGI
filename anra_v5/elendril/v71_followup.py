"""Predeclared within-world route surface interventions, no model inference."""
from dataclasses import replace
from itertools import product

from .codec import canonical, digest
from .cognition import CognitiveState, PublicEnvironment, Truth, belief
from .v6_data import Surface, _partition, _supervision, case_to_record, public_action, render_public
from .v71_data import ChainSurface, assert_training_disjoint, build_suite as v71_suite

QUESTIONS = ('route2', 'firsthop', 'clamped', 'directlookup')
TIMES = ('original', 'shifted')
CLUTTER = ('full', 'route-only')
ORDERS = ('original', 'reverse')


def _key(question, time, clutter, order):
    return '|'.join((question, time, clutter, order))


def _base(seed):
    cases = tuple(c for c in v71_suite(seed=seed)['base'] if c.kind == 'route')
    if len(cases) != 16:
        raise ValueError('expected sixteen original V7.1 base route worlds')
    return tuple(sorted(cases, key=lambda c: (c.world_id, c.case_id)))


def _prompt(case):
    state = CognitiveState(case.goal).observe(case.environment.initial)
    return render_public(state, tuple(case.environment.channels), case.surface, case.memory is not None)


def build_suite(seed=17071):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    suite = {}
    original = _base(seed)
    for question, time, clutter, order in product(QUESTIONS, TIMES, CLUTTER, ORDERS):
        key = _key(question, time, clutter, order)
        result = []
        for case in original:
            start = case.goal.subject
            first = belief(case.environment.initial, start)
            if first.status != 'derived':
                raise ValueError('original first edge not uniquely determined')
            middle = first.value
            terminal = belief(case.environment.initial, middle)
            if terminal.status != 'derived' or terminal.value != case.truth.expected:
                raise ValueError('original terminal edge not uniquely determined')
            if case.environment.channels or case.memory is not None:
                raise ValueError('route baseline must have no channels or memory')
            events = tuple(e for e in case.environment.initial if clutter == 'full' or e.subject in (start, middle))
            if time == 'shifted':
                events = tuple(replace(e, tick=e.tick + 9000) for e in events)
            subject = middle if question in ('clamped', 'directlookup') else start
            family = 'revision' if question == 'directlookup' else 'route'
            hops = 0 if question == 'directlookup' else 2 if question == 'route2' else 1
            expected = middle if question == 'firsthop' else terminal.value
            ident = digest(['v71-followup-v1', case.case_id, key])
            goal = replace(case.goal, episode=ident, family=family, subject=subject)
            surface_args = dict(template=case.surface.template, aliases=case.surface.aliases,
                                hops=hops, scratch=case.surface.scratch, order=order)
            surface = ChainSurface(**surface_args) if hops == 1 else Surface(**surface_args)
            env = PublicEnvironment(events, {})
            truth = Truth(ident, expected, tuple(e.identity for e in events))
            result.append(replace(case, case_id=ident, family=family, goal=goal, environment=env, truth=truth,
                                  surface=surface, pair_id=digest(['v71-followup-pair', case.world_id, question]), panel=key))
        suite[key] = tuple(result)
    return suite


def planned_contrasts(suite):
    expected = {_key(*p) for p in product(QUESTIONS, TIMES, CLUTTER, ORDERS)}
    if set(suite) != expected:
        raise ValueError('missing or unexpected followup cohort')
    comparisons = []
    for q, t, c, o in product(QUESTIONS, TIMES, CLUTTER, ORDERS):
        for factor, enabled, alternate in [('time', t == TIMES[0], (q, TIMES[1], c, o)),
                                            ('clutter', c == CLUTTER[0], (q, t, CLUTTER[1], o)),
                                            ('order', o == ORDERS[0], (q, t, c, ORDERS[1]))]:
            if not enabled:
                continue
            left, right = _key(q,t,c,o), _key(*alternate)
            lcases, rcases = suite[left], suite[right]
            if len(lcases) != 16 or len(rcases) != 16 or [x.world_id for x in lcases] != [x.world_id for x in rcases]:
                raise ValueError('planned comparison missing aligned worlds')
            comparisons.append({'factor': factor, 'left': left, 'right': right,
                                'world_ids': [x.world_id for x in lcases], 'pairs': 16})
    return comparisons


def audit(suite, *, seed=17071, train=None):
    """Rebuild every variant, verify public teacher and report partition limits.

    Supply the actual training dataset to check exact public decision-state and
    world disjointness. A train partition assignment alone is not evidence of an
    actual overlap; without the dataset this check remains NOT_RUN.
    """
    rebuilt = build_suite(seed=seed)
    if set(suite) != set(rebuilt):
        raise ValueError('cohort mismatch')
    for key in rebuilt:
        if canonical([case_to_record(c) for c in suite[key]]) != canonical([case_to_record(c) for c in rebuilt[key]]):
            raise ValueError('followup case or factor differs from construction')
    contrasts = planned_contrasts(suite)
    for key, cases in suite.items():
        for case in cases:
            state = CognitiveState(case.goal).observe(case.environment.initial)
            if public_action(state, (), case.surface, False) != ('ANSWER', case.truth.expected):
                raise ValueError('public graph does not derive declared label')
            prompt = _prompt(case)
            if len(prompt.encode('utf-8')) + 6 > 512:
                raise ValueError('public prompt exceeds V6 context limit')
            # Header is determined solely by public question/advertisements, with
            # no output label appended. It may naturally contain the same symbol.
            entity = case.surface.aliases + case.goal.subject
            question = (f'Current value of {entity}?' if case.goal.family == 'revision'
                        else f'Follow {case.surface.hops} links from {entity}.')
            if case.surface.aliases:
                question += ' Output the letter after the alias.'
            if prompt.splitlines()[1] != question + ' Query=0 Memory=0':
                raise ValueError('question header is not public-only')
    originals = _base(seed)
    baseline = suite[_key('route2', 'original', 'full', 'original')]
    if any(_prompt(a) != _prompt(b) for a,b in zip(originals, baseline)):
        raise ValueError('original baseline prompt changed')
    partitions = {}
    for cases in suite.values():
        for case in cases:
            for key in _supervision(case)[1]:
                partition = _partition(key)
                partitions[partition] = partitions.get(partition, 0) + 1
    disjoint = {'status': 'NOT_RUN', 'reason': 'actual training dataset not supplied'} if train is None else assert_training_disjoint(train, suite)
    return {'schema': 'elendril-v71-followup-audit-v1', 'status': 'PASS', 'seed': seed,
            'cohorts': len(suite), 'cases': sum(map(len, suite.values())), 'planned_contrasts': len(contrasts),
            'partition_counts': dict(sorted(partitions.items())), 'actual_training_disjoint': disjoint,
            'cases_sha256': digest({k: [case_to_record(c) for c in suite[k]] for k in sorted(suite)}),
            'limitations': ['Only factor-one comparisons within the same question are predeclared; cross-question differences change the task.',
                'Clamped changes the goal and is not injected internal state.',
                'Original V7.1 independent chains had monotone hop-ordered rows and ticks; last-row/max-tick copying can solve them, so prior hop-4 accuracy is not proof of composition.',
                'Transformed variants may map to the training partition; check exact overlap against the actual training dataset.']}
