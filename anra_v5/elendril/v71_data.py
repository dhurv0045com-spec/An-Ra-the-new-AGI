"""Deterministic V7.1 diagnostic case construction; no actor sees truth."""
from dataclasses import asdict, dataclass, replace
import random

from .codec import canonical, digest
from .cognition import CognitiveState, ExperienceEvent, Goal, MemoryVersion, PublicEnvironment, Truth
from .v6_data import (CHANNEL, SYMBOLS, Surface, V6Case, _dataset, _partition, _state_key, _supervision,
                      case_from_record, case_to_record, make_dataset, panel_cases)


@dataclass(frozen=True, slots=True)
class ChainSurface(Surface):
    """V6-compatible renderer surface permitting the two additional hop counts.

    V6's case_from_record accepts only its original Surface; a V7.1 reader must
    reconstruct this subclass for hop 1/4 records.
    """
    def __post_init__(self):
        Surface(self.template, self.aliases, 2, self.scratch, self.order)
        if self.hops not in (1, 2, 3, 4):
            raise ValueError('invalid chain length')


def _admitted(env, memory):
    return tuple(e.identity for e in (*env.initial, *(x for p in env.channels.values() for x in p),
                                       *(memory.events if memory else ())))


def _chain(seed, hops, count, used, *, split='validation'):
    rng = random.Random(int(digest(['v71-chain', seed, hops, split]), 16))
    cases = []
    while len(cases) < count:
        nodes = rng.sample(SYMBOLS, hops + 3)
        tick = rng.randrange(1, 10000)
        world = digest(['v71-chain-world', seed, hops, split, nodes, tick])
        scope, reuse = 'world-' + world, 'memory-' + world
        goal = Goal(world, 'route', nodes[0], scope, reuse, tuple(nodes))
        events = tuple(ExperienceEvent(scope, nodes[i], 'map', tick + i, nodes[i+1]) for i in range(hops))
        state = CognitiveState(goal).observe(events)
        key = _state_key(state, (), False)
        if _partition(key) != split or key in used:
            continue
        used.add(key)
        env = PublicEnvironment(events, {})
        cases.append(V6Case(world, world, 'route', 'route', goal, env,
                            Truth(world, nodes[hops], _admitted(env, None)), ChainSurface(hops=hops), panel='v71-chain'))
    return tuple(cases)


def make_route_training(seed=27171, worlds=512, hops=3):
    """Independently sample a partition-audited B task with complete paths."""
    if type(seed) is not int or not 0 <= seed < 2**32 or type(worlds) is not int or not 1 <= worlds <= 20000 or hops != 3:
        raise ValueError('invalid bounded three-hop training request')
    cases = _chain(seed, hops, worlds, set(), split='train')
    # Three hops uses the V6-qualified surface, keeping V6 dataset serialization.
    cases = tuple(replace(c, surface=Surface(hops=3), panel='v71-route-training') for c in cases)
    return _dataset('train', cases, seed=seed)


def _variant(case, panel, env, memory, expected=None, *, preserve_expected=True, pair_id=''):
    new_id = digest([case.case_id, panel])
    truth = Truth(new_id, case.truth.expected if preserve_expected else expected, _admitted(env, memory))
    return replace(case, case_id=new_id, goal=replace(case.goal, episode=new_id), environment=env,
                   memory=memory, truth=truth, pair_id=pair_id, panel=panel)


def _hints(chains):
    result = []
    for case in chains:
        first = case.environment.initial[0].value
        final = case.truth.expected
        unused = [s for s in SYMBOLS if s not in case.goal.slots]
        subject = unused[0]
        sham = next(s for s in SYMBOLS if s not in (first, final))
        pair = digest(['v71-hint-pair', case.world_id])
        for panel, value in [('first-intermediate', first), ('sham-intermediate', sham)]:
            event = ExperienceEvent(case.goal.scope, subject, 'hint', 0, value)
            env = PublicEnvironment((*case.environment.initial, event), {})
            result.append(_variant(case, panel, env, None, pair_id=pair))
    return tuple(result)


def _clamped(chains):
    result = []
    for case in chains:
        pair = digest(['v71-clamped-pair', case.world_id])
        result.append(replace(case, pair_id=pair, panel='two-hop-original'))
        clamped = _variant(case, 'first-hop-clamped', case.environment, None, pair_id=pair)
        clamped = replace(clamped, goal=replace(clamped.goal, subject=case.environment.initial[0].value),
                          surface=ChainSurface(hops=1))
        result.append(clamped)
    return tuple(result)


def _payloads(base, count):
    candidates = [c for c in base if c.kind in ('missing', 'conflict') and c.environment.channels]
    result = []
    # A cyclic permutation of a fixed sequence gives exactly equal original and
    # alternative marginal histograms, including small non-multiple-of-16 runs.
    labels = [SYMBOLS[i % 16] for i in range(count)]
    for i in range(count):
        case = candidates[i % len(candidates)]
        original, alternative = labels[i], labels[(i + 1) % count]
        pair = digest(['v71-balanced-payload', case.world_id, i])
        for tag, value in [('payload-original', original), ('payload-alternative', alternative)]:
            channel = {k: tuple(replace(e, value=value) for e in p) for k, p in case.environment.channels.items()}
            env = PublicEnvironment(case.environment.initial, channel)
            result.append(_variant(case, tag + '-' + str(i), env, case.memory,
                                   expected=value, preserve_expected=False, pair_id=pair))
    return tuple(result)


def _memories(base):
    out = []
    for case in (c for c in base if c.memory is not None):
        pair = digest(['v71-memory', case.world_id])
        out.append(replace(case, pair_id=pair, panel='memory-original'))
        out.append(_variant(case, 'memory-removed', case.environment, None, pair_id=pair))
        event = case.memory.events[0]
        unused = next(s for s in SYMBOLS if s not in case.goal.slots and s != case.truth.expected)
        irrelevant = MemoryVersion((replace(event, subject=unused),))
        out.append(_variant(case, 'memory-irrelevant', case.environment, irrelevant, pair_id=pair))
        stale_value = next(s for s in SYMBOLS if s != event.value)
        # Both rows belong to the same source and the correction is later.
        stale = replace(event, tick=event.tick, value=stale_value)
        correction = replace(event, tick=event.tick + 1)
        corrected = MemoryVersion((stale, correction))
        out.append(_variant(case, 'memory-stale-corrected', case.environment, corrected, pair_id=pair))
    return tuple(out)


def _irrelevant_queries(base):
    out = []
    for case in (c for c in base if c.kind == 'known'):
        unused = next(s for s in SYMBOLS if s not in case.goal.slots and s != case.truth.expected)
        event = ExperienceEvent(case.goal.scope, unused, 'audit', 9999, unused)
        env = PublicEnvironment(case.environment.initial, {CHANNEL: (event,)})
        out.append(_variant(case, 'query-irrelevant', env, case.memory))
    return tuple(out)


def _query_crossed(base):
    out = []
    for case in (c for c in base if c.kind in ('known', 'missing', 'conflict') and c.environment.channels):
        advertised = _irrelevant_queries((case,))[0] if case.kind == 'known' else case
        out.append(replace(advertised, panel='query-advertised-known' if case.kind == 'known' else 'query-advertised-needed'))
        env = PublicEnvironment(case.environment.initial, {})
        # With no remaining evidence, missing/conflict worlds have no safe value.
        out.append(_variant(case, 'query-unavailable-known' if case.kind == 'known' else 'query-unavailable-needed',
                            env, case.memory, expected=None, preserve_expected=case.kind == 'known'))
    return tuple(out)


def build_suite(seed=17071, worlds_per_kind=16):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(worlds_per_kind) is not int or not 2 <= worlds_per_kind <= 16:
        raise ValueError('seed must be uint32 and worlds_per_kind must be 2..16')
    data = make_dataset('validation', worlds=7 * worlds_per_kind, seed=seed)
    used = set()
    chains = {f'hop-{h}': _chain(seed, h, worlds_per_kind, used) for h in (1, 2, 3, 4)}
    suite = {'base': tuple(data.cases), **chains,
             'intermediate-vs-sham': _hints(chains['hop-2']),
             'hop-2-clamped': _clamped(chains['hop-2']),
             'balanced-payload': _payloads(data.cases, worlds_per_kind),
             'memory-manipulations': _memories(data.cases),
             'query-irrelevant': _irrelevant_queries(data.cases),
             'query-crossed': _query_crossed(data.cases)}
    for name in ('template', 'row-order', 'distractor', 'alias'):
        suite['robustness-' + name] = panel_cases(data, name)
    if sum(map(len, suite.values())) > 1200:
        raise ValueError('suite exceeds bounded case count')
    return suite


def suite_manifest(suite):
    cohorts = {name: {'count': len(cases), 'cases_sha256': digest([case_to_record(c) for c in cases]),
                       'case_ids': [c.case_id for c in cases]} for name, cases in sorted(suite.items())}
    return {'schema': 'elendril-v71-suite-v1', 'cohorts': cohorts,
            'suite_sha256': digest(cohorts), 'total_cases': sum(c['count'] for c in cohorts.values()),
            'qualifications': [
                'Synthetic chains are independently generated; their initial states are validation-partition checked, not an external benchmark.',
                'Hop 1 and 4 use ChainSurface; V6 case_from_record cannot reconstruct these surfaces.',
                'First-intermediate cues are extra unrelated rows with no instruction to use them; the matched sham is a distractor control, not explicit state injection. Both avoid the terminal answer.',
                'Hop-2-clamped changes the question to one hop from the true first intermediate; it is engineered task decomposition, not the same prompt or proof of an internal mechanism.',
                'Memory removed and irrelevant preserve the original target for a causal availability contrast; without that memory the original target is not publicly solvable, so abstention is not scored as correct in those conditions.',
                'Stale-corrected memory has an earlier wrong row and a later correct row from the same source.',
                'Query-unavailable missing/conflict cases are safe-abstention worlds; querying cost repricing is not learned cost conditioning.',
                'Robustness panels and synthetic distributions are related to V6; no claim of natural-world generalization.']}


def suite_to_record(suite):
    return {'schema': 'elendril-v71-portable-suite-v1', 'manifest': suite_manifest(suite),
            'cohorts': {k: [case_to_record(c) for c in v] for k, v in sorted(suite.items())}}


def suite_from_record(record):
    if record.get('schema') != 'elendril-v71-portable-suite-v1':
        raise ValueError('invalid suite schema')
    suite = {}
    for name, rows in record['cohorts'].items():
        cases = []
        for row in rows:
            hops = row['surface']['hops']
            if hops in (1, 4):
                intermediate = {**row, 'surface': {**row['surface'], 'hops': 2}}
                case = case_from_record(intermediate)
                case = replace(case, surface=ChainSurface(**row['surface']))
            else:
                case = case_from_record(row)
            if canonical(case_to_record(case)) != canonical(row):
                raise ValueError('case reconstruction mismatch')
            cases.append(case)
        suite[name] = tuple(cases)
    if suite_manifest(suite) != record['manifest']:
        raise ValueError('suite identity mismatch')
    return suite


def assert_training_disjoint(train, suite):
    """Compare actual public decision-state keys and world IDs, not seeds alone."""
    train_cases = tuple(train.cases if hasattr(train, 'cases') else train)
    train_worlds = {c.world_id for c in train_cases}
    train_states = {key for c in train_cases for key in _supervision(c)[1]}
    for cases in suite.values():
        for case in cases:
            if case.world_id in train_worlds or train_states.intersection(_supervision(case)[1]):
                raise ValueError('V7.1 diagnostic world or decision state overlaps training')
    return {'status': 'PASS', 'training_worlds': len(train_worlds), 'training_states': len(train_states)}
