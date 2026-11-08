import copy
import json
import unittest
from collections import Counter

from anra_v5.elendril.cognition import CognitiveState, belief
from anra_v5.elendril.v6_data import make_dataset, public_action, render_public
from anra_v5.elendril.v71_data import (assert_training_disjoint, build_suite, suite_from_record,
                                        suite_manifest, suite_to_record, make_route_training)


class V71DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = build_suite()

    def test_deterministic_portable_and_bounded(self):
        manifest = suite_manifest(self.suite)
        self.assertEqual(manifest, suite_manifest(build_suite()))
        self.assertEqual(manifest['total_cases'], 1008)
        record = json.loads(json.dumps(suite_to_record(self.suite)))
        self.assertEqual(manifest, suite_manifest(suite_from_record(record)))
        bad = copy.deepcopy(record)
        bad['cohorts']['hop-4'][0]['truth']['expected'] = 'Z'
        with self.assertRaises(ValueError):
            suite_from_record(bad)
        train = make_dataset('train', worlds=32, seed=17071)
        self.assertEqual(assert_training_disjoint(train, self.suite)['status'], 'PASS')
        with self.assertRaises(ValueError):
            assert_training_disjoint(self.suite['base'][:1], self.suite)

    def test_hops_and_intermediate(self):
        for hops in range(1, 5):
            for case in self.suite[f'hop-{hops}']:
                state = CognitiveState(case.goal).observe(case.environment.initial)
                node = case.goal.subject
                for _ in range(hops):
                    found = belief(state.events, node)
                    self.assertEqual(found.status, 'derived')
                    node = found.value
                self.assertEqual(node, case.truth.expected)
                self.assertEqual(public_action(state, (), case.surface), ('ANSWER', node))
        rows = self.suite['intermediate-vs-sham']
        for actual, sham in zip(rows[::2], rows[1::2]):
            self.assertEqual(actual.pair_id, sham.pair_id)
            self.assertEqual(actual.environment.initial[:-1], sham.environment.initial[:-1])
            self.assertEqual(actual.environment.initial[-1].value, actual.environment.initial[0].value)
            self.assertNotEqual(sham.environment.initial[-1].value, actual.environment.initial[-1].value)
            self.assertNotEqual(actual.environment.initial[-1].value, actual.truth.expected)
            self.assertNotEqual(sham.environment.initial[-1].value, actual.truth.expected)
            prompts = [render_public(CognitiveState(c.goal).observe(c.environment.initial), (), c.surface) for c in (actual, sham)]
            self.assertEqual(len(prompts[0].encode()), len(prompts[1].encode()))
        for original, clamped in zip(*[iter(self.suite['hop-2-clamped'])]*2):
            self.assertEqual(original.pair_id, clamped.pair_id)
            self.assertEqual(original.environment, clamped.environment)
            self.assertEqual(clamped.goal.subject, original.environment.initial[0].value)
            self.assertEqual(clamped.truth.expected, original.truth.expected)
            state = CognitiveState(clamped.goal).observe(clamped.environment.initial)
            self.assertEqual(public_action(state, (), clamped.surface), ('ANSWER', original.truth.expected))

    def test_balanced_payload_and_memory(self):
        rows = self.suite['balanced-payload']
        left, right = rows[::2], rows[1::2]
        self.assertEqual(Counter(c.truth.expected for c in left), Counter(c.truth.expected for c in right))
        for a, b in zip(left, right):
            self.assertEqual(a.pair_id, b.pair_id)
            self.assertEqual(a.environment.initial, b.environment.initial)
            self.assertNotEqual(a.truth.expected, b.truth.expected)
            for case in (a, b):
                self.assertEqual({e.value for p in case.environment.channels.values() for e in p}, {case.truth.expected})
        for original, removed, irrelevant, corrected in zip(*[iter(self.suite['memory-manipulations'])]*4):
            self.assertIsNone(removed.memory)
            self.assertEqual(original.truth.expected, removed.truth.expected)
            self.assertNotIn(irrelevant.memory.events[0].subject, irrelevant.goal.slots)
            events = (*corrected.environment.initial, *corrected.memory.events)
            state = CognitiveState(corrected.goal).observe(events)
            self.assertEqual(public_action(state, (), corrected.surface), ('ANSWER', corrected.truth.expected))
            for c in (original, removed, irrelevant, corrected):
                admitted = {e.identity for e in c.environment.initial}
                if c.memory:
                    admitted.update(e.identity for e in c.memory.events)
                self.assertEqual(set(c.truth.admitted_public), admitted)

    def test_crossed_query_truth(self):
        for advertised, unavailable in zip(*[iter(self.suite['query-crossed'])]*2):
            self.assertTrue(advertised.environment.channels)
            self.assertFalse(unavailable.environment.channels)
            if unavailable.panel == 'query-unavailable-needed':
                self.assertIsNone(unavailable.truth.expected)
                state = CognitiveState(unavailable.goal).observe(unavailable.environment.initial)
                self.assertEqual(public_action(state, (), unavailable.surface), ('ABSTAIN', None))
            else:
                self.assertEqual(advertised.truth.expected, unavailable.truth.expected)

    def test_independent_three_hop_training(self):
        train = make_route_training(worlds=512)
        self.assertEqual(train.identity, make_route_training(worlds=512).identity)
        self.assertEqual(len(train.cases), 512)
        self.assertEqual(assert_training_disjoint(train, self.suite)['status'], 'PASS')
        for case in train.cases:
            self.assertEqual(case.surface.hops, 3)
            state = CognitiveState(case.goal).observe(case.environment.initial)
            self.assertEqual(public_action(state, (), case.surface), ('ANSWER', case.truth.expected))


if __name__ == '__main__':
    unittest.main()
