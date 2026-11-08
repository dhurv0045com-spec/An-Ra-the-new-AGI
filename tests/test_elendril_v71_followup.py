import unittest
from dataclasses import replace

from anra_v5.elendril.cognition import CognitiveState
from anra_v5.elendril.v6_data import make_dataset, public_action
from anra_v5.elendril.v71_followup import audit, build_suite, planned_contrasts


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = build_suite()

    def test_deterministic_size_and_baseline(self):
        a = audit(self.suite)
        self.assertEqual((a['cohorts'], a['cases'], a['planned_contrasts']), (32, 512, 48))
        self.assertEqual(a['cases_sha256'], audit(build_suite())['cases_sha256'])
        self.assertEqual(a['actual_training_disjoint']['status'], 'NOT_RUN')

    def test_all_public_teacher_correct(self):
        for cases in self.suite.values():
            for case in cases:
                state = CognitiveState(case.goal).observe(case.environment.initial)
                self.assertEqual(public_action(state, (), case.surface), ('ANSWER', case.truth.expected))
                self.assertEqual(set(case.truth.admitted_public), {e.identity for e in case.environment.initial})

    def test_contrasts_change_one_factor_and_align_worlds(self):
        contrasts = planned_contrasts(self.suite)
        self.assertEqual(len(contrasts), 48)
        for p in contrasts:
            left, right = p['left'].split('|'), p['right'].split('|')
            self.assertEqual(left[0], right[0])
            changed = [i for i in range(4) if left[i] != right[i]]
            self.assertEqual(changed, [{'time':1, 'clutter':2, 'order':3}[p['factor']]])
            self.assertEqual(len(p['world_ids']), 16)
            for a, b in zip(self.suite[p['left']], self.suite[p['right']]):
                self.assertEqual(a.world_id, b.world_id)
                self.assertEqual(a.truth.expected, b.truth.expected)
                if p['factor'] == 'time':
                    self.assertEqual([e.tick + 9000 for e in a.environment.initial], [e.tick for e in b.environment.initial])
                    self.assertEqual([(e.subject,e.source,e.value) for e in a.environment.initial], [(e.subject,e.source,e.value) for e in b.environment.initial])
                elif p['factor'] == 'clutter':
                    self.assertTrue(set(b.environment.initial) <= set(a.environment.initial))
                else:
                    self.assertEqual(a.environment, b.environment)
                    self.assertEqual((a.surface.order, b.surface.order), ('original', 'reverse'))

    def test_tampering_is_detected(self):
        suite = dict(self.suite)
        name = next(iter(suite))
        row = suite[name][0]
        suite[name] = (replace(row, truth=replace(row.truth, expected='P' if row.truth.expected != 'P' else 'A')), *suite[name][1:])
        with self.assertRaises(ValueError):
            audit(suite)

    def test_actual_training_disjoint_check(self):
        train = make_dataset('train', worlds=32, seed=606)
        self.assertEqual(audit(self.suite, train=train)['actual_training_disjoint']['status'], 'PASS')
        with self.assertRaises(ValueError):
            audit(self.suite, train=self.suite['route2|original|full|original'])


if __name__ == '__main__':
    unittest.main()
