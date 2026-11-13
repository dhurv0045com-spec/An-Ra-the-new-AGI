import copy
import math
import unittest
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v6_evaluation import ControlPolicy, independent_rescore
from anra_v5.elendril.v7_measurement import evaluate, verify_legacy, verify_record


class V7MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_dataset('validation', worlds=32)
        cls.record = evaluate(cls.data, ControlPolicy('always-query'))

    def test_exact_tenths_and_legacy_reduction(self):
        r = copy.deepcopy(self.record)
        group = r['metrics']['all']
        self.assertGreater(group['verified_queries'], 0)
        self.assertEqual(group['utility_numerator'], 10 * group['correct'] - group['verified_queries'])
        self.assertEqual(group['utility'], group['utility_numerator'] / 10)
        legacy = r['raw']['metrics']['all']
        # Use a distinct valid reduction on both naive and compensated sums.
        legacy['utility'] = (group['utility'] if legacy['utility'] != group['utility']
                             else math.nextafter(group['utility'], math.inf))
        with self.assertRaises(ValueError): independent_rescore(self.data, r['raw'])
        self.assertEqual(verify_legacy(self.data, r['raw']), r['metrics'])
        self.assertEqual(verify_record(self.data, r), r['metrics'])

    def test_corruption_rejected(self):
        for change in ('utility', 'nan', 'confidence', 'action', 'cost', 'source', 'stable'):
            r = copy.deepcopy(self.record)
            if change == 'utility': r['raw']['metrics']['all']['utility'] += .01
            elif change == 'nan': r['raw']['metrics']['all']['utility'] = float('nan')
            elif change == 'confidence': r['raw']['episodes'][0]['confidence'] = .123
            elif change == 'action': r['raw']['episodes'][0]['trace']['decisions'][0]['operation'] = 'ANSWER'
            elif change == 'cost': r['raw']['costs']['verified_query'] = .2
            elif change == 'source': r['raw']['source_sha256'] = '0' * 64
            else: r['metrics']['all']['utility_numerator'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError): verify_record(self.data, r)
