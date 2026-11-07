import copy
import unittest
from dataclasses import replace
from anra_v5.elendril import codec
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_data import make_dataset, panel_cases
from anra_v5.elendril.v6_registry import PANELS, MODEL_PROFILES, LEGACY_MODEL_PROFILES
from anra_v5.elendril.v6_evaluation import (ControlPolicy, evaluate, independent_rescore, parse_action, action_tokens, diagnose, panel_report, diagnostic_cases, fit_calibration, validation_risk_bound)

class V6EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_dataset('validation', worlds=32)
    def test_registry_and_grammar(self):
        self.assertEqual(len(PANELS), 15)
        self.assertEqual([x[0] for x in PANELS], [f'P{i:02}' for i in range(1,16)])
        self.assertEqual(codec.VOCAB_SIZE, 46)
        for name, context in [('S3',512),('S8',512),('S20',640)]:
            self.assertEqual(MODEL_PROFILES[name].context_length, context)
            self.assertEqual(MODEL_PROFILES[name].vocabulary_size, 260)
            self.assertEqual(LEGACY_MODEL_PROFILES[name].vocabulary_size, 46)
        self.assertEqual(parse_action(action_tokens('ANSWER','A'), ()), ('ANSWER','A'))
        self.assertEqual(parse_action(action_tokens('ANSWER','A')[:-1], ())[0], 'INVALID')
        self.assertEqual(parse_action((*action_tokens('ANSWER','A'), UTF8.eos_id), ())[0], 'INVALID')
    def test_replay_and_corruption(self):
        record = evaluate(self.data, ControlPolicy('fixed'))
        self.assertEqual(independent_rescore(self.data,record),record['metrics'])
        corrupt = copy.deepcopy(record)
        corrupt['episodes'][0]['trace']['answer'] = 'Z'
        with self.assertRaises(ValueError): independent_rescore(self.data,corrupt)
        corrupt = copy.deepcopy(record); corrupt['source_sha256'] = '0'*64
        with self.assertRaises(ValueError): independent_rescore(self.data,corrupt)

    def test_model_factory_gets_no_truth_channel_payload_or_memory_values(self):
        class Probe:
            origin='model'
            def for_case(inner,public,**kwargs):
                self.assertFalse(hasattr(public,'truth'))
                self.assertFalse(hasattr(public,'environment'))
                self.assertFalse(hasattr(public.memory,'events'))
                return ControlPolicy('fixed',surface=public.surface,memory_available=kwargs['memory_available'])
        evaluate(self.data,Probe())
    def test_interventions_and_statuses(self):
        for name in ('query-swap','time-order'):
            cases = panel_cases(self.data,name)
            self.assertTrue(cases)
            for i in range(0,len(cases),2):
                self.assertEqual(cases[i].pair_id,cases[i+1].pair_id)
                self.assertNotEqual(cases[i].truth.expected,cases[i+1].truth.expected)
        report = diagnose(self.data,ControlPolicy('query-blind'))
        self.assertEqual(report['panels']['P10']['status'],'FAIL')
        self.assertTrue(all(p['status'] in {'PASS','FAIL','NOT_QUALIFIED','SKIPPED'} for p in report['panels'].values()))
    def test_legacy_model_raw_eos_replay(self):
        from anra_v5.elendril.cognition import OperationCandidate
        from anra_v5.elendril.representation import LEGACY
        class LegacyModel:
            origin = "model"
            representation = "legacy-ascii"
            model_identity = "legacy-test-model"
            def for_case(self, case, **kwargs): return self
            def propose(self, state, legal, channels, positions):
                return OperationCandidate("ANSWER", "A", .7, 2, 1,
                    (*LEGACY.encode("A"), LEGACY.eos_id))
        record = evaluate((self.data.cases[0],), LegacyModel())
        self.assertEqual(record["decoder_schema"], "legacy-ascii")
        self.assertEqual(independent_rescore((self.data.cases[0],), record), record["metrics"])
        bad = copy.deepcopy(record)
        bad["episodes"][0]["trace"]["decisions"][0]["raw_tokens"] = list(LEGACY.encode("A"))
        with self.assertRaises(ValueError): independent_rescore((self.data.cases[0],), bad)
        self.assertEqual(parse_action((*LEGACY.encode("-A"), LEGACY.eos_id), (), "legacy-ascii"), ("DELIBERATE", "A"))
        self.assertEqual(parse_action((*LEGACY.encode("-A"), LEGACY.eos_id), (), "utf8-raw")[0], "INVALID")
    def test_truth_firewall_and_memory_cost(self):
        case = next(c for c in self.data.cases if c.memory is not None)
        altered = replace(case, truth=replace(case.truth, expected="P" if case.truth.expected != "P" else "A"))
        one = evaluate((case,), ControlPolicy("fixed"), memory_mode="prefetch")
        two = evaluate((altered,), ControlPolicy("fixed"), memory_mode="prefetch")
        self.assertEqual(one["episodes"][0]["trace"], two["episodes"][0]["trace"])
        row = one["episodes"][0]
        self.assertEqual(row["utility"], float(row["outcome"]["correct"]))
        self.assertEqual(one["metrics"]["all"]["memory_requests"], 1)
    def test_paired_interventions_and_all_variant_replay(self):
        from anra_v5.elendril.cognition import OperationCandidate
        from anra_v5.elendril.representation import LEGACY
        from anra_v5.elendril.v6_data import action_text
        class PublicModel:
            origin = 'model'
            representation = 'legacy-ascii'
            model_identity = 'frozen-test-checkpoint'
            def __init__(self, inner=None): self.inner = inner
            def for_case(self, case, *, memory_available=None):
                return PublicModel(ControlPolicy('fixed').for_case(case, memory_available=memory_available))
            def propose(self, state, legal, channels, positions):
                choice = self.inner.propose(state, legal, channels, positions)
                text = action_text(choice.operation, choice.argument).replace('~','-')
                return OperationCandidate(choice.operation, choice.argument, .8, 2, 1, (*LEGACY.encode(text), LEGACY.eos_id))
        result = diagnose(self.data, PublicModel(), positive_control={'source':'synthetic-test','passed':True},
                          weight_identity={'before':'frozen-test-checkpoint','after':'frozen-test-checkpoint'})
        self.assertEqual(result['panels']['P06']['status'], 'PASS')
        self.assertEqual(result['panels']['P14']['status'], 'PASS')
        # Tensor bytes and the checkpoint-manifest SHA are different identities.
        separate = panel_report(result['base'], result['variants'], positive_control={'source':'test','passed':True},
                                weight_identity={'before':'tensor-hash','after':'tensor-hash',
                                                 'checkpoint':'frozen-test-checkpoint'})
        self.assertEqual(separate['P14']['status'], 'PASS')
        separate = panel_report(result['base'], result['variants'], positive_control={'source':'test','passed':True},
                                weight_identity={'before':'tensor-hash','after':'tensor-hash','checkpoint':'wrong-checkpoint'})
        self.assertEqual(separate['P14']['status'], 'NOT_QUALIFIED')
        blind = diagnose(self.data, ControlPolicy('query-blind'), variant_names=('payload-swap',),
                         positive_control={'source':'test','passed':True})
        self.assertEqual(blind['panels']['P06']['status'], 'FAIL')
        for name, record in result['variants'].items():
            cases = diagnostic_cases(self.data, name)
            self.assertEqual(independent_rescore(cases, record), record['metrics'])
        for name in ('payload-swap','memory-swap'):
            cases = diagnostic_cases(self.data, name)
            self.assertTrue(cases)
            for left, right in zip(cases[::2], cases[1::2]):
                self.assertEqual(left.environment.initial, right.environment.initial)
                self.assertEqual(left.pair_id, right.pair_id)
                self.assertNotEqual(left.truth.expected, right.truth.expected)
        broken = panel_report(result['base'], result['variants'], positive_control={'source':'test','passed':True},
                              weight_identity={'before':'frozen-test-checkpoint','after':'changed'})
        self.assertEqual(broken['P14']['status'], 'NOT_QUALIFIED')
    def test_panel_denominators_and_controls(self):
        base = evaluate(self.data, ControlPolicy('fixed'))
        controls = {k: evaluate(self.data, ControlPolicy(k)) for k in ('fixed', 'always-query', 'no-query')}
        reports = panel_report(base, positive_control={'source': 'test', 'passed': True}, controls=controls)
        for panel, kinds in {'P02': ('known',), 'P03': ('missing',), 'P04': ('conflict',), 'P05': ('null','unavailable')}.items():
            self.assertEqual(reports[panel]['samples'], sum(r['kind'] in kinds for r in base['episodes']))
            self.assertEqual(len(reports[panel]['raw_predictions']), reports[panel]['metrics']['episodes'])
        self.assertEqual(reports['P07']['status'], 'FAIL') # fixed cannot exceed itself
        controls['fixed']['cases_sha256'] = 'corrupted'
        self.assertEqual(panel_report(base, positive_control={'source':'test','passed':True}, controls=controls)['P07']['status'], 'NOT_QUALIFIED')
    def test_calibration_is_validation_only(self):
        from anra_v5.elendril.cognition import OperationCandidate
        class Model:
            origin = "model"
            model_identity = "test-confidence-model"
            def for_case(self, case, **kwargs): return self
            def propose(self, state, legal, channels, positions):
                return OperationCandidate("ANSWER", "A", .5, 2, 1, action_tokens("ANSWER", "A"))
        validation = evaluate(self.data, Model())
        calibration = fit_calibration(validation)
        confirmation = make_dataset('confirmation', worlds=16)
        heldout = evaluate(confirmation, Model())
        result = panel_report(heldout, positive_control={'source':'test','passed':True}, calibration=calibration)
        self.assertIn(result['P08']['status'], ('PASS','FAIL'))
        with self.assertRaises(ValueError): panel_report(validation, calibration=calibration)
    def test_validation_firewall(self):
        r = evaluate(self.data, ControlPolicy('fixed'))
        with self.assertRaises(ValueError): validation_risk_bound(r)

if __name__ == '__main__': unittest.main()
