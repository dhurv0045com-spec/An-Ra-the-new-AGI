import copy
import unittest
from dataclasses import replace

from anra_v5.elendril.cognition import OperationCandidate
from anra_v5.elendril.v6_data import make_dataset, render_public
from anra_v5.elendril.v6_evaluation import ControlPolicy, action_tokens
from anra_v5.elendril.v7_diagnostics import _cohorts, diagnose, verify_diagnosis


class PublicModel:
    origin = 'model'
    model_identity = 'frozen-test-checkpoint'

    def for_case(self, public, *, memory_available=None):
        assert not hasattr(public, 'truth') and not hasattr(public, 'environment')
        result = PublicModel()
        result.public = public
        result.memory_available = memory_available
        result.inner = ControlPolicy('fixed').for_case(public, memory_available=memory_available)
        result.prompts = []
        return result

    def propose(self, state, legal, channels, positions):
        self.prompts.append(render_public(state, channels, self.public.surface,
                                          self.memory_available and 'RETRIEVE' in legal))
        choice = self.inner.propose(state, legal, channels, positions)
        return OperationCandidate(choice.operation, choice.argument, .8, 1, 1,
                                  action_tokens(choice.operation, choice.argument))


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_dataset('validation', worlds=14)
        cls.result = diagnose(cls.data, PublicModel())

    def test_payload_and_swap_construction(self):
        cohorts = _cohorts(self.data)
        for original, supplied in zip(cohorts['interactive'], cohorts['supplied']):
            payload = tuple(e for p in original.environment.channels.values() for e in p)
            self.assertEqual(supplied.environment.initial, original.environment.initial + payload)
            self.assertEqual(dict(supplied.environment.channels), {})
            self.assertEqual(supplied.truth, original.truth)
        swaps = cohorts['interactive-swap']
        for left, right in zip(swaps[::2], swaps[1::2]):
            self.assertEqual(left.environment.initial, right.environment.initial)
            self.assertNotEqual(left.truth.expected, right.truth.expected)
            self.assertEqual(left.pair_id, right.pair_id)

    def test_replay_and_corruption(self):
        self.assertEqual(verify_diagnosis(self.data, self.result), self.result['findings'])
        self.assertTrue(any(x['queried_then_prompt_identical'] is True for x in self.result['episodes']))
        for change in ('trace', 'summary', 'payload'):
            bad = copy.deepcopy(self.result)
            if change == 'trace':
                bad['records']['interactive']['raw']['episodes'][0]['trace']['answer'] = 'Z'
            elif change == 'summary':
                bad['findings']['interactive']['nonnull_correct'] += 1
            else:
                bad['records']['supplied']['raw']['episodes'][0]['trace']['state']['events'][0]['value'] = 'Z'
            with self.assertRaises(ValueError):
                verify_diagnosis(self.data, bad)

    def test_fixed_control_and_truth_firewall(self):
        result = diagnose(self.data, ControlPolicy('fixed'))
        self.assertEqual(result['findings']['interactive']['nonnull_correct'], result['findings']['interactive']['nonnull_total'])
        case = self.data.cases[0]
        altered = replace(case, truth=replace(case.truth, expected='P' if case.truth.expected != 'P' else 'A'))
        before = diagnose((case,), PublicModel())
        after = diagnose((altered,), PublicModel())
        for key in ('interactive', 'supplied'):
            self.assertEqual(before['records'][key]['raw']['episodes'][0]['trace'], after['records'][key]['raw']['episodes'][0]['trace'])

    def test_bounded_posthoc_probe(self):
        import torch
        from types import SimpleNamespace
        from anra_v5.elendril.representation import UTF8
        class Backend:
            binding = SimpleNamespace(checkpoint_sha256='frozen-test-checkpoint', identity='other', context_length=512)
            def next_distribution(self, ids):
                self_assert = not torch.is_grad_enabled()
                if not self_assert:
                    raise AssertionError('probe must disable gradients')
                return torch.full((UTF8.vocabulary_size,), 1 / UTF8.vocabulary_size)
        result = diagnose(self.data, PublicModel(), backend=Backend())
        self.assertEqual(len(result['token_probes']), 8)
        self.assertEqual(verify_diagnosis(self.data, result, backend=Backend()), result['findings'])
        missing = copy.deepcopy(result)
        missing['token_probes'] = []
        with self.assertRaises(ValueError):
            verify_diagnosis(self.data, missing, backend=Backend())
        changed = copy.deepcopy(result)
        changed['token_probes'][0]['teacher_forced_eos_probability'] = .25
        changed['token_probes'][0]['teacher_forced_value_to_eos_nll'] = -__import__('math').log(changed['token_probes'][0]['gold_first_probability'] * .25)
        with self.assertRaises(ValueError):
            verify_diagnosis(self.data, changed, backend=Backend())
        result['token_probes'][0]['gold_first_nll'] = 0
        with self.assertRaises(ValueError):
            verify_diagnosis(self.data, result)


if __name__ == '__main__':
    unittest.main()
