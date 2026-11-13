from types import SimpleNamespace
import pytest
import torch

from anra_v5.elendril.cognition import CognitiveState
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v71_policy import GrammarPolicy


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = []
        self.binding = SimpleNamespace(tokenizer_sha256=UTF8.identity,
            vocabulary_size=UTF8.vocabulary_size, checkpoint_sha256='fake-checkpoint',
            identity='fake', context_length=512)
    def next_distribution(self, ids):
        self.inputs.append(tuple(ids))
        return self.outputs[len(self.inputs)-1]


def dist(**values):
    x = torch.full((UTF8.vocabulary_size,), .001, dtype=torch.float64)
    for token, probability in values.items():
        x[UTF8.eos_id if token == 'EOS' else UTF8.encode(token)[0]] = probability
    return x


def test_grammar_keeps_original_probabilities_and_wrong_answer():
    case = next(c for c in make_dataset('validation', worlds=14).cases
                if c.truth.expected not in (None, 'A'))
    state = CognitiveState(case.goal).observe(case.environment.initial)
    backend = FakeBackend([dist(A=.20, Q=.10), dist(EOS=.25)])
    actor = GrammarPolicy(backend).for_case(case, memory_available=False)
    result = actor.propose(state, {'ANSWER', 'ABSTAIN'}, (), 2000)
    assert (result.operation, result.argument) == ('ANSWER', 'A')
    assert result.argument != case.truth.expected
    assert result.raw_tokens == (UTF8.encode('A')[0], UTF8.eos_id)
    assert result.predicted_success == pytest.approx(.20 * .25)
    assert result.model_calls == 2
    assert result.model_positions == sum(map(len, backend.inputs))
    assert len(actor.prompts) == 1


def test_grammar_deliberation_three_tokens_and_budget():
    case = make_dataset('validation', worlds=7).cases[0]
    state = CognitiveState(case.goal).observe(case.environment.initial)
    backend = FakeBackend([dist(**{'~': .4}), dist(B=.3), dist(EOS=.2)])
    actor = GrammarPolicy(backend).for_case(case, memory_available=False)
    result = actor.propose(state, {'DELIBERATE'}, (), 2000)
    assert (result.operation, result.argument) == ('DELIBERATE', 'B')
    assert result.raw_tokens == (*UTF8.encode('~B'), UTF8.eos_id)
    assert result.model_calls == 3
    assert result.model_positions == sum(map(len, backend.inputs))
    assert result.predicted_success == pytest.approx(.4*.3*.2)
    assert GrammarPolicy(FakeBackend([])).for_case(case).propose(state, {'ANSWER'}, (), 1) is None
