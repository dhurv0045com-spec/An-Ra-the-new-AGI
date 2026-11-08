"""Offline audit rejects altered science while declaring binary64 roundoff."""
import copy
import math
import pytest
import torch
from types import SimpleNamespace
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v6_evaluation import NeuralPolicy
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v71_evaluation import evaluate_suite
from tools.analyze_elendril_v71_results import portable_replay


@pytest.fixture
def evidence():
    suite={'base':make_dataset('validation',worlds=7).cases}
    class Backend:
        binding=SimpleNamespace(tokenizer_sha256=UTF8.identity,vocabulary_size=UTF8.vocabulary_size,
            checkpoint_sha256='engineering-fixture',identity='fixture',context_length=512)
        def next_distribution(self,ids):
            d=torch.full((UTF8.vocabulary_size,),.0001,dtype=torch.float64)
            d[UTF8.eos_id if ids[-1]==UTF8.encode('A')[0] else UTF8.encode('A')[0]]=.9
            return d
    return suite,evaluate_suite(suite,NeuralPolicy(Backend()))


def test_two_ulp_derived_brier_allowance_preserves_original(evidence):
    suite,report=evidence
    value=report['metrics']['base']['all']['calibration']['brier']
    report['metrics']['base']['all']['calibration']['brier']=math.nextafter(value,math.inf)
    original=copy.deepcopy(report)
    differences=portable_replay(suite,report,name='engineering-fixture')
    assert len(differences)==1 and report==original


@pytest.mark.parametrize('mutation',['brier','count','confidence'])
def test_changed_metrics_and_raw_confidence_rejected(evidence,mutation):
    suite,report=evidence
    if mutation=='brier': report['metrics']['base']['all']['calibration']['brier']+=.001
    elif mutation=='count': report['metrics']['base']['all']['correct']+=1
    else: report['records']['base']['raw']['episodes'][0]['confidence']=.314159
    with pytest.raises(ValueError): portable_replay(suite,report)
