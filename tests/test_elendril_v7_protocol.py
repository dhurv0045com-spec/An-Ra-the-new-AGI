import math
import pytest
from anra_v5.elendril.codec import digest
from anra_v5.elendril.v7_protocol import PROTOCOL_SHA,protocol,jobs,milestones,validate_job


def test_frozen_design_balanced_and_rounds_actual_updates():
    assert digest(protocol())==PROTOCOL_SHA
    assert len(jobs())==9 and len({j['job_id'] for j in jobs()})==9
    for i,seed in enumerate((7,19,41)):
        assert {j['method'] for j in jobs()[3*i:3*i+3]}=={'uniform','blocked','interleaved'}
        assert all(j['seed']==seed for j in jobs()[3*i:3*i+3])
    assert milestones(9472)==(74,148,222,296,592,1184,2368)
    assert milestones(17)[-1]==4*math.ceil(17/16)
    with pytest.raises(ValueError): validate_job({'job_id':'invented','method':'uniform','seed':7})
    with pytest.raises(ValueError): milestones(0)
