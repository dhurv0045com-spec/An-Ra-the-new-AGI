"""No models: validate offline replay and detection of the archived tool-state defect."""
from copy import deepcopy
from hashlib import sha256

import pytest

from tools.analyze_elendril_v8_results import audit_episode


def episode(*, stale=False):
    initial = [{'subject':'B','source':'map','tick':1,'value':'C'}]
    payload = [{'subject':'A','source':'s0','tick':2,'value':'F'}]
    row = {'goal_subject':'A','hops':1,'initial_facts':initial,
           'allowed_public_tools':{'query_available':True,'memory_available':False},
           'query_payload':payload,'memory_payload':[],'expected_answer':'F',
           'mode':'raw','correct':True,'exact_match':True,'first_divergence_step':None,'steps':[]}
    for idx,(op,arg) in enumerate([('QUERY',None),('ANSWER','F')]):
        q = int(idx==0 or stale)
        text = f'Current value of A? Query={q} Memory=0\n(map,1,B,C)'
        if idx: text += '; (s0,2,A,F)'
        p = {'operation':op,'argument':arg}
        row['steps'].append({'public_text_utf8':text,'public_text_sha256':sha256(text.encode()).hexdigest(),
                             'ideal_operation':op,'ideal_argument':arg,'executed_proposal':p,
                             'raw_proposal':dict(p),'tool_responses':payload if idx==0 else []})
    return row


def test_correct_tool_transition_replays_without_contract_violations():
    observations, violations = audit_episode(episode())
    assert len(observations)==2
    assert violations == {}


def test_stale_tool_flag_is_detected_even_if_final_answer_is_correct():
    _, violations = audit_episode(episode(stale=True))
    assert violations == {'stale_availability_steps':1}


@pytest.mark.parametrize('mutation', ['score','fact','teacher'])
def test_replay_rejects_false_evidence(mutation):
    row=deepcopy(episode())
    if mutation=='score': row['correct']=False
    elif mutation=='fact': row['initial_facts'][0]['value']='D'
    else: row['steps'][1]['ideal_argument']='H'
    with pytest.raises(ValueError): audit_episode(row)
