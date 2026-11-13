from collections import Counter

import pytest

from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_data import make_dataset
from anra_v5.elendril.v717_measurement import audit_targets
from tools.analyze_elendril_v717_results import audit_measurement, analyze, small_measurement


def perfect(data):
    values=set(UTF8.encode('ABCDEFGHIJKLMNOP')); rows=[]; decoded=[]
    for s in data.samples:
        start=s.eligible.index(True); target=list(s.tokens[start:])
        decoded.append(dict(task_id=s.task_id,category=s.category,expected=target,generated=target,exact=True))
        for i in range(start,len(s.tokens)):
            token=s.tokens[i]
            rows.append((s.category,token,i==start,token in values,token==UTF8.eos_id))
    def summary(rows):
        result={}
        for role,position in (('all',None),('first_action',2),('value',3),('eos',4)):
            counts=Counter(str(r[1]) for r in rows if position is None or r[position]); n=sum(counts.values())
            result[role]=dict(tokens=n,correct=n,accuracy=1. if n else None,cross_entropy=0. if n else None,
                target_distribution=dict(sorted(counts.items())),confusion={k+'->'+k:v for k,v in sorted(counts.items())})
        return result
    n=len(data.samples)
    return dict(examples=n,audit=audit_targets(data),teacher_forced=summary(rows),
        by_kind={k:summary([r for r in rows if r[0]==k]) for k in sorted({r[0] for r in rows})},
        teacher_forced_exact_actions=dict(correct=n,examples=n,accuracy=1.),
        decoded=dict(status='MEASURED',examples=n,records=decoded,correct=n,accuracy=1.),
        teacher_minus_decode_exact_accuracy=0.)


@pytest.fixture(scope='module')
def data(): return make_dataset(worlds=8)


def test_independent_target_and_decoded_arithmetic(data):
    audit_measurement(perfect(data),data,tuple(range(len(data.samples))),decoded=True)


def test_final_summary_observation_stage_counts(data):
    p=perfect(data); r={'train':{'teacher':p,'decoded':p},'heldout':{'teacher':p,'decoded':p}}
    summary=small_measurement(r,final=True,datasets={'train':data,'heldout':data})
    for label in ('train','heldout'):
        rows=summary[label]['decoded_by_observation_stage'].values()
        assert sum(v['examples'] for v in rows)==len(data.samples)
        assert all(v['examples']==v['correct'] for v in rows)


@pytest.mark.parametrize('corruption',['count','target','prediction','accuracy','loss'])
def test_measurement_corruption_is_rejected(data,corruption):
    r=perfect(data)
    if corruption=='count': r['teacher_forced']['eos']['tokens']+=1
    elif corruption=='target': r['decoded']['records'][0]['expected']=[999]
    elif corruption=='prediction': r['decoded']['records'][0]['generated']=[999]
    elif corruption=='accuracy': r['teacher_forced']['first_action']['accuracy']=0.
    else: r['teacher_forced']['value']['cross_entropy']=float('nan')
    with pytest.raises(ValueError): audit_measurement(r,data,tuple(range(len(data.samples))),decoded=True)


def test_unexpected_protocol_stops_before_scientific_analysis(tmp_path,monkeypatch):
    monkeypatch.setattr('tools.analyze_elendril_v717_results.verify_zip',lambda _:dict(source='a'*40,protocol_sha256='b'*64,campaign_status='EXECUTED'))
    with pytest.raises(ValueError,match='identity'): analyze(tmp_path/'archive.zip')
