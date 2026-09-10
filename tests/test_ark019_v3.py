from __future__ import annotations
import importlib.util
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'experiments'/'ARK-019'/'ark019_v3_core.py'
spec=importlib.util.spec_from_file_location('ark019_v3_core_test',P); C=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(C)

def m(c=1.0,o=1.0,q=1.0):
    return {'canonical':c,'order_only':o,'query_order':q,'qualified':c>=.90 and o>=.85 and q>=.85,'healthy_margin':c>=.95 and o>=.90 and q>=.90}

def test_static_replay_treatment_is_1of64_level():
    c=C.initial_controller('STATIC_REPLAY_1OF64'); assert C.treatment('STATIC_REPLAY_1OF64',c,1.0)==(64,None)

def test_margin_then_failure_escalation():
    c=C.initial_controller('GUARDIAN_REPLAY')
    C.update_controller('GUARDIAN_REPLAY',c,100,m(o=.89,q=.89),m()); assert c['state']=='PLASTIC'
    C.update_controller('GUARDIAN_REPLAY',c,200,m(o=.89,q=.89),m()); assert c['state']=='SPARSE64'
    C.update_controller('GUARDIAN_REPLAY',c,300,m(c=.89,o=.84,q=.84),m()); assert c['state']=='REPLAY32'

def test_hybrid_persistent_failure_adds_cap():
    c=C.initial_controller('GUARDIAN_HYBRID'); c['state']='REPLAY32'
    C.update_controller('GUARDIAN_HYBRID',c,100,m(c=.89,o=.84,q=.84),m()); assert c['state']=='REPLAY32'
    C.update_controller('GUARDIAN_HYBRID',c,200,m(c=.89,o=.84,q=.84),m()); assert c['state']=='EMERGENCY_CAP16X'
    assert C.treatment('GUARDIAN_HYBRID',c,.123)==(32,.123)

def test_recovery_deescalates():
    c=C.initial_controller('GUARDIAN_HYBRID'); c['state']='EMERGENCY_CAP16X'
    for step in (100,200): C.update_controller('GUARDIAN_HYBRID',c,step,m(),m(c=.1,o=.1,q=.1))
    assert c['state']=='EMERGENCY_CAP16X'
    C.update_controller('GUARDIAN_HYBRID',c,300,m(),m(c=.1,o=.1,q=.1)); assert c['state']=='SPARSE64'

def test_skill_b_confirmation_requires_three():
    c=C.initial_controller('GUARDIAN_REPLAY')
    for step in (100,200): C.update_controller('GUARDIAN_REPLAY',c,step,m(),m())
    assert c['skill_b_confirmation_step'] is None
    C.update_controller('GUARDIAN_REPLAY',c,300,m(),m()); assert c['skill_b_qualification_step']==100 and c['skill_b_confirmation_step']==300

def test_nonidentity_perm_never_identity():
    for i in range(100): assert C.nonidentity_perm(1,i)!=(0,1,2)

def test_query_order_changes_canonical_when_query_was_last():
    facts=((1,4),(2,5),(3,6)); assert C.query_order_perm(facts,3)!=(0,1,2)
