from __future__ import annotations
import importlib.util
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'experiments'/'ARK-019'/'ark019_v4_core.py'
spec=importlib.util.spec_from_file_location('ark019_v4_core_test',P); C=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(C)

def m(c=1.0,o=1.0,q=1.0): return {'canonical':c,'order_only':o,'query_order':q}

def test_dose_selects_smallest_passing_both_parents():
    fail=[{'status':'FAILED','confirmation_step':None,'validation_qualified':False} for _ in range(2)]
    ok=[{'status':'QUALIFIED','confirmation_step':1700,'validation_qualified':True} for _ in range(2)]
    r=C.choose_b_slots({8:fail,12:ok,16:ok}); assert r['status']=='PASS' and r['selected_b_slots']==12

def test_dose_fails_closed_without_two_parent_pass():
    one=[{'status':'QUALIFIED','confirmation_step':1200,'validation_qualified':True}]
    r=C.choose_b_slots({8:one,12:one,16:one}); assert r['status']=='FAIL' and r['selected_b_slots'] is None

def test_static_treatments():
    c=C.initial_controller('PLASTIC_HIGH')
    assert C.treatment('PLASTIC_HIGH',c,.2,1)==(0,None)
    assert C.treatment('STATIC_REPLAY_1OF64',c,.2,1)==(64,None)
    assert C.treatment('STATIC_REPLAY_1OF32',c,.2,1)==(32,None)
    assert C.treatment('STATIC_CAP16X',c,.2,1)==(0,.2)

def test_guardian_warns_before_formal_failure():
    c=C.initial_controller('GUARDIAN_REPLAY')
    C.update_controller('GUARDIAN_REPLAY',c,25,m(.94,.94,.94))
    assert c['state']=='SPARSE64'

def test_guardian_formal_failure_escalates_to_replay32():
    c=C.initial_controller('GUARDIAN_REPLAY')
    C.update_controller('GUARDIAN_REPLAY',c,25,m(.89,.84,.84))
    assert c['state']=='REPLAY32'

def test_hybrid_uses_cap_after_persistent_replay32_failure():
    c=C.initial_controller('GUARDIAN_HYBRID'); c['state']='REPLAY32'
    C.update_controller('GUARDIAN_HYBRID',c,25,m(.89,.84,.84)); assert c['state']=='REPLAY32'
    C.update_controller('GUARDIAN_HYBRID',c,50,m(.89,.84,.84)); assert c['state']=='EMERGENCY_CAP16X'
    assert C.treatment('GUARDIAN_HYBRID',c,.123,51)==(32,.123)

def test_recovery_returns_to_sparse_then_plastic():
    c=C.initial_controller('GUARDIAN_REPLAY'); c['state']='REPLAY32'
    for st in (25,50,75): C.update_controller('GUARDIAN_REPLAY',c,st,m())
    assert c['state']=='REPLAY32'
    C.update_controller('GUARDIAN_REPLAY',c,100,m()); assert c['state']=='SPARSE64'
    for st in (125,150,175): C.update_controller('GUARDIAN_REPLAY',c,st,m())
    assert c['state']=='SPARSE64'
    C.update_controller('GUARDIAN_REPLAY',c,200,m()); assert c['state']=='PLASTIC'

def test_b_confirmation_requires_three_50_step_hits():
    c=C.initial_controller('GUARDIAN_REPLAY')
    C.update_b_confirmation(c,50,m()); C.update_b_confirmation(c,100,m()); assert c['b_confirmation_step'] is None
    C.update_b_confirmation(c,150,m()); assert c['b_qualification_step']==50 and c['b_confirmation_step']==150

def test_dedupe_resume_rows_keeps_one_per_step_and_truncates():
    rows=[{'step':0,'x':0},{'step':100,'x':1},{'step':100,'x':2},{'step':200,'x':3}]
    assert C.dedupe_trajectory(rows,100)==[{'step':0,'x':0},{'step':100,'x':2}]

def test_main_b_failure_has_specific_inconclusive_verdict():
    arms={}
    for ps in C.PRETRAIN_SEEDS:
      for os in C.MAIN_ORDER_SEEDS:
        for arm in C.ARMS:
          arms[f'{ps}:{os}:{arm}']={'parent_seed':ps,'order_seed':os,'arm':arm,'b_confirmation_step':None,
              'final_b_sealed':m(.33,.33,.33),'final_a_sealed':m(), 'control_trace':[], 'measurement_trace':[],
              'final_science_sealed_nll':4.0,'counters':{'protection_updates':0,'replay_slots':0}}
    r=C.decide(arms); assert r['verdict']=='INCONCLUSIVE_MAIN_B_FORMATION_INSTABILITY'

def test_successful_guardian_can_be_promoted_without_zero_failure_requirement():
    arms={}
    for ps in C.PRETRAIN_SEEDS:
      for os in C.MAIN_ORDER_SEEDS:
        for arm in C.ARMS:
          is_ph=arm=='PLASTIC_HIGH'
          is_g=arm.startswith('GUARDIAN')
          ctrl=[]
          for st in range(25,201,25):
            if is_ph: aa=m(.2,.2,.2)
            elif is_g and st<=25: aa=m(.8,.8,.8)
            else: aa=m()
            ctrl.append({'step':st,'a_control':aa})
          meas=[]
          for st in range(0,201,100):
            aa=m(.1,.1,.1) if is_ph and st>0 else m()
            meas.append({'step':st,'a_sealed':aa})
          final_a=m(.1,.1,.1) if is_ph else m()
          arms[f'{ps}:{os}:{arm}']={'parent_seed':ps,'order_seed':os,'arm':arm,'b_confirmation_step':300,
              'final_b_sealed':m(), 'final_a_sealed':final_a,'control_trace':ctrl,'measurement_trace':meas,
              'final_science_sealed_nll':4.0,'counters':{'protection_updates':400 if is_g else 0,'replay_slots':200 if is_g else 0}}
    r=C.decide(arms); assert r['verdict']=='GUARDIAN_CONTINUAL_PROXY_CANDIDATE'; assert r['authorized']
