import json
import pytest

from bramastra_lab.research.contracts import Action, PublicObservation
from bramastra_lab.research.environments import (
    EnvironmentError, InventoryEnvironment, ProgramEnvironment, ProgramTask,
    SwitchEnvironment, SwitchTask, inventory_fixture, parity_synergy_information,
)
from bramastra_lab.research.environments.qualification import fixture_tasks, qualification


def act(obs, kind, args): return Action(obs.episode_id,obs.step_id,kind,args,"test-policy")


def test_shared_contract_replay_and_public_record_has_no_hidden_references():
    envs=[SwitchEnvironment(fixture_tasks()["training"][0]), InventoryEnvironment(inventory_fixture()),
          ProgramEnvironment(ProgramTask("add",2,4))]
    for env in envs:
        a=env.reset(7); b=env.reset(7)
        assert isinstance(a,PublicObservation) and a.to_dict()==b.to_dict()
        public=json.dumps(a.to_dict())
        assert "gates" not in public and "constant" not in public and "pending" not in public
        assert not any(hasattr(v,"__dict__") for v in a.to_dict().values())


def test_switch_truth_target_protection_repeat_typing_and_budget():
    task=SwitchTask(("x","y"),(("z","xor",("x","y")),),(1,1),2)
    env=SwitchEnvironment(task); obs=env.reset(1)
    with pytest.raises(EnvironmentError): env.step(act(obs,"query",{"input":[1,1]}))
    with pytest.raises(EnvironmentError): env.step(act(obs,"query",{"input":[True,0]}))
    result=env.step(act(obs,"query",{"input":[1,0]})); obs=result.observation
    with pytest.raises(EnvironmentError): env.step(act(obs,"query",{"input":[1,0]}))
    result=env.step(act(obs,"predict",{"value":0}))
    assert result.termination and result.reward==1 and result.observation.remaining_budget==0
    with pytest.raises(EnvironmentError): env.step(act(result.observation,"predict",{"value":0}))


def test_inventory_prerequisite_delayed_effect_and_overwrite_require_memory():
    env=InventoryEnvironment(inventory_fixture()); obs=env.reset(2)
    r=env.step(act(obs,"apply",{"operation":"smelt"})); assert not r.observation.feedback["accepted"]
    obs=r.observation; obs=env.step(act(obs,"apply",{"operation":"gather"})).observation
    obs=env.step(act(obs,"apply",{"operation":"smelt"})).observation
    # Delayed bar arrives on next step, then paint overwrites it.
    obs=env.step(act(obs,"apply",{"operation":"paint"})).observation
    obs=env.step(act(obs,"inspect",{})).observation; assert obs.feedback["slot"]=="red"


def test_program_finite_interpreter_target_repeat_and_scoring():
    env=ProgramEnvironment(ProgramTask("mul",3,4,3)); obs=env.reset(9)
    with pytest.raises(EnvironmentError): env.step(act(obs,"test",{"input":4}))
    obs=env.step(act(obs,"test",{"input":2})).observation; assert obs.feedback["output"]==6
    with pytest.raises(EnvironmentError): env.step(act(obs,"test",{"input":2}))
    r=env.step(act(obs,"predict",{"value":12})); assert r.reward==1 and r.termination


def test_semantic_splits_and_structural_holdouts_are_deduplicated():
    q=qualification(); ids=[]
    for split,tasks in fixture_tasks().items():
        for task in tasks: ids.append(task.semantic_id)
    assert len(ids)==len(set(ids)) and q["semantic_duplicate_count"]==0
    assert q["shortcut_diagnostics"]["held_out_composition_count"]==1


def test_exact_parity_synergy_and_oracle_learnability():
    gains=parity_synergy_information(); assert gains[(1,0)]==0 and gains[(0,1)]==0
    after=parity_synergy_information((((1,0),1),)); assert after[(0,1)]==1
    q=qualification()
    assert all(x["oracle_success_rate"]==1 for x in q["families"].values())
    assert all(x["ambiguous_tasks"]==0 for x in q["families"].values())
