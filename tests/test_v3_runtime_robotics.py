from __future__ import annotations

import torch

from inference.kv_cache import TieredKVCache
from inference.prefix_cache import PrefixCache
from inference.speculative import SpeculativeBenchmark, accept_draft_prefix
from robotics.contracts import SkillGoal, SkillResult, Workflow, WorkflowState
from robotics.domain_randomization import sample_domain
from robotics.workflow import WorkflowExecutor
from robotics.world_model import PredictiveWorldModel


def test_awkc_never_evicts_identity_critical_segment() -> None:
    cache = TieredKVCache(max_tokens=4)
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn(1, 2, 2, 4)
    cache.update(0, key, value, token_start=0, identity_critical=True)
    cache.update(0, key, value, token_start=2, salience=0.0)
    cache.update(0, key, value, token_start=4, salience=1.0)
    segments = cache.layers[0]
    assert any(segment.identity_critical for segment in segments)
    assert sum(segment.tokens for segment in segments) <= 4


def test_prefix_and_speculative_promotion_contracts() -> None:
    cache = PrefixCache(max_entries=2)
    cache.put("m", [1, 2], {"kv": True})
    assert cache.get("m", [1, 2]) == {"kv": True}
    assert accept_draft_prefix([1, 2, 8], [1, 2, 3]) == 2
    benchmark = SpeculativeBenchmark(100, 40, 3.0, 1.5)
    assert benchmark.promotion_allowed


def test_robotics_workflow_and_world_model() -> None:
    condition_state = {"ready": True, "done": True}
    executor = WorkflowExecutor(
        lambda goal: SkillResult(success=True),
        lambda condition: condition_state.get(condition, False),
    )
    workflow = Workflow(
        workflow_id="w1",
        goal="test",
        skills=[
            SkillGoal(
                "move",
                {},
                ("ready",),
                ("done",),
                1.0,
            )
        ],
    )
    result = executor.execute(workflow)
    assert result.state == WorkflowState.COMPLETED

    model = PredictiveWorldModel(state_dim=8, action_dim=4, hidden_dim=16)
    prediction = model(torch.randn(2, 8), torch.randn(2, 4))
    assert prediction["next_state"].shape == (2, 8)
    assert (prediction["epistemic_uncertainty"] >= 0).all()


def test_domain_randomization_is_seeded() -> None:
    assert sample_domain(42) == sample_domain(42)
