from __future__ import annotations

import torch

from memory.memory_router import MemoryRouter  # noqa: F401 - establishes legacy import order
from multimodal.vision import InHouseVisionEncoder, VisionSoftTokenProjector
from retrieval import RetrievalQuery
from retrieval.trained import TwoTowerRetriever
from robotics.rollout import rollout_actions
from robotics.world_model import PredictiveWorldModel
from self_modification.proposal_ladder import SelfDevelopmentProposal, evaluate_proposal_only
from training.moonshot_architectures import LatentReasoningChannel, StateSpaceMixer
from verification import DEFAULT_VERIFIER_REGISTRY


def test_m1_and_m3_pilot_modules_preserve_shapes() -> None:
    values = torch.randn(2, 5, 8)
    assert StateSpaceMixer(8)(values).shape == values.shape
    assert LatentReasoningChannel(8, latent_steps=2)(values).shape == (2, 8)


def test_m2_in_house_vision_emits_soft_tokens() -> None:
    tokens = InHouseVisionEncoder(width=8, patch_size=4)(torch.randn(2, 3, 8, 8))
    assert VisionSoftTokenProjector(8, 12)(tokens).shape == (2, 4, 12)


def test_m4_rollout_is_offline_only() -> None:
    report = rollout_actions(
        PredictiveWorldModel(state_dim=8, action_dim=4, hidden_dim=16),
        {"x": 1},
        [{"move": "left"}],
    )
    assert report["offline_only"] is True and len(report["steps"]) == 1


def test_m5_m6_m7_stay_behind_their_existing_safety_protocols() -> None:
    retriever = TwoTowerRetriever(4)
    retriever.index([("a", "alpha", torch.tensor([1.0, 0.0, 0.0, 0.0]))])
    assert retriever.search(RetrievalQuery("q", limit=1, vector=[1, 0, 0, 0]))[0].id == "a"
    proof = DEFAULT_VERIFIER_REGISTRY.verify(
        "formal_proof",
        {
            "premises": ["a"],
            "rules": ["a -> b"],
            "steps": ["b"],
            "conclusion": "b",
        },
    )
    assert proof.score == 1.0
    injected = DEFAULT_VERIFIER_REGISTRY.verify(
        "formal_proof",
        {"premises": ["a"], "rules": [], "steps": ["b"], "conclusion": "b"},
    )
    assert injected.score == 0.0
    outcome = evaluate_proposal_only(SelfDevelopmentProposal("p1", "tests", 0.1, True))
    assert outcome["eligible_for_human_review"] is True and outcome["auto_apply"] is False
