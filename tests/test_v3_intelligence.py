from __future__ import annotations

import torch
from torch import nn

from intelligence.competence import CalibratedCompetenceModel
from intelligence.curiosity import CuriosityCandidate, CuriosityEngine
from intelligence.hgp import (
    HierarchicalGoalPlanner,
    MissionNode,
    MissionTree,
    VerificationRule,
    WorkflowStep,
)
from intelligence.ogrs import OnlineGoalRegulationSystem
from intelligence.proof_memory import CausalProofMemory, ProofRecord
from intelligence.verifier_search import VerificationOutcome, VerifierSearch
from training.continual import attach_candidate_adapters


def test_hgp_validates_and_compiles_workflow() -> None:
    child = MissionNode(
        node_id="task-1",
        title="Inspect",
        objective="inspect workspace",
        level=1,
        verification=(VerificationRule("test", "workspace inspected"),),
    )
    tree = MissionTree(
        goal="complete task",
        root=MissionNode(
            node_id="root",
            title="Mission",
            objective="complete task",
            level=0,
            children=[child],
        ),
    )
    planner = HierarchicalGoalPlanner()
    workflow = planner.compile_workflow(
        tree,
        lambda node: WorkflowStep(
            skill="inspect",
            parameters={},
            preconditions=(),
            postconditions=("inspected",),
            timeout_seconds=30,
            recovery="replan",
            source_node=node.node_id,
        ),
    )
    assert workflow[0].source_node == "task-1"


def test_competence_curiosity_and_ogrs_are_bounded() -> None:
    competence = CalibratedCompetenceModel()
    for _ in range(30):
        competence.update("math", correct=True, confidence=0.9, verified=True)
    assert competence.policy("math") in {"direct", "verify"}
    curiosity = CuriosityEngine().propose(
        [CuriosityCandidate("math", old_loss=1.0, new_loss=0.7, novelty=0.8, verifiability=1.0)]
    )
    assert curiosity and curiosity["priority"] == "below_owner"
    regulation = OnlineGoalRegulationSystem().regulate(0.8)
    assert regulation.weight_updates_allowed is False
    assert 0.0 <= regulation.rim_scale <= 1.0


def test_proof_memory_and_verifier_search(tmp_path) -> None:
    memory = CausalProofMemory(tmp_path / "proofs.jsonl")
    memory.add(
        ProofRecord(
            proof_id="p1",
            claim="2+2=4",
            evidence=("arithmetic",),
            assumptions=(),
            derivation="addition",
            verifier="symbolic",
            confidence=1.0,
        )
    )
    assert memory.active()[0].proof_id == "p1"
    search = VerifierSearch()
    search.register(
        "symbolic",
        lambda claim: VerificationOutcome(claim == "2+2=4", 1.0, "symbolic", "checked"),
    )
    assert search.verify("2+2=4", "math").passed


def test_continual_adapters_freeze_base_and_train_only_candidates() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    attached = attach_candidate_adapters(model, rank=2)
    assert attached == ["0", "2"]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert all("lora_" in name or "magnitude" in name for name in trainable)
    output = model(torch.randn(3, 4))
    assert output.shape == (3, 2)
