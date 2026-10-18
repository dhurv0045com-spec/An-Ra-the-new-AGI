from __future__ import annotations

from pathlib import Path

import pytest

from discovery_gym.baselines import (
    InformationGainPolicy,
    NoExperimentPolicy,
    OraclePolicy,
    RandomPolicy,
    _target_candidates,
    policy_factories,
)
from discovery_gym.contracts import Action, ActionCommitment, Policy, PublicEvent, content_hash
from discovery_gym.environment import DiscoveryEnvironment
from discovery_gym.evaluation import run_episode, run_suite, score_episode
from discovery_gym.firewall import (
    FirewallViolation,
    assert_commitment_precedes_outcome,
    assert_no_forbidden_values,
    assert_no_hidden_values,
    assert_public_payload,
    assert_public_view,
)
from discovery_gym.generator import generate_suite, suite_manifest
from discovery_gym.protocol import freeze_dg0_protocol, protocol_template
from discovery_gym.receipts import make_identity, no_checkpoint_identity, result_receipt, write_json_exclusive
from discovery_gym.rules import all_assignments, overridden_assignment
from discovery_gym.audits import audit_suite


def test_generator_splits_are_structurally_disjoint_and_deterministic():
    first = generate_suite(seed=1701, worlds_per_split=6)
    second = generate_suite(seed=1701, worlds_per_split=6)
    assert first == second
    manifest = suite_manifest(first, seed=1701)
    assert manifest.structure_ids["train"] == ["and2", "threshold2", "xor2"]
    assert manifest.structure_ids["transfer"] == ["mux", "neg_threshold3", "xor_and"]
    assert manifest.affordance_ids["transfer"] == ["read,probe,submit,stop"]
    assert all("intervene" not in world.legal_action_kinds for world in first["transfer"])
    structure_sets = [set(values) for values in manifest.structure_ids.values()]
    for index, left in enumerate(structure_sets):
        for right in structure_sets[index + 1 :]:
            assert not left & right
    semantic_sets = [
        {world.rule.semantic_sha256 for world in first[split]}
        for split in first
    ]
    for index, left in enumerate(semantic_sets):
        for right in semantic_sets[index + 1 :]:
            assert not left & right


def test_public_view_has_no_hidden_truth_and_target_is_not_directly_queryable():
    world = generate_suite(seed=1701)["train"][0]
    environment = DiscoveryEnvironment(world, budget=8)
    view = environment.reset()
    assert_public_view(view)
    assert_public_payload(view.to_dict())
    assert_no_hidden_values(view, world)
    assert all(tuple(descriptor.arguments.get("context", ())) != world.target_assignment for descriptor in view.legal_actions)
    assert world.rule.structure_id not in repr(view.to_dict())
    assert world.target_assignment not in view.to_dict()["probe_contexts"]
    assert world.target_assignment in world.manifest.target_candidate_contexts
    assert set(world.probe_contexts).isdisjoint(world.manifest.target_candidate_contexts)


def test_action_identity_and_binary_contracts_reject_mutation():
    action = Action.make("probe", context=[0, 0, 0], prediction=0)
    serialized = action.to_dict()
    serialized["arguments"]["context"] = [0, 0, 1]
    with pytest.raises(ValueError):
        Action.from_mapping(serialized)
    world = generate_suite(seed=1701)["train"][0]
    with pytest.raises(ValueError):
        world.rule.evaluate((0.0, 0, 0))
    with pytest.raises(ValueError):
        overridden_assignment((0.0, 0, 0), 0, 0)
    with pytest.raises(ValueError):
        Action.make("submit", answer=0, prediction=0.0)


def test_public_firewall_checks_structured_hidden_values_and_commitments():
    world = generate_suite(seed=1701)["train"][0]
    with pytest.raises(FirewallViolation):
        assert_no_forbidden_values({"safe": list(world.target_assignment)}, [list(world.target_assignment)])
    environment = DiscoveryEnvironment(world, budget=8)
    environment.reset()
    read = next(item for item in environment.public_view().legal_actions if item.kind == "read")
    environment.step(Action.make("read", prediction=0, **dict(read.arguments)))
    commitment_data = environment.commitments[0].to_dict()
    commitment_data["available_view"]["total_cost"] = 99
    commitment_data["available_view_sha256"] = content_hash(commitment_data["available_view"])
    tampered = ActionCommitment(
        step_id=commitment_data["step_id"],
        action=Action.from_mapping(commitment_data["action"]),
        available_view=commitment_data["available_view"],
        available_view_sha256=commitment_data["available_view_sha256"],
    )
    with pytest.raises(FirewallViolation):
        assert_commitment_precedes_outcome((tampered,), environment.transcript)


def test_target_candidates_respect_public_probe_complement():
    world = generate_suite(seed=1701)["train"][0]
    view = DiscoveryEnvironment(world, budget=8).reset()
    candidates = set(_target_candidates(view))
    assert candidates.isdisjoint(set(view.probe_contexts))
    assert candidates | set(view.probe_contexts) == set(all_assignments())


def test_intervention_affordance_cannot_reach_any_target_candidate():
    world = generate_suite(seed=1701)["train"][0]
    assert set(world.intervention_contexts).issubset(set(world.probe_contexts))
    for context in world.intervention_contexts:
        for variable in range(3):
            for value in (0, 1):
                assert overridden_assignment(context, variable, value) not in world.manifest.target_candidate_contexts


class InvalidStopThenStopPolicy(Policy):
    identity = "invalid-stop-then-stop/v1"
    uses_hidden_truth = False

    def __init__(self) -> None:
        self._first = True

    def act(self, view):
        if self._first:
            self._first = False
            return Action.make("stop", prediction=0)
        return Action.make("stop")


def test_repeated_and_target_actions_are_rejected_and_charged():
    world = generate_suite(seed=1701)["train"][0]
    environment = DiscoveryEnvironment(world, budget=8)
    view = environment.reset()
    descriptor = next(item for item in view.legal_actions if item.kind == "probe")
    action = Action.make("probe", prediction=0, **dict(descriptor.arguments))
    assert environment.step(action).accepted
    repeated = environment.step(Action.make("probe", prediction=0, **dict(descriptor.arguments)))
    assert not repeated.accepted
    assert repeated.event.cost == 2
    target = Action.make("probe", prediction=0, context=list(world.target_assignment))
    rejected = environment.step(target)
    assert not rejected.accepted
    assert rejected.event.cost == 2
    assert environment.total_cost == 6
    assert sum(event.cost for event in environment.transcript) == environment.total_cost


def test_zero_cost_rejection_is_recomputed_without_corrupting_budget_ledger():
    world = generate_suite(seed=1701)["train"][0]
    record = run_episode(world, InvalidStopThenStopPolicy(), budget=2)
    assert record["total_cost"] == 0
    assert record["events"][0]["accepted"] is False
    assert record["evaluator_recomputation"]["agreement"]
    assert record["legality_recomputation"]["agreement"]


def test_commitment_is_written_before_feedback_and_independent_evaluator_recomputes():
    world = generate_suite(seed=1701)["train"][0]
    with pytest.raises(ValueError):
        run_episode(world, OraclePolicy(world), budget=8)
    record = run_episode(world, OraclePolicy(world), budget=8, allow_hidden_truth=True)
    assert record["success"]
    assert "success" not in record["events"][-1]["feedback"]
    assert record["evaluator_recomputation"]["agreement"]
    assert record["legality_recomputation"]["agreement"]
    assert len(record["commitments"]) == len(record["events"])
    assert record["commitments"][0]["available_view"]["step_id"] == 0
    assert "feedback" in record["events"][0]
    event_data = record["events"][0]
    tampered_feedback = dict(event_data["feedback"])
    tampered_feedback["evaluator_label"] = "forged"
    tampered_event = PublicEvent(
        step_id=event_data["step_id"],
        action=Action.from_mapping(event_data["action"]),
        accepted=event_data["accepted"],
        cost=event_data["cost"],
        feedback=tampered_feedback,
    )
    commitment_data = record["commitments"][0]
    commitment = ActionCommitment(
        step_id=commitment_data["step_id"],
        action=Action.from_mapping(commitment_data["action"]),
        available_view=commitment_data["available_view"],
        available_view_sha256=commitment_data["available_view_sha256"],
    )
    tampered = score_episode(world, (tampered_event,), (commitment,), policy=OraclePolicy(world), budget=8)
    assert not tampered["evaluator_recomputation"]["agreement"]


def test_transparent_information_gain_uses_legal_evidence_without_oracle_access():
    suite = generate_suite(seed=1701, worlds_per_split=6)
    run = run_suite(
        suite,
        {
            "information_gain": lambda world: InformationGainPolicy(),
            "no_experiment": lambda world: NoExperimentPolicy(),
            "random": lambda world: RandomPolicy(1702),
            "oracle": lambda world: OraclePolicy(world),
        },
        budget=8,
        allow_hidden_truth=True,
    )
    summary = run["summary"]["overall"]
    assert summary["transparent-information-gain/v1"]["success_rate"] > summary["random-legal/v1"]["success_rate"]
    assert summary["transparent-information-gain/v1"]["mean_experiment_cost"] > 0.0
    assert summary["evaluator-oracle/v1"]["success_rate"] == 1.0
    assert summary["transparent-information-gain/v1"]["budget_overrun_rate"] == 0.0
    with pytest.raises(ValueError):
        run_suite(
            {"train": (next(iter(suite["train"])),)},
            {
                "first": lambda world: InformationGainPolicy(),
                "second": lambda world: InformationGainPolicy(),
            },
            budget=8,
        )


def test_dg0_audit_passes_and_records_shortcut_rates():
    suite = generate_suite(seed=1701, worlds_per_split=6)
    manifest = suite_manifest(suite, seed=1701)
    protocol = freeze_dg0_protocol(manifest, budget=8).to_dict()
    run = run_suite(suite, policy_factories(1701), budget=8, allow_hidden_truth=True)
    audit = audit_suite(suite, run, protocol)
    assert audit["status"] == "PASS"
    legality = next(item for item in audit["checks"] if item["check"] == "legality_and_cost_accounting")
    assert legality["details"]["worlds_checked"] == 24
    assert next(item for item in audit["checks"] if item["check"] == "shortcut_baseline_audit")["details"]["oracle_success_rate"] == 1.0
    assert protocol["model_status"] == "NOT_RUN"
    assert protocol["prospective_thresholds"]["scientific_learning_threshold"] == "not_defined_in_DG0"


def test_protocol_and_receipt_identities_are_stable(tmp_path: Path):
    suite = generate_suite(seed=1701, worlds_per_split=2)
    manifest = suite_manifest(suite, seed=1701)
    first = freeze_dg0_protocol(manifest, budget=8).to_dict()
    second = freeze_dg0_protocol(manifest, budget=8).to_dict()
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert protocol_template()["stage"] == "DG1"
    identity = make_identity(
        source_commit="abc",
        source_closure_sha256="closure",
        generator_sha256="generator",
        evaluator_sha256="evaluator",
        protocol_sha256=first["protocol_sha256"],
        policy_id="transparent",
        model_id="none",
        seed=1701,
    )
    receipt = result_receipt(
        identity=identity,
        protocol=first,
        suite_manifest=manifest.to_dict(),
        run={"records": []},
        audit={"status": "PASS"},
        command="test",
    )
    assert receipt["scientific_status"] == "no_model_learning_claim"
    assert receipt["protocol_sha256"] == first["protocol_sha256"]
    assert receipt["identity"]["protocol_sha256"] == first["protocol_sha256"]
    tampered_protocol = dict(first)
    tampered_protocol["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        result_receipt(
            identity=identity,
            protocol=tampered_protocol,
            suite_manifest=manifest.to_dict(),
            run={"records": []},
            audit={"status": "PASS"},
            command="test",
        )
    assert no_checkpoint_identity(experiment_id=first["protocol_id"], protocol_sha256=first["protocol_sha256"])["status"] == "NOT_CREATED_DG0"
    target = tmp_path / "receipt.json"
    write_json_exclusive(target, receipt)
    with pytest.raises(FileExistsError):
        write_json_exclusive(target, receipt)
    assert content_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"}) == receipt["receipt_sha256"]
