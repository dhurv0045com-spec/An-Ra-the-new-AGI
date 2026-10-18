from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import ACTION_COSTS, PUBLIC_SCHEMA_ID, Action, content_hash
from .environment import DiscoveryEnvironment
from .firewall import assert_no_hidden_values, assert_public_view
from .generator import ALL_CONTEXTS, GENERATOR_VERSION, GeneratedWorld, WORLD_SCHEMA, world_id_for
from .rules import catalog_digest, overridden_assignment


def _check(name: str, passed: bool, details: Mapping[str, Any]) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "details": dict(details)}


def audit_generator(suite: Mapping[str, tuple[GeneratedWorld, ...]]) -> dict[str, Any]:
    worlds = [world for values in suite.values() for world in values]
    semantic_by_split = {
        split: {world.rule.semantic_sha256 for world in values}
        for split, values in suite.items()
    }
    structure_by_split = {
        split: {world.structure_id for world in values}
        for split, values in suite.items()
    }
    graph_by_split = {
        split: {world.rule.graph_signature for world in values}
        for split, values in suite.items()
    }
    semantic_overlap = {
        left: sorted(semantic_by_split[left] & semantic_by_split[right])
        for index, left in enumerate(sorted(suite))
        for right in sorted(suite)[index + 1 :]
        if semantic_by_split[left] & semantic_by_split[right]
    }
    structure_overlap = {
        left: sorted(structure_by_split[left] & structure_by_split[right])
        for index, left in enumerate(sorted(suite))
        for right in sorted(suite)[index + 1 :]
        if structure_by_split[left] & structure_by_split[right]
    }
    graph_overlap = {
        left: sorted(graph_by_split[left] & graph_by_split[right])
        for index, left in enumerate(sorted(suite))
        for right in sorted(suite)[index + 1 :]
        if graph_by_split[left] & graph_by_split[right]
    }
    ids = [world.world_id for world in worlds]
    target_probe_leaks = [
        world.world_id
        for world in worlds
        if world.target_assignment in world.probe_contexts
    ]
    provenance_errors: list[str] = []
    affordance_errors: list[str] = []
    for world in worlds:
        expected_inputs = content_hash(world.generation_inputs)
        expected_target = content_hash(list(world.target_assignment))
        expected_id, expected_id_basis = world_id_for(
            split=world.split,
            seed=int(world.generation_inputs["seed"]),
            index=int(world.generation_inputs["index"]),
        )
        expected_generation_inputs = {
            "generator_version": GENERATOR_VERSION,
            "seed": world.generation_inputs.get("seed"),
            "split": world.split,
            "index": world.generation_inputs.get("index"),
            "structure_id": world.structure_id,
            "variant": world.rule.variant,
            "rendering_id": world.rendering_id,
            "legal_action_kinds": list(world.legal_action_kinds),
            "target_assignment": list(world.target_assignment),
            "probe_contexts": [list(item) for item in world.probe_contexts],
            "target_candidate_contexts": [list(item) for item in world.manifest.target_candidate_contexts],
            "intervention_contexts": [list(item) for item in world.intervention_contexts],
            "catalog_sha256": catalog_digest(),
        }
        if world.world_id != expected_id or world.manifest.world_id_basis_sha256 != expected_id_basis:
            provenance_errors.append(f"{world.world_id}:world-id-basis")
        if world.manifest.schema != WORLD_SCHEMA or world.manifest.public_schema_id != PUBLIC_SCHEMA_ID:
            provenance_errors.append(f"{world.world_id}:schema-version")
        if world.generation_inputs != expected_generation_inputs:
            provenance_errors.append(f"{world.world_id}:generation-input-fields")
        if world.manifest.generation_inputs_sha256 != expected_inputs:
            provenance_errors.append(f"{world.world_id}:generation-inputs")
        if world.manifest.target_assignment_sha256 != expected_target:
            provenance_errors.append(f"{world.world_id}:target-hash")
        if world.manifest.rule_semantic_sha256 != world.rule.semantic_sha256:
            provenance_errors.append(f"{world.world_id}:rule-hash")
        if (
            world.target_assignment not in world.manifest.target_candidate_contexts
            or set(world.probe_contexts) & set(world.manifest.target_candidate_contexts)
            or len(world.probe_contexts) != 4
            or len(world.manifest.target_candidate_contexts) != 4
            or set(world.probe_contexts) | set(world.manifest.target_candidate_contexts) != set(ALL_CONTEXTS)
            or len(world.intervention_contexts) != 1
            or not set(world.intervention_contexts).issubset(set(world.probe_contexts))
            or any(
                overridden_assignment(context, variable, value) in world.manifest.target_candidate_contexts
                for context in world.intervention_contexts
                for variable in range(3)
                for value in (0, 1)
            )
        ):
            provenance_errors.append(f"{world.world_id}:target-partition")
        if world.manifest.probe_contexts != world.probe_contexts:
            provenance_errors.append(f"{world.world_id}:probe-contexts")
        if world.manifest.variable_names != world.variable_names:
            provenance_errors.append(f"{world.world_id}:variable-names")
        if world.manifest.action_costs != ACTION_COSTS:
            affordance_errors.append(f"{world.world_id}:action-costs")
        if world.manifest.legal_action_kinds != world.legal_action_kinds:
            affordance_errors.append(f"{world.world_id}:affordance")
        if world.world_id in world.structure_id or world.world_id in world.split:
            provenance_errors.append(f"{world.world_id}:identity-leak")
    passed = (
        len(ids) == len(set(ids))
        and not semantic_overlap
        and not structure_overlap
        and not graph_overlap
        and not target_probe_leaks
        and not provenance_errors
        and not affordance_errors
    )
    return _check(
        "generator_and_split_integrity",
        passed,
        {
            "world_count": len(worlds),
            "semantic_overlap": semantic_overlap,
            "structure_overlap": structure_overlap,
            "graph_overlap": graph_overlap,
            "duplicate_world_ids": len(ids) != len(set(ids)),
            "target_probe_leaks": target_probe_leaks,
            "provenance_errors": provenance_errors,
            "affordance_errors": affordance_errors,
            "affordance_ids": {split: sorted({",".join(world.legal_action_kinds) for world in values}) for split, values in suite.items()},
            "independence_boundary": "one latent causal program and all its renderings stay in one split",
        },
    )


def audit_firewall(suite: Mapping[str, tuple[GeneratedWorld, ...]]) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for worlds in suite.values():
        for world in worlds:
            environment = DiscoveryEnvironment(world, budget=8)
            view = environment.reset()
            try:
                assert_public_view(view)
                assert_no_hidden_values(view, world)
                probe = next((item for item in view.legal_actions if item.kind == "probe"), None)
                if probe is not None:
                    environment.step(Action.make("probe", prediction=0, **dict(probe.arguments)))
                    assert_public_view(environment.public_view())
                    assert_no_hidden_values(environment.public_view(), world)
                terminal = DiscoveryEnvironment(world, budget=8)
                terminal.reset()
                terminal.step(Action.make("stop"))
                assert_public_view(terminal.public_view())
                assert_no_hidden_values(terminal.public_view(), world)
                if "success" in terminal.transcript[-1].feedback:
                    raise ValueError("terminal feedback exposes an evaluator label")
            except ValueError as exc:
                failures.append(f"{world.world_id}:{exc}")
            checked += 1
    return _check(
        "public_evidence_firewall",
        not failures,
        {"worlds_checked": checked, "failures": failures},
    )


def _audit_legality_world(world: GeneratedWorld) -> dict[str, Any]:
    environment = DiscoveryEnvironment(world, budget=8)
    view = environment.reset()
    legal_costs_match = all(item.cost == ACTION_COSTS[item.kind] for item in view.legal_actions)
    initial_target_leak_actions = [
        item.to_dict()
        for item in view.legal_actions
        if item.kind in {"probe", "intervene"} and tuple(item.arguments.get("context", ())) == world.target_assignment
    ]
    probe_descriptor = next((item for item in view.legal_actions if item.kind == "probe"), None)
    repeated_rejected = False
    target_rejected = False
    target_intervention_rejected = False
    target_effective_intervention_available = False
    state_unchanged_after_rejection = False
    state_unchanged_after_target = False
    if probe_descriptor is not None:
        first = Action.make("probe", prediction=0, **dict(probe_descriptor.arguments))
        first_result = environment.step(first)
        state_before_repeat = environment.evidence_fingerprint()
        repeated = environment.step(Action.make("probe", prediction=0, **dict(probe_descriptor.arguments)))
        repeated_rejected = (
            first_result.accepted
            and not repeated.accepted
            and repeated.event.cost == ACTION_COSTS["probe"]
            and environment.evidence_fingerprint() == state_before_repeat
        )
        state_before_target = environment.evidence_fingerprint()
        target_action = Action.make("probe", prediction=0, context=list(world.target_assignment))
        target_result = environment.step(target_action)
        target_rejected = (
            not target_result.accepted
            and target_result.event.cost == ACTION_COSTS["probe"]
            and environment.evidence_fingerprint() == state_before_target
        )
    intervention_environment = DiscoveryEnvironment(world, budget=8)
    intervention_environment.reset()
    for context in world.intervention_contexts:
        for variable, variable_name in enumerate(world.variable_names):
            for value in (0, 1):
                if overridden_assignment(context, variable, value) != world.target_assignment:
                    continue
                target_effective_intervention_available = True
                before_intervention = intervention_environment.evidence_fingerprint()
                intervention_result = intervention_environment.step(
                    Action.make(
                        "intervene",
                        prediction=0,
                        context=list(context),
                        variable=variable_name,
                        value=value,
                    )
                )
                target_intervention_rejected = (
                    not intervention_result.accepted
                    and intervention_result.event.cost == ACTION_COSTS["intervene"]
                    and intervention_environment.evidence_fingerprint() == before_intervention
                )
                break
            if target_intervention_rejected:
                break
        if target_effective_intervention_available:
            break
    if not target_effective_intervention_available:
        target_intervention_rejected = True
    events = environment.transcript
    cost_sum = sum(event.cost for event in events)
    state_unchanged_after_rejection = repeated_rejected
    state_unchanged_after_target = target_rejected
    passed = (
        legal_costs_match
        and repeated_rejected
        and target_rejected
        and target_intervention_rejected
        and state_unchanged_after_rejection
        and state_unchanged_after_target
        and cost_sum == environment.total_cost
        and not initial_target_leak_actions
    )
    return {
        "world_id": world.world_id,
        "passed": passed,
        "legal_costs_match": legal_costs_match,
        "repeated_action_rejected_and_charged": repeated_rejected,
        "target_probe_rejected_and_charged": target_rejected,
        "target_intervention_rejected_or_unreachable": target_intervention_rejected,
        "target_effective_intervention_available": target_effective_intervention_available,
        "rejected_state_unchanged": state_unchanged_after_rejection and state_unchanged_after_target,
        "event_cost_sum": cost_sum,
        "environment_total_cost": environment.total_cost,
        "direct_target_actions_legal": initial_target_leak_actions,
    }


def audit_legality_and_cost(worlds: Iterable[GeneratedWorld]) -> dict[str, Any]:
    checks = [_audit_legality_world(world) for world in worlds]
    failures = [item for item in checks if not item["passed"]]
    return _check(
        "legality_and_cost_accounting",
        not failures,
        {
            "worlds_checked": len(checks),
            "failures": failures,
            "charging_rule": "every attempted action is charged; rejected actions do not mutate hidden state",
        },
    )


def audit_evaluator(suite: Mapping[str, tuple[GeneratedWorld, ...]], run: Mapping[str, Any]) -> dict[str, Any]:
    records = list(run.get("records", []))
    disagreements = [
        record["world_id"]
        for record in records
        if not record.get("evaluator_recomputation", {}).get("agreement", False)
        or not record.get("legality_recomputation", {}).get("agreement", False)
    ]
    oracle_records = [record for record in records if record.get("uses_hidden_truth")]
    oracle_success = all(record.get("success") for record in oracle_records) if oracle_records else False
    return _check(
        "independent_evaluator_recomputation",
        bool(records) and not disagreements and oracle_success,
        {
            "records_checked": len(records),
            "disagreements": disagreements,
            "legality_disagreements": [
                record["world_id"]
                for record in records
                if not record.get("legality_recomputation", {}).get("agreement", False)
            ],
            "oracle_records": len(oracle_records),
            "oracle_success": oracle_success,
            "labels_recomputed_from": "hidden rule and target assignment, never model output",
        },
    )


def audit_shortcuts(run: Mapping[str, Any]) -> dict[str, Any]:
    overall = run.get("summary", {}).get("overall", {})
    shortcut_ids = ("surface-position-shortcut/v1", "task-identity-shortcut/v1", "no-experiment/v1")
    shortcut_rates = {
        policy_id: overall.get(policy_id, {}).get("success_rate")
        for policy_id in shortcut_ids
    }
    oracle_rate = overall.get("evaluator-oracle/v1", {}).get("success_rate")
    missing_controls = [policy_id for policy_id in shortcut_ids if policy_id not in overall]
    no_shortcut_ceiling = not missing_controls and all(
        isinstance(rate, (int, float)) and rate < 0.9 for rate in shortcut_rates.values()
    )
    return _check(
        "shortcut_baseline_audit",
        no_shortcut_ceiling and oracle_rate is not None,
        {
            "shortcut_success_rates": shortcut_rates,
            "missing_controls": missing_controls,
            "oracle_success_rate": oracle_rate,
            "interpretation": "shortcut controls are reported as controls, never as learned discovery",
        },
    )


def audit_confirmation_use(protocol: Mapping[str, Any]) -> dict[str, Any]:
    passed = (
        protocol.get("model_status") == "NOT_RUN"
        and protocol.get("tuning_policy") == "no policy, threshold, or generator choice may use confirmation outcomes"
        and protocol.get("confirmation_policy") == "confirmation worlds are generated and audited but no model is trained or tuned"
    )
    return _check(
        "confirmation_data_firewall",
        passed,
        {
            "model_status": protocol.get("model_status"),
            "tuning_policy": protocol.get("tuning_policy"),
            "confirmation_policy": protocol.get("confirmation_policy"),
        },
    )


def audit_suite(
    suite: Mapping[str, tuple[GeneratedWorld, ...]],
    run: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        audit_generator(suite),
        audit_firewall(suite),
        audit_legality_and_cost(world for values in suite.values() for world in values),
        audit_evaluator(suite, run),
        audit_shortcuts(run),
        audit_confirmation_use(protocol),
    ]
    passed = all(check["status"] == "PASS" for check in checks)
    return {
        "schema": "anra-discovery-gym/dg0-audit/v1",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "shortcut_audit_is_informational": False,
        "claim_ceiling": "DG0 establishes an auditable deterministic instrument only; it does not establish model learning, causal discovery by a model, transfer, retention, or AGI.",
    }
