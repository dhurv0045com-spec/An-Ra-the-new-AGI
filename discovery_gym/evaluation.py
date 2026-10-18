from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .baselines import Policy
from .contracts import ACTION_COSTS, PolicyFactory, content_hash
from .environment import DiscoveryEnvironment
from .firewall import assert_commitment_precedes_outcome, assert_public_view
from .generator import GeneratedWorld
from .rules import evaluate_rule, overridden_assignment


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_binary(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def _event_actual(event: Any, world: GeneratedWorld) -> int | None:
    feedback = event.feedback
    if feedback.get("kind") in {"probe", "intervene"}:
        value = feedback.get("outcome")
        return value if _is_binary(value) else None
    if feedback.get("kind") == "read":
        value = feedback.get("value")
        return value if _is_binary(value) else None
    if feedback.get("kind") == "submit":
        return world.hidden_answer
    return None


def _recompute_events(world: GeneratedWorld, events: tuple[Any, ...]) -> dict[str, Any]:
    expected_events: list[dict[str, Any]] = []
    agreement = True
    for event in events:
        feedback = event.feedback
        kind = feedback.get("kind")
        if not event.accepted:
            matches = (
                kind == "invalid_action"
                and event.cost == ACTION_COSTS.get(event.action.kind, 1)
            )
            expected_events.append({"step_id": event.step_id, "kind": "rejected", "matches": matches})
            agreement = agreement and matches
            continue
        expected: int | None = None
        matches = True
        try:
            if kind == "read" and event.action.kind == "read":
                index = world.variable_names.index(feedback["variable"])
                expected = world.target_assignment[index]
                matches = (
                    feedback.get("variable") == event.action.argument("variable")
                    and _is_binary(feedback.get("value"))
                    and expected == feedback.get("value")
                )
            elif kind == "probe" and event.action.kind == "probe":
                context = tuple(feedback["context"])
                expected = evaluate_rule(world.rule, context)
                matches = (
                    list(event.action.argument("context", ())) == list(context)
                    and _is_binary(feedback.get("outcome"))
                    and expected == feedback.get("outcome")
                )
            elif kind == "intervene" and event.action.kind == "intervene":
                context = tuple(feedback["context"])
                variable = world.variable_names.index(feedback["intervention"]["variable"])
                value = feedback["intervention"]["value"]
                expected = evaluate_rule(world.rule, overridden_assignment(context, variable, value))
                matches = (
                    list(event.action.argument("context", ())) == list(context)
                    and event.action.argument("variable") == feedback["intervention"]["variable"]
                    and event.action.argument("value") == value
                    and _is_binary(value)
                    and _is_binary(feedback.get("outcome"))
                    and expected == feedback.get("outcome")
                )
            elif kind == "submit" and event.action.kind == "submit":
                expected = world.hidden_answer
                submitted = feedback.get("submitted_answer")
                matches = (
                    set(feedback) == {"kind", "mode", "submitted_answer"}
                    and _is_binary(submitted)
                    and event.action.argument("answer") == submitted
                )
            elif kind == "stop" and event.action.kind == "stop":
                matches = feedback.get("abstained") is True and not event.action.arguments
            else:
                matches = False
        except (KeyError, TypeError, ValueError, IndexError):
            matches = False
        agreement = agreement and matches
        expected_events.append({"step_id": event.step_id, "kind": kind, "expected": expected, "matches": matches})
    return {"agreement": agreement, "events": expected_events}


def _replay_legality(
    world: GeneratedWorld,
    events: tuple[Any, ...],
    budget: int,
) -> dict[str, Any]:
    environment = DiscoveryEnvironment(world, budget=budget)
    environment.reset()
    checks: list[dict[str, Any]] = []
    agreement = True
    for event in events:
        try:
            replayed = environment.step(event.action)
            matches = (
                replayed.accepted == event.accepted
                and replayed.event.cost == event.cost
                and replayed.event.feedback == event.feedback
            )
        except Exception as exc:
            matches = False
            replayed = None
            checks.append({"step_id": event.step_id, "matches": False, "error": str(exc)})
            agreement = False
            continue
        checks.append({"step_id": event.step_id, "matches": matches})
        agreement = agreement and matches
    agreement = agreement and environment.terminated
    return {"agreement": agreement, "events": checks}


def score_episode(
    world: GeneratedWorld,
    events: tuple[Any, ...],
    commitments: tuple[Any, ...],
    *,
    policy: Policy,
    budget: int,
) -> dict[str, Any]:
    assert_commitment_precedes_outcome(commitments, events)
    recomputation = _recompute_events(world, events)
    legality_recomputation = _replay_legality(world, events, budget)
    submit_events = [event for event in events if event.accepted and event.feedback.get("kind") == "submit"]
    final_event = submit_events[-1] if submit_events else None
    final_answer = final_event.feedback.get("submitted_answer") if final_event else None
    success = _is_binary(final_answer) and final_answer == world.hidden_answer
    total_cost = sum(event.cost for event in events)
    experiment_events = [
        event
        for event in events
        if event.accepted and event.feedback.get("kind") in {"read", "probe", "intervene"}
    ]
    invalid_events = [event for event in events if not event.accepted]
    prediction_rows: list[dict[str, Any]] = []
    for event in experiment_events + submit_events:
        actual = _event_actual(event, world)
        if event.action.prediction is None or actual is None:
            continue
        prediction_rows.append(
            {
                "step_id": event.step_id,
                "kind": event.action.kind,
                "prediction": event.action.prediction,
                "actual": actual,
                "correct": int(event.action.prediction == actual),
                "brier": (event.action.prediction - actual) ** 2,
            }
        )
    by_kind: dict[str, int] = defaultdict(int)
    for event in experiment_events:
        by_kind[event.action.kind] += 1
    return {
        "schema": "anra-discovery-gym/episode-result/v1",
        "episode_id": world.world_id,
        "world_id": world.world_id,
        "independence_unit": "latent_world",
        "split": world.split,
        "structure_id": world.structure_id,
        "policy_id": policy.identity,
        "uses_hidden_truth": policy.uses_hidden_truth,
        "budget": budget,
        "total_cost": total_cost,
        "environment_cost": total_cost,
        "inference_cost": 0,
        "experiment_cost": sum(event.cost for event in experiment_events),
        "invalid_cost": sum(event.cost for event in invalid_events),
        "experiment_count": len(experiment_events),
        "experiment_counts_by_kind": dict(sorted(by_kind.items())),
        "invalid_action_count": len(invalid_events),
        "budget_overrun": total_cost > budget,
        "final_answer": final_answer,
        "success": success,
        "abstained": final_event is None,
        "prediction_rows": prediction_rows,
        "prediction_accuracy": _mean([float(row["correct"]) for row in prediction_rows]),
        "outcome_prediction_accuracy": _mean([float(row["correct"]) for row in prediction_rows]),
        "prediction_brier": _mean([float(row["brier"]) for row in prediction_rows]),
        "calibration_brier": _mean([float(row["brier"]) for row in prediction_rows]),
        "information_gain_per_cost": None,
        "information_gain_status": "not_computed_without_a_declared_belief_trace",
        "causal_rule_recovery": "not_measured_in_DG0",
        "transfer_result": "not_evaluated_in_DG0",
        "events": [event.to_dict() for event in events],
        "commitments": [commitment.to_dict() for commitment in commitments],
        "evaluator_recomputation": recomputation,
        "legality_recomputation": legality_recomputation,
        "component_attribution": {
            "action_selection": "evaluator_oracle" if policy.uses_hidden_truth else policy.identity,
            "world_generation": "evaluator",
            "outcome_execution": "environment",
            "belief_update": "transparent_baseline" if not policy.uses_hidden_truth else "oracle_diagnostic",
            "transfer": "not_evaluated_in_dg0",
        },
        "claim_status": "instrument_result_only",
    }


def run_episode(
    world: GeneratedWorld,
    policy: Policy,
    *,
    budget: int = 8,
    allow_hidden_truth: bool = False,
) -> dict[str, Any]:
    if policy.uses_hidden_truth and not allow_hidden_truth:
        raise ValueError("hidden-truth policies require an explicit diagnostic opt-in")
    environment = DiscoveryEnvironment(world, budget=budget)
    policy.reset(world.world_id)
    view = environment.reset()
    assert_public_view(view)
    while not environment.terminated:
        action = policy.act(view)
        environment.step(action)
        view = environment.public_view()
        assert_public_view(view)
    return score_episode(
        world,
        environment.transcript,
        environment.commitments,
        policy=policy,
        budget=budget,
    )


def _aggregate_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(records),
        "success_rate": _mean([float(record["success"]) for record in records]),
        "final_task_success": _mean([float(record["success"]) for record in records]),
        "prediction_accuracy": _mean([float(record["prediction_accuracy"]) for record in records if record["prediction_accuracy"] is not None]),
        "outcome_prediction_accuracy": _mean([float(record["outcome_prediction_accuracy"]) for record in records if record["outcome_prediction_accuracy"] is not None]),
        "prediction_brier": _mean([float(record["prediction_brier"]) for record in records if record["prediction_brier"] is not None]),
        "calibration_brier": _mean([float(record["calibration_brier"]) for record in records if record["calibration_brier"] is not None]),
        "information_gain_per_cost": None,
        "information_gain_status": "not_computed_without_a_declared_belief_trace",
        "mean_total_cost": _mean([float(record["total_cost"]) for record in records]),
        "mean_environment_cost": _mean([float(record["environment_cost"]) for record in records]),
        "mean_inference_cost": _mean([float(record["inference_cost"]) for record in records]),
        "mean_experiment_cost": _mean([float(record["experiment_cost"]) for record in records]),
        "invalid_action_rate": _mean([float(record["invalid_action_count"]) for record in records]),
        "budget_overrun_rate": _mean([float(record["budget_overrun"]) for record in records]),
        "abstention_rate": _mean([float(record["abstained"]) for record in records]),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["policy_id"]].append(record)
    overall = {policy_id: _aggregate_group(values) for policy_id, values in sorted(grouped.items())}
    oracle_rate = overall.get("evaluator-oracle/v1", {}).get("success_rate")
    if oracle_rate is not None:
        for summary in overall.values():
            summary["regret_vs_oracle"] = oracle_rate - float(summary["success_rate"])
    by_split: dict[str, dict[str, Any]] = {}
    for policy_id, values in sorted(grouped.items()):
        by_split[policy_id] = {}
        for split in sorted({record["split"] for record in values}):
            by_split[policy_id][split] = _aggregate_group([record for record in values if record["split"] == split])
    by_structure: dict[str, dict[str, Any]] = {}
    for policy_id, values in sorted(grouped.items()):
        by_structure[policy_id] = {}
        for structure in sorted({record["structure_id"] for record in values}):
            by_structure[policy_id][structure] = _aggregate_group([record for record in values if record["structure_id"] == structure])
    return {
        "schema": "anra-discovery-gym/summary/v1",
        "independence_unit": "latent_world",
        "overall": overall,
        "by_split": by_split,
        "by_structure": by_structure,
    }


def run_suite(
    suite: Mapping[str, tuple[GeneratedWorld, ...]],
    factories: Mapping[str, PolicyFactory],
    *,
    budget: int = 8,
    allow_hidden_truth: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    identity_factories: dict[str, str] = {}
    for factory_id, factory in sorted(factories.items()):
        for worlds in suite.values():
            for world in worlds:
                policy = factory(world)
                if not isinstance(policy.identity, str) or not policy.identity:
                    raise ValueError("policy identities must be nonempty strings")
                existing_factory = identity_factories.get(policy.identity)
                if existing_factory is not None and existing_factory != factory_id:
                    raise ValueError(
                        f"policy identity collision: {policy.identity} used by {existing_factory} and {factory_id}"
                    )
                identity_factories.setdefault(policy.identity, factory_id)
                records.append(
                    run_episode(
                        world,
                        policy,
                        budget=budget,
                        allow_hidden_truth=allow_hidden_truth,
                    )
                )
    return {
        "schema": "anra-discovery-gym/suite-run/v1",
        "budget": budget,
        "allow_hidden_truth_diagnostic": allow_hidden_truth,
        "records": records,
        "summary": aggregate_records(records),
        "record_sha256": content_hash(records),
    }
