from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import (
    ACTION_COSTS,
    HIDDEN_KEYS,
    PUBLIC_SCHEMA_ID,
    Action,
    ActionCommitment,
    PublicView,
    canonical_json,
    content_hash,
    walk_keys,
)
from .generator import GeneratedWorld


class FirewallViolation(ValueError):
    pass


def _contains_exact(value: Any, forbidden_values: Iterable[Any]) -> bool:
    for forbidden in forbidden_values:
        try:
            if canonical_json(value) == canonical_json(forbidden):
                return True
        except TypeError:
            continue
    if isinstance(value, Mapping):
        return any(_contains_exact(item, forbidden_values) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_exact(item, forbidden_values) for item in value)
    return False


def assert_no_forbidden_values(payload: Any, forbidden_values: Iterable[Any]) -> None:
    if _contains_exact(payload, forbidden_values):
        raise FirewallViolation("a forbidden value entered a public payload")


def _forbidden_world_values(world: GeneratedWorld) -> tuple[Any, ...]:
    return (
        list(world.target_assignment),
        list(world.manifest.target_candidate_contexts),
        list(world.rule.truth_table()),
        world.generation_inputs,
        world.manifest.to_dict(),
        world.rule.semantic_sha256,
        world.world_id_basis_sha256 if hasattr(world, "world_id_basis_sha256") else world.manifest.world_id_basis_sha256,
        world.structure_id,
        world.split,
        world.rendering_id,
    )


def assert_public_view(view: PublicView) -> None:
    payload = view.to_dict()
    keys = {key.lower() for key in walk_keys(payload)}
    leaked = sorted(keys & HIDDEN_KEYS)
    if leaked:
        raise FirewallViolation(f"hidden fields entered the policy view: {leaked}")
    episode_id = str(payload.get("episode_id", ""))
    if not episode_id.startswith("dgw-"):
        raise FirewallViolation("episode identity is not an opaque world identity")


def assert_public_payload(payload: Mapping[str, Any]) -> None:
    keys = {key.lower() for key in walk_keys(payload)}
    leaked = sorted(keys & HIDDEN_KEYS)
    if leaked:
        raise FirewallViolation(f"hidden fields entered a public payload: {leaked}")


def assert_no_hidden_values(view: PublicView, world: GeneratedWorld) -> None:
    assert_public_view(view)
    assert_no_forbidden_values(view.to_dict(), _forbidden_world_values(world))


def _assert_public_snapshot(snapshot: Mapping[str, Any], expected_step: int) -> None:
    expected_fields = {
        "schema",
        "episode_id",
        "step_id",
        "goal",
        "variable_names",
        "target_context_id",
        "probe_contexts",
        "legal_actions",
        "transcript",
        "remaining_budget",
        "total_cost",
        "belief_space_id",
    }
    if set(snapshot) != expected_fields:
        raise FirewallViolation("pre-action public view has an unexpected schema")
    assert_public_payload(snapshot)
    if snapshot.get("schema") != PUBLIC_SCHEMA_ID:
        raise FirewallViolation("pre-action public view has the wrong schema")
    if snapshot.get("step_id") != expected_step:
        raise FirewallViolation("pre-action public view has the wrong step")
    transcript = snapshot.get("transcript")
    if not isinstance(transcript, list) or any(not isinstance(item, Mapping) for item in transcript):
        raise FirewallViolation("pre-action public transcript is malformed")
    if [item.get("step_id") for item in transcript] != list(range(expected_step)):
        raise FirewallViolation("pre-action public transcript is not causally ordered")
    total_cost = snapshot.get("total_cost")
    remaining_budget = snapshot.get("remaining_budget")
    if (
        isinstance(total_cost, bool)
        or not isinstance(total_cost, int)
        or total_cost < 0
        or isinstance(remaining_budget, bool)
        or not isinstance(remaining_budget, int)
        or remaining_budget < 0
    ):
        raise FirewallViolation("pre-action public budget is malformed")
    if sum(item.get("cost", -1) for item in transcript) != total_cost:
        raise FirewallViolation("pre-action public cost ledger is malformed")
    legal_actions = snapshot.get("legal_actions")
    if not isinstance(legal_actions, list):
        raise FirewallViolation("pre-action legal-action set is malformed")


def assert_commitment_precedes_outcome(
    commitments: tuple[ActionCommitment, ...],
    events: tuple[Any, ...],
) -> None:
    if len(commitments) != len(events):
        raise FirewallViolation("every action must have one pre-outcome commitment")
    for expected_step, (commitment, event) in enumerate(zip(commitments, events)):
        if not isinstance(commitment, ActionCommitment) or not isinstance(event.action, Action):
            raise FirewallViolation("commitment and event must contain actions")
        if commitment.step_id != expected_step or event.step_id != expected_step:
            raise FirewallViolation("commitment and event steps are not aligned")
        if commitment.action.action_id != event.action.action_id:
            raise FirewallViolation("event action differs from its commitment")
        if content_hash(commitment.available_view) != commitment.available_view_sha256:
            raise FirewallViolation("pre-action public view hash is not bound to its snapshot")
        _assert_public_snapshot(commitment.available_view, expected_step)


def audit_firewall(world: GeneratedWorld, environment: Any) -> dict[str, Any]:
    view = environment.reset()
    assert_public_view(view)
    assert_no_hidden_values(view, world)
    assert_commitment_precedes_outcome(environment.commitments, environment.transcript)
    return {
        "schema": "anra-discovery-gym/firewall-audit/v1",
        "world_id": world.world_id,
        "public_view_sha256": content_hash(view.to_dict()),
        "passed": True,
        "checked": ("public_keys", "hidden_values", "commitment_order"),
    }
