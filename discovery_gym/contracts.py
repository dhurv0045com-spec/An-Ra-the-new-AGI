from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol


CONTRACT_VERSION = "anra-discovery-gym/contracts/v2"
PUBLIC_SCHEMA_ID = "anra-discovery-gym/public-state/v2"
ACTION_COSTS = {
    "read": 1,
    "probe": 2,
    "intervene": 3,
    "submit": 1,
    "stop": 0,
}
SPLITS = ("train", "development", "confirmation", "transfer")
HIDDEN_KEYS = frozenset(
    {
        "answer_key",
        "causal_program",
        "evaluator_label",
        "future_outcome",
        "generation_inputs",
        "generation_inputs_sha256",
        "generation_seed",
        "gold",
        "gold_answer",
        "hidden_answer",
        "hidden_rule",
        "mechanism_id",
        "program",
        "rule",
        "rule_spec",
        "rule_semantic_sha256",
        "rendering_id",
        "seed",
        "split",
         "structure_id",
         "success",
         "correct",
         "verdict",
         "reward",
         "target_assignment",
         "target_assignment_sha256",
         "target_candidate_contexts",
         "target_value",
         "truth",
         "world_family",
         "world_id_basis_sha256",
    }
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError("non-finite values are not permitted")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("mapping keys must be strings")
            _validate_json_value(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)
        return
    raise TypeError(f"unsupported argument type: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    arguments: tuple[tuple[str, Any], ...] = ()
    prediction: int | None = None
    action_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in ACTION_COSTS:
            raise ValueError(f"unknown action kind: {self.kind}")
        names = [name for name, _ in self.arguments]
        if any(not isinstance(name, str) for name in names):
            raise TypeError("action argument names must be strings")
        if len(names) != len(set(names)):
            raise ValueError("action arguments must have unique names")
        for _, value in self.arguments:
            _validate_json_value(value)
        if self.prediction is not None:
            if (
                isinstance(self.prediction, bool)
                or not isinstance(self.prediction, int)
                or self.prediction not in (0, 1)
            ):
                raise ValueError("predictions must be binary integers")
        object.__setattr__(self, "arguments", tuple(sorted(self.arguments, key=lambda item: item[0])))
        payload = {
            "kind": self.kind,
            "arguments": dict(self.arguments),
            "prediction": self.prediction,
        }
        expected_action_id = content_hash(payload)[:20]
        if not self.action_id:
            object.__setattr__(self, "action_id", expected_action_id)
        elif self.action_id != expected_action_id:
            raise ValueError("action_id does not match action contents")

    @classmethod
    def make(cls, kind: str, *, prediction: int | None = None, **arguments: Any) -> "Action":
        return cls(kind, tuple(sorted(arguments.items())), prediction)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Action":
        if not isinstance(value, Mapping):
            raise TypeError("serialized action must be a mapping")
        unknown = set(value) - {"kind", "arguments", "prediction", "action_id"}
        if unknown:
            raise ValueError(f"unknown serialized action fields: {sorted(unknown)}")
        if "kind" not in value or not isinstance(value["kind"], str):
            raise TypeError("serialized action kind must be a string")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise TypeError("serialized action arguments must be a mapping")
        action_id = value.get("action_id", "")
        if not isinstance(action_id, str):
            raise TypeError("serialized action_id must be a string")
        return cls(
            value["kind"],
            tuple(arguments.items()),
            value.get("prediction"),
            action_id,
        )

    def argument(self, name: str, default: Any = None) -> Any:
        for key, value in self.arguments:
            if key == name:
                return value
        return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "arguments": dict(self.arguments),
            "prediction": self.prediction,
            "action_id": self.action_id,
        }


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    kind: str
    arguments: dict[str, Any]
    cost: int
    outcome_domain: tuple[int, ...]
    mode: str
    preconditions: tuple[str, ...]
    state_changes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in ACTION_COSTS:
            raise ValueError(f"unknown descriptor kind: {self.kind}")
        if self.cost != ACTION_COSTS[self.kind]:
            raise ValueError("descriptor cost disagrees with the contract")
        _validate_json_value(self.arguments)
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "outcome_domain", tuple(self.outcome_domain))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "state_changes", tuple(self.state_changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "arguments": self.arguments,
            "cost": self.cost,
            "outcome_domain": list(self.outcome_domain),
            "mode": self.mode,
            "preconditions": list(self.preconditions),
            "state_changes": list(self.state_changes),
        }


@dataclass(frozen=True, slots=True)
class PublicEvent:
    step_id: int
    action: Action
    accepted: bool
    cost: int
    feedback: dict[str, Any]

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("event step_id must be nonnegative")
        if not isinstance(self.action, Action):
            raise TypeError("event action must be an Action")
        if not isinstance(self.accepted, bool):
            raise TypeError("event accepted must be boolean")
        if isinstance(self.cost, bool) or not isinstance(self.cost, int) or self.cost < 0:
            raise ValueError("event cost must be a nonnegative integer")
        _validate_json_value(self.feedback)
        object.__setattr__(self, "feedback", dict(self.feedback))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action.to_dict(),
            "accepted": self.accepted,
            "cost": self.cost,
            "feedback": self.feedback,
        }


@dataclass(frozen=True, slots=True)
class ActionCommitment:
    step_id: int
    action: Action
    available_view: dict[str, Any]
    available_view_sha256: str

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("commitment step_id must be nonnegative")
        if not isinstance(self.action, Action):
            raise TypeError("commitment action must be an Action")
        if not isinstance(self.available_view, Mapping):
            raise TypeError("commitment available view must be a mapping")
        if not isinstance(self.available_view_sha256, str) or len(self.available_view_sha256) != 64:
            raise ValueError("commitment view hash must be a SHA-256 hex digest")
        _validate_json_value(self.available_view)
        object.__setattr__(self, "available_view", dict(self.available_view))
        if content_hash(self.available_view) != self.available_view_sha256:
            raise ValueError("pre-action public view hash does not match its contents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action.to_dict(),
            "available_view": self.available_view,
            "available_view_sha256": self.available_view_sha256,
        }


@dataclass(frozen=True, slots=True)
class PublicView:
    schema: str
    episode_id: str
    step_id: int
    goal: str
    variable_names: tuple[str, ...]
    target_context_id: str
    probe_contexts: tuple[tuple[int, ...], ...]
    legal_actions: tuple[ActionDescriptor, ...]
    transcript: tuple[PublicEvent, ...]
    remaining_budget: int
    total_cost: int
    belief_space_id: str

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_SCHEMA_ID:
            raise ValueError("unsupported public schema")
        if self.step_id < 0:
            raise ValueError("view step_id must be nonnegative")
        if self.remaining_budget < 0 or self.total_cost < 0:
            raise ValueError("budget values cannot be negative")
        variable_names = tuple(self.variable_names)
        if not variable_names or any(not isinstance(name, str) for name in variable_names):
            raise ValueError("public view requires named variables")
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("public variable names must be unique")
        probe_contexts = tuple(tuple(item) for item in self.probe_contexts)
        if any(
            len(context) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1) for value in context)
            for context in probe_contexts
        ):
            raise ValueError("public probe contexts must contain three binary integers")
        if len(set(probe_contexts)) != len(probe_contexts):
            raise ValueError("public probe contexts must be unique")
        transcript = tuple(self.transcript)
        if any(not isinstance(event, PublicEvent) for event in transcript):
            raise TypeError("public transcript must contain PublicEvent values")
        if tuple(event.step_id for event in transcript) != tuple(range(self.step_id)):
            raise ValueError("public transcript steps must precede the current view step")
        if sum(event.cost for event in transcript) != self.total_cost:
            raise ValueError("public view cost does not match its transcript")
        object.__setattr__(self, "variable_names", variable_names)
        object.__setattr__(self, "probe_contexts", probe_contexts)
        object.__setattr__(self, "legal_actions", tuple(self.legal_actions))
        object.__setattr__(self, "transcript", transcript)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "goal": self.goal,
            "variable_names": list(self.variable_names),
            "target_context_id": self.target_context_id,
            "probe_contexts": [list(item) for item in self.probe_contexts],
            "legal_actions": [item.to_dict() for item in self.legal_actions],
            "transcript": [item.to_dict() for item in self.transcript],
            "remaining_budget": self.remaining_budget,
            "total_cost": self.total_cost,
            "belief_space_id": self.belief_space_id,
        }


class Policy(ABC):
    identity: str = "unnamed-policy/v1"
    uses_hidden_truth: bool = False

    def reset(self, episode_id: str) -> None:
        return None

    @abstractmethod
    def act(self, view: PublicView) -> Action:
        raise NotImplementedError


class PolicyFactory(Protocol):
    def __call__(self, world: Any) -> Policy:
        ...


@dataclass(frozen=True, slots=True)
class SuiteManifest:
    schema: str
    generator_version: str
    seed: int
    split_seeds: dict[str, int]
    worlds: dict[str, list[dict[str, Any]]]
    split_identities: dict[str, str]
    structure_ids: dict[str, list[str]]
    affordance_ids: dict[str, list[str]]
    rendering_ids: dict[str, list[str]]
    variable_role_ids: dict[str, list[str]]
    catalog_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "split_seeds": self.split_seeds,
            "worlds": self.worlds,
            "split_identities": self.split_identities,
            "structure_ids": self.structure_ids,
            "affordance_ids": self.affordance_ids,
            "rendering_ids": self.rendering_ids,
            "variable_role_ids": self.variable_role_ids,
            "catalog_sha256": self.catalog_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    schema: str
    source_commit: str
    source_closure_sha256: str
    generator_sha256: str
    evaluator_sha256: str
    protocol_sha256: str
    policy_id: str
    model_id: str
    seed: int
    checkpoint_sha256: str | None = None
    resume_state_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_commit": self.source_commit,
            "source_closure_sha256": self.source_closure_sha256,
            "generator_sha256": self.generator_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "protocol_sha256": self.protocol_sha256,
            "policy_id": self.policy_id,
            "model_id": self.model_id,
            "seed": self.seed,
            "checkpoint_sha256": self.checkpoint_sha256,
            "resume_state_sha256": self.resume_state_sha256,
        }


def walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_keys(item)


def sequence_hash(values: Sequence[Any]) -> str:
    return content_hash(list(values))
