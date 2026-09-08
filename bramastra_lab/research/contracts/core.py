"""Strict, dependency-free contracts for BRAMASTRA research artifacts."""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import sys
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


class ContractError(ValueError):
    """A value does not satisfy its declared research contract."""


SPLIT_ALIASES = MappingProxyType({
    "train": "training", "training": "training", "training_gain_probe": "training_gain_probe",
    "dev": "development", "development": "development",
    "strategy_validation": "strategy_validation", "test": "confirmation", "confirmation": "confirmation",
})
SUPPORTED_SPLITS = frozenset(SPLIT_ALIASES.values())
PROMOTION_DECISIONS = frozenset({"accept", "reject", "diagnose"})


def _clean(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"nonfinite number at {path}")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ContractError(f"binary data is not a JSON value at {path}")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError(f"JSON object keys must be strings at {path}")
        return {key: _clean(value[key], f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_clean(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ContractError(f"unsupported JSON type at {path}: {type(value).__name__}")


def _freeze(value: Any, path: str = "$") -> Any:
    cleaned = _clean(value, path)
    if isinstance(cleaned, dict):
        return MappingProxyType({key: _freeze(item, f"{path}.{key}") for key, item in cleaned.items()})
    if isinstance(cleaned, list):
        return tuple(_freeze(item, f"{path}[]") for item in cleaned)
    return cleaned


def canonical_json(value: Any) -> bytes:
    return json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def content_identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _byteorder(dtype: Any) -> str:
    order = getattr(dtype, "byteorder", None)
    if order == "=":
        return sys.byteorder
    if order == "<":
        return "little"
    if order == ">":
        return "big"
    if order == "|":
        return "not-applicable"
    if str(dtype).startswith("torch."):
        return sys.byteorder
    raise ContractError("tensor dtype does not declare a supported byte order")


def tensor_identity(tensor: Any) -> str:
    """Hash dtype, shape, explicit byte order, and contiguous C-order bytes."""
    dtype, shape = getattr(tensor, "dtype", None), getattr(tensor, "shape", None)
    if dtype is None or shape is None:
        raise ContractError("tensor must expose dtype and shape")
    if getattr(dtype, "hasobject", False) or getattr(dtype, "fields", None) is not None:
        raise ContractError("object and structured tensor dtypes are unsupported")
    dimensions = []
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise ContractError("tensor shape must contain nonnegative integer dimensions")
        dimensions.append(dimension)
    try:
        raw = tensor.tobytes(order="C")
    except TypeError:
        raw = tensor.tobytes()
    except AttributeError:
        try:
            raw = tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
        except Exception as exc:
            raise ContractError("tensor does not provide contiguous CPU bytes") from exc
    if not isinstance(raw, (bytes, bytearray)):
        raise ContractError("tensor byte representation must be bytes")
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        _integer(itemsize, "tensor dtype itemsize", 1)
        expected_nbytes = itemsize
        for dimension in dimensions:
            expected_nbytes *= dimension
        if len(raw) != expected_nbytes:
            raise ContractError("tensor bytes do not match dtype itemsize and shape")
    descriptor = {"dtype": str(dtype), "shape": dimensions, "byteorder": _byteorder(dtype),
                  "layout": "C-contiguous", "nbytes": len(raw)}
    return hashlib.sha256(canonical_json(descriptor) + b"\0" + bytes(raw)).hexdigest()


def _strict(data: Mapping[str, Any], allowed: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ContractError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in data):
        raise ContractError(f"{name} keys must be strings")
    extra = set(data) - allowed
    if extra:
        raise ContractError(f"{name} has unknown fields: {sorted(extra)}")
    return dict(data)


def _id(value: Any, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")


def _integer(value: Any, name: str, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be at least {minimum}")


def _number(value: Any, name: str, minimum: float | None = None, maximum: float | None = None) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{name} must be at most {maximum}")


def _mapping(value: Any, name: str, *, nonempty: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    result = _freeze(value, name)
    if nonempty and not result:
        raise ContractError(f"{name} must not be empty")
    return result


def _ids(value: Any, name: str) -> Mapping[str, str]:
    result = _mapping(value, name, nonempty=True)
    for key, item in result.items():
        _id(item, f"{name}.{key}")
    return result


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ContractError(f"{name} must be a nonempty array")
    result = tuple(value)
    for index, item in enumerate(result):
        _id(item, f"{name}[{index}]")
    return result


def normalize_split(value: Any) -> str:
    if not isinstance(value, str) or value not in SPLIT_ALIASES:
        raise ContractError("unsupported split")
    return SPLIT_ALIASES[value]


class Record:
    schema: ClassVar[str]
    required: ClassVar[frozenset[str]]
    optional: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["schema"] = self.schema
        return _clean(result)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        values = _strict(data, cls.required | cls.optional | {"schema"}, cls.schema)
        if values.get("schema") != cls.schema:
            raise ContractError(f"unsupported schema/version for {cls.schema}")
        missing = cls.required - set(values)
        if missing:
            raise ContractError(f"{cls.schema} missing fields: {sorted(missing)}")
        values.pop("schema")
        return cls(**values)

    def identity(self) -> str:
        return content_identity(self.to_dict())


def _validate_public_schema(schema: Any, path: str = "public payload schema") -> Mapping[str, Any]:
    """Validate the supported closed-schema dialect before applying it to data."""
    spec = _strict(schema, frozenset({"type", "properties", "required", "items", "enum"}), path)
    expected = spec.get("type")
    if expected not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ContractError(f"{path} has unsupported type")
    allowed_by_type = {
        "object": {"type", "properties", "required", "enum"},
        "array": {"type", "items", "enum"},
        "string": {"type", "enum"}, "integer": {"type", "enum"},
        "number": {"type", "enum"}, "boolean": {"type", "enum"},
        "null": {"type", "enum"},
    }
    invalid = set(spec) - allowed_by_type[expected]
    if invalid:
        raise ContractError(f"{path} has fields invalid for {expected}: {sorted(invalid)}")
    if "enum" in spec:
        enum = spec["enum"]
        if not isinstance(enum, (list, tuple)) or not enum:
            raise ContractError(f"{path} enum must be a nonempty array")
        _clean(enum, f"{path}.enum")
        if len({canonical_json(item) for item in enum}) != len(enum):
            raise ContractError(f"{path} enum values must be unique by JSON type and value")
    if expected == "object":
        properties, required = spec.get("properties"), spec.get("required", ())
        if not isinstance(properties, Mapping) or any(not isinstance(key, str) for key in properties):
            raise ContractError(f"{path} object requires string-keyed properties")
        if not isinstance(required, (list, tuple)) or any(not isinstance(key, str) for key in required):
            raise ContractError(f"{path} required must be an array of strings")
        if len(set(required)) != len(required) or not set(required) <= set(properties):
            raise ContractError(f"{path} required fields must be unique declared properties")
        for key, child in properties.items():
            _validate_public_schema(child, f"{path}.properties.{key}")
    elif expected == "array":
        if "items" not in spec:
            raise ContractError(f"{path} array requires an items schema")
        _validate_public_schema(spec["items"], f"{path}.items")
    if "enum" in spec:
        base_schema = {key: value for key, value in spec.items() if key != "enum"}
        for index, choice in enumerate(spec["enum"]):
            try:
                validate_public_payload(choice, base_schema)
            except ContractError as exc:
                raise ContractError(f"{path}.enum[{index}] violates its declared type") from exc
    return spec


def validate_public_payload(payload: Any, schema: Mapping[str, Any]) -> None:
    """Validate public JSON against a recursive, closed field allowlist."""
    _clean(payload)
    spec = _validate_public_schema(schema)
    expected = spec.get("type")
    if "enum" in spec:
        enum = spec["enum"]
        # Records freeze JSON arrays/objects to tuples/mapping proxies. Compare
        # their canonical JSON form so freezing preserves enum semantics while
        # _clean still keeps booleans distinct from integers.
        if not any(canonical_json(payload) == canonical_json(choice) for choice in enum):
            raise ContractError("public payload value is outside enum")
    if expected == "object":
        if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
            raise ContractError("public payload must be a string-keyed object")
        properties, required = spec["properties"], spec.get("required", ())
        unknown, missing = set(payload) - set(properties), set(required) - set(payload)
        if unknown:
            raise ContractError(f"public payload has undeclared fields: {sorted(unknown)}")
        if missing:
            raise ContractError(f"public payload is missing fields: {sorted(missing)}")
        for key, item in payload.items():
            validate_public_payload(item, properties[key])
    elif expected == "array":
        if not isinstance(payload, (list, tuple)):
            raise ContractError("public payload must be an array")
        for item in payload:
            validate_public_payload(item, spec["items"])
    elif expected == "string" and not isinstance(payload, str):
        raise ContractError("public payload must be a string")
    elif expected == "integer" and (not isinstance(payload, int) or isinstance(payload, bool)):
        raise ContractError("public payload must be an integer")
    elif expected == "number" and (not isinstance(payload, (int, float)) or isinstance(payload, bool)
                                    or not math.isfinite(payload)):
        raise ContractError("public payload must be a finite number")
    elif expected == "boolean" and not isinstance(payload, bool):
        raise ContractError("public payload must be a boolean")
    elif expected == "null" and payload is not None:
        raise ContractError("public payload must be null")


@dataclass(frozen=True)
class TaskSpec(Record):
    semantic_id: str; family: str; generator_hash: str; public_schema_id: str; split: str
    difficulty: Mapping[str, Any]; resource_limits: Mapping[str, Any]
    schema: ClassVar[str] = "TaskSpec/v1"
    required: ClassVar[frozenset[str]] = frozenset({"semantic_id", "family", "generator_hash", "public_schema_id", "split", "difficulty", "resource_limits"})

    def __post_init__(self):
        for name in ("semantic_id", "family", "generator_hash", "public_schema_id"):
            _id(getattr(self, name), name)
        object.__setattr__(self, "split", normalize_split(self.split))
        object.__setattr__(self, "difficulty", _mapping(self.difficulty, "difficulty"))
        object.__setattr__(self, "resource_limits", _mapping(self.resource_limits, "resource_limits"))


@dataclass(frozen=True)
class PublicObservation(Record):
    task_id: str; episode_id: str; step_id: int; observable_values: Any
    legal_action_schema: Mapping[str, Any]; feedback: Any; remaining_budget: int
    schema: ClassVar[str] = "PublicObservation/v1"
    required: ClassVar[frozenset[str]] = frozenset({"task_id", "episode_id", "step_id", "observable_values", "legal_action_schema", "feedback", "remaining_budget"})

    def __post_init__(self):
        _id(self.task_id, "task_id"); _id(self.episode_id, "episode_id")
        _integer(self.step_id, "step_id", 0); _integer(self.remaining_budget, "remaining_budget", 0)
        object.__setattr__(self, "observable_values", _freeze(self.observable_values, "observable_values"))
        object.__setattr__(self, "feedback", _freeze(self.feedback, "feedback"))
        object.__setattr__(self, "legal_action_schema", _mapping(self.legal_action_schema, "legal_action_schema", nonempty=True))

    def validate_public(self, *, observable_schema: Mapping[str, Any],
                        feedback_schema: Mapping[str, Any]) -> None:
        """Apply environment-reviewed closed schemas to both public data payloads."""
        validate_public_payload(self.observable_values, observable_schema)
        validate_public_payload(self.feedback, feedback_schema)
        _validate_public_schema(self.legal_action_schema, "legal_action_schema")


@dataclass(frozen=True)
class Action(Record):
    episode_id: str; step_id: int; action_kind: str; arguments: Mapping[str, Any]; policy_identity: str
    schema: ClassVar[str] = "Action/v1"
    required: ClassVar[frozenset[str]] = frozenset({"episode_id", "step_id", "action_kind", "arguments", "policy_identity"})

    def __post_init__(self):
        _id(self.episode_id, "episode_id"); _integer(self.step_id, "step_id", 0)
        _id(self.action_kind, "action_kind"); _id(self.policy_identity, "policy_identity")
        object.__setattr__(self, "arguments", _mapping(self.arguments, "arguments"))


@dataclass(frozen=True)
class Transition(Record):
    observation: Mapping[str, Any]; action: Mapping[str, Any]; next_observation: Mapping[str, Any]
    reward: float; cost: float; termination: bool; truncation: bool; behavior_probability: float | None
    schema: ClassVar[str] = "Transition/v1"
    required: ClassVar[frozenset[str]] = frozenset({"observation", "action", "next_observation", "reward", "cost", "termination", "truncation", "behavior_probability"})

    def __post_init__(self):
        observation = PublicObservation.from_dict(self.observation)
        action = Action.from_dict(self.action)
        next_observation = PublicObservation.from_dict(self.next_observation)
        _number(self.reward, "reward"); _number(self.cost, "cost", 0)
        if not isinstance(self.termination, bool) or not isinstance(self.truncation, bool):
            raise ContractError("termination and truncation must be booleans")
        if self.termination and self.truncation:
            raise ContractError("termination and truncation are distinct")
        if self.behavior_probability is not None:
            _number(self.behavior_probability, "behavior_probability", 0, 1)
            if self.behavior_probability == 0:
                raise ContractError("selected action behavior probability must be positive")
        if observation.episode_id != action.episode_id or next_observation.episode_id != action.episode_id:
            raise ContractError("transition episode IDs disagree")
        if observation.task_id != next_observation.task_id:
            raise ContractError("transition task IDs disagree")
        if observation.step_id != action.step_id:
            raise ContractError("observation and action step IDs disagree")
        if next_observation.step_id != observation.step_id + 1:
            raise ContractError("next observation step must follow current step")
        if next_observation.remaining_budget > observation.remaining_budget:
            raise ContractError("remaining budget cannot increase")
        object.__setattr__(self, "observation", _freeze(observation.to_dict()))
        object.__setattr__(self, "action", _freeze(action.to_dict()))
        object.__setattr__(self, "next_observation", _freeze(next_observation.to_dict()))


@dataclass(frozen=True)
class Episode(Record):
    task_semantic_id: str; transitions: tuple[Mapping[str, Any], ...]
    collection_policy: str; seed: int; reset_scope: str
    schema: ClassVar[str] = "Episode/v1"
    required: ClassVar[frozenset[str]] = frozenset({"task_semantic_id", "transitions", "collection_policy", "seed", "reset_scope"})

    def __post_init__(self):
        _id(self.task_semantic_id, "task_semantic_id"); _id(self.collection_policy, "collection_policy")
        _id(self.reset_scope, "reset_scope"); _integer(self.seed, "seed", 0)
        if not isinstance(self.transitions, (list, tuple)) or not self.transitions:
            raise ContractError("episode transitions must be a nonempty array")
        result, episode_id, previous_next, done = [], None, None, False
        for raw in self.transitions:
            if done:
                raise ContractError("events after termination/truncation")
            transition = Transition.from_dict(raw)
            observation = PublicObservation.from_dict(transition.observation)
            next_observation = PublicObservation.from_dict(transition.next_observation)
            if observation.task_id != self.task_semantic_id:
                raise ContractError("episode task semantic ID disagrees with transition task")
            if episode_id is None:
                episode_id = observation.episode_id
                if observation.step_id != 0:
                    raise ContractError("episode must begin at step zero")
            if observation.episode_id != episode_id or next_observation.episode_id != episode_id:
                raise ContractError("episode contains multiple episode IDs")
            if previous_next is not None and observation.to_dict() != previous_next.to_dict():
                raise ContractError("each transition must begin with the preceding next observation")
            previous_next = next_observation
            done = transition.termination or transition.truncation
            result.append(_freeze(transition.to_dict()))
        if not done:
            raise ContractError("episode must end in termination or truncation")
        object.__setattr__(self, "transitions", tuple(result))


@dataclass(frozen=True)
class TrainingBatch(Record):
    tensor_content_ids: Mapping[str, str]; source_ids: tuple[str, ...]; masks: Mapping[str, str]
    lengths: tuple[int, ...]; target_count: int; sampler_cursor: Mapping[str, Any]
    schema: ClassVar[str] = "TrainingBatch/v1"
    required: ClassVar[frozenset[str]] = frozenset({"tensor_content_ids", "source_ids", "masks", "lengths", "target_count", "sampler_cursor"})

    def __post_init__(self):
        object.__setattr__(self, "tensor_content_ids", _ids(self.tensor_content_ids, "tensor_content_ids"))
        object.__setattr__(self, "source_ids", _string_tuple(self.source_ids, "source_ids"))
        object.__setattr__(self, "masks", _ids(self.masks, "masks"))
        if not isinstance(self.lengths, (list, tuple)) or len(self.lengths) != len(self.source_ids):
            raise ContractError("lengths must have one entry per source")
        for index, length in enumerate(self.lengths):
            _integer(length, f"lengths[{index}]", 0)
        object.__setattr__(self, "lengths", tuple(self.lengths)); _integer(self.target_count, "target_count", 0)
        object.__setattr__(self, "sampler_cursor", _mapping(self.sampler_cursor, "sampler_cursor", nonempty=True))


@dataclass(frozen=True)
class Checkpoint(Record):
    model_spec_identity: str; weights_identity: str; optimizer_identity: str; schedule_identity: str
    rng_state_identities: Mapping[str, str]; sampler_state_identity: str; replay_cursor_identity: str
    replay_index_identity: str; parent_identity: str | None; data_identity: str; code_identity: str
    runtime_identity: str
    schema: ClassVar[str] = "Checkpoint/v1"
    required: ClassVar[frozenset[str]] = frozenset({"model_spec_identity", "weights_identity", "optimizer_identity", "schedule_identity", "rng_state_identities", "sampler_state_identity", "replay_cursor_identity", "replay_index_identity", "parent_identity", "data_identity", "code_identity", "runtime_identity"})

    def __post_init__(self):
        for name in ("model_spec_identity", "weights_identity", "optimizer_identity", "schedule_identity", "sampler_state_identity", "replay_cursor_identity", "replay_index_identity", "data_identity", "code_identity", "runtime_identity"):
            _id(getattr(self, name), name)
        _id(self.parent_identity, "parent_identity", optional=True)
        object.__setattr__(self, "rng_state_identities", _ids(self.rng_state_identities, "rng_state_identities"))


@dataclass(frozen=True)
class Experiment(Record):
    hypothesis: str; arms: tuple[str, ...]; allocation: Mapping[str, int]; seeds: tuple[int, ...]
    splits: tuple[str, ...]; metrics: tuple[str, ...]; stop_rules: Mapping[str, Any]
    code_identity: str; data_identity: str; runtime_identity: str; revision: int
    schema: ClassVar[str] = "Experiment/v1"
    required: ClassVar[frozenset[str]] = frozenset({"hypothesis", "arms", "allocation", "seeds", "splits", "metrics", "stop_rules", "code_identity", "data_identity", "runtime_identity", "revision"})

    def __post_init__(self):
        _id(self.hypothesis, "hypothesis"); object.__setattr__(self, "arms", _string_tuple(self.arms, "arms"))
        if len(set(self.arms)) != len(self.arms):
            raise ContractError("experiment arms must be unique")
        allocation = _mapping(self.allocation, "allocation", nonempty=True)
        if set(allocation) != set(self.arms):
            raise ContractError("allocation must name every and only experiment arm")
        for arm, count in allocation.items():
            _integer(count, f"allocation.{arm}", 0)
        object.__setattr__(self, "allocation", allocation)
        if not isinstance(self.seeds, (list, tuple)) or not self.seeds:
            raise ContractError("seeds must be a nonempty array")
        for index, seed in enumerate(self.seeds):
            _integer(seed, f"seeds[{index}]", 0)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not isinstance(self.splits, (list, tuple)) or not self.splits:
            raise ContractError("splits must be a nonempty array")
        object.__setattr__(self, "splits", tuple(normalize_split(split) for split in self.splits))
        if len(set(self.splits)) != len(self.splits):
            raise ContractError("experiment splits must be unique after normalization")
        object.__setattr__(self, "metrics", _string_tuple(self.metrics, "metrics"))
        object.__setattr__(self, "stop_rules", _mapping(self.stop_rules, "stop_rules", nonempty=True))
        for name in ("code_identity", "data_identity", "runtime_identity"):
            _id(getattr(self, name), name)
        _integer(self.revision, "revision", 1)


@dataclass(frozen=True)
class Outcome(Record):
    world_id: str; target: str; arm: str; budget: int; prediction: float; label: int
    cost: float | None; failure_reason: str | None; timing_seconds: float | None
    code_identity: str; data_identity: str
    schema: ClassVar[str] = "Outcome/v1"
    required: ClassVar[frozenset[str]] = frozenset({"world_id", "target", "arm", "budget", "prediction", "label", "cost", "failure_reason", "timing_seconds", "code_identity", "data_identity"})

    def __post_init__(self):
        for name in ("world_id", "target", "arm", "code_identity", "data_identity"):
            _id(getattr(self, name), name)
        _integer(self.budget, "budget", 0); _number(self.prediction, "prediction", 0, 1)
        if not isinstance(self.label, int) or isinstance(self.label, bool) or self.label not in (0, 1):
            raise ContractError("label must be integer 0 or 1")
        if self.cost is not None:
            _number(self.cost, "cost", 0)
        if self.failure_reason is not None:
            _id(self.failure_reason, "failure_reason")
        if self.timing_seconds is not None:
            _number(self.timing_seconds, "timing_seconds", 0)


@dataclass(frozen=True)
class Promotion(Record):
    parent_identity: str; child_identity: str; examiner_protocol_identity: str
    transfer_comparison_identity: str; retention_comparison_identity: str
    cumulative_history_identity: str; decision: str
    schema: ClassVar[str] = "Promotion/v1"
    required: ClassVar[frozenset[str]] = frozenset({"parent_identity", "child_identity", "examiner_protocol_identity", "transfer_comparison_identity", "retention_comparison_identity", "cumulative_history_identity", "decision"})

    def __post_init__(self):
        for field_name in self.required - {"decision"}:
            _id(getattr(self, field_name), field_name)
        if not isinstance(self.decision, str) or self.decision not in PROMOTION_DECISIONS:
            raise ContractError(f"decision must be one of {sorted(PROMOTION_DECISIONS)}")
        if self.parent_identity == self.child_identity:
            raise ContractError("promotion parent and child must differ")


def validate_semantic_splits(assignments: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(assignments, Mapping) or not assignments:
        raise ContractError("split assignments must be a nonempty object")
    if any(not isinstance(key, str) for key in assignments):
        raise ContractError("split assignment keys must be strings")
    normalized: dict[str, list[dict[str, str]]] = {}; seen: dict[str, str] = {}
    for declared_split, raw_tasks in assignments.items():
        split = normalize_split(declared_split)
        if split in normalized:
            raise ContractError(f"duplicate aliases declare split {split}")
        if not isinstance(raw_tasks, (list, tuple)):
            raise ContractError(f"split {declared_split} must contain an array")
        entries = []
        for raw_task in raw_tasks:
            task = raw_task if isinstance(raw_task, TaskSpec) else TaskSpec.from_dict(raw_task)
            if task.split != split:
                raise ContractError(f"task {task.semantic_id} declares {task.split} but is assigned to {split}")
            prior = seen.get(task.semantic_id)
            if prior is not None:
                raise ContractError(f"semantic task {task.semantic_id} appears in both {prior} and {split}")
            seen[task.semantic_id] = split
            entries.append({"semantic_id": task.semantic_id, "task_identity": task.identity()})
        normalized[split] = sorted(entries, key=lambda item: (item["semantic_id"], item["task_identity"]))
    inventory = {"schema": "SemanticSplitInventory/v1", "algorithm": "semantic_id/v1", "splits": {key: normalized[key] for key in sorted(normalized)}}
    inventory["identity"] = content_identity(inventory)
    return inventory


def adapt_discovery_outcome(row: Mapping[str, Any], *, code_identity: str = "unavailable", data_identity: str = "unavailable") -> Outcome:
    """Adapt an actual discovery evaluation row without inventing evidence."""
    values = _strict(row, frozenset({"world_id", "family", "target", "policy", "budget", "correct", "probability", "label", "queries", "actions"}), "discovery outcome row")
    missing = {"world_id", "target", "policy", "budget", "probability", "label"} - set(values)
    if missing:
        raise ContractError(f"discovery outcome row missing fields: {sorted(missing)}")
    target = values["target"]
    if not isinstance(target, (str, int)) or isinstance(target, bool) or str(target) == "":
        raise ContractError("discovery target must be a string or integer")
    if "queries" in values and values["queries"] != values["budget"]:
        raise ContractError("discovery queries and budget disagree")
    if "actions" in values:
        if not isinstance(values["actions"], (list, tuple)):
            raise ContractError("discovery actions must be an array")
        if len(values["actions"]) != values["budget"]:
            raise ContractError("discovery action count and budget disagree")
        _clean(values["actions"], "discovery actions")
    if "correct" in values:
        if not isinstance(values["correct"], bool):
            raise ContractError("discovery correct must be a boolean")
        predicted_label = int(values["probability"] >= 0.5) if isinstance(values["probability"], (int, float)) and not isinstance(values["probability"], bool) else None
        if predicted_label is not None and values["correct"] != (predicted_label == values["label"]):
            raise ContractError("discovery correct disagrees with prediction and label")
    return Outcome(values["world_id"], str(target), values["policy"], values["budget"],
                   values["probability"], values["label"], None, None, None,
                   code_identity, data_identity)
