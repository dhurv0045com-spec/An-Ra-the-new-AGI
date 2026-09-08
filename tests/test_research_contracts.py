import json
import math
from pathlib import Path

import numpy as np
import pytest

from bramastra_lab.research.contracts import (
    Action, Checkpoint, ContractError, Episode, Experiment, Outcome, Promotion,
    PublicObservation, TaskSpec, TrainingBatch, Transition,
    adapt_discovery_outcome, canonical_json, content_identity, tensor_identity,
    validate_public_payload, validate_semantic_splits,
)


ROOT = Path(__file__).resolve().parents[1]
ACTION_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"],
}


def task(semantic="rule-a", split="training", text="surface"):
    return TaskSpec(
        semantic, "parity", "gen-hash", "obs/v1", split,
        {"surface": text, "nested": {"levels": [1, 2]}}, {"steps": 4},
    )


def observation(step=0, budget=2, *, episode="episode-a", task_id="rule-a", value=None):
    return PublicObservation(
        task_id, episode, step, {"x": [1, 0]} if value is None else value,
        ACTION_SCHEMA, None, budget,
    )


def transition(step=0, *, terminal=False, current=None, next_value=None):
    current = current or observation(step, 2 - step)
    following = observation(step + 1, 1 - step, value=next_value)
    action = Action("episode-a", step, "choose", {"choice": step}, "policy-a")
    return Transition(
        current.to_dict(), action.to_dict(), following.to_dict(),
        1.0 if terminal else 0.0, 1.0, terminal, False, 0.5,
    )


def checkpoint(**changes):
    values = {
        "model_spec_identity": "model", "weights_identity": "weights",
        "optimizer_identity": "optimizer", "schedule_identity": "schedule",
        "rng_state_identities": {"python": "py-rng", "torch": "torch-rng"},
        "sampler_state_identity": "sampler", "replay_cursor_identity": "cursor",
        "replay_index_identity": "index", "parent_identity": None,
        "data_identity": "data", "code_identity": "code", "runtime_identity": "runtime",
    }
    values.update(changes)
    return Checkpoint(**values)


def test_canonical_json_is_strict_and_type_preserving():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert content_identity({"a": 1}) != content_identity({"a": "1"})
    with pytest.raises(ContractError, match="keys must be strings"):
        canonical_json({1: "integer-key"})
    with pytest.raises(ContractError, match="binary data"):
        canonical_json({"ordinary": b"cannot-use-a-colliding-sentinel"})
    with pytest.raises(ContractError, match="nonfinite"):
        canonical_json({"x": math.inf})


def test_records_reject_unknown_fields_and_unsupported_versions():
    raw = task().to_dict()
    assert TaskSpec.from_dict(raw) == task()
    with pytest.raises(ContractError, match="unknown fields"):
        TaskSpec.from_dict({**raw, "private_state": {}})
    with pytest.raises(ContractError, match="unsupported schema"):
        TaskSpec.from_dict({**raw, "schema": "TaskSpec/v2"})
    with pytest.raises(ContractError, match="missing fields"):
        TaskSpec.from_dict({"schema": "TaskSpec/v1"})


@pytest.mark.parametrize("constructor", [
    lambda: TaskSpec("x", "f", "g", "s", "holdout", {}, {}),
    lambda: TaskSpec("", "f", "g", "s", "train", {}, {}),
    lambda: PublicObservation("t", "e", True, {}, ACTION_SCHEMA, None, 1),
    lambda: PublicObservation("t", "e", 0, {}, ACTION_SCHEMA, None, -1),
    lambda: Action("e", 0, "", {}, "policy"),
    lambda: TrainingBatch({}, ("source",), {"loss": "mask"}, (1,), 1, {"offset": 0}),
    lambda: Experiment("h", ("a",), {"a": 1}, (False,), ("training",), ("m",), {"n": 1}, "c", "d", "r", 1),
    lambda: Outcome("w", "t", "a", 0, math.nan, 0, None, None, None, "c", "d"),
])
def test_direct_constructors_enforce_types_counts_ids_and_splits(constructor):
    with pytest.raises(ContractError):
        constructor()


def test_nested_payloads_are_deeply_immutable_and_detached_from_callers():
    difficulty = {"nested": {"values": [1, 2]}}
    spec = TaskSpec("id", "family", "generator", "schema", "train", difficulty, {"steps": 2})
    difficulty["nested"]["values"].append(3)
    assert spec.difficulty["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        spec.difficulty["nested"]["new"] = 1

    values = {"nested": [{"value": 1}]}
    obs = PublicObservation("id", "episode", 0, values, ACTION_SCHEMA, {"seen": [1]}, 2)
    values["nested"][0]["value"] = 9
    assert obs.observable_values["nested"][0]["value"] == 1
    with pytest.raises(TypeError):
        obs.observable_values["nested"][0]["value"] = 2

    arguments = {"nested": {"choice": [1]}}
    action = Action("episode", 0, "choose", arguments, "policy")
    arguments["nested"]["choice"].append(2)
    assert action.arguments["nested"]["choice"] == (1,)
    with pytest.raises(TypeError):
        checkpoint().rng_state_identities["python"] = "changed"


def test_public_payloads_require_explicit_closed_reviewed_schemas():
    observable_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "integer"},
            "trace": {"type": "array", "items": {"type": "boolean"}},
        },
        "required": ["answer", "trace"],
    }
    obs = observation(value={"answer": 1, "trace": [True, False]})
    obs.validate_public(observable_schema=observable_schema, feedback_schema={"type": "null"})
    with pytest.raises(ContractError, match="undeclared"):
        validate_public_payload({"answer": 1, "trace": [], "hidden": 7}, observable_schema)
    with pytest.raises(ContractError, match="missing"):
        validate_public_payload({"answer": 1}, observable_schema)
    with pytest.raises(ContractError, match="must be an integer"):
        validate_public_payload({"answer": True, "trace": []}, observable_schema)
    with pytest.raises(ContractError, match="unknown fields"):
        validate_public_payload({}, {"type": "object", "properties": {}, "additionalProperties": True})
    with pytest.raises(ContractError, match="enum values must be unique"):
        validate_public_payload(1, {"type": "integer", "enum": [1, 1]})


def test_public_enum_values_survive_record_freezing_and_round_trip():
    array_schema = {"type": "array", "items": {"type": "integer"}, "enum": [[1, 2]]}
    object_schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "enum": [{"x": 1}],
    }
    validate_public_payload([1, 2], array_schema)
    validate_public_payload({"x": 1}, object_schema)
    for payload, schema in (([1, 2], array_schema), ({"x": 1}, object_schema)):
        obs = PublicObservation("task", "episode", 0, payload, ACTION_SCHEMA, None, 1)
        obs.validate_public(observable_schema=schema, feedback_schema={"type": "null"})
        restored = PublicObservation.from_dict(obs.to_dict())
        restored.validate_public(observable_schema=schema, feedback_schema={"type": "null"})


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_outcome_rejects_float_labels(label):
    with pytest.raises(ContractError, match="label must be integer"):
        Outcome("world", "target", "arm", 1, 0.5, label, None, None, None, "code", "data")


def test_transition_and_episode_validate_nested_identity_and_order_rules():
    first = transition(0, next_value={"x": [0, 1]})
    second = transition(1, terminal=True, current=PublicObservation.from_dict(first.next_observation))
    episode = Episode("rule-a", (first.to_dict(), second.to_dict()), "policy-a", 7, "task")
    assert Episode.from_dict(episode.to_dict()) == episode

    bad_action = Action("different", 0, "choose", {"choice": 0}, "policy-a")
    with pytest.raises(ContractError, match="episode IDs disagree"):
        Transition(first.observation, bad_action.to_dict(), first.next_observation, 0, 1, False, False, 0.5)
    with pytest.raises(ContractError, match="preceding next observation"):
        Episode("rule-a", (first.to_dict(), transition(1, terminal=True).to_dict()), "p", 0, "task")
    with pytest.raises(ContractError, match="begin at step zero"):
        Episode("rule-a", (transition(1, terminal=True).to_dict(),), "p", 0, "task")
    with pytest.raises(ContractError, match="must end"):
        Episode("rule-a", (first.to_dict(),), "p", 0, "task")
    with pytest.raises(ContractError, match="events after"):
        Episode("rule-a", (transition(0, terminal=True).to_dict(), second.to_dict()), "p", 0, "task")
    with pytest.raises(ContractError, match="positive"):
        Transition(first.observation, first.action, first.next_observation, 0, 1, False, False, 0.0)


def test_tensor_identity_covers_bytes_dtype_shape_and_explicit_byteorder():
    native = np.array([1, 2], dtype=np.int32)
    changed = native.copy()
    changed[0] = 9
    assert tensor_identity(native) != tensor_identity(changed)
    assert tensor_identity(native) != tensor_identity(native.astype(np.int64))
    assert tensor_identity(native) != tensor_identity(native.reshape(1, 2))
    assert tensor_identity(np.array([1, 2], dtype="<i4")) != tensor_identity(np.array([1, 2], dtype=">i4"))
    with pytest.raises(ContractError, match="object and structured"):
        tensor_identity(np.array([object()], dtype=object))


def test_semantic_split_inventory_is_normalized_order_independent_and_closed():
    a, b, dev = task("a", "train"), task("b", "train"), task("c", "dev")
    left = validate_semantic_splits({"train": [a, b], "dev": [dev]})
    right = validate_semantic_splits({"development": [dev], "training": [b, a]})
    assert left == right and left["identity"] == right["identity"]
    with pytest.raises(ContractError, match="appears in both"):
        validate_semantic_splits({"train": [a], "dev": [task("a", "dev", "different words")]})
    with pytest.raises(ContractError, match="declares development"):
        validate_semantic_splits({"training": [dev]})
    with pytest.raises(ContractError, match="duplicate aliases"):
        validate_semantic_splits({"train": [a], "training": [b]})


def test_checkpoint_requires_every_resume_state_identity():
    complete = checkpoint()
    assert Checkpoint.from_dict(complete.to_dict()) == complete
    for field in Checkpoint.required:
        raw = complete.to_dict()
        raw.pop(field)
        with pytest.raises(ContractError, match="missing fields"):
            Checkpoint.from_dict(raw)
    with pytest.raises(ContractError):
        checkpoint(weights_identity="")
    with pytest.raises(ContractError):
        checkpoint(rng_state_identities={})
    with pytest.raises(ContractError):
        checkpoint(rng_state_identities={"python": 3})


def test_outer_record_subclasses_enforce_their_own_semantics():
    experiment = Experiment(
        "paired", ("parent", "child"), {"parent": 4, "child": 4}, (1, 2),
        ("dev",), ("brier",), {"max_worlds": 8}, "code", "data", "runtime", 1,
    )
    assert experiment.splits == ("development",)
    with pytest.raises(ContractError, match="allocation"):
        Experiment("h", ("a", "b"), {"a": 1}, (1,), ("dev",), ("m",), {"n": 1}, "c", "d", "r", 1)
    with pytest.raises(ContractError, match="unique"):
        Experiment("h", ("a", "a"), {"a": 1}, (1,), ("dev",), ("m",), {"n": 1}, "c", "d", "r", 1)
    with pytest.raises(ContractError, match="label"):
        Outcome("w", "t", "a", 1, 0.5, True, None, None, None, "c", "d")

    accepted = Promotion("parent", "child", "examiner", "transfer", "retention", "history", "accept")
    assert Promotion.from_dict(accepted.to_dict()) == accepted
    for decision in ("promote", "accepted", True, ""):
        with pytest.raises(ContractError, match="decision"):
            Promotion("parent", "child", "examiner", "transfer", "retention", "history", decision)
    with pytest.raises(ContractError, match="must differ"):
        Promotion("same", "same", "examiner", "transfer", "retention", "history", "reject")
    with pytest.raises(ContractError):
        Promotion("parent", "child", "", "transfer", "retention", "history", "reject")


def test_actual_discovery_dev_701_rows_adapt_without_fabricated_evidence():
    run = ROOT / "artifacts" / "bramastra" / "discovery_dev_701"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((run / "initial" / "evaluation_rows.json").read_text(encoding="utf-8"))
    code_id = content_identity(manifest["source_sha256"])
    data_id = content_identity(rows)
    outcomes = [adapt_discovery_outcome(row, code_identity=code_id, data_identity=data_id) for row in rows]
    assert len(outcomes) == len(rows) > 100
    assert (outcomes[0].world_id, outcomes[0].target, outcomes[0].arm, outcomes[0].prediction) == (
        rows[0]["world_id"], str(rows[0]["target"]), rows[0]["policy"], rows[0]["probability"],
    )
    assert all(o.cost is None and o.failure_reason is None and o.timing_seconds is None for o in outcomes)
    assert all(o.code_identity == code_id and o.data_identity == data_id for o in outcomes)

    missing = dict(rows[0])
    missing.pop("probability")
    with pytest.raises(ContractError, match="missing fields"):
        adapt_discovery_outcome(missing, code_identity=code_id, data_identity=data_id)
    with pytest.raises(ContractError, match="unknown fields"):
        adapt_discovery_outcome({**rows[0], "case_id": rows[0]["world_id"]})


def test_handoff_fixture_dataset_validates_for_downstream_consumers():
    fixture = json.loads((ROOT / "engineering" / "reports" / "W01" / "fixture_dataset.json").read_text(encoding="utf-8"))
    inventory = validate_semantic_splits(fixture["split_assignments"])
    assert inventory["schema"] == "SemanticSplitInventory/v1"
    assert Episode.from_dict(fixture["episode_shard"]).task_semantic_id == "fixture-rule-training"
    assert TrainingBatch.from_dict(fixture["replay_descriptor"]).target_count == 1
    assert Checkpoint.from_dict(fixture["checkpoint_manifest"]).weights_identity == "fixture-weights-sha256"
