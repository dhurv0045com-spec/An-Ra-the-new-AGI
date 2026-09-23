"""Versioned, lossless public-state projections shared by training and inference.

This module does not discover what is public. Callers must pass only the goal,
received history, workspace evidence and declared budgets admitted by their
environment. Once admitted, the codec preserves every JSON field and value;
Python tuples normalize to JSON arrays at this wire boundary, while unsupported
objects reject instead of being stringified or sliced.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from bramastra_lab.research.contracts.core import content_identity


PUBLIC_STATE_SCHEMA = "bramastra-public-state/v3"
PUBLIC_HISTORY_SCHEMA = "bramastra-public-history/v3"
PUBLIC_EVIDENCE_SCHEMA = "bramastra-public-evidence/v2"
PUBLIC_MEMORY_SCHEMA = "bramastra-public-memory/v1"

ACTION_KEY_CODES = {
    "kind": "k", "variable": "var", "item": "itm", "input": "inp",
    "value": "val", "container": "cnt", "switch": "sw",
    "answer": "ans", "target": "tgt", "column": "col",
    "equals": "eq", "op": "op", "operand": "opr",
}
FEEDBACK_KEY_CODES = {
    "kind": "k", "variable": "var", "value": "val", "item": "itm",
    "requires": "req", "input": "inp", "result": "res",
    "total": "tot", "matched": "mat", "written": "wri",
    "submitted_answer": "sub", "correct": "cor",
    "contains_item": "cnt_itm", "changed": "chg", "state": "state",
    "success": "suc", "error": "err", "is_dependency": "is_dep",
    "table": "tbl", "filtered_rows": "flt", "sum": "sum",
}

PUBLIC_STATE_CONTRACT = {
    "schema": PUBLIC_STATE_SCHEMA,
    "history_schema": PUBLIC_HISTORY_SCHEMA,
    "evidence_schema": PUBLIC_EVIDENCE_SCHEMA,
    "memory_schema": PUBLIC_MEMORY_SCHEMA,
    "action_key_codes": ACTION_KEY_CODES,
    "feedback_key_codes": FEEDBACK_KEY_CODES,
    "encoding": "canonical-json-events/compact-inspection-v1",
    "instruction_tag": "instruction:{v:3}",
    "projection": "typed inspect/observation tuples; otherwise versioned aliases; all unknown keys retained as pairs; no implicit coercion or slicing; Python tuples normalize to JSON arrays",
    "optional_context": "omit whole evidence records with receipt IDs",
    "required_context": "schema, goal, ordered received history, remaining budgets",
    "teacher_budget_rule": "one prior decision call per teacher action; full node budget before search",
}


class PublicStateError(ValueError):
    """A public-state value is invalid or cannot be represented losslessly."""


def public_state_identity() -> str:
    """Identity for checkpoints and compiled-row sidecars."""
    return content_identity(PUBLIC_STATE_CONTRACT)


def _json_value(value: Any, *, path: str) -> Any:
    """Copy a JSON value while rejecting implicit type/key conversions."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PublicStateError(f"{path} contains a nonfinite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise PublicStateError(
                    f"{path} has a non-string object key {key!r}")
            result[key] = _json_value(child, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(child, path=f"{path}[{index}]")
                for index, child in enumerate(value)]
    raise PublicStateError(
        f"{path} contains unsupported {type(value).__name__}; only JSON-compatible "
        "values are admitted so types cannot be silently stringified")


def compact_history_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Encode history compactly with a versioned, collision-free envelope."""
    if not isinstance(entry, Mapping):
        raise PublicStateError("history entries must be JSON objects")
    payload = _json_value(entry, path="history")
    action = payload.get("action")
    feedback = payload.get("feedback")
    # K8 inquiry trajectories repeat the same inspection operation and
    # variable in both action and observation. This exact-shape form stores
    # the variable and observed value once; every other shape uses the
    # general lossless codec below. Typed comparison matters because Python
    # treats False == 0 even though they are different JSON observations.
    if _is_inspection_observation(action, feedback):
        return {"v": 3, "i": [
            _json_value(action["variable"], path="history.action.variable"),
            _json_value(feedback["value"], path="history.feedback.value"),
        ]}
    # Preserve direct observation records and malformed/unrecognized wrapper
    # shapes whole. Known action/feedback keys are abbreviated only inside
    # their nested maps; unknown keys live in pair arrays and cannot collide
    # with any codec field.
    if (("action" not in payload and "feedback" not in payload)
            or ("action" in payload and not isinstance(action, Mapping))
            or ("feedback" in payload and not isinstance(feedback, Mapping))):
        return {"v": 3, "p": payload}
    mask = (1 if "action" in payload else 0) | (2 if "feedback" in payload else 0)
    result: dict[str, Any] = {"v": 3, "m": mask}
    if mask & 1:
        result["a"] = _compact_fields(action, ACTION_KEY_CODES,
                                      path="history.action")
    if mask & 2:
        result["f"] = _compact_fields(feedback, FEEDBACK_KEY_CODES,
                                      path="history.feedback")
    outer = [[key, value] for key, value in payload.items()
             if key not in {"action", "feedback"}]
    if outer:
        result["u"] = outer
    return result


def expand_history_entry(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Invert a v3 history envelope; ambiguous unversioned rows reject."""
    if not isinstance(compact, Mapping) or compact.get("v") != 3:
        raise PublicStateError(
            "history envelope is unversioned, unsupported, or has unknown fields")
    if set(compact) == {"v", "i"}:
        pair = compact.get("i")
        if not isinstance(pair, list) or len(pair) != 2:
            raise PublicStateError(
                "inspection-observation envelope must contain a two-value list")
        variable = _json_value(pair[0], path="history.variable")
        value = _json_value(pair[1], path="history.value")
        return {
            "action": {"kind": "inspect",
                       "variable": _json_value(variable, path="history.action.variable")},
            "feedback": {"kind": "observation",
                         "variable": _json_value(variable, path="history.feedback.variable"),
                         "value": value},
        }
    if set(compact) == {"v", "p"}:
        payload = compact.get("p")
        if not isinstance(payload, Mapping):
            raise PublicStateError("direct history payload must be a JSON object")
        return _json_value(payload, path="history.payload")
    mask = compact.get("m")
    if (not isinstance(mask, int) or isinstance(mask, bool)
            or mask not in (1, 2, 3)):
        raise PublicStateError("history wrapper has an invalid presence mask")
    expected = {"v", "m"}
    if mask & 1:
        expected.add("a")
    if mask & 2:
        expected.add("f")
    if "u" in compact:
        expected.add("u")
    if set(compact) != expected:
        raise PublicStateError("history wrapper fields disagree with its presence mask")
    payload: dict[str, Any] = {}
    if mask & 1:
        payload["action"] = _expand_fields(
            compact["a"], ACTION_KEY_CODES, path="history.action")
    if mask & 2:
        payload["feedback"] = _expand_fields(
            compact["f"], FEEDBACK_KEY_CODES, path="history.feedback")
    _restore_pairs(payload, compact.get("u", []),
                   reserved={"action", "feedback"}, path="history")
    return payload


def _is_inspection_observation(action: Any, feedback: Any) -> bool:
    """Whether this exact K8 inspect/observation shape has the tuple form."""
    if (not isinstance(action, Mapping) or not isinstance(feedback, Mapping)
            or set(action) != {"kind", "variable"}
            or set(feedback) != {"kind", "variable", "value"}
            or action.get("kind") != "inspect"
            or feedback.get("kind") != "observation"):
        return False
    left = json.dumps(action["variable"], sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)
    right = json.dumps(feedback["variable"], sort_keys=True,
                       separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False)
    return left == right


def _compact_fields(value: Mapping[str, Any], aliases: Mapping[str, str], *,
                    path: str) -> dict[str, Any]:
    copied = _json_value(value, path=path)
    reverse = {code: key for key, code in aliases.items()}
    if len(reverse) != len(aliases):
        raise PublicStateError("field-code table is not one-to-one")
    result: dict[str, Any] = {}
    unknown: list[list[Any]] = []
    for key, item in copied.items():
        code = aliases.get(key)
        if code is None:
            unknown.append([key, item])
        else:
            result[code] = item
    if unknown:
        result["u"] = unknown
    return result


def _expand_fields(value: Any, aliases: Mapping[str, str], *,
                   path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicStateError(f"{path} compact value must be an object")
    reverse = {code: key for key, code in aliases.items()}
    if len(reverse) != len(aliases):
        raise PublicStateError("field-code table is not one-to-one")
    compact = dict(value)
    unknown = compact.pop("u", [])
    expanded = {reverse.get(key, key): _json_value(item, path=f"{path}.{key}")
                for key, item in compact.items()}
    _restore_pairs(expanded, unknown, reserved=set(expanded), path=path)
    return expanded


def _restore_pairs(target: dict[str, Any], pairs: Any, *,
                   reserved: set[str], path: str) -> None:
    if not isinstance(pairs, list):
        raise PublicStateError(f"{path} extension fields must be a pair list")
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
            raise PublicStateError(f"{path} extension must contain [key, value] pairs")
        key, value = pair
        if key in seen or key in reserved:
            raise PublicStateError(f"{path} extension key collides with {key!r}")
        seen.add(key)
        target[key] = _json_value(value, path=f"{path}.{key}")


def compact_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap complete workspace evidence without colliding with user keys."""
    if not isinstance(record, Mapping):
        raise PublicStateError("evidence records must be JSON objects")
    return {"v": 2, "r": _json_value(record, path="evidence")}


def expand_evidence_record(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Invert a v2 evidence envelope exactly."""
    if not isinstance(compact, Mapping) or set(compact) != {"v", "r"} \
            or compact.get("v") != 2:
        raise PublicStateError(
            "evidence envelope is unversioned, unsupported, or has unknown fields")
    record = compact.get("r")
    if not isinstance(record, Mapping):
        raise PublicStateError("evidence payload must be a JSON object")
    return _json_value(record, path="evidence.record")


def compact_memory_content(content: str) -> dict[str, Any]:
    """Serialize one eligible memory item as a compact, typed prompt event.

    MemoryIndex stores text so it can index both prose and structured examples.
    Structured JSON objects/arrays are restored as JSON values to avoid
    escaping their full representation as a string; other content stays text.
    """
    if not isinstance(content, str) or not content:
        raise PublicStateError("memory content must be nonempty text")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = content
    if not isinstance(parsed, (Mapping, list)):
        parsed = content
    try:
        payload = _json_value(parsed, path="memory.content")
    except PublicStateError:
        # A string that resembles invalid/non-finite JSON remains plain text;
        # it must not make the entire retrieval pass fail.
        payload = content
    return {"v": 1, "m": payload}


def public_state_prefix_events(
    goal: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
    budgets: Mapping[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Build the mandatory training/inference event prefix from public inputs."""
    if not isinstance(goal, Mapping):
        raise PublicStateError("goal must be a JSON object")
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise PublicStateError("history must be an ordered sequence of JSON objects")
    goal_payload = _json_value(goal, path="goal")
    events: list[tuple[str, dict[str, Any]]] = [
        # The role plus version is the compact wire tag. The full schema
        # identity remains in the checkpoint/compiled-row sidecar contract.
        ("instruction", {"v": 3}),
        ("goal", goal_payload),
    ]
    events.extend(("observation", compact_history_entry(entry))
                  for entry in history)
    budget_event = remaining_budget_event(budgets)
    if budget_event is not None:
        events.append(("budget", budget_event))
    return events


def remaining_budget_event(budgets: Mapping[str, Any] | None
                           ) -> dict[str, Any] | None:
    """Validate and version the runtime budget event without dropping fields."""
    if budgets is None:
        return None
    if not isinstance(budgets, Mapping):
        raise PublicStateError("budgets must be a JSON object")
    copied = _json_value(budgets, path="budgets")
    for key in ("actions_left", "calls_left", "nodes_left"):
        if key not in copied:
            continue
        value = copied[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PublicStateError(f"budget {key!r} must be a nonnegative integer")
    return {"v": 2, "r": copied}
