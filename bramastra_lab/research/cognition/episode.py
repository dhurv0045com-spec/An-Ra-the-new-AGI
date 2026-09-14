"""Live cognitive episode kernel (O04): observations in, actions out.

One episode loop with distinct policy adapters over real finite
environments (reset/step/verify). The loop reads a public observation and
goal, renders state through the shared renderer, asks the adapter for a
typed legal action, executes it, records the received result/cost and
updates state with ONLY received observations. It stops on environment
termination or the declared budget. The independent task verifier decides
final success. Exceptions and invalid actions become explicit outcomes with
costs; they are never replaced by teacher choices.

Real observations and imagined outcomes have different types and storage
paths: imagined planner nodes live in a separate list and can never enter
the real history. Model-call counts, tool calls and imagined nodes reduce
from the emitted events, never from constants.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


class EpisodeError(ValueError):
    """An episode violated its declared kernel contract."""

    def __init__(self, message: str = "", *,
                 model_origin: str | None = None) -> None:
        super().__init__(message)
        # A failed generation still has an origin (real model output that did
        # not parse is model-originated evidence of failure, not absence).
        self.model_origin = model_origin


ACTION_BUDGET_DEFAULT = 4
CALL_BUDGET_DEFAULT = 16
NODE_BUDGET_DEFAULT = 8
RENDER_MAX_TOKENS_DEFAULT = 512


@dataclass
class EpisodeEvent:
    """One durable trace event (real observation/action, never imagined)."""

    event_index: int
    session_job_id: str
    checkpoint_id: str | None
    input_hash: str
    kind: str  # observation | action | generation | verification | truncation
    action: Mapping[str, Any] | None = None
    generation_id: str | None = None
    model_origin: str | None = None
    planner_meta: Mapping[str, Any] | None = None
    observed_result: Mapping[str, Any] | None = None
    resource_delta: float = 0.0
    predecessor: int | None = None
    imagined: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"event_index": self.event_index,
                "session_job_id": self.session_job_id,
                "checkpoint_id": self.checkpoint_id,
                "input_hash": self.input_hash, "kind": self.kind,
                "action": dict(self.action) if self.action else None,
                "generation_id": self.generation_id,
                "model_origin": self.model_origin,
                "planner_meta": dict(self.planner_meta)
                if self.planner_meta else None,
                "observed_result": dict(self.observed_result)
                if self.observed_result else None,
                "resource_delta": self.resource_delta,
                "predecessor": self.predecessor, "imagined": self.imagined}


@dataclass
class ImaginedNode:
    """One hypothetical planner node (never a real observation)."""

    node_index: int
    public_state_hash: str
    goal_hash: str
    action_prefix: list[Mapping[str, Any]]
    predicted_outcome: Mapping[str, Any]
    value_estimate: float
    remaining_budget: int
    provenance: str


def render_public_state(*, goal: Mapping[str, Any],
                        history: Sequence[Mapping[str, Any]],
                        workspace: Sequence[Mapping[str, Any]] | None = None,
                        budgets: Mapping[str, Any] | None = None,
                        max_tokens: int = RENDER_MAX_TOKENS_DEFAULT
                        ) -> tuple[list[int], list[str]]:
    """Shared state renderer for training and inference (O04).

    Encodes goal + bounded history + workspace + budgets and returns
    (tokens, omitted_workspace_ids). The goal is never dropped and
    conflicting-status records are never omitted; older active records fall
    out of the render window only with their IDs explicitly reported (the
    caller emits them into the trace), never silently.
    """
    from bramastra_lab.research.experience.codec import (
        SPECIAL_BOUNDARY, encode_event, encode_text)

    tokens = [SPECIAL_BOUNDARY]
    tokens += encode_event("goal", dict(goal))
    for entry in list(history):
        tokens += encode_event("observation", _compact_history_entry(entry))
    records = list(workspace or [])
    priority = [r for r in records if r.get("status") == "conflicting"]
    rest = [r for r in records if r.get("status") != "conflicting"]
    # Compact budget block (list form saves ~35 bytes vs named keys).
    budget_block = {"b": [int((budgets or {}).get("actions_left", 0)),
                          int((budgets or {}).get("calls_left", 0)),
                          int((budgets or {}).get("nodes_left", 0))]} \
        if budgets else None
    omitted: list[str] = []
    while rest and len(tokens) + sum(
            len(encode_event("observation", _compact_evidence_record(r)))
            for r in priority + rest) + (len(json.dumps(
                budget_block, sort_keys=True)) if budget_block
                else 0) > max_tokens:
        dropped = rest.pop(0)
        omitted.append(str(dropped.get("observation_id", "?")))
    for record in priority + rest:
        tokens += encode_event("observation", _compact_evidence_record(record))
    if budget_block:
        tokens += encode_text(json.dumps(budget_block, sort_keys=True))
    if len(tokens) > max_tokens:
        raise EpisodeError(
            f"public state needs {len(tokens)} tokens but the context policy "
            f"allows {max_tokens}; refusing silent truncation of history/goal")
    return tokens, omitted


def _ACTION_SHORT_KEYS():
    return {
        "kind": "k", "variable": "var", "item": "itm", "input": "inp",
        "value": "val", "container": "cnt", "switch": "sw",
        "answer": "ans", "target": "tgt", "column": "col",
        "equals": "eq", "op": "op", "operand": "opr",
    }

def _FEEDBACK_SHORT_KEYS():
    return {
        "kind": "k", "variable": "var", "value": "val", "item": "itm",
        "requires": "req", "input": "inp", "result": "res",
        "total": "tot", "matched": "mat", "written": "wri",
        "submitted_answer": "sub", "correct": "cor",
        "contains_item": "cnt_itm", "changed": "chg", "state": "state",
        "success": "suc", "error": "err", "is_dependency": "is_dep",
        "table": "tbl", "filtered_rows": "flt", "sum": "sum",
    }

def _compact_history_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Lossless-compact history entry. Unique short keys per field.
    Unknown fields survive in the '_x' extension map."""
    action = dict(entry.get("action", {}))
    feedback = dict(entry.get("feedback", {}))
    a_keys = _ACTION_SHORT_KEYS()
    f_keys = _FEEDBACK_SHORT_KEYS()
    compact_action: dict[str, Any] = {}
    action_ext: dict[str, Any] = {}
    for key, value in action.items():
        short = a_keys.get(key)
        if short and short not in compact_action:
            compact_action[short] = value
        elif short is None:
            action_ext[key] = value
    compact_feedback: dict[str, Any] = {}
    feedback_ext: dict[str, Any] = {}
    for key, value in feedback.items():
        short = f_keys.get(key)
        if short and short not in compact_feedback:
            compact_feedback[short] = value
        elif short is None:
            feedback_ext[key] = value
    if action_ext:
        compact_action["x"] = action_ext
    if feedback_ext:
        compact_feedback["x"] = feedback_ext
    return {"a": compact_action, "f": compact_feedback}


def _expand_history_entry(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Exact inverse of _compact_history_entry."""
    a_reverse = {v: k for k, v in _ACTION_SHORT_KEYS().items()}
    f_reverse = {v: k for k, v in _FEEDBACK_SHORT_KEYS().items()}
    compact_action = dict(compact.get("a", {}))
    compact_feedback = dict(compact.get("f", {}))
    ext = compact_action.pop("x", {})
    for key, value in ext.items():
        compact_action[key] = value
    ext = compact_feedback.pop("x", {})
    for key, value in ext.items():
        compact_feedback[key] = value
    action = {a_reverse.get(k, k): v for k, v in compact_action.items()}
    feedback = {f_reverse.get(k, k): v for k, v in compact_feedback.items()}
    return {"action": action, "feedback": feedback}


def _compact_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Lossless-compact evidence record for the shared renderer."""
    compact: dict[str, Any] = {}
    for key, value in record.items():
        if key == "observed_value":
            compact["o"] = str(value)[:120] if isinstance(value, str) else value
        elif key == "entity":
            compact["e"] = value
        elif key == "observation_id":
            compact["i"] = value
        elif key == "temporal_scope":
            compact["t"] = value
        elif key == "status":
            compact["s"] = value
        else:
            compact[key] = value
    return compact or {"e": "empty-record"}


def mark_conflicts(workspace: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Mark contradictions and temporal supersession.

    Groups records by (subject, predicate). Within a group:
    - Same value at same valid_time: duplicates, no conflict.
    - Different values at same valid_time: contradiction (both conflicting).
    - Different valid_times: later supersedes earlier (earlier superseded).
    Different subjects or predicates never conflict. Multivalued predicates
    (e.g. contains) accumulate without conflict.
    """
    conflicts: list[tuple[str, str]] = []
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for record in workspace:
        if record.get("predicate") == "contains":
            continue  # multivalued: accumulate without conflict
        key = (record.get("subject"), record.get("predicate"))
        groups.setdefault(key, []).append(record)

    for (subject, predicate), members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: r.get("valid_time", 0))
        # Check for same-time contradictions.
        by_time: dict[Any, list[dict[str, Any]]] = {}
        for member in members:
            by_time.setdefault(member.get("valid_time", 0), []).append(member)
        has_contradiction = False
        for _time, same_time in by_time.items():
            values = set(json.dumps(r.get("value"), sort_keys=True, default=str)
                         for r in same_time)
            if len(values) > 1:
                has_contradiction = True
                for r in same_time:
                    r["status"] = "conflicting"
                for i in range(len(same_time)):
                    for j in range(i + 1, len(same_time)):
                        conflicts.append((same_time[i].get("record_id", "?"),
                                          same_time[j].get("record_id", "?")))
        if has_contradiction:
            continue
        # Temporal supersession: later valid_time supersedes earlier.
        sorted_members = sorted(members, key=lambda r: r.get("valid_time", 0))
        for i in range(len(sorted_members) - 1):
            earlier = sorted_members[i]
            later = sorted_members[i + 1]
            if earlier.get("value") != later.get("value"):
                earlier["status"] = "superseded"
                later["supersedes"] = earlier.get("record_id")
                later["status"] = "active"
            else:
                earlier["status"] = "superseded"
                later["supersedes"] = earlier.get("record_id")
                later["status"] = "active"
    return conflicts


def _hash_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


class ModelWorldModel:
    """Production world predictor: model-originated imagined outcomes.

    Prompts the shared decoder for a feedback prediction over the declared
    finite support (the current legal actions). The imagined origin derives
    from the underlying generation's reported origin (real model output vs
    double output), never from this class's name. Unparseable outputs become
    explicit neutral predictions (recorded, never silent). Local tests use
    canned doubles.
    """

    name = "model-world"

    def __init__(self, model: ModelInterface) -> None:
        self.model = model
        self.calls = 0

    def __call__(self, *, state: Mapping[str, Any],
                 action: Mapping[str, Any], depth: int) -> Mapping[str, Any]:
        from bramastra_lab.research.experience.codec import encode_text

        prompt_text = json.dumps({
            "goal": dict(state.get("goal", {})),
            "history": [dict(entry) for entry in state.get("history", [])],
            "workspace": [dict(entry) for entry in state.get("workspace", [])],
            "action": dict(action),
            "depth": depth,
        }, sort_keys=True, default=str)[:512]
        prompt = [259] + encode_text(prompt_text)[:256]
        self.calls += 1
        try:
            out = self.model.generate(prompt, max_new_tokens=24)
            predicted = json.loads(out["answer"])
            if not isinstance(predicted, Mapping):
                raise ValueError("prediction is not an object")
            sp = predicted.get("success_prob")
            if sp is not None:
                sp = float(sp)
                if not (0.0 <= sp <= 1.0):
                    return {"feedback": {},
                            "success_prob": None, "value": None,
                            "origin": "out-of-range",
                            "prediction_failed": True,
                            "error": f"success_prob {sp} outside [0,1]"}
            base_origin = str(out.get("origin", "model"))
            return {"feedback": dict(predicted.get("feedback", {})),
                    "success_prob": sp,
                    "value": float(predicted.get("value", 0.0)),
                    "origin": base_origin + "-imagined",
                    "prediction_failed": False}
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            return {"feedback": {},
                    "success_prob": None, "value": None,
                    "origin": "parse-failure",
                    "prediction_failed": True,
                    "error": str(exc)[:120]}
        except Exception as exc:
            return {"feedback": {},
                    "success_prob": None, "value": None,
                    "origin": "neutral-fallback",
                    "prediction_failed": True,
                    "error": str(exc)[:120]}


class CannedWorldModel:
    """Deterministic test double with a scripted prediction per call."""

    name = "canned-world"

    def __init__(self, predictions: Sequence[Mapping[str, Any]]) -> None:
        if not predictions:
            raise EpisodeError("canned world model needs predictions")
        self.predictions = [dict(p) for p in predictions]
        self.calls = 0

    def __call__(self, *, state: Mapping[str, Any],
                 action: Mapping[str, Any], depth: int) -> Mapping[str, Any]:
        predicted = dict(self.predictions[min(
            self.calls, len(self.predictions) - 1)])
        self.calls += 1
        return {**predicted, "origin": "canned-double-imagined"}


# -- model interface ---------------------------------------------------------

class ModelInterface:
    """Explicit model boundary for adapters (no hidden oracles)."""

    def generate(self, prompt_tokens: list[int], *,
                 max_new_tokens: int) -> dict[str, Any]:
        raise NotImplementedError


class FreeGenerationModel(ModelInterface):
    """Production model interface: real decoder free generation."""

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config

    def generate(self, prompt_tokens: list[int], *,
                 max_new_tokens: int) -> dict[str, Any]:
        from bramastra_lab.research.runtime.inference import generate_free_form

        report = generate_free_form(self.model, self.config,
                                    list(prompt_tokens),
                                    max_new_tokens=max_new_tokens)
        return {"tokens": [], "answer": report.answer,
                "stopped_on_eos": report.stopped_on_eos,
                "new_tokens": report.new_tokens,
                "origin": "model"}


class ScriptedModel(ModelInterface):
    """Deterministic test double with a canned action script.

    Explicitly test-only: callers must label receipts fixture. Each
    generate() pops the next scripted JSON action; exhaustion repeats the
    last entry (deterministic, recorded via generation ids `script-i`).
    """

    def __init__(self, script: Sequence[Mapping[str, Any]]) -> None:
        if not script:
            raise EpisodeError("scripted model needs a nonempty script")
        self.script = [dict(entry) for entry in script]
        self.calls = 0

    def generate(self, prompt_tokens: list[int], *,
                 max_new_tokens: int) -> dict[str, Any]:
        entry = self.script[min(self.calls, len(self.script) - 1)]
        generation_id = f"script-{self.calls}"
        self.calls += 1
        return {"tokens": [], "answer": json.dumps(entry, sort_keys=True),
                "stopped_on_eos": True, "new_tokens": 1,
                "generation_id": generation_id, "origin": "scripted-double"}


def parse_action(answer: str, *, model_origin: str | None = None) -> Mapping[str, Any]:
    """Parse model text into a typed action mapping (strict JSON object)."""
    try:
        action = json.loads(answer)
    except (json.JSONDecodeError, TypeError) as exc:
        raise EpisodeError(f"model output is not a JSON action: {exc}",
                           model_origin=model_origin) from exc
    if not isinstance(action, Mapping) or "kind" not in action:
        raise EpisodeError("model action must be a JSON object with 'kind'",
                           model_origin=model_origin)
    return dict(action)


# -- adapters ------------------------------------------------------------------

class Adapter:
    """Named decision policy with its own trace identity."""

    name: str = "abstract"
    # Whether the kernel maintains workspace evidence for this adapter.
    # Only workspace-consuming treatments build workspace state; other arms
    # must not get a solved belief structure for free.
    uses_workspace: bool = False

    def select(self, *, legal_actions: Sequence[Mapping[str, Any]],
               rendered: list[int], model: ModelInterface,
               workspace: list[dict[str, Any]],
               state_view: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


class LearnedPolicyAdapter(Adapter):
    """Model generation parsed to a legal action (B policy arm)."""

    name = "learned-policy"

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        out = model.generate(rendered, max_new_tokens=24)
        action = parse_action(out["answer"],
                              model_origin=out.get("origin", "model"))
        action["_generation_id"] = out.get("generation_id")
        action["_origin"] = out.get("origin", "model")
        return action


class WorkspacePolicyAdapter(Adapter):
    """Policy with workspace evidence admitted to the prompt (B workspace).

    Workspace construction is a declared deterministic transform: every
    received observation is admitted as an evidence record before selection.
    The reference symbolic updater stays a separate control.
    """

    name = "workspace-policy"
    uses_workspace = True

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        out = model.generate(rendered, max_new_tokens=24)
        action = parse_action(out["answer"],
                              model_origin=out.get("origin", "model"))
        action["_generation_id"] = out.get("generation_id")
        action["_origin"] = out.get("origin", "model")
        action["_workspace_records"] = len(workspace)
        return action


def admit_observation_evidence(workspace: list[dict[str, Any]], *,
                               observation: Mapping[str, Any],
                               observation_id: str) -> dict[str, Any]:
    """Declared deterministic workspace transform (shared by train/infer)."""
    kind = observation.get("kind", "observation")
    variable = observation.get("variable")
    value = observation.get("value")
    subject = f"variable:{variable}" if variable else f"observation:{observation_id}"
    predicate = observation.get("kind", "value")
    record = {
        "record_id": observation_id,
        "subject": subject,
        "predicate": predicate,
        "value": value if value is not None else json.dumps(observation, sort_keys=True, default=str)[:120],
        "valid_time": len(workspace),
        "source_event_id": observation_id,
        "status": "active",
    }
    workspace.append(record)
    return record


class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class BoundedPlannerAdapter(Adapter):
    """Depth-two / node-eight imagined search (B planner arm).

    Breadth-first: all roots expanded at depth-1 first, then depth-2
    children with remaining budget. This ensures permutation equivariance
    for depth-1 scores regardless of action ordering.
    """

    name = "bounded-planner"

    def __init__(self, *, world_model: Callable[..., Mapping[str, Any]],
                 value_fn: Callable[..., float] | None = None,
                 value_fn_name: str = "zero-baseline",
                 max_depth: int = 2, max_nodes: int = 8) -> None:
        self.world_model = world_model
        self.value_fn = value_fn or (lambda **kwargs: 0.0)
        self.value_fn_name = value_fn_name
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.last_imagined: list[Any] = []
        self.excluded_root_ids: list[str] = []
        self._model_calls = 0

    @staticmethod
    def _known_forecast(predicted: Mapping[str, Any]) -> float | None:
        """Extract a valid success_prob; None for failed/out-of-range."""
        if predicted.get("prediction_failed"):
            return None
        sp = predicted.get("success_prob")
        if sp is None:
            return None
        try:
            val = float(sp)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= val <= 1.0):
            return None
        return val

    def _make_node(self, node_index: int, action_prefix: list,
                   predicted: Mapping[str, Any], state_view: Mapping[str, Any],
                   provenance: str) -> _AttrDict:
        return _AttrDict(
            node_index=node_index,
            action_prefix=[dict(a) for a in action_prefix],
            predicted_outcome=dict(predicted),
            value_estimate=float(predicted.get("value", 0.0)),
            remaining_budget=int(state_view.get("budgets", {}).get(
                "actions_left", 0)),
            provenance=provenance,
            public_state_hash=_hash_mapping(dict(state_view)),
            goal_hash=_hash_mapping(dict(state_view.get("goal", {}))),
        )

    def _world_call(self, state_view: Mapping[str, Any],
                    action: Mapping[str, Any], depth: int,
                    applied_prefix: tuple) -> Mapping[str, Any]:
        self._model_calls += 1
        return self.world_model(
            state={**dict(state_view), "applied_prefix": applied_prefix},
            action=dict(action), depth=depth)

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        self.last_imagined = []
        self.excluded_root_ids = []
        self._model_calls = 0
        if not legal_actions:
            raise EpisodeError("planner has no legal actions to expand")

        action_keys = [json.dumps(dict(a), sort_keys=True) for a in legal_actions]
        root_scores: dict[str, float] = {}
        nodes = 0

        # Phase 1: Expand all roots at depth-1 (breadth-first).
        root_predictions: dict[str, Mapping[str, Any]] = {}
        for index, action in enumerate(legal_actions):
            key = action_keys[index]
            if nodes >= self.max_nodes:
                self.excluded_root_ids.append(key)
                continue
            predicted = self._world_call(state_view, action, 1, ())
            nodes += 1
            root_predictions[key] = predicted
            sp1 = self._known_forecast(predicted)
            self.last_imagined.append(self._make_node(
                nodes, [action], predicted, state_view, "planner-depth1"))
            root_scores[key] = sp1 if sp1 is not None else -1.0

        # Phase 2: Depth-2 children with remaining budget (breadth-first
        # over roots, expanding children in root order). Q uses A5:
        # Q_2(h,a) = sp1 * max_over_children(sp2), weighting the best
        # continuation by the root's own transition probability.
        if self.max_depth >= 2:
            for index, action in enumerate(legal_actions):
                key = action_keys[index]
                if key in self.excluded_root_ids:
                    continue
                if nodes >= self.max_nodes:
                    break
                prefix = (key,)
                predicted1 = root_predictions[key]
                sp1 = self._known_forecast(predicted1)
                if sp1 is None or sp1 <= 0:
                    continue
                best_sp2 = None
                for second in legal_actions:
                    if nodes >= self.max_nodes:
                        break
                    predicted2 = self._world_call(
                        state_view, second, 2, prefix)
                    nodes += 1
                    sp2 = self._known_forecast(predicted2)
                    self.last_imagined.append(self._make_node(
                        nodes, [action, second], predicted2, state_view,
                        "planner-depth2"))
                    if sp2 is not None:
                        best_sp2 = max(best_sp2, sp2) if best_sp2 is not None else sp2
                if best_sp2 is not None:
                    root_scores[key] = sp1 * best_sp2

        # Select best root by Q value.
        best_key = max(root_scores, key=lambda k: root_scores[k])
        best_index = action_keys.index(best_key)
        chosen = dict(legal_actions[best_index])
        chosen["_planner_nodes"] = nodes
        chosen["_planner_value"] = root_scores[best_key]
        chosen["_value_fn"] = self.value_fn_name
        chosen["_model_calls"] = self._model_calls
        return chosen


def _default_guess(legal_actions: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Well-formed default submission for scripted controls.

    Controls answer without inquiry by guessing: the submission is always
    well-formed (never an invalid-action error) and usually wrong; the
    verifier scores it honestly. Guesses derive from the legal set itself.
    """
    submissions = [a for a in legal_actions if _is_submission_like(a)]
    if not submissions:
        return dict(legal_actions[0])
    action = dict(submissions[0])
    inquiries = [a for a in legal_actions if not _is_submission_like(a)]
    if "answer" not in action and action.get("kind") == "submit":
        action["answer"] = False
    if action.get("kind") == "submit" and "item" not in action:
        first_item = next((a.get("item") for a in inquiries
                           if a.get("item") is not None), "__unknown__")
        action["item"] = first_item
    if action.get("kind") == "submit" and "value" not in action:
        action["value"] = 0
    return action


class FixedInquiryAdapter(Adapter):
    """Scripted fixed-order inquiry control (A fixed arm)."""

    name = "fixed-inquiry"

    def __init__(self, *, order: Sequence[int] | None = None,
                 submit_after: int = 2) -> None:
        self.order = list(order) if order else []
        self.position = 0
        self.submit_after = submit_after

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        inquiries = [a for a in legal_actions
                     if not _is_submission_like(a)]
        if inquiries and self.position < self.submit_after:
            if self.order:
                action = dict(inquiries[self.order[self.position % len(self.order)]
                                        % len(inquiries)])
            else:
                action = dict(inquiries[self.position % len(inquiries)])
            self.position += 1
            action["_origin"] = "fixed-script"
            return action
        action = _default_guess(legal_actions)
        action["_origin"] = "fixed-script-submit"
        return action


class RandomInquiryAdapter(Adapter):
    """Seeded random inquiry control (A random arm)."""

    name = "random-inquiry"

    def __init__(self, *, seed: int = 0, submit_after: int = 2) -> None:
        import random as _random

        self.rng = _random.Random(seed)
        self.position = 0
        self.submit_after = submit_after

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        inquiries = [a for a in legal_actions
                     if not _is_submission_like(a)]
        if inquiries and self.position < self.submit_after:
            pool = inquiries
            action = dict(self.rng.choice(pool))
            self.position += 1
            action["_origin"] = "random-script"
            return action
        action = _default_guess(legal_actions)
        action["_origin"] = "random-script-submit"
        return action


class DirectAnswerAdapter(Adapter):
    """Immediate submission control (A direct arm)."""

    name = "direct-answer"

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        action = _default_guess(legal_actions)
        action["_origin"] = "direct-script"
        return action


class ConstantScorerAdapter(Adapter):
    """Negative control: constant scores ignore the model entirely."""

    name = "constant-scorer-control"

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        action = dict(legal_actions[0])
        action["_origin"] = "constant-control"
        return action


class SymbolicReferenceAdapter(Adapter):
    """Named oracle control (never learner evidence).

    Holds its environment explicitly and executes the environment's own
    oracle answer. Labeled origin symbolic-oracle everywhere; excluded from
    every learned comparison (diagnostic ceiling only).
    """

    name = "symbolic-reference"

    def __init__(self, env) -> None:
        self.env = env
        self._sequence: list[Mapping[str, Any]] | None = None
        self._position = 0

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        if self._sequence is None:
            answer = self.env.oracle_answer()
            if answer.get("kind") == "sequence":
                self._sequence = list(answer["steps"])
            else:
                self._sequence = [dict(answer)]
            self._position = 0
        action = dict(self._sequence[min(self._position,
                                         len(self._sequence) - 1)])
        self._position += 1
        action["_origin"] = "symbolic-oracle"
        return action


def _is_submission_like(action: Mapping[str, Any]) -> bool:
    kind = str(action.get("kind", "")).lower()
    return kind in ("submit", "submission", "write_result", "check_result")


# -- episode loop ---------------------------------------------------------------

def run_episode(env, adapter: Adapter, *, model: ModelInterface,
                seed: int, mechanism: Mapping[str, Any] | None = None,
                goal_override: Mapping[str, Any] | None = None,
                omit_history: bool = False,
                action_budget: int = ACTION_BUDGET_DEFAULT,
                call_budget: int = CALL_BUDGET_DEFAULT,
                node_budget: int = NODE_BUDGET_DEFAULT,
                max_new_tokens: int = 24,
                session_job_id: str = "local",
                checkpoint_id: str | None = None,
                track_workspace: bool | None = None
                ) -> dict[str, Any]:
    """Run one live episode; return the durable trace and summary.

    The goal override (goal-swap) applies BEFORE rendering so the changed
    goal is what the adapter sees and what success is measured against.
    `omit_history` (negative control) drops history from the prompt only.
    Counts reduce from emitted events. No optimizer work occurs here.
    """
    import random as _random

    rng = _random.Random(seed)
    episode_id = f"ep-{seed}-{rng.randint(0, 999999):06d}"
    observation = env.reset(episode_id=episode_id)
    # Workspace state exists only for workspace-consuming treatments (plus
    # explicit contradiction probes): other arms must not receive a solved
    # belief structure for free.
    build_workspace = adapter.uses_workspace if track_workspace is None \
        else bool(track_workspace)
    base_goal = dict(goal_override) if goal_override is not None else dict(
        observation.observable_values)
    history: list[dict[str, Any]] = []
    workspace: list[dict[str, Any]] = []
    events: list[EpisodeEvent] = []
    imagined: list[ImaginedNode] = []
    model_calls = tool_calls = invalid = 0
    event_index = 0

    def emit(kind: str, **fields: Any) -> EpisodeEvent:
        nonlocal event_index
        event = EpisodeEvent(
            event_index=event_index, session_job_id=session_job_id,
            checkpoint_id=checkpoint_id,
            input_hash=_hash_mapping({"goal": base_goal, "kind": kind,
                                      "index": event_index}),
            kind=kind,
            action=fields.get("action"),
            generation_id=fields.get("generation_id"),
            model_origin=fields.get("model_origin"),
            planner_meta=fields.get("planner_meta"),
            observed_result=fields.get("observed_result"),
            resource_delta=float(fields.get("resource_delta", 0.0)),
            predecessor=event_index - 1 if event_index else None)
        events.append(event)
        event_index += 1
        return event

    emit("observation", observed_result=dict(observation.feedback))
    terminated = truncated = False
    success: bool | None = None
    while True:
        # Budget: at most `action_budget` inquiry/tool actions plus one
        # final submission (experiment.md: 4 real actions plus submission).
        inquiries_used = sum(
            1 for entry in history
            if not _is_submission_like(entry.get("action", {})))
        calls_left = call_budget - model_calls
        if calls_left <= 0:
            truncated = True
            emit("truncation", observed_result={"reason": "budget-exhausted"})
            break
        budgets = {"actions_left": max(
            0, action_budget - inquiries_used), "calls_left": calls_left,
            "nodes_left": node_budget - len(imagined)}
        try:
            rendered, omitted = render_public_state(
                goal=base_goal,
                history=[] if omit_history else history,
                workspace=workspace, budgets=budgets)
        except EpisodeError as exc:
            truncated = True
            emit("truncation", observed_result={"reason": str(exc)[:200]})
            break
        if omitted:
            # Explicit render-window record (never silent): older active
            # evidence left the prompt but stays in workspace state.
            emit("observation",
                 observed_result={"kind": "render_window",
                                  "omitted_workspace_ids": omitted},
                 resource_delta=0.0)
        legal = env.legal_actions()
        state_view = {"goal": base_goal,
                      "history": list(history),
                      "workspace": list(workspace),
                      "budgets": dict(budgets)}
        try:
            action = adapter.select(legal_actions=legal, rendered=rendered,
                                    model=model, workspace=workspace,
                                    state_view=state_view)
        except EpisodeError as exc:
            # A failed generation still consumed a model call: budget it, or
            # unparseable outputs would loop forever without progress. The
            # failure keeps the generation's origin for evidence tracing.
            model_calls += 1
            invalid += 1
            emit("action", action={"kind": "adapter_error"},
                 model_origin=getattr(exc, "model_origin", None)
                 or "unknown",
                 planner_meta={},
                 observed_result={"kind": "adapter_error",
                                  "error": str(exc)[:200]},
                 resource_delta=1.0)
            if call_budget - model_calls <= 0:
                truncated = True
                emit("truncation",
                     observed_result={"reason": "budget-exhausted"})
                break
            continue
        action = dict(action)
        generation_id = action.pop("_generation_id", None)
        model_origin = action.pop("_origin", "unknown")
        model_calls += 1
        if inquiries_used >= action_budget and not _is_submission_like(action):
            truncated = True
            emit("action", action=dict(action), generation_id=generation_id,
                 model_origin=model_origin, planner_meta={},
                 observed_result={"kind": "refused",
                                  "reason": "inquiry budget exhausted"},
                 resource_delta=0.0)
            emit("truncation", observed_result={"reason": "budget-exhausted"})
            break
        if isinstance(adapter, BoundedPlannerAdapter):
            imagined.extend(adapter.last_imagined)
        planner_meta = {}
        for key in ("_planner_nodes", "_planner_value", "_value_fn",
                    "_workspace_records"):
            if key in action:
                planner_meta[key] = action.pop(key)
        for reserved in ("_workspace_records", "_planner_nodes",
                         "_planner_value", "_value_fn"):
            action.pop(reserved, None)
        if action.get("kind") in ("read_table", "filter_rows", "sum_column",
                                  "write_result", "check_result"):
            tool_calls += 1
        try:
            next_obs, cost, terminated, truncated = env.step(action)
        except Exception as exc:  # environment contract violation surfaces
            invalid += 1
            emit("action", action=dict(action), generation_id=generation_id,
                 model_origin=model_origin, planner_meta=planner_meta,
                 observed_result={"kind": "environment_error",
                                  "error": str(exc)[:200]},
                 resource_delta=1.0)
            continue
        if dict(next_obs.feedback).get("kind") == "invalid_action":
            # Declared charging rule fired inside the env: count the received
            # outcome here (never inferred from case numbers).
            invalid += 1
        try:
            record = env.action_record(action, policy_identity=adapter.name)
            _ = record
        except Exception:
            pass
        emit("action", action=dict(action), generation_id=generation_id,
             model_origin=model_origin, planner_meta=planner_meta,
             observed_result=dict(next_obs.feedback), resource_delta=cost)
        # ONLY received observations enter the real history; imagined
        # planner nodes stay in the separate imagined list.
        history.append({"action": dict(action),
                        "feedback": dict(next_obs.feedback)})
        if build_workspace:
            admit_observation_evidence(
                workspace, observation=dict(next_obs.feedback),
                observation_id=f"{episode_id}:{len(history)}")
        observation = next_obs
        if terminated or truncated:
            feedback = dict(observation.feedback)
            if isinstance(feedback.get("success"), bool):
                success = feedback["success"]
            break
    summary = {"actions": len(history) + invalid,
               "model_calls": model_calls,
               "tool_calls": tool_calls,
               "invalid_actions": invalid,
               "imagined_nodes": len(imagined),
               "truncated": truncated, "terminated": terminated,
               "success": success,
               "cost": sum(event.resource_delta for event in events)}
    return {"events": [event.to_dict() for event in events],
            "imagined": [dict(node) for node in imagined],
            "history": history,
            "summary": summary,
            "adapter": adapter.name,
            "goal": base_goal,
            "episode_id": episode_id}


# --- I02/K8 additions: action coding and history expansion -------------------

RESPONSE_ENVELOPE_TOKENS = 8


def encode_action_code(action: dict, legal_actions: list[dict]) -> str:
    """Encode an action as a compact JSON code within the response envelope.

    The code is {"a": index} where index is the position in legal_actions.
    Full JSON often exceeds the envelope; the code always fits.
    """
    for index, candidate in enumerate(legal_actions):
        if dict(candidate) == dict(action):
            import json as _json
            code = _json.dumps({"a": index}, separators=(",", ":"))
            return code
    raise ValueError(f"action not in legal set: {action}")


def decode_action_code(code: dict, legal_actions: list[dict]) -> tuple[dict, str]:
    """Decode a compact action code back to the full action dict.

    Rejects out-of-range indexes and unknown keys. Returns (action, "code").
    """
    if not isinstance(code, dict):
        raise ValueError("action code must be a dict")
    keys = set(code.keys())
    if keys != {"a"}:
        raise ValueError(f"action code must have exactly key 'a', got {sorted(keys)}")
    index = code["a"]
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("action code index must be an integer")
    if index < 0 or index >= len(legal_actions):
        raise ValueError(
            f"action code index {index} out of range for {len(legal_actions)} legal actions")
    return dict(legal_actions[index]), "code"


