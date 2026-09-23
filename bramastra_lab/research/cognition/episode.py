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
import math
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
    value_estimate: float | None
    remaining_budget: int
    provenance: str


def render_public_state(*, goal: Mapping[str, Any],
                        history: Sequence[Mapping[str, Any]],
                        workspace: Sequence[Mapping[str, Any]] | None = None,
                        memory_context: Any | None = None,
                        budgets: Mapping[str, Any] | None = None,
                        max_tokens: int = RENDER_MAX_TOKENS_DEFAULT
                        ) -> tuple[list[int], list[str]]:
    """Render versioned public state; omit only whole optional evidence rows.

    The mandatory prefix is shared with the trajectory compiler. Goal,
    received history and remaining budgets are never sliced or silently
    coerced. Conflicting evidence is mandatory; other evidence may be omitted
    whole and is returned by ID for the episode trace.
    """
    from bramastra_lab.research.experience.codec import (
        SPECIAL_BOUNDARY, EncodingError, encode_event)
    from bramastra_lab.research.experience.public_state import (
        PublicStateError, compact_memory_content,
        public_state_prefix_events)

    if (not isinstance(max_tokens, int) or isinstance(max_tokens, bool)
            or max_tokens < 1):
        raise EpisodeError("max_tokens must be a positive integer")
    tokens = [SPECIAL_BOUNDARY]
    try:
        for role, content in public_state_prefix_events(
                goal, history, budgets=budgets):
            tokens += encode_event(role, content)
    except (EncodingError, PublicStateError) as exc:
        raise EpisodeError(
            f"required public state cannot be represented losslessly: {exc}") from exc

    records = list(workspace or [])
    if memory_context is not None:
        # Retrieval provenance stays in the trace; the prompt receives only
        # stable episode-local aliases and eligible public training content.
        # Retrieval returns best-first. Optional rows are removed from the
        # front under pressure, so append memory worst-first to retain the
        # strongest retrieved exemplar.
        for memory_record in reversed(memory_context.records):
            alias = _memory_prompt_alias(memory_context.index_identity,
                                         memory_record)
            records.append({
                "observation_id": f"memory-{alias}",
                "predicate": "retrieved_training_example",
                "value": memory_record.content,
                "status": "memory"})
    if any(not isinstance(record, Mapping) for record in records):
        raise EpisodeError("workspace evidence must be a sequence of JSON objects")
    priority = [record for record in records
                if record.get("status") == "conflicting"]
    optional = [record for record in records
                if record.get("status") != "conflicting"]
    omitted: list[str] = []

    def encode_evidence(record: Mapping[str, Any]) -> list[int]:
        try:
            if record.get("status") == "memory":
                return encode_event(
                    "observation", compact_memory_content(record["value"]))
            return encode_event("observation", _compact_evidence_record(record))
        except (EncodingError, EpisodeError, PublicStateError) as exc:
            raise EpisodeError(
                f"evidence {record.get('observation_id', '?')!r} is not "
                f"representable within the public event limit: {exc}") from exc

    priority_tokens = [encode_evidence(record) for record in priority]
    optional_tokens: list[tuple[Mapping[str, Any], list[int]]] = []
    for record in optional:
        try:
            optional_tokens.append((record, encode_evidence(record)))
        except EpisodeError:
            omitted.append(str(record.get("observation_id", "?")))
    while optional_tokens and len(tokens) + sum(map(len, priority_tokens)) \
            + sum(len(encoded) for _, encoded in optional_tokens) > max_tokens:
        dropped, _encoded = optional_tokens.pop(0)
        omitted.append(str(dropped.get("observation_id", "?")))

    for encoded in priority_tokens:
        tokens += encoded
    for _record, encoded in optional_tokens:
        tokens += encoded
    if len(tokens) > max_tokens:
        raise EpisodeError(
            f"public state needs {len(tokens)} tokens but the context policy "
            f"allows {max_tokens}; refusing to truncate required context")
    return tokens, omitted


def _compact_history_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Version and preserve a complete received event without key aliasing."""
    from bramastra_lab.research.experience.public_state import (
        PublicStateError, compact_history_entry)

    try:
        return compact_history_entry(entry)
    except PublicStateError as exc:
        raise EpisodeError(str(exc)) from exc


def _expand_history_entry(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Invert a v3 history envelope; ambiguous unversioned rows reject."""
    from bramastra_lab.research.experience.public_state import (
        PublicStateError, expand_history_entry)

    try:
        return expand_history_entry(compact)
    except PublicStateError as exc:
        raise EpisodeError(str(exc)) from exc


def _compact_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Version and preserve a complete evidence record without key aliases."""
    from bramastra_lab.research.experience.public_state import (
        PublicStateError, compact_evidence_record)

    try:
        return compact_evidence_record(record)
    except PublicStateError as exc:
        raise EpisodeError(str(exc)) from exc


def _expand_evidence_record(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Invert a v2 workspace evidence envelope."""
    from bramastra_lab.research.experience.public_state import (
        PublicStateError, expand_evidence_record)

    try:
        return expand_evidence_record(compact)
    except PublicStateError as exc:
        raise EpisodeError(str(exc)) from exc


def _hash_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def _memory_prompt_alias(index_identity: str, record: Any) -> str:
    """Stable episode-local label without exposing a source identity."""
    return hashlib.sha256(
        f"{index_identity}:{record.content_identity()}".encode()
    ).hexdigest()[:12]


def mark_conflicts(workspace: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Mark contradictions and temporal supersession.

    Groups records by (subject, predicate). Within a group:
    - Same value at same valid_time: duplicates, no conflict.
    - Different values at same valid_time: contradiction (both conflicting).
    - Different valid_times: later supersedes earlier (earlier superseded).
    - Missing/unknown valid_time: no temporal relation is inferred.
    Different subjects or predicates never conflict. Multivalued predicates
    (e.g. contains) accumulate without conflict.
    """
    conflicts: list[tuple[str, str]] = []

    def time_order(value: Any) -> tuple[int, Any]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, float(value))
        return (1, str(value))

    # Recompute derived labels from the append-only evidence ledger on every
    # call. This avoids stale "active" or "superseded" marks when a later
    # observation changes how a time group is resolved.
    for record in workspace:
        if record.get("status", "active") in {
                "active", "conflicting", "superseded"}:
            record["status"] = "active"
            record.pop("supersedes", None)

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for record in workspace:
        if record.get("status") == "retracted":
            continue
        if record.get("predicate") == "contains":
            continue  # multivalued: accumulate without conflict
        key = (record.get("subject"), record.get("predicate"))
        groups.setdefault(key, []).append(record)

    for (subject, predicate), members in groups.items():
        if len(members) < 2:
            continue
        by_time: dict[Any, list[dict[str, Any]]] = {}
        for member in members:
            valid_time = member.get("valid_time")
            if valid_time is None:
                # Unknown time cannot establish simultaneity or supersession.
                continue
            if (isinstance(valid_time, bool)
                    or not isinstance(valid_time, (int, float, str))
                    or (isinstance(valid_time, float)
                        and not math.isfinite(valid_time))
                    or (isinstance(valid_time, str) and not valid_time)):
                raise EpisodeError(
                    f"evidence has invalid valid_time {valid_time!r}")
            by_time.setdefault(valid_time, []).append(member)
        ordered_times = sorted(by_time, key=time_order)
        for time_index, valid_time in enumerate(ordered_times):
            same_time = by_time[valid_time]
            if time_index > 0:
                prior_time_records = by_time[ordered_times[time_index - 1]]
                superseded_id = prior_time_records[0].get("record_id", "?")
                for record in same_time:
                    record["supersedes"] = superseded_id
            from bramastra_lab.research.experience.codec import (
                EncodingError, canonical_event_bytes)

            def value_identity(record: Mapping[str, Any]) -> bytes:
                try:
                    return canonical_event_bytes({
                        "present": record.get("value_present", True),
                        "value": record.get("value")})
                except EncodingError as exc:
                    raise EpisodeError(
                        f"evidence value is not canonical JSON: {exc}") from exc

            identities = [value_identity(record) for record in same_time]
            if len(set(identities)) > 1:
                for r in same_time:
                    r["status"] = "conflicting"
                for i in range(len(same_time)):
                    for j in range(i + 1, len(same_time)):
                        if identities[i] != identities[j]:
                            conflicts.append((same_time[i].get("record_id", "?"),
                                              same_time[j].get("record_id", "?")))
            elif time_index == len(ordered_times) - 1:
                # Exact duplicates at the current time remain separately
                # auditable and jointly active; they do not corroborate a
                # proposition more than one source root.
                for record in same_time:
                    record["status"] = "active"
            else:
                for record in same_time:
                    record["status"] = "superseded"
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
        from bramastra_lab.research.experience.codec import encode_event

        # Complete canonical representation: the SAME shared renderer used
        # for training and for policy inference (goal + history + workspace
        # + the applied action event). No character slicing, no token
        # slicing; overflow fails explicitly instead of discarding the
        # decision-relevant state.
        try:
            state_tokens, _omitted = render_public_state(
                goal=dict(state.get("goal", {})),
                history=[dict(entry) for entry in state.get("history", [])],
                workspace=[dict(entry) for entry in state.get("workspace", [])],
                budgets=state.get("budgets"))
        except EpisodeError as exc:
            return {"feedback": {},
                    "success_prob": None, "value": None,
                    "origin": "representation-overflow",
                    "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "prediction_failed": True,
                    "error": f"world-model prompt refused: {exc}"}
        try:
            action_tokens = encode_event("action", dict(action))
        except EncodingError as exc:
            return {"feedback": {}, "success_prob": None, "value": None,
                    "origin": "representation-overflow",
                    "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "prediction_failed": True,
                    "error": f"world-model action refused: {exc}"}
        prompt = state_tokens + action_tokens
        if len(prompt) + 1 > RENDER_MAX_TOKENS_DEFAULT:
            return {"feedback": {},
                    "success_prob": None, "value": None,
                    "origin": "representation-overflow",
                    "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "prediction_failed": True,
                    "error": (f"world-model prompt needs {len(prompt)} "
                              f"tokens; policy allows "
                              f"{RENDER_MAX_TOKENS_DEFAULT}; refusing to "
                              "truncate the imagined-transition input")}
        self.calls += 1
        out: Mapping[str, Any] = {}
        base_origin = "unknown"
        output_tokens = 0
        try:
            raw_output = self.model.generate(prompt, max_new_tokens=24)
            if not isinstance(raw_output, Mapping):
                raise ValueError("model prediction response is not an object")
            out = raw_output
            base_origin = str(out.get("origin", "model"))
            raw_token_count = out.get("new_tokens", 0)
            if (isinstance(raw_token_count, int)
                    and not isinstance(raw_token_count, bool)
                    and raw_token_count >= 0):
                output_tokens = raw_token_count
            predicted = json.loads(out["answer"])
            if not isinstance(predicted, Mapping):
                raise ValueError("prediction is not an object")
            feedback = predicted.get("feedback", {})
            if not isinstance(feedback, Mapping):
                raise ValueError("predicted feedback is not an object")
            sp = predicted.get("success_prob")
            if sp is not None:
                if isinstance(sp, bool):
                    raise ValueError("success_prob must be numeric, not boolean")
                sp = float(sp)
                if not (0.0 <= sp <= 1.0):
                    return {"feedback": {},
                            "success_prob": None, "value": None,
                            "origin": base_origin + "-out-of-range",
                            "generation_origin": base_origin,
                            "model_calls": 1, "input_tokens": len(prompt),
                            "output_tokens": output_tokens,
                            "prediction_failed": True,
                            "error": f"success_prob {sp} outside [0,1]"}
            raw_value = predicted.get("value", 0.0)
            if isinstance(raw_value, bool):
                raise ValueError("predicted value must be numeric, not boolean")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("predicted value must be finite")
            return {"feedback": dict(feedback),
                    "success_prob": sp,
                    "value": value,
                    "origin": base_origin + "-imagined",
                    "generation_origin": base_origin,
                    "model_calls": 1, "input_tokens": len(prompt),
                    "output_tokens": output_tokens,
                    "prediction_failed": False}
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            return {"feedback": {},
                    "success_prob": None, "value": None,
                    "origin": base_origin + "-parse-failure",
                    "generation_origin": base_origin,
                    "model_calls": 1, "input_tokens": len(prompt),
                    "output_tokens": output_tokens,
                    "prediction_failed": True,
                    "error": str(exc)[:120]}
        except Exception as exc:
            return {"feedback": {},
                    "success_prob": None, "value": None,
                    "origin": base_origin + "-neutral-fallback",
                    "generation_origin": base_origin,
                    "model_calls": 1, "input_tokens": len(prompt),
                    "output_tokens": output_tokens,
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
        return {**predicted, "origin": "canned-double-imagined",
                "model_calls": 0, "input_tokens": 0, "output_tokens": 0}


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
    if not isinstance(observation, Mapping):
        raise EpisodeError("observation must be a JSON object")
    if not isinstance(observation_id, str) or not observation_id:
        raise EpisodeError("observation_id must be a nonempty string")
    kind = observation.get("kind", "observation")
    if not isinstance(kind, str) or not kind:
        raise EpisodeError("observation kind must be a nonempty string")
    variable = observation.get("variable")
    value_present = "value" in observation
    value = observation.get("value")
    if variable is not None:
        try:
            variable_key = json.dumps(
                variable, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EpisodeError("observation variable must be a JSON value") from exc
        subject = f"variable:{variable_key}"
    else:
        subject = f"observation:{observation_id}"
    predicate = kind
    explicit_time = "valid_time" in observation
    # The ordered episode-step fallback is declared, deterministic logical
    # time; an explicitly supplied None remains genuinely unknown.
    valid_time = observation.get("valid_time", len(workspace))
    if (valid_time is not None and (isinstance(valid_time, bool)
            or not isinstance(valid_time, (int, float, str))
            or (isinstance(valid_time, float)
                and not math.isfinite(valid_time))
            or (isinstance(valid_time, str) and not valid_time))):
        raise EpisodeError("observation valid_time must be a finite scalar or None")
    record = {
        "record_id": observation_id,
        "subject": subject,
        "predicate": predicate,
        "value_present": value_present,
        "value": value,
        "valid_time": valid_time,
        "time_basis": ("environment" if explicit_time and valid_time is not None
                       else "unknown" if explicit_time else "episode_step"),
        "source_event_id": observation_id,
        "status": "active",
    }
    workspace.append(record)
    return record


def admit_typed_observation(cognitive_workspace: Any,
                            workspace: list[dict[str, Any]], *,
                            observation: Mapping[str, Any],
                            observation_id: str) -> dict[str, Any]:
    """Admit one received event to the typed ledger and its prompt projection.

    The event identifier is retained only as typed ancestry. Model-visible
    records use workspace-local aliases, so episode/task provenance does not
    leak through the renderer.
    """
    record = admit_observation_evidence(
        workspace, observation=observation, observation_id=observation_id)
    source_event_id = str(record.pop("source_event_id"))
    typed = cognitive_workspace.admit_evidence(
        {"subject": record["subject"], "predicate": record["predicate"],
         "value_present": record["value_present"],
         "value": record["value"], "valid_time": record["valid_time"],
         "time_basis": record["time_basis"]},
        ancestry=(source_event_id,))
    record["record_id"] = typed.alias
    record["observation_id"] = typed.alias
    return record


def sync_typed_conflict_states(cognitive_workspace: Any,
                               workspace: Sequence[Mapping[str, Any]]) -> None:
    """Copy the renderer's deterministic conflict/supersession labels."""
    for record in workspace:
        alias = str(record["record_id"])
        state = str(record.get("status", "active"))
        if state not in {"active", "conflicting", "superseded"}:
            raise EpisodeError(f"unexpected evidence render state {state!r}")
        cognitive_workspace.set_evidence_conflict_state(alias, state)


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
        if (not isinstance(max_depth, int) or isinstance(max_depth, bool)
                or max_depth not in (1, 2)):
            raise EpisodeError("max_depth must be 1 or 2")
        if (not isinstance(max_nodes, int) or isinstance(max_nodes, bool)
                or max_nodes < 1):
            raise EpisodeError("max_nodes must be a positive integer")
        self.last_imagined: list[Any] = []
        self.excluded_root_ids: list[str] = []
        self._model_calls = 0
        self._inference_tokens = 0
        self._root_cursor = 0

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
        if isinstance(sp, bool):
            return None
        if not (0.0 <= val <= 1.0):
            return None
        return val

    @staticmethod
    def _known_value(predicted: Mapping[str, Any]) -> float | None:
        """Extract a finite value estimate; None for failed/unknown."""
        if predicted.get("prediction_failed"):
            return None
        value = predicted.get("value")
        if value is None or isinstance(value, bool):
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if val != val or val in (float("inf"), float("-inf")):
            return None
        return val

    def _make_node(self, node_index: int, action_prefix: list,
                   predicted: Mapping[str, Any], state_view: Mapping[str, Any],
                   provenance: str) -> _AttrDict:
        return _AttrDict(
            node_index=node_index,
            action_prefix=[dict(a) for a in action_prefix],
            predicted_outcome=dict(predicted),
            value_estimate=self._known_value(predicted),
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
        try:
            predicted = self.world_model(
                state={**dict(state_view), "applied_prefix": applied_prefix},
                action=dict(action), depth=depth)
        except Exception as exc:
            return {"feedback": {}, "success_prob": None, "value": None,
                    "origin": "world-model-error", "prediction_failed": True,
                    "error": f"{type(exc).__name__}: {exc}"[:160]}
        if isinstance(predicted, Mapping):
            token_count = predicted.get("input_tokens", 0)
            if (isinstance(token_count, int) and not isinstance(token_count, bool)
                    and token_count >= 0):
                self._inference_tokens += token_count
            return predicted
        return {"feedback": {}, "success_prob": None, "value": None,
                "origin": "invalid-world-model-output",
                "prediction_failed": True}

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        self.last_imagined = []
        self.excluded_root_ids = []
        self._model_calls = 0
        self._inference_tokens = 0
        if not legal_actions:
            raise EpisodeError("planner has no legal actions to expand")

        action_keys = [json.dumps(dict(a), sort_keys=True) for a in legal_actions]
        if len(set(action_keys)) != len(action_keys):
            raise EpisodeError("planner requires distinct canonical legal actions")
        budgets = state_view.get("budgets", {})
        if not isinstance(budgets, Mapping):
            raise EpisodeError("planner budgets must be a mapping")
        nodes_left = budgets.get("nodes_left", self.max_nodes)
        calls_left = budgets.get("calls_left", self.max_nodes)
        for label, value in (("nodes_left", nodes_left),
                             ("calls_left", calls_left)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise EpisodeError(f"planner {label} must be a nonnegative integer")
        node_limit = min(self.max_nodes, nodes_left, calls_left)
        if node_limit == 0:
            chosen = dict(_default_guess(legal_actions))
            chosen.update({"_origin": "fallback",
                           "_planner_nodes": 0,
                           "_planner_value": None,
                           "_value_fn": self.value_fn_name,
                           "_model_calls": 0,
                           "_inference_tokens": 0,
                           "_planner_fallback_reason": "planning_budget_exhausted"})
            return chosen

        # When a resumed episode has only a fraction of its node budget left,
        # rotate the root subset across decisions instead of always excluding
        # the same tail of the legal-action list.
        start = self._root_cursor % len(legal_actions)
        root_order = [(start + offset) % len(legal_actions)
                      for offset in range(len(legal_actions))]
        if node_limit < len(legal_actions):
            self._root_cursor = (start + node_limit) % len(legal_actions)
        else:
            self._root_cursor = 0
        root_scores: dict[str, float] = {}
        nodes = 0

        # Phase 1: Expand all roots at depth-1 (breadth-first).
        root_predictions: dict[str, Mapping[str, Any]] = {}
        for index in root_order:
            action = legal_actions[index]
            key = action_keys[index]
            if nodes >= node_limit:
                self.excluded_root_ids.append(key)
                continue
            predicted = self._world_call(state_view, action, 1, ())
            nodes += 1
            root_predictions[key] = predicted
            sp1 = self._known_forecast(predicted)
            self.last_imagined.append(self._make_node(
                nodes, [action], predicted, state_view, "planner-depth1"))
            root_scores[key] = sp1 if sp1 is not None else -1.0

        valid_roots = [key for key in root_scores
                       if self._known_forecast(root_predictions[key]) is not None]
        if not valid_roots:
            chosen = dict(_default_guess(legal_actions))
            chosen.update({"_origin": "fallback",
                           "_planner_nodes": nodes,
                           "_planner_value": None,
                           "_value_fn": self.value_fn_name,
                           "_model_calls": self._model_calls,
                           "_inference_tokens": self._inference_tokens,
                           "_planner_fallback_reason": "all_root_predictions_failed"})
            return chosen

        # Phase 2: Depth-2 children with remaining budget (breadth-first
        # over roots, expanding children in root order). Q uses A5:
        # Q_2(h,a) = sp1 * max_over_children(sp2), weighting the best
        # continuation by the root's own transition probability.
        if self.max_depth >= 2:
            eligible_roots: dict[str, tuple[Mapping[str, Any], Mapping[str, Any], float]] = {}
            for index in root_order:
                key = action_keys[index]
                if key not in valid_roots:
                    continue
                predicted1 = root_predictions[key]
                feedback1 = predicted1.get("feedback")
                sp1 = self._known_forecast(predicted1)
                if sp1 is None or sp1 <= 0 or not isinstance(feedback1, Mapping):
                    continue
                # The next prediction must be conditioned on the hypothetical
                # action and its predicted observation. This state is local to
                # search; it is never appended to the real episode history.
                forecast = {"kind": "imagined_transition",
                            "predicted_feedback": dict(feedback1),
                            "success_prob": sp1,
                            "value": predicted1.get("value"),
                            "origin": predicted1.get("origin")}
                successor = {**dict(state_view),
                             "history": [*list(state_view.get("history", [])),
                                         {"action": dict(legal_actions[index]),
                                          "feedback": forecast,
                                          "imagined": True}]}
                eligible_roots[key] = (legal_actions[index], successor, sp1)
            best_second: dict[str, float] = {}
            # Round-robin children across roots so one early action cannot
            # consume every remaining node before its peers receive a rollout.
            for second in legal_actions:
                for key, (first, successor, sp1) in eligible_roots.items():
                    if nodes >= node_limit:
                        break
                    predicted2 = self._world_call(
                        successor, second, 2, (key,))
                    nodes += 1
                    sp2 = self._known_forecast(predicted2)
                    self.last_imagined.append(self._make_node(
                        nodes, [first, second], predicted2, successor,
                        "planner-depth2"))
                    if sp2 is not None:
                        best_second[key] = max(best_second.get(key, -1.0), sp2)
                if nodes >= node_limit:
                    break
            for key, best_sp2 in best_second.items():
                root_scores[key] = eligible_roots[key][2] * best_sp2

        # Select best root by Q value.
        best_key = max(root_scores, key=lambda k: root_scores[k])
        best_index = action_keys.index(best_key)
        chosen = dict(legal_actions[best_index])
        chosen_prediction = root_predictions[best_key]
        chosen["_planner_nodes"] = nodes
        chosen["_planner_value"] = root_scores[best_key]
        chosen["_value_fn"] = self.value_fn_name
        chosen["_model_calls"] = self._model_calls
        chosen["_inference_tokens"] = self._inference_tokens
        chosen["_origin"] = str(chosen_prediction.get("origin", "unknown"))
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
                track_workspace: bool | None = None,
                memory_index: Any | None = None,
                memory_top_k: int = 2,
                memory_token_budget: int = 192
                ) -> dict[str, Any]:
    """Run one live episode; return the durable trace and summary.

    The goal override (goal-swap) applies BEFORE rendering so the changed
    goal is what the adapter sees and what success is measured against.
    `omit_history` (negative control) drops history from the prompt only.
    Counts reduce from emitted events. No optimizer work occurs here.
    """
    import random as _random

    rng = _random.Random(seed)
    if memory_index is not None:
        from bramastra_lab.research.memory.store import MemoryIndex

        if not isinstance(memory_index, MemoryIndex):
            raise TypeError("memory_index must be a frozen MemoryIndex")
        if not isinstance(memory_top_k, int) or memory_top_k <= 0:
            raise ValueError("memory_top_k must be a positive integer")
        if not isinstance(memory_token_budget, int) or memory_token_budget < 0:
            raise ValueError("memory_token_budget must be a nonnegative integer")
    episode_id = f"ep-{seed}-{rng.randint(0, 999999):06d}"
    observation = env.reset(episode_id=episode_id)
    # Workspace state exists only for workspace-consuming treatments (plus
    # explicit contradiction probes): other arms must not receive a solved
    # belief structure for free.
    build_workspace = adapter.uses_workspace if track_workspace is None \
        else bool(track_workspace)
    base_goal = dict(goal_override) if goal_override is not None else dict(
        observation.observable_values)
    cognitive_workspace = None
    if build_workspace:
        from bramastra_lab.research.cognition.workspace import CognitiveWorkspace

        cognitive_workspace = CognitiveWorkspace(
            goal=base_goal, success_predicate="environment.verified_success",
            budget=action_budget)
    history: list[dict[str, Any]] = []
    workspace: list[dict[str, Any]] = []
    events: list[EpisodeEvent] = []
    imagined: list[ImaginedNode] = []
    model_calls = tool_calls = invalid = 0
    inference_input_tokens = memory_retrievals = memory_records_read = 0
    memory_token_cost = 0
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
        memory_context = None
        memory_budget_limited = False
        memory_query = None
        if memory_index is not None:
            query = json.dumps({
                "goal": base_goal,
                "history": [] if omit_history else history,
                "workspace": workspace}, sort_keys=True, default=str)
            memory_query = query
            excluded_episodes = set()
            if mechanism is not None and mechanism.get("mechanism_id"):
                excluded_episodes.add(str(mechanism["mechanism_id"]))
            # Retrieval budgets exact model-visible memory events, including
            # their event role and version envelope. Preserve the same 24-token
            # action-generation headroom used by the adapters; the renderer is
            # still the final authority when workspace evidence also competes.
            base_rendered, _base_omitted = render_public_state(
                goal=base_goal,
                history=[] if omit_history else history,
                workspace=workspace, budgets=budgets)
            available = max(0, RENDER_MAX_TOKENS_DEFAULT
                            - len(base_rendered) - 24)
            retrieval_budget = min(memory_token_budget, available)
            memory_context = memory_index.retrieve(
                query, scope_allowlist={"training", "controller"},
                top_k=memory_top_k, exclude_episodes=excluded_episodes,
                token_budget=retrieval_budget)
        try:
            rendered, omitted = render_public_state(
                goal=base_goal,
                history=[] if omit_history else history,
                workspace=workspace, memory_context=memory_context,
                budgets=budgets)
        except EpisodeError:
            if memory_index is None:
                truncated = True
                emit("truncation", observed_result={"reason":
                     "public context overflow"})
                break
            # Retrieval is optional computation. If a complete memory item
            # cannot fit alongside the current state, omit it as a whole and
            # report the budget decision; never slice its content.
            memory_context = memory_index.retrieve(
                memory_query or "", scope_allowlist={"training", "controller"},
                top_k=memory_top_k, token_budget=0)
            memory_budget_limited = True
            try:
                rendered, omitted = render_public_state(
                    goal=base_goal,
                    history=[] if omit_history else history,
                    workspace=workspace, memory_context=memory_context,
                    budgets=budgets)
            except EpisodeError as exc:
                truncated = True
                emit("truncation", observed_result={"reason": str(exc)[:200]})
                break
        if memory_index is not None:
            memory_retrievals += 1
            omitted_ids = set(omitted)
            visible_records = [
                record for record in memory_context.records
                if f"memory-{_memory_prompt_alias(memory_context.index_identity, record)}"
                not in omitted_ids]
            from bramastra_lab.research.experience.codec import encode_event
            from bramastra_lab.research.experience.public_state import (
                compact_memory_content)

            visible_memory_tokens = sum(
                len(encode_event("observation",
                                 compact_memory_content(record.content)))
                for record in visible_records)
            memory_records_read += len(visible_records)
            memory_token_cost += visible_memory_tokens
            memory_budget_limited = (memory_budget_limited
                                     or bool(memory_context.omitted_record_ids)
                                     or len(visible_records)
                                     < len(memory_context.records))
            emit("memory_retrieval", planner_meta={
                "origin": "fixed_scope_lexical",
                "index_identity": memory_context.index_identity,
                "retrieval_rule_identity": memory_context.retrieval_rule_identity,
                "record_count": len(visible_records),
                "candidate_count": (len(visible_records)
                                    + len(memory_context.omitted_record_ids)
                                    + len([item for item in omitted_ids
                                           if item.startswith("memory-")])),
                "record_aliases": [
                    _memory_prompt_alias(memory_context.index_identity, record)
                    for record in visible_records],
                "token_cost": visible_memory_tokens,
                "budget_limited": memory_budget_limited})
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
                      "memory_index_identity": (
                          memory_context.index_identity
                          if memory_context is not None else None),
                      "budgets": dict(budgets)}
        if cognitive_workspace is not None:
            state_view["cognitive_workspace"] = cognitive_workspace.rendered_view(
                budget=64 * 1024)
        try:
            action = adapter.select(legal_actions=legal, rendered=rendered,
                                    model=model, workspace=workspace,
                                    state_view=state_view)
        except EpisodeError as exc:
            # A failed generation still consumed a model call: budget it, or
            # unparseable outputs would loop forever without progress. The
            # failure keeps the generation's origin for evidence tracing.
            consumed_calls = (max(1, adapter._model_calls)
                              if isinstance(adapter, BoundedPlannerAdapter) else 1)
            model_calls += consumed_calls
            consumed_tokens = (adapter._inference_tokens
                               if isinstance(adapter, BoundedPlannerAdapter)
                               else len(rendered))
            inference_input_tokens += consumed_tokens
            invalid += 1
            emit("action", action={"kind": "adapter_error"},
                 model_origin=getattr(exc, "model_origin", None)
                 or "unknown",
                 planner_meta={"model_calls": consumed_calls,
                               "inference_input_tokens": consumed_tokens},
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
        requested_calls = action.pop("_model_calls", None)
        requested_tokens = action.pop("_inference_tokens", None)
        if requested_calls is None:
            model_calls += 1
            inference_input_tokens += len(rendered)
        else:
            if (not isinstance(requested_calls, int)
                    or isinstance(requested_calls, bool) or requested_calls < 0):
                raise EpisodeError("adapter model-call count must be nonnegative")
            if (not isinstance(requested_tokens, int)
                    or isinstance(requested_tokens, bool) or requested_tokens < 0):
                raise EpisodeError("adapter inference-token count must be nonnegative")
            model_calls += requested_calls
            inference_input_tokens += requested_tokens
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
                    "_workspace_records", "_planner_fallback_reason"):
            if key in action:
                planner_meta[key] = action.pop(key)
        if requested_calls is not None:
            planner_meta["model_calls"] = requested_calls
            planner_meta["inference_input_tokens"] = requested_tokens
        for reserved in ("_workspace_records", "_planner_nodes",
                         "_planner_value", "_value_fn",
                         "_planner_fallback_reason"):
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
        if cognitive_workspace is not None:
            admit_typed_observation(
                cognitive_workspace, workspace,
                observation=dict(next_obs.feedback),
                observation_id=f"{episode_id}:{len(history)}")
            mark_conflicts(workspace)
            sync_typed_conflict_states(cognitive_workspace, workspace)
            cognitive_workspace.step_counter += 1
        observation = next_obs
        if terminated or truncated:
            feedback = dict(observation.feedback)
            if isinstance(feedback.get("success"), bool):
                success = feedback["success"]
            break
    cognitive_state = cognitive_workspace.to_dict() \
        if cognitive_workspace is not None else None
    cognitive_identity = None
    if cognitive_state is not None:
        from bramastra_lab.research.contracts.core import content_identity

        cognitive_identity = content_identity(cognitive_state)
    summary = {"actions": len(history) + invalid,
               "model_calls": model_calls,
               "inference_input_tokens": inference_input_tokens,
               "memory": {
                   "enabled": memory_index is not None,
                   "origin": "fixed_scope_lexical"
                   if memory_index is not None else None,
                   "index_identity": memory_index.identity
                   if memory_index is not None else None,
                   "retrieval_rule_identity": memory_index.retrieval_rule_identity
                   if memory_index is not None else None,
                   "retrievals": memory_retrievals,
                   "records_read": memory_records_read,
                   "token_cost": memory_token_cost},
               "tool_calls": tool_calls,
               "invalid_actions": invalid,
               "imagined_nodes": len(imagined),
               "cognition": {
                   "enabled": cognitive_workspace is not None,
                   "state_identity": cognitive_identity,
                   "evidence_count": len(cognitive_workspace.evidence)
                   if cognitive_workspace is not None else 0,
                   "conflicting_evidence": sum(
                       record.conflict_state == "conflicting"
                       for record in cognitive_workspace.evidence.values())
                   if cognitive_workspace is not None else 0,
                   "superseded_evidence": sum(
                       record.conflict_state == "superseded"
                       for record in cognitive_workspace.evidence.values())
                   if cognitive_workspace is not None else 0,
                   "step_counter": cognitive_workspace.step_counter
                   if cognitive_workspace is not None else 0},
               "truncated": truncated, "terminated": terminated,
               "success": success,
               "cost": sum(event.resource_delta for event in events)}
    return {"events": [event.to_dict() for event in events],
            "imagined": [dict(node) for node in imagined],
            "history": history,
            "summary": summary,
            "adapter": adapter.name,
            "goal": base_goal,
            "episode_id": episode_id,
            "cognitive_state": cognitive_state,
            "cognitive_state_identity": cognitive_identity}


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
            return _json.dumps({"a": index}, separators=(",", ":"))
    # Fallback: match by kind when the action carries additional payload
    # fields (e.g. submit with answer) not present in the template.
    kind = action.get("kind")
    for index, candidate in enumerate(legal_actions):
        if candidate.get("kind") == kind:
            import json as _json
            return _json.dumps({"a": index}, separators=(",", ":"))
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
