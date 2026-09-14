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


def _compact_history_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Lossless-compact history entry (semantic fields preserved).

    Short keys keep multi-step traces inside the byte-level context policy;
    the mapping is bijective over the whitelisted fields (see _expand key
    table in receipts: a=action, f=feedback, k=kind, v=value/variable,
    n=name, r=result/requires, c=correct, s=submitted, m=matched,
    w=written, t=total).
    """
    action = dict(entry.get("action", {}))
    feedback = dict(entry.get("feedback", {}))
    compact_action = {"k": action.get("kind")}
    for key, short in (("variable", "v"), ("item", "n"), ("input", "v"),
                       ("value", "v"), ("container", "n"), ("switch", "n"),
                       ("answer", "s")):
        if action.get(key) is not None:
            compact_action[short] = action[key]
    compact_feedback = {"k": feedback.get("kind")}
    for key, short in (("variable", "v"), ("value", "v"), ("item", "n"),
                       ("requires", "r"), ("input", "v"), ("result", "r"),
                       ("total", "t"), ("matched", "m"), ("written", "w"),
                       ("submitted_answer", "s"), ("correct", "c"),
                       ("contains_item", "v"), ("changed", "c"),
                       ("state", "v")):
        if feedback.get(key) is not None:
            compact_feedback[short] = feedback[key]
    return {"a": compact_action, "f": compact_feedback}


def _compact_evidence_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Lossless-compact evidence record for the shared renderer."""
    compact: dict[str, Any] = {}
    for key, short in (("entity", "e"), ("observed_value", "o"),
                       ("observation_id", "i"), ("temporal_scope", "t"),
                       ("status", "s")):
        if record.get(key) is not None:
            value = record[key]
            compact[short] = str(value)[:120] if key == "observed_value" \
                else value
    return compact or {"e": "empty-record"}


def mark_conflicts(workspace: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Mark contradictory evidence without rewriting history.

    Two records conflict when they share an entity with different observed
    values. Both keep their source observations; their status becomes
    "conflicting" (a revision of support, not of history). Returns the
    conflicting observation-id pairs.
    """
    conflicts: list[tuple[str, str]] = []
    for i, left in enumerate(workspace):
        for right in workspace[i + 1:]:
            if left.get("entity") and left.get("entity") == right.get("entity") \
                    and left.get("observed_value") != right.get("observed_value"):
                left["status"] = "conflicting"
                right["status"] = "conflicting"
                conflicts.append((str(left.get("observation_id")),
                                  str(right.get("observation_id"))))
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

        prompt_text = json.dumps({"state_goal": dict(state.get("goal", {})),
                                  "action": dict(action),
                                  "depth": depth}, sort_keys=True)[:256]
        prompt = [259] + encode_text(prompt_text)[:64]
        self.calls += 1
        try:
            out = self.model.generate(prompt, max_new_tokens=24)
            predicted = json.loads(out["answer"])
            if not isinstance(predicted, Mapping):
                raise ValueError("prediction is not an object")
            base_origin = str(out.get("origin", "model"))
            return {"feedback": dict(predicted.get("feedback", {})),
                    "success_prob": float(predicted.get("success_prob", 0.5)),
                    "value": float(predicted.get("value", 0.0)),
                    "origin": base_origin + "-imagined",
                    "prediction_failed": False}
        except Exception as exc:
            return {"feedback": {},
                    "success_prob": 0.5, "value": 0.0,
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
    compact_obs = _compact_history_entry({"action": {}, "feedback": dict(
        observation)})["f"]
    record = {"entity": str(compact_obs.get("k", "observation")),
              "observed_value": json.dumps(compact_obs, sort_keys=True,
                                           default=str)[:120],
              "observation_id": observation_id,
              "temporal_scope": "episode",
              "status": "active"}
    workspace.append(record)
    return record


class BoundedPlannerAdapter(Adapter):
    """Depth-two / node-eight imagined search (B planner arm).

    Expands legal actions, predicts outcomes through the declared world
    model, and ranks first actions by predicted success minus action cost.
    All predictions stay in the imagined list with provenance; the proposed
    action executes for real in the loop like any other adapter.
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
        self.last_imagined: list[ImaginedNode] = []

    def select(self, *, legal_actions, rendered, model, workspace,
               state_view) -> Mapping[str, Any]:
        self.last_imagined = []
        if not legal_actions:
            raise EpisodeError("planner has no legal actions to expand")
        scored: list[tuple[float, Mapping[str, Any]]] = []
        nodes = 0
        state_hash = _hash_mapping(dict(state_view.get("goal", {})))
        for action in list(legal_actions):
            if nodes >= self.max_nodes:
                break
            predicted = self.world_model(state=dict(state_view),
                                         action=dict(action), depth=1)
            nodes += 1
            self.last_imagined.append(ImaginedNode(
                node_index=nodes, public_state_hash=state_hash,
                goal_hash=_hash_mapping(dict(state_view.get("goal", {}))),
                action_prefix=[dict(action)],
                predicted_outcome=dict(predicted),
                value_estimate=float(predicted.get("value", 0.0)),
                remaining_budget=int(state_view.get("budgets", {}).get(
                    "actions_left", 0)),
                provenance="planner-depth1"))
            best_value = float(predicted.get("success_prob", 0.0)) \
                + self.value_fn(state=dict(state_view), action=dict(action))
            if self.max_depth >= 2:
                for second in list(legal_actions):
                    if nodes >= self.max_nodes:
                        break
                    predicted2 = self.world_model(
                        state=dict(state_view), action=dict(second), depth=2)
                    nodes += 1
                    self.last_imagined.append(ImaginedNode(
                        node_index=nodes, public_state_hash=state_hash,
                        goal_hash=_hash_mapping(
                            dict(state_view.get("goal", {}))),
                        action_prefix=[dict(action), dict(second)],
                        predicted_outcome=dict(predicted2),
                        value_estimate=float(predicted2.get("value", 0.0)),
                        remaining_budget=int(state_view.get("budgets", {}).get(
                            "actions_left", 0)),
                        provenance="planner-depth2"))
                    candidate = float(predicted2.get("success_prob", 0.0)) \
                        + self.value_fn(state=dict(state_view),
                                        action=dict(second)) - 0.05
                    best_value = max(best_value, candidate)
            cost = 1.0
            scored.append((best_value - 0.1 * cost, dict(action),
                         str(predicted.get("origin", "unknown"))))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = dict(scored[0][1])
        chosen["_planner_nodes"] = nodes
        chosen["_planner_value"] = scored[0][0]
        chosen["_value_fn"] = self.value_fn_name
        chosen["_origin"] = scored[0][2]
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
            "imagined": [node.__dict__ for node in imagined],
            "history": history,
            "summary": summary,
            "adapter": adapter.name,
            "goal": base_goal,
            "episode_id": episode_id}
