from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import (
    ACTION_COSTS,
    PUBLIC_SCHEMA_ID,
    Action,
    ActionCommitment,
    ActionDescriptor,
    PublicEvent,
    PublicView,
    content_hash,
)
from .generator import GeneratedWorld
from .rules import catalog_digest, evaluate_rule, overridden_assignment


class EnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StepResult:
    accepted: bool
    event: PublicEvent
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "event": self.event.to_dict(),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class DiscoveryEnvironment:
    def __init__(self, world: GeneratedWorld, *, budget: int = 8) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.world = world
        self.budget = budget
        self._reset_state()

    def _reset_state(self) -> None:
        self._step_id = 0
        self._remaining = self.budget
        self._spent = 0
        self._terminated = False
        self._truncated = False
        self._read_variables: set[int] = set()
        self._probe_contexts: set[tuple[int, int, int]] = set()
        self._interventions: set[tuple[tuple[int, int, int], int, int]] = set()
        self._events: list[PublicEvent] = []
        self._commitments: list[ActionCommitment] = []

    @property
    def step_id(self) -> int:
        return self._step_id

    @property
    def remaining_budget(self) -> int:
        return self._remaining

    @property
    def total_cost(self) -> int:
        return self._spent

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def transcript(self) -> tuple[PublicEvent, ...]:
        return tuple(self._events)

    @property
    def commitments(self) -> tuple[ActionCommitment, ...]:
        return tuple(self._commitments)

    def evidence_fingerprint(self) -> str:
        payload = {
            "rule": self.world.rule.semantic_sha256,
            "target_assignment": list(self.world.target_assignment),
            "read_variables": sorted(self._read_variables),
            "probe_contexts": [list(item) for item in sorted(self._probe_contexts)],
            "interventions": [
                {"context": list(context), "variable": variable, "value": value}
                for context, variable, value in sorted(self._interventions)
            ],
        }
        return content_hash(payload)

    def state_fingerprint(self) -> str:
        return content_hash(
            {
                "evidence": self.evidence_fingerprint(),
                "spent": self._spent,
                "remaining": self._remaining,
            }
        )

    def _variable_index(self, name: str) -> int:
        try:
            return self.world.variable_names.index(name)
        except ValueError as exc:
            raise EnvironmentError("unknown variable") from exc

    def _normalize_context(self, value: Any) -> tuple[int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise EnvironmentError("context must contain three values")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item not in (0, 1)
            for item in value
        ):
            raise EnvironmentError("context values must be binary integers")
        return tuple(int(item) for item in value)

    def _normalize_arguments(self, action: Action) -> dict[str, Any]:
        values = dict(action.arguments)
        if action.kind == "read":
            if set(values) != {"variable"} or not isinstance(values["variable"], str):
                raise EnvironmentError("read requires one variable")
            self._variable_index(values["variable"])
        elif action.kind == "probe":
            if set(values) != {"context"}:
                raise EnvironmentError("probe requires one context")
            values["context"] = self._normalize_context(values["context"])
        elif action.kind == "intervene":
            if set(values) != {"context", "variable", "value"}:
                raise EnvironmentError("intervention requires context, variable, and value")
            values["context"] = self._normalize_context(values["context"])
            if not isinstance(values["variable"], str):
                raise EnvironmentError("intervention variable must be a name")
            self._variable_index(values["variable"])
            if (
                isinstance(values["value"], bool)
                or not isinstance(values["value"], int)
                or values["value"] not in (0, 1)
            ):
                raise EnvironmentError("intervention value must be a binary integer")
        elif action.kind == "submit":
            if (
                set(values) != {"answer"}
                or isinstance(values["answer"], bool)
                or not isinstance(values["answer"], int)
                or values["answer"] not in (0, 1)
            ):
                raise EnvironmentError("submit requires a binary answer")
        elif action.kind == "stop":
            if values:
                raise EnvironmentError("stop takes no arguments")
        return values

    def _is_withheld_target_query(self, kind: str, arguments: Mapping[str, Any]) -> bool:
        if kind == "probe":
            return tuple(arguments["context"]) == self.world.target_assignment
        if kind == "intervene":
            variable = self._variable_index(arguments["variable"])
            return overridden_assignment(arguments["context"], variable, arguments["value"]) == self.world.target_assignment
        return False

    def _descriptor_for(self, kind: str, arguments: Mapping[str, Any]) -> ActionDescriptor:
        cost = ACTION_COSTS[kind]
        if kind == "read":
            return ActionDescriptor(
                kind=kind,
                arguments=dict(arguments),
                cost=cost,
                outcome_domain=(0, 1),
                mode="observational",
                preconditions=("variable is public", "variable has not been read"),
                state_changes=("target input becomes observed",),
            )
        if kind == "probe":
            return ActionDescriptor(
                kind=kind,
                arguments=dict(arguments),
                cost=cost,
                outcome_domain=(0, 1),
                mode="observational",
                preconditions=("context is public", "context is not the withheld target", "context is unused"),
                state_changes=("one probe result becomes observed",),
            )
        if kind == "intervene":
            return ActionDescriptor(
                kind=kind,
                arguments=dict(arguments),
                cost=cost,
                outcome_domain=(0, 1),
                mode="interventional",
                preconditions=("context is public", "resulting assignment is not the withheld target", "intervention is unused"),
                state_changes=("one counterfactual result becomes observed",),
            )
        if kind == "submit":
            return ActionDescriptor(
                kind=kind,
                arguments=dict(arguments),
                cost=cost,
                outcome_domain=(),
                mode="evaluative",
                preconditions=("answer is binary",),
                state_changes=("episode terminates",),
            )
        return ActionDescriptor(
            kind=kind,
            arguments={},
            cost=cost,
            outcome_domain=(),
            mode="control",
            preconditions=("episode is active",),
            state_changes=("episode terminates without an answer",),
        )

    def legal_actions(self) -> tuple[ActionDescriptor, ...]:
        if self._terminated:
            return ()
        values: list[ActionDescriptor] = []
        if self._remaining >= ACTION_COSTS["read"]:
            for index, name in enumerate(self.world.variable_names):
                if index not in self._read_variables:
                    values.append(self._descriptor_for("read", {"variable": name}))
        if self._remaining >= ACTION_COSTS["probe"]:
            for context in self.world.probe_contexts:
                if context not in self._probe_contexts:
                    values.append(self._descriptor_for("probe", {"context": list(context)}))
        if "intervene" in self.world.legal_action_kinds and self._remaining >= ACTION_COSTS["intervene"]:
            for context in self.world.intervention_contexts:
                for variable in range(3):
                    for value in (0, 1):
                        if (context, variable, value) not in self._interventions:
                            values.append(
                                self._descriptor_for(
                                    "intervene",
                                    {
                                        "context": list(context),
                                        "variable": self.world.variable_names[variable],
                                        "value": value,
                                    },
                                )
                            )
        if self._remaining >= ACTION_COSTS["submit"]:
            values.append(self._descriptor_for("submit", {"answer": 0}))
            values.append(self._descriptor_for("submit", {"answer": 1}))
        values.append(self._descriptor_for("stop", {}))
        return tuple(values)

    def _matches_legal(self, action: Action, arguments: Mapping[str, Any]) -> bool:
        for descriptor in self.legal_actions():
            if descriptor.kind != action.kind:
                continue
            descriptor_values = descriptor.arguments
            if set(arguments) != set(descriptor_values):
                continue
            matches = True
            for key, expected in descriptor_values.items():
                actual = arguments[key]
                if isinstance(expected, tuple):
                    actual = tuple(actual)
                if isinstance(expected, list):
                    actual = list(actual)
                if actual != expected:
                    matches = False
                    break
            if matches:
                return True
        return False

    def public_view(self) -> PublicView:
        return PublicView(
            schema=PUBLIC_SCHEMA_ID,
            episode_id=self.world.world_id,
            step_id=self._step_id,
            goal="predict the hidden binary output for the current state",
            variable_names=self.world.variable_names,
            target_context_id=self.world.manifest.target_context_id,
            probe_contexts=self.world.probe_contexts,
            legal_actions=self.legal_actions(),
            transcript=self.transcript,
            remaining_budget=self._remaining,
            total_cost=self._spent,
            belief_space_id=catalog_digest(),
        )

    def _charge(self, cost: int) -> None:
        self._spent += cost
        self._remaining = max(0, self._remaining - cost)

    def _finish_if_exhausted(self) -> None:
        if self._remaining == 0 and not self._terminated:
            self._truncated = True
            self._terminated = True

    def step(self, action: Action) -> StepResult:
        if self._terminated:
            raise EnvironmentError("episode is already terminated")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        before = self.public_view()
        commitment = ActionCommitment(
            step_id=self._step_id,
            action=action,
            available_view=before.to_dict(),
            available_view_sha256=content_hash(before.to_dict()),
        )
        self._commitments.append(commitment)
        try:
            arguments = self._normalize_arguments(action)
            if action.prediction is not None and action.kind in {"read", "probe", "intervene"}:
                if action.prediction not in (0, 1):
                    raise EnvironmentError("prediction is outside the outcome domain")
            if action.kind == "submit" and action.prediction not in (None, arguments["answer"]):
                raise EnvironmentError("submitted prediction must equal the answer")
            if action.kind == "stop" and action.prediction is not None:
                raise EnvironmentError("stop cannot carry an outcome prediction")
            if self._is_withheld_target_query(action.kind, arguments):
                raise EnvironmentError("action is not legal in the current state")
            if not self._matches_legal(action, arguments):
                raise EnvironmentError("action is not legal in the current state")
            cost = ACTION_COSTS[action.kind]
            self._charge(cost)
            accepted = True
            feedback = self._apply(action, arguments)
        except (EnvironmentError, TypeError, ValueError) as exc:
            cost = ACTION_COSTS.get(action.kind, 1)
            self._charge(cost)
            accepted = False
            feedback = {"kind": "invalid_action", "reason": str(exc), "charged_cost": cost}
        event = PublicEvent(
            step_id=self._step_id,
            action=action,
            accepted=accepted,
            cost=cost,
            feedback=feedback,
        )
        self._events.append(event)
        self._step_id += 1
        if accepted and action.kind in {"submit", "stop"}:
            self._terminated = True
        else:
            self._finish_if_exhausted()
        return StepResult(
            accepted=accepted,
            event=event,
            terminated=self._terminated,
            truncated=self._truncated,
        )

    def _apply(self, action: Action, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if action.kind == "read":
            variable = arguments["variable"]
            index = self._variable_index(variable)
            self._read_variables.add(index)
            return {
                "kind": "read",
                "mode": "observational",
                "variable": variable,
                "value": self.world.target_assignment[index],
            }
        if action.kind == "probe":
            context = arguments["context"]
            self._probe_contexts.add(context)
            return {
                "kind": "probe",
                "mode": "observational",
                "context": list(context),
                "outcome": evaluate_rule(self.world.rule, context),
            }
        if action.kind == "intervene":
            context = arguments["context"]
            variable_name = arguments["variable"]
            variable = self._variable_index(variable_name)
            value = arguments["value"]
            self._interventions.add((context, variable, value))
            counterfactual = overridden_assignment(context, variable, value)
            return {
                "kind": "intervene",
                "mode": "interventional",
                "context": list(context),
                "intervention": {"variable": variable_name, "value": value},
                "outcome": evaluate_rule(self.world.rule, counterfactual),
            }
        if action.kind == "submit":
            answer = arguments["answer"]
            return {
                "kind": "submit",
                "mode": "evaluative",
                "submitted_answer": answer,
            }
        return {"kind": "stop", "mode": "control", "abstained": True}

    def reset(self) -> PublicView:
        self._reset_state()
        return self.public_view()
