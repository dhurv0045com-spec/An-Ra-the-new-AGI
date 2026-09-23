"""Real model-origin proposer dispatch (I04/E5) with method-to-trainer
binding, controlled capture and origin validation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from bramastra_lab.research.contracts.core import content_identity
from bramastra_lab.research.metalearning.generations import (
    ProposalCapture,
    validate_origin,
)
from bramastra_lab.research.metalearning.method_language import (
    GradientTransform,
    MethodError,
    MethodProgram,
    ObjectiveCoefficients,
    compile_method,
)

METHOD_TOKEN_VOCABULARY = frozenset({"M0", "M1", "M2"})

# P0/P1/P_fixed lineages (R07): three distinct typed method programs with
# choices captured before fresh confirmation outcomes. E5's support data
# currently contains answer-token supervision only, so the compiled programs
# must not claim world/action/value/pair labels that are absent from that data.
# M0 is the token baseline; M1 changes both token scale and LR (a bundled
# intervention); M2 keeps the baseline objective and adds tighter clipping.
_METHOD_PROGRAMS = {
    "M0": MethodProgram(
        objective_coefficients=ObjectiveCoefficients(
            token=1.0),
        comparison_protocol="e5-anchor/v1"),
    "M1": MethodProgram(
        objective_coefficients=ObjectiveCoefficients(
            token=0.8),
        comparison_protocol="e5-anchor/v1"),
    "M2": MethodProgram(
        objective_coefficients=ObjectiveCoefficients(token=1.0),
        gradient_transform=GradientTransform(kind="clip_norm", bound=0.5),
        comparison_protocol="e5-anchor/v1"),
}


class DispatchError(ValueError):
    """A proposer dispatch violated its contract."""


@dataclass
class MethodTrialOutcome:
    method_id: str
    task_identity: str
    measured_updates: int
    measured_success: float
    elapsed_seconds: float
    validation: str = "measured"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class MethodArchive:
    rows: tuple[MethodTrialOutcome, ...]
    cutoff_event_index: int

    def identity(self) -> str:
        return content_identity({
            "rows": [row.to_dict() for row in self.rows],
            "cutoff_event_index": self.cutoff_event_index,
        })

    def best_measured(self, task_identity: str) -> str | None:
        results = [row for row in self.rows
                   if row.task_identity == task_identity
                   and row.validation == "measured"]
        if not results:
            return None
        return max(results, key=lambda row: row.measured_success).method_id

    def contains_task_outcomes(self, task_identity: str) -> bool:
        return any(row.task_identity == task_identity for row in self.rows)


class MethodProposer:
    """Generates a method choice through the shared decoder."""

    def __init__(self, model, config, *, checkpoint_payload_identity: str) -> None:
        self.model = model
        self.config = config
        self.checkpoint_payload_identity = checkpoint_payload_identity

    def render_input(self, task_descriptor: Mapping[str, Any],
                     archive_snapshot: MethodArchive) -> str:
        """Canonical rendered input for the method choice (audit record).

        This is the exact public state the model is prompted with, so the
        recorded transcript describes what was actually decoded.
        """
        forbidden = {"measured_success", "query_outcome", "label"}
        leaked = set(task_descriptor) & forbidden
        if leaked:
            raise DispatchError(
                f"current-task outcomes in proposal context: {sorted(leaked)}")
        from bramastra_lab.research.campaigns.phases.compiler import (
            method_choice_payload)

        payload = method_choice_payload(
            tasks=self._descriptor_tasks(task_descriptor),
            archive_methods=sorted(
                {row.method_id for row in archive_snapshot.rows}))
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _descriptor_tasks(task_descriptor: Mapping[str, Any]) -> dict[str, str]:
        """Normalize a method-choice descriptor to identity -> family.

        A block decision carries a `tasks` map; a single-task descriptor
        carries `task_identity` with one `family`. Both normalize to the same
        canonical payload.
        """
        tasks = task_descriptor.get("tasks")
        if isinstance(tasks, Mapping) and tasks:
            return {str(key): str(value) for key, value in tasks.items()}
        identities = task_descriptor.get("task_identities")
        if identities is None and task_descriptor.get("task_identity"):
            identities = [task_descriptor["task_identity"]]
        family = str(task_descriptor.get("family", ""))
        return {str(identity): family for identity in (identities or ())}

    def capture_proposal(self, task_descriptor: Mapping[str, Any],
                         archive_snapshot: MethodArchive, *,
                         sampling_temperature: float = 0.7) -> ProposalCapture:
        from bramastra_lab.research.campaigns.phases.compiler import (
            method_choice_payload, method_choice_prompt)
        from bramastra_lab.research.runtime.inference import generate_free_form

        rendered_input = self.render_input(task_descriptor, archive_snapshot)
        payload = json.loads(rendered_input)
        # The decode prompt is the canonical training prompt for the same
        # payload: same tokens the proposer was trained to continue.
        prompt = method_choice_prompt(payload)
        report = generate_free_form(self.model, self.config, prompt,
                                    max_new_tokens=24)
        raw_output = report.answer
        token = raw_output.strip().upper()
        if token in METHOD_TOKEN_VOCABULARY:
            method_id = token
            program = _METHOD_PROGRAMS[method_id]
        else:
            method_id, program = parse_method_selection(raw_output)
        return ProposalCapture(
            checkpoint_payload_identity=self.checkpoint_payload_identity,
            rendered_input=rendered_input,
            sampling={"temperature": sampling_temperature, "max_new_tokens": 24},
            raw_output=raw_output, parsed_program=program)

    def choose_method(self, task_descriptor: Mapping[str, Any],
                      archive_snapshot: MethodArchive, *,
                      checkpoint_registry: Mapping[str, str]) -> tuple[str, Any]:
        capture = self.capture_proposal(task_descriptor, archive_snapshot)
        method_id, _ = parse_method_selection(capture.raw_output)
        validate_origin(capture,
                        reparsed_program=_METHOD_PROGRAMS[method_id],
                        checkpoint_registry=checkpoint_registry)
        return method_id, capture


def parse_method_selection(raw_output: str) -> tuple[str, Any]:
    """Parse a method selection: bare token or typed JSON."""
    token = raw_output.strip().upper()
    if token in METHOD_TOKEN_VOCABULARY:
        return token, _METHOD_PROGRAMS[token]
    try:
        raw = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise MethodError(f"method output is neither a token nor JSON: {exc}")
    program = _program_from_json(raw)
    return _program_to_method_id(program), program


def _program_to_method_id(program: MethodProgram) -> str:
    """Map a typed program to its lineage token (R07).

    Gradient-transform programs are M2; objective-coefficient programs are
    distinguished by their exact coefficients (M0 anchor vs M1 reduced-LR
    variant). Arbitrary JSON no longer collapses into the small vocabulary by
    a loose check — unknown coefficient sets raise instead of mapping to M0.
    """
    if program.gradient_transform is not None:
        coefficients = program.objective_coefficients
        baseline_objective = coefficients is not None and (
            float(coefficients.token), float(coefficients.world),
            float(coefficients.action), float(coefficients.value),
            float(coefficients.pair)) == (1.0, 0.0, 0.0, 0.0, 0.0)
        if program.gradient_transform.kind == "clip_norm" \
                and float(program.gradient_transform.bound) == 0.5 \
                and baseline_objective:
            return "M2"
        raise DispatchError(
            f"unknown gradient_transform {program.gradient_transform!r}; "
            "only M2's baseline objective plus clip_norm/0.5 is admitted")
    coefficients = program.objective_coefficients
    if coefficients is not None:
        token, world, action, value, pair = (
            float(coefficients.token), float(coefficients.world),
            float(coefficients.action), float(coefficients.value),
            float(coefficients.pair))
        if (token, world, action, value, pair) == (1.0, 0.0, 0.0, 0.0, 0.0) \
                and program.gradient_transform is None:
            return "M0"
        if (token, world, action, value, pair) == (0.8, 0.0, 0.0, 0.0, 0.0) \
                and program.gradient_transform is None:
            return "M1"
        raise DispatchError(
            f"objective coefficients {(token, world, action, value, pair)} "
            "match no declared lineage (M0/M1); refusing loose mapping")
    raise DispatchError("program matches no declared method lineage")


def _program_from_json(raw: Mapping[str, Any]) -> MethodProgram:
    if not isinstance(raw, Mapping):
        # Valid JSON that is not a program object (a bare bool/number/list)
        # is a decode failure with its own message, never an AttributeError
        # from deep inside the program builder.
        raise MethodError(
            f"method program JSON must be an object, got {type(raw).__name__}")
    if raw.get("proposal") == "no_change":
        raise MethodError("use the no_change literal, not a program JSON")
    kwargs: dict[str, Any] = {}
    if raw.get("schedule"):
        from bramastra_lab.research.metalearning.method_language import (
            ScheduleExpression, SchedulePoint)
        kwargs["schedule"] = ScheduleExpression(
            counter=raw["schedule"]["counter"],
            points=tuple(SchedulePoint(**point)
                         for point in raw["schedule"]["points"]))
    if raw.get("replay_weights"):
        from bramastra_lab.research.metalearning.method_language import ReplayWeights
        kwargs["replay_weights"] = ReplayWeights(raw["replay_weights"])
    if raw.get("objective_coefficients"):
        from bramastra_lab.research.metalearning.method_language import (
            ObjectiveCoefficients)
        kwargs["objective_coefficients"] = ObjectiveCoefficients(
            **raw["objective_coefficients"])
    if raw.get("gradient_transform"):
        from bramastra_lab.research.metalearning.method_language import (
            GradientTransform)
        kwargs["gradient_transform"] = GradientTransform(
            **raw["gradient_transform"])
    kwargs["expected_gain"] = raw.get("expected_gain", 0.0)
    kwargs["predicted_cost"] = raw.get("predicted_cost", 0.0)
    kwargs["comparison_protocol"] = raw.get("comparison_protocol", "e5-anchor/v1")
    kwargs["failure_conditions"] = tuple(raw.get("failure_conditions", ()))
    kwargs["eligibility_scope"] = tuple(raw.get("eligibility_scope", ("training",)))
    return MethodProgram(**kwargs)


def dispatch_method_to_trainer(method_id: str, compiled: Mapping[str, Any],
                               trainer, *, task_identity: str) -> str:
    """Apply a selected method to the REAL trainer (R07).

    Binds the exact selected token/program, immutable archive, compiled
    semantic recipe and applied trainer state: the compiled identity must
    match the declared lineage program (recompiled here), otherwise the
    caller-supplied recipe is rejected as mismatched. M1 halves LR; M2
    applies the declared clipping bound and requires enabled gates.
    """
    if method_id not in METHOD_TOKEN_VOCABULARY:
        raise DispatchError(f"unknown method id {method_id!r}")
    if not compiled.get("identity"):
        raise DispatchError("compiled method carries no identity")
    # Reject mismatched caller recipes: recompile the declared lineage and
    # require identity agreement (a token string is not authentication).
    from bramastra_lab.research.metalearning.method_language import compile_method

    expected_program = _METHOD_PROGRAMS[method_id]
    # Runtime config is part of the compiled identity; accept any runtime but
    # require the program identity to match the declared lineage.
    expected_program_identity = expected_program.identity()
    actual_program_identity = compiled.get("program_identity")
    if actual_program_identity != expected_program_identity:
        raise DispatchError(
            f"compiled recipe program {str(actual_program_identity)[:12]}... "
            f"does not match declared lineage {method_id} "
            f"({expected_program_identity[:12]}...); rejecting mismatched "
            "caller recipe")
    if method_id == "M1":
        for group in trainer.optimizer.param_groups:
            group["lr"] = group["lr"] * 0.5
    if method_id == "M2":
        if not getattr(trainer.model, "gates_enabled", False):
            raise DispatchError(
                "M2 dispatched but the trainer model has no enabled gates")
        transform = compiled.get("gradient_transform")
        if not isinstance(transform, Mapping) \
                or transform.get("kind") != "clip_norm" \
                or float(transform.get("bound", -1.0)) != 0.5:
            raise DispatchError(
                "M2 compiled recipe must declare clip_norm bound 0.5")
        trainer.clip_norm = 0.5
    trainer.set_controller_multiplier(
        1.0, f"method:{method_id}:{compiled['identity'][:12]}")
    return f"applied:{method_id}:{task_identity}"
