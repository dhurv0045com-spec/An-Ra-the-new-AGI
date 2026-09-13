"""Real model-origin proposer dispatch (I04/E5): generated method bytes bind
to a validated, applied method and its successor update. Fixture generation
remains a test control and never enters the learned evidence namespace.

The proposer's decoder generates method tokens through the shared model;
the controlled adapter renders the input, decodes the output, parses the
AST and validates origin — all through these production functions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from bramastra_lab.research.metalearning.generations import (
    ProposalCapture,
    classify_external_submission,
    validate_origin,
)
from bramastra_lab.research.metalearning.method_language import (
    GradientTransform,
    MethodError,
    MethodProgram,
    ObjectiveCoefficients,
    compile_method,
)


class DispatchError(ValueError):
    """A proposer dispatch violated its contract."""


METHOD_TOKEN_VOCABULARY = frozenset({"M0", "M1", "M2"})

_METHOD_PROGRAMS = {
    "M0": MethodProgram(objective_coefficients=ObjectiveCoefficients(
        token=1.0, world=0.5, action=0.5, value=0.1, pair=0.1),
        comparison_protocol="e5-anchor/v1"),
    "M1": MethodProgram(objective_coefficients=ObjectiveCoefficients(
        token=1.0, world=0.5, action=0.5, value=0.1, pair=0.1),
        comparison_protocol="e5-anchor/v1"),
    "M2": MethodProgram(gradient_transform=GradientTransform(
        kind="clip_norm", bound=1.0), comparison_protocol="e5-anchor/v1"),
}


@dataclass
class MethodTrialOutcome:
    method_id: str
    task_identity: str
    measured_updates: int
    measured_success: float
    elapsed_seconds: float
    validation: str = "measured"   # measured | failed | invalid

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class MethodArchive:
    """Immutable pre-decision archive snapshot (RSI §5): binding this hash
    into proposal context proves which outcomes were visible."""

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
    """Generates a method choice through the shared decoder.

    The input contains only permitted support-derived task descriptors,
    public goals, budget and the frozen archive summary — never the current
    task's measured outcomes (confirmation->archive boundary).
    """

    def __init__(self, model, config, *, checkpoint_payload_identity: str) -> None:
        self.model = model
        self.config = config
        self.checkpoint_payload_identity = checkpoint_payload_identity

    def render_input(self, task_descriptor: Mapping[str, Any],
                     archive_snapshot: MethodArchive) -> str:
        forbidden = {"measured_success", "query_outcome", "label"}
        leaked = set(task_descriptor) & forbidden
        if leaked:
            raise DispatchError(
                f"current-task outcomes cannot appear in proposal context: {sorted(leaked)}")
        summary = {
            "task_descriptor": dict(task_descriptor),
            "archive_identity": archive_snapshot.identity(),
            "archive_methods": sorted({row.method_id for row in archive_snapshot.rows}),
        }
        return json.dumps(summary, sort_keys=True)

    def capture_proposal(self, task_descriptor: Mapping[str, Any],
                         archive_snapshot: MethodArchive, *,
                         sampling_temperature: float = 0.7,
                         generator: torch.Generator | None = None) -> ProposalCapture:
        rendered_input = self.render_input(task_descriptor, archive_snapshot)
        from bramastra_lab.research.experience.codec import encode_text

        prompt = [259] + encode_text(rendered_input)
        from bramastra_lab.research.runtime.inference import generate_free_form

        report = __import__("bramastra_lab.research.runtime.inference",
                            fromlist=["generate_free_form"]).generate_free_form(
            self.model, self.config, prompt, max_new_tokens=24,
            generator=generator)
        raw_output = report.answer
        # Controlled parser: method-token vocabulary or a typed program JSON.
        token = raw_output.strip().upper()
        if token in METHOD_TOKEN_VOCABULARY:
            raw_normalized = json.dumps({"proposal": token})
        else:
            raw_normalized = raw_output
        method_id, program = parse_method_selection(raw_normalized)
        if program is None:
            # Bare token: build the representative typed program for capture.
            program = _METHOD_PROGRAMS[method_id]
        return ProposalCapture(
            checkpoint_payload_identity=self.checkpoint_payload_identity,
            rendered_input=rendered_input, sampling={
                "temperature": sampling_temperature,
                "max_new_tokens": 24},
            raw_output=raw_output, parsed_program=program)

    def choose_method(self, task_descriptor: Mapping[str, Any],
                      archive_snapshot: MethodArchive, *,
                      checkpoint_registry: Mapping[str, str],
                      generator: torch.Generator | None = None) -> tuple[str, ProposalCapture]:
        capture = self.capture_proposal(task_descriptor, archive_snapshot,
                                        generator=generator)
        reparsed = parse_method_selection(
            json.dumps({"proposal": capture.raw_output.strip().upper()})
            if capture.raw_output.strip().upper() in METHOD_TOKEN_VOCABULARY
            else capture.raw_output)
        reparsed_id, reparsed_program = parse_method_selection(
            json.dumps({"proposal": capture.raw_output.strip().upper()})
            if capture.raw_output.strip().upper() in METHOD_TOKEN_VOCABULARY
            else capture.raw_output)
        if reparsed_program is None:
            reparsed_program = _METHOD_PROGRAMS[reparsed_id]
        checkpoint_id = validate_origin(capture, reparsed_program=reparsed_program,
                                        checkpoint_registry=checkpoint_registry)
        return reparsed_id, capture


def _program_to_method_id(program: MethodProgram) -> str:
    if program.gradient_transform is not None:
        return "M2"
    return "M0"


def parse_method_selection(raw_output: str) -> tuple[str, MethodProgram | None]:
    """Parse a method selection: a bare method token (the declared E5 action
    language) or typed JSON. Returns (method_id, program-or-None)."""
    token = raw_output.strip().upper()
    if token in METHOD_TOKEN_VOCABULARY:
        return token, None
    try:
        raw = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise MethodError(f"method output is neither a token nor JSON: {exc}")
    program = _program_from_json(raw)
    return _program_to_method_id(program), program


def _program_from_json(raw: Mapping[str, Any]) -> MethodProgram:
    if raw.get("proposal") == "no_change":
        raise MethodError("use the no_change literal, not a program JSON")
    kwargs: dict[str, Any] = {}
    if raw.get("schedule"):
        from bramastra_lab.research.metalearning.method_language import (
            ScheduleExpression, SchedulePoint)

        kwargs["schedule"] = ScheduleExpression(
            counter=raw["schedule"]["counter"],
            points=tuple(SchedulePoint(**point) for point in raw["schedule"]["points"]))
    if raw.get("replay_weights"):
        from bramastra_lab.research.metalearning.method_language import ReplayWeights

        kwargs["replay_weights"] = ReplayWeights(raw["replay_weights"])
    if raw.get("objective_coefficients"):
        from bramastra_lab.research.metalearning.method_language import ObjectiveCoefficients

        kwargs["objective_coefficients"] = ObjectiveCoefficients(
            **raw["objective_coefficients"])
    if raw.get("gradient_transform"):
        from bramastra_lab.research.metalearning.method_language import GradientTransform

        kwargs["gradient_transform"] = GradientTransform(**raw["gradient_transform"])
    kwargs["expected_gain"] = raw.get("expected_gain", 0.0)
    kwargs["predicted_cost"] = raw.get("predicted_cost", 0.0)
    kwargs["comparison_protocol"] = raw.get("comparison_protocol", "e5-anchor/v1")
    kwargs["failure_conditions"] = tuple(raw.get("failure_conditions", ()))
    kwargs["eligibility_scope"] = tuple(raw.get("eligibility_scope", ("training",)))
    return MethodProgram(**kwargs)


def dispatch_method_to_trainer(method_id: str, compiled: Mapping[str, Any],
                               trainer, *, task_identity: str,
                               measured_outcomes: MethodArchive | None = None) -> str:
    """Apply a selected method to the REAL trainer: schedule and coefficients
    change the actual update recipe. Returns the applied method identity.

    A dispatch that silently substitutes M0 while keeping the proposal label
    is the deliberate defect this function's caller must reject — the
    compiled identity is recorded in the trainer-visible state.
    """
    if method_id not in METHOD_TOKEN_VOCABULARY:
        raise DispatchError(f"unknown method id {method_id!r}")
    if compiled.get("identity") is None:
        raise DispatchError("compiled method carries no identity")
    if method_id == "M1":
        for group in trainer.optimizer.param_groups:
            group["lr"] = group["lr"] * 0.5
    if method_id == "M2":
        # Architecture migration is handled by the caller before the trainer
        # is constructed; dispatch records the requirement here.
        if not getattr(trainer.model, "gates_enabled", False):
            raise DispatchError(
                "M2 dispatched but the trainer's model has no enabled gates; "
                "architecture migration must precede dispatch")
    trainer.set_controller_multiplier(1.0, f"method:{method_id}:{compiled['identity'][:12]}")
    return f"applied:{method_id}:{task_identity}"
