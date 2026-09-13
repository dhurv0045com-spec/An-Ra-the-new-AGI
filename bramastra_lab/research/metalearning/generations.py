"""Model-origin proposals and generation transactions (M23/M24).

Proposal capture is produced by a controlled model-execution adapter and
binds checkpoint payload identity, rendered input, sampling settings, raw
output and the parsed AST. The validator reparses the captured output and
compares the AST before compilation; transcript hashing alone cannot qualify
an arbitrary external submission. The generation transaction links proposer,
compiled method, matched comparison and chief approval; the accepted
successor — never a substituted external proposer — begins the next
generation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import content_identity
from bramastra_lab.research.metalearning.method_language import (
    MethodError,
    MethodProgram,
    compile_method,
)


class OriginError(ValueError):
    """A proposal origin or generation transaction violated its contract."""


PROPOSER_VERSION = "method-parser/v1"


@dataclass(frozen=True)
class ProposalCapture:
    """Controlled-execution receipt for one model-origin proposal."""

    checkpoint_payload_identity: str
    rendered_input: str
    sampling: Mapping[str, Any]
    raw_output: str
    parsed_program: MethodProgram
    parser_version: str = PROPOSER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_payload_identity, str) \
                or not self.checkpoint_payload_identity:
            raise OriginError("checkpoint payload identity must be recorded")
        if not isinstance(self.rendered_input, str) or not self.rendered_input:
            raise OriginError("rendered input must be recorded")
        if not isinstance(self.sampling, Mapping) or "temperature" not in self.sampling:
            raise OriginError("sampling settings (at least temperature) must be recorded")
        if not isinstance(self.raw_output, str) or not self.raw_output:
            raise OriginError("raw output must be recorded")
        if not isinstance(self.parsed_program, MethodProgram):
            raise OriginError("parsed program must be a MethodProgram")
        if self.parser_version != PROPOSER_VERSION:
            raise OriginError("unknown parser version")

    @property
    def transcript_hash(self) -> str:
        return content_identity({"input": self.rendered_input,
                                 "output": self.raw_output,
                                 "sampling": dict(self.sampling),
                                 "checkpoint": self.checkpoint_payload_identity})


def parse_model_output(raw_output: str) -> MethodProgram:
    """Controlled parser for the typed method language (JSON serialization).

    A parsing failure is an invalid proposal, not a silent no-op.
    """
    try:
        raw = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise MethodError(f"proposal output is not valid JSON: {exc}")
    if not isinstance(raw, Mapping):
        raise MethodError("proposal output must be a JSON object")
    if raw.get("proposal") == "no_change":
        raise MethodError(
            "no-change proposals are represented by the no_change literal; "
            "a MethodProgram JSON is required here")
    kwargs: dict[str, Any] = {}
    if "schedule" in raw and raw["schedule"] is not None:
        from bramastra_lab.research.metalearning.method_language import (
            ScheduleExpression,
            SchedulePoint,
        )

        kwargs["schedule"] = ScheduleExpression(
            counter=raw["schedule"]["counter"],
            points=tuple(SchedulePoint(**point) for point in raw["schedule"]["points"]))
    if "replay_weights" in raw and raw["replay_weights"] is not None:
        from bramastra_lab.research.metalearning.method_language import ReplayWeights

        kwargs["replay_weights"] = ReplayWeights(raw["replay_weights"])
    if "objective_coefficients" in raw and raw["objective_coefficients"] is not None:
        from bramastra_lab.research.metalearning.method_language import ObjectiveCoefficients

        kwargs["objective_coefficients"] = ObjectiveCoefficients(
            **raw["objective_coefficients"])
    if "gradient_transform" in raw and raw["gradient_transform"] is not None:
        from bramastra_lab.research.metalearning.method_language import GradientTransform

        kwargs["gradient_transform"] = GradientTransform(**raw["gradient_transform"])
    kwargs["expected_gain"] = raw.get("expected_gain", 0.0)
    kwargs["predicted_cost"] = raw.get("predicted_cost", 0.0)
    kwargs["comparison_protocol"] = raw.get("comparison_protocol", "")
    kwargs["failure_conditions"] = tuple(raw.get("failure_conditions", ()))
    kwargs["eligibility_scope"] = tuple(raw.get("eligibility_scope", ("training",)))
    return MethodProgram(**kwargs)


def validate_origin(capture: ProposalCapture, *, reparsed_program: MethodProgram,
                    checkpoint_registry: Mapping[str, str]) -> str:
    """Origin validation: the captured AST must equal the controlled runner's
    reparse, and the checkpoint reference must exist in the registry. Returns
    the validated proposer checkpoint identity. Transcript hashing alone
    cannot qualify arbitrary external output."""
    if capture.parsed_program.identity() != reparsed_program.identity():
        raise OriginError(
            "captured AST differs from the controlled reparse; origin validation "
            "failed")
    checkpoint_id = capture.checkpoint_payload_identity
    if checkpoint_id not in checkpoint_registry:
        raise OriginError(
            f"proposal references checkpoint {checkpoint_id[:12]}... which is not "
            "in the registry; a token string is not proposer authentication")
    if checkpoint_registry[checkpoint_id] != "model":
        raise OriginError(
            f"checkpoint {checkpoint_id[:12]}... is registered as "
            f"{checkpoint_registry[checkpoint_id]!r}; only model-origin captures "
            "qualify for model-origin proposals")
    return checkpoint_id


def classify_external_submission(raw_output: str, *,
                                 checkpoint_registry: Mapping[str, str]) -> str:
    """An external transcript with a forged or missing checkpoint reference
    is labeled external/assisted regardless of its content quality."""
    try:
        program = parse_model_output(raw_output)
    except MethodError:
        return "external_invalid"
    capture = ProposalCapture(
        checkpoint_payload_identity="claimed-checkpoint",
        rendered_input="unknown", sampling={"temperature": 1.0},
        raw_output=raw_output, parsed_program=program)
    try:
        validate_origin(capture, reparsed_program=program,
                        checkpoint_registry=checkpoint_registry)
    except OriginError:
        return "external_assisted"
    return "model_origin"


# --- generation transaction (M24) --------------------------------------------

GENERATION_STATUSES = frozenset({"proposed", "validated", "compiled", "compared",
                                 "confirmed", "accepted", "rejected", "crashed"})


@dataclass
class GenerationReceipt:
    generation_id: str
    predecessor_receipt_id: str | None
    proposer_checkpoint: str
    proposer_origin: str
    parent_method: Mapping[str, Any]
    candidate_program: MethodProgram | None
    compiled_identity: str | None
    comparison_identity: str | None
    attempted_candidates: tuple[str, ...] = ()
    consumed_resources: Mapping[str, float] = field(default_factory=dict)
    selected_candidate_identity: str | None = None
    chief_approval_hash: str | None = None
    status: str = "proposed"
    fixture: bool = True
    migration_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in GENERATION_STATUSES:
            raise OriginError(f"unknown generation status {self.status!r}")
        if not self.fixture and self.status == "accepted" and \
                self.chief_approval_hash is None:
            raise OriginError(
                "a learned parent publication requires a chief approval bound to "
                "the evidence hashes; fixture registries are separate")

    def identity(self) -> str:
        return content_identity({
            "generation_id": self.generation_id,
            "predecessor": self.predecessor_receipt_id,
            "proposer": self.proposer_checkpoint,
            "proposer_origin": self.proposer_origin,
            "parent_method": dict(self.parent_method),
            "candidate": None if self.candidate_program is None
            else self.candidate_program.to_dict(),
            "compiled_identity": self.compiled_identity,
            "comparison_identity": self.comparison_identity,
            "attempted": list(self.attempted_candidates),
            "resources": dict(self.consumed_resources),
            "selected": self.selected_candidate_identity,
            "approval": self.chief_approval_hash,
            "status": self.status,
            "fixture": self.fixture,
            "migration": dict(self.migration_state),
        })


class GenerationRegistry:
    """Separate fixture and learned registries; fixtures can never publish a
    learned parent, and every attempt (including failures) is retained."""

    def __init__(self) -> None:
        self.receipts: dict[str, GenerationReceipt] = {}
        self.accepted_fixture_parents: dict[str, str] = {}
        self.accepted_learned_parents: dict[str, str] = {}

    def record(self, receipt: GenerationReceipt) -> str:
        receipt_id = receipt.identity()
        self.receipts[receipt_id] = receipt
        return receipt_id

    def publish(self, receipt: GenerationReceipt, *,
                chief_approval_hash: str | None,
                successor_proposer: str) -> str:
        """Publish an accepted successor into the chain.

        - A fixture acceptance writes only the fixture registry.
        - A learned acceptance requires chief approval and a model-origin
          proposer; the successor proposer must be the accepted model itself.
        """
        if receipt.status != "confirmed":
            raise OriginError("only confirmed generations can be published")
        if receipt.candidate_program is None:
            # No-change: the parent method continues; the successor proposer
            # is the same checkpoint.
            successor = successor_proposer
            parent_method = dict(receipt.parent_method)
        else:
            if chief_approval_hash is None:
                raise OriginError(
                    "publishing an accepted candidate requires chief approval "
                    "bound to the evidence hashes")
            if receipt.fixture:
                if receipt.proposer_origin != "model":
                    raise OriginError(
                        "fixture chain attribution error: the receipt claims a "
                        "non-model proposer")
                successor = successor_proposer
                parent_method = {**receipt.parent_method,
                                 **receipt.candidate_program.to_dict()}
            else:
                if receipt.proposer_origin != "model":
                    raise OriginError(
                        "a learned parent cannot be published from a non-model "
                        "proposer; external assistance is recorded as assisted")
                successor = successor_proposer
                parent_method = {**receipt.parent_method,
                                 **receipt.candidate_program.to_dict()}
        receipt.chief_approval_hash = chief_approval_hash
        receipt.status = "accepted"
        receipt.selected_candidate_identity = (
            receipt.candidate_program.identity()
            if receipt.candidate_program is not None else "no_change")
        receipt_id = self.record(receipt)
        if receipt.fixture:
            self.accepted_fixture_parents[receipt.generation_id] = successor
        else:
            self.accepted_learned_parents[receipt.generation_id] = successor
        return receipt_id

    def successor_proposer(self, generation_id: str) -> str:
        for registry in (self.accepted_fixture_parents, self.accepted_learned_parents):
            if generation_id in registry:
                return registry[generation_id]
        raise OriginError(f"generation {generation_id!r} has no accepted successor")


def run_fixture_generation(generation_id: str, predecessor_receipt_id: str | None,
                           proposer_checkpoint: str, parent_method: Mapping[str, Any],
                           candidate_program: MethodProgram | None, *,
                           comparison_identity: str | None,
                           confirmed: bool, crash: bool = False,
                           registry: GenerationRegistry | None = None) -> GenerationReceipt:
    """One fixture generation transaction (reject/crash/retry/no-change
    paths). Fixture evidence cannot update any learned parent."""
    registry = registry or GenerationRegistry()
    receipt = GenerationReceipt(
        generation_id=generation_id, predecessor_receipt_id=predecessor_receipt_id,
        proposer_checkpoint=proposer_checkpoint, proposer_origin="model",
        parent_method=dict(parent_method), candidate_program=candidate_program,
        compiled_identity=(compile_method(candidate_program, runtime_config={"profile": "tiny"})
                           ["identity"] if candidate_program is not None else None),
        comparison_identity=comparison_identity,
        attempted_candidates=(candidate_program.identity(),)
        if candidate_program is not None else (),
        consumed_resources={"model_calls": 3, "inference_tokens": 240},
        fixture=True)
    if crash:
        receipt.status = "crashed"
        registry.record(receipt)
        return receipt
    if not confirmed:
        receipt.status = "rejected"
        registry.record(receipt)
        return receipt
    receipt.status = "confirmed"
    registry.record(receipt)
    registry.publish(receipt, chief_approval_hash="fixture-approval",
                     successor_proposer=proposer_checkpoint)
    return receipt
