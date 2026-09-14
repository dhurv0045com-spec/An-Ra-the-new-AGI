"""Fail-closed implementation readiness for the K8 campaign.

This gate describes reviewed repository implementation state.  It deliberately
does not infer readiness from imports, CUDA availability, or the presence of a
notebook.  A disposition is cleared only by an explicit source change to this
reviewed manifest together with the focused production-path evidence named by
the disposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


IMPLEMENTATION_READINESS_SCHEMA = "bramastra-k8-implementation-readiness/v1"
REVIEWED_SOURCE_REVISION = "5c0567f"


@dataclass(frozen=True)
class PhaseDisposition:
    """Reviewed status for one required campaign phase."""

    phase: str
    status: str
    blockers: tuple[str, ...]
    required_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "blockers": list(self.blockers),
            "required_evidence": list(self.required_evidence),
        }


# This is intentionally explicit and immutable.  It is a reviewed disposition,
# not a heuristic scan of source files.  Updating a status requires the chief to
# review the corresponding production path and focused evidence.
REVIEWED_DISPOSITIONS: tuple[PhaseDisposition, ...] = (
    PhaseDisposition(
        phase="E1",
        status="blocked",
        blockers=(
            "No accepted production E1 executor binds answer-only and unified "
            "objective configurations to the canonical update window.",
            "The E1 checkpoint API has not been verified for matched parent, "
            "stream, and checkpoint-fraction lineage.",
        ),
        required_evidence=(
            "production E1 trace for both arms and both seeds with identical "
            "initial model/optimizer/stream identities",
            "objective-weight and disabled-term receipt through the real trainer",
            "25/50/75/100 percent checkpoints with restore validation",
        ),
    ),
    PhaseDisposition(
        phase="E2",
        status="blocked",
        blockers=(
            "No accepted production E2 executor loads frozen E1 parents through "
            "the learned policy, workspace, and planner path.",
            "Model/parent identity and bounded action/call/node accounting are "
            "not verified at the phase boundary.",
        ),
        required_evidence=(
            "frozen E1 checkpoint load receipt bound to every E2 mode",
            "production mini-environment trace using model outputs and real "
            "inquiry/tool budgets",
            "negative control proving the symbolic reference is not the learned arm",
        ),
    ),
    PhaseDisposition(
        phase="E3",
        status="blocked",
        blockers=(
            "No accepted parent-restore executor proves isolated E3 children from "
            "the same E1-B parent.",
            "Replay mixture, protected-task exclusion, and fresh child optimizer "
            "state are not bound to a restorable parent identity.",
        ),
        required_evidence=(
            "parent hash and fresh optimizer receipt for isolated T0/T1 children",
            "protected-task exclusion and actual tool receipt validation for E3",
        ),
    ),
    PhaseDisposition(
        phase="E4",
        status="blocked",
        blockers=(
            "No accepted parent-restore executor proves independent E4 S0/S1 "
            "forks from the same E1-B parent.",
            "Architecture migration, gate state, and treatment ordering are not "
            "bound to restorable parent identities.",
        ),
        required_evidence=(
            "parent hash and fresh optimizer receipt for independent S0/S1 arms",
            "zero-gate equivalence, gate gradients, migration, and restore evidence for E4",
        ),
    ),
    PhaseDisposition(
        phase="E5",
        status="blocked",
        blockers=(
            "The reviewed path still has fixture/archive shortcuts and no accepted "
            "production evidence for a measured proposer update.",
            "A canonical trainer dispatch, equal anchor trials, and P0/P1/P_fixed "
            "lineage with frozen pre-decision counts are not verified.",
        ),
        required_evidence=(
            "production archive pipeline with hashed, explicitly labeled local "
            "test outcomes including successes and failures; learned outcomes "
            "remain an E5 runtime requirement",
            "model-origin proposal transcript bound to checkpoint, archive cutoff, "
            "and parsed method",
            "trainer-double trace proving applied P0 choice, fixed M0 control, and "
            "six fresh confirmation tasks captured before outcomes",
        ),
    ),
    PhaseDisposition(
        phase="E6",
        status="blocked",
        blockers=(
            "No accepted production export proves a complete, restorable K8 result "
            "bundle from all phase outputs.",
        ),
        required_evidence=(
            "export manifest covering source/data/protocol, ledger, failures, "
            "comparisons, transcripts, and required checkpoints",
            "hash verification plus successful fresh-process load of each required "
            "payload",
            "export failure propagated as campaign failure rather than success",
        ),
    ),
)


class ImplementationReadinessError(RuntimeError):
    """Raised when K8 implementation readiness is not explicitly satisfied."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        blocked = ", ".join(self.report.get("blocked_phases", ()))
        super().__init__(f"K8 implementation readiness blocked: {blocked}")


def implementation_readiness(*, source_identity: str | None = None) -> dict[str, Any]:
    """Return the reviewed implementation disposition without side effects.

    ``source_identity`` is recorded as context only.  It cannot make the gate
    ready and an unrelated git/source change does not create a new block.
    """
    dispositions = [item.to_dict() for item in REVIEWED_DISPOSITIONS]
    report: dict[str, Any] = {
        "schema": IMPLEMENTATION_READINESS_SCHEMA,
        "reviewed_source_revision": REVIEWED_SOURCE_REVISION,
        "ready": False,
        "blocked_phases": [item.phase for item in REVIEWED_DISPOSITIONS
                            if item.status == "blocked"],
        "dispositions": dispositions,
    }
    if source_identity is not None:
        report["source_identity"] = source_identity
    return report


def require_implementation_ready(*, source_identity: str | None = None) -> dict[str, Any]:
    """Return readiness or raise a typed fail-closed error."""
    report = implementation_readiness(source_identity=source_identity)
    if not report["ready"]:
        raise ImplementationReadinessError(report)
    return report
