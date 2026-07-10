"""M7 proposal-only authority ladder; it cannot write or merge repository changes."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SelfDevelopmentProposal:
    proposal_id: str
    scope: str
    predicted_delta: float
    tests_passed: bool
    human_approved: bool = False


def evaluate_proposal_only(proposal: SelfDevelopmentProposal) -> dict[str, object]:
    allowed_scopes = {"docs", "tests", "eval_additions", "verifier"}
    eligible = (
        proposal.scope in allowed_scopes
        and proposal.tests_passed
        and proposal.predicted_delta > 0
    )
    return {
        "proposal": asdict(proposal),
        "eligible_for_human_review": eligible,
        "auto_apply": False,
        "merged": False,
    }
