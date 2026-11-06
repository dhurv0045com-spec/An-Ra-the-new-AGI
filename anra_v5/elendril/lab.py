"""Bounded intervention and independent promotion-review contracts.

The verifier authenticates receipts with a key held outside the candidate process.
The module validates evidence for review and never mutates a canonical pointer.
"""

from dataclasses import asdict, dataclass
import hmac
import re

from .codec import canonical, digest


SHA256 = re.compile(r"[0-9a-f]{64}")
INTERVENTIONS = frozenset({"memory", "weights", "architecture"})


def _hashes(*identities: str) -> None:
    if any(not isinstance(value, str) or SHA256.fullmatch(value) is None for value in identities):
        raise ValueError("manifest identities must be lowercase SHA-256 values")


@dataclass(frozen=True, slots=True)
class Budget:
    training_tokens: int
    inference_positions: int
    wall_ms: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.counters()):
            raise ValueError("work counters must be nonnegative integers")

    def counters(self) -> tuple[int, int, int]:
        return self.training_tokens, self.inference_positions, self.wall_ms

    def covers(self, used: "Budget") -> bool:
        return all(actual <= cap for cap, actual in zip(self.counters(), used.counters()))


@dataclass(frozen=True, slots=True)
class Proposal:
    candidate_actor: str
    intervention: str
    parent_sha256: str
    recipe_sha256: str
    policy_sha256: str
    budget: Budget

    def __post_init__(self) -> None:
        _hashes(self.parent_sha256, self.recipe_sha256, self.policy_sha256)
        if not self.candidate_actor or self.intervention not in INTERVENTIONS:
            raise ValueError("proposal requires an actor and a supported intervention")

    def sha256(self) -> str:
        return digest(["elendril-proposal-v1", asdict(self)])


@dataclass(frozen=True, slots=True)
class Verification:
    verifier: str
    proposal_sha256: str
    child_sha256: str
    policy_sha256: str
    evidence_sha256: str
    work: Budget
    capability_pass: bool
    retention_pass: bool
    confirmation_pass: bool

    def __post_init__(self) -> None:
        _hashes(self.proposal_sha256, self.child_sha256, self.policy_sha256, self.evidence_sha256)
        if not self.verifier or any(type(gate) is not bool for gate in self.gates()):
            raise ValueError("verification requires a verifier and explicit gate decisions")

    def gates(self) -> tuple[bool, bool, bool]:
        return self.capability_pass, self.retention_pass, self.confirmation_pass

    def payload(self) -> bytes:
        return canonical(["elendril-verification-v1", asdict(self)])


def review(
    proposal: Proposal,
    verification: Verification,
    signature: str,
    *,
    trusted_verifier: str,
    verifier_key: bytes,
) -> dict:
    """Return an owner review record binding the immutable parent as rollback target."""
    if len(verifier_key) < 32:
        raise ValueError("a trusted verifier key of at least 32 bytes is required")
    if verification.verifier != trusted_verifier or trusted_verifier == proposal.candidate_actor:
        raise ValueError("verification must be issued by a distinct trusted actor")
    expected = hmac.digest(verifier_key, verification.payload(), "sha256").hex()
    if not isinstance(signature, str) or SHA256.fullmatch(signature) is None or not hmac.compare_digest(expected, signature):
        raise ValueError("verification authentication failed")
    if (verification.proposal_sha256 != proposal.sha256()
            or verification.policy_sha256 != proposal.policy_sha256
            or verification.child_sha256 == proposal.parent_sha256):
        raise ValueError("verification does not bind the proposed intervention")
    eligible = proposal.budget.covers(verification.work) and all(verification.gates())
    return {"schema": "elendril-owner-review-v1", "status": "eligible" if eligible else "rejected",
            "proposal_sha256": proposal.sha256(), "evidence_sha256": verification.evidence_sha256,
            "candidate_sha256": verification.child_sha256, "rollback_sha256": proposal.parent_sha256}
