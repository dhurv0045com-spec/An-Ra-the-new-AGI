"""Candidate improvement transactions (M12/A8): propose -> construct child ->
evaluate -> validate -> record ACCEPT/REJECT/INCONCLUSIVE with crash-safe
publication. Fixture transactions use a separate registry namespace and can
never write the learned accepted-parent pointer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from bramastra_lab.research.contracts.core import content_identity

TRANSACTION_STATUSES = frozenset({"proposed", "child_constructed", "evaluated",
                                  "accepted", "rejected", "inconclusive", "crashed"})
CHANGE_KINDS = frozenset({"weights", "memory", "task_selection", "method", "config"})
EVIDENCE_CLASSES = frozenset({"fixture", "random_model", "learned"})


class TransactionError(ValueError):
    """A candidate transaction violated its protocol."""


@dataclass
class CandidateTransaction:
    transaction_id: str
    parent_identity: str
    child_identity: str | None
    change_kind: str
    evidence_class: str
    execution_mode: str
    learned_updates: int
    comparison_identity: str | None = None
    chief_approval_hash: str | None = None
    status: str = "proposed"
    retries: int = 0
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in TRANSACTION_STATUSES:
            raise TransactionError(f"unknown status {self.status!r}")
        if self.change_kind not in CHANGE_KINDS:
            raise TransactionError(
                f"change kind must be one of {sorted(CHANGE_KINDS)}")
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise TransactionError(
                f"evidence class must be one of {sorted(EVIDENCE_CLASSES)}")
        if self.evidence_class == "learned" and self.learned_updates <= 0 \
                and self.change_kind == "weights":
            raise TransactionError(
                "a weight-improvement claim requires actual recorded child "
                "updates; zero-update changes need their own change kind")

    def identity(self) -> str:
        return content_identity({
            "transaction_id": self.transaction_id,
            "parent": self.parent_identity, "child": self.child_identity,
            "change_kind": self.change_kind, "evidence_class": self.evidence_class,
            "execution_mode": self.execution_mode,
            "learned_updates": self.learned_updates,
            "comparison": self.comparison_identity,
            "approval": self.chief_approval_hash,
            "status": self.status, "evidence": list(self.evidence_ids),
        })


class CandidateRegistry:
    """Crash-safe state machine with idempotent retries. Fixture and learned
    registries are disjoint namespaces."""

    def __init__(self, *, namespace: str = "fixture") -> None:
        if namespace not in ("fixture", "learned"):
            raise TransactionError("namespace must be 'fixture' or 'learned'")
        self.namespace = namespace
        self.transactions: dict[str, CandidateTransaction] = {}
        self.accepted_parent: str | None = None

    def record(self, transaction: CandidateTransaction) -> CandidateTransaction:
        existing = self.transactions.get(transaction.transaction_id)
        if existing is not None:
            if existing.identity() != transaction.identity():
                raise TransactionError(
                    "transaction id reuse with different content; retries must "
                    "be idempotent")
            return existing
        self.transactions[transaction.transaction_id] = transaction
        return transaction

    def _get(self, transaction_id: str) -> CandidateTransaction:
        transaction = self.transactions.get(transaction_id)
        if transaction is None:
            raise TransactionError(f"unknown transaction {transaction_id!r}")
        return transaction

    def construct_child(self, transaction_id: str, child_identity: str) -> CandidateTransaction:
        transaction = self._get(transaction_id)
        if transaction.status != "proposed":
            raise TransactionError(
                f"child construction requires status 'proposed', got "
                f"{transaction.status!r}")
        transaction.child_identity = child_identity
        transaction.status = "child_constructed"
        return self.record(transaction)

    def record_comparison(self, transaction_id: str, comparison_identity: str,
                          learned_updates: int | None = None) -> CandidateTransaction:
        transaction = self._get(transaction_id)
        if transaction.comparison_identity is not None:
            # Idempotent retry of the SAME comparison is allowed at any later
            # status; changing the preregistered identity is not.
            if transaction.comparison_identity != comparison_identity:
                raise TransactionError(
                    "a comparison cannot change its own preregistered identity")
            transaction.retries += 1
            return self.record(transaction)
        if transaction.status != "child_constructed":
            raise TransactionError("comparison requires a constructed child")
        transaction.comparison_identity = comparison_identity
        if learned_updates is not None:
            transaction.learned_updates = learned_updates
        transaction.status = "evaluated"
        return self.record(transaction)

    def decide(self, transaction_id: str, decision: str, *,
               chief_approval_hash: str | None = None) -> CandidateTransaction:
        transaction = self._get(transaction_id)
        if transaction.status != "evaluated":
            raise TransactionError(
                f"decision requires status 'evaluated', got {transaction.status!r}")
        if decision not in ("accepted", "rejected", "inconclusive"):
            raise TransactionError("decision must be accepted/rejected/inconclusive")
        if decision == "accepted":
            if transaction.evidence_class == "learned":
                if chief_approval_hash is None:
                    raise TransactionError(
                        "a learned candidate publication requires chief approval "
                        "bound to the evidence hashes")
                if transaction.change_kind == "weights" and transaction.learned_updates <= 0:
                    raise TransactionError(
                        "zero-update weight-learning claims are rejected by the "
                        "publication API itself")
            transaction.chief_approval_hash = chief_approval_hash
            transaction.status = "accepted"
            if self.namespace == "learned":
                self.accepted_parent = transaction.child_identity
        else:
            transaction.status = decision
        return self.record(transaction)

    def crash(self, transaction_id: str) -> CandidateTransaction:
        transaction = self._get(transaction_id)
        transaction.status = "crashed"
        return self.record(transaction)
