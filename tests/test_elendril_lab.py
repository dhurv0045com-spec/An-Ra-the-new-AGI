from dataclasses import replace
import hmac

import pytest

from anra_v5.elendril.lab import Budget, Proposal, Verification, review


def test_review_authenticates_independent_binding_and_enforces_budget():
    key = b"test-only-verifier-key".ljust(32, b"0")
    proposal = Proposal("learner", "weights", "a"*64, "b"*64, "c"*64, Budget(32, 48, 60))
    receipt = Verification("verifier", proposal.sha256(), "d"*64, "c"*64, "e"*64,
                           Budget(31, 48, 59), True, True, True)

    def decide(record):
        signature = hmac.digest(key, record.payload(), "sha256").hex()
        return review(proposal, record, signature, trusted_verifier="verifier", verifier_key=key)

    assert decide(receipt)["rollback_sha256"] == "a"*64
    assert decide(receipt)["status"] == "eligible"
    assert decide(replace(receipt, work=Budget(33, 48, 59)))["status"] == "rejected"
    assert decide(replace(receipt, retention_pass=False))["status"] == "rejected"
    with pytest.raises(ValueError, match="distinct"):
        decide(replace(receipt, verifier="learner"))
    with pytest.raises(ValueError, match="does not bind"):
        decide(replace(receipt, policy_sha256="f"*64))
    with pytest.raises(ValueError, match="authentication failed"):
        review(proposal, receipt, "0"*64, trusted_verifier="verifier", verifier_key=key)
