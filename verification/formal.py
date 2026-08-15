"""M6 deterministic certificate verifier for small propositional proof chains."""

from __future__ import annotations

import re

from verification.builtins import BuiltinVerificationResult
from verification.registry import VerifierRequest, register_verifier

_ATOM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _atom(value: object) -> str | None:
    candidate = str(value).strip()
    return candidate if _ATOM.fullmatch(candidate) else None


def _rule(value: object) -> tuple[str, str] | None:
    parts = str(value).split("->")
    if len(parts) != 2:
        return None
    left, right = (_atom(part) for part in parts)
    return (left, right) if left is not None and right is not None else None


@register_verifier("formal_proof")
def verify_formal_proof(request: VerifierRequest) -> BuiltinVerificationResult:
    """Verify an explicit finite modus-ponens certificate.

    Facts may only enter through ``premises``. Implications may only enter
    through ``rules``. Each proof step must either repeat a known fact or be
    the consequent of a supplied rule whose antecedent is already known.
    """
    premise_rows = list(request.payload.get("premises", ()))
    rule_rows = list(request.payload.get("rules", ()))
    step_rows = list(request.payload.get("steps", ()))
    premises = [_atom(item) for item in premise_rows]
    rules = [_rule(item) for item in rule_rows]
    steps = [_atom(item) for item in step_rows]
    conclusion = _atom(request.payload.get("conclusion", ""))
    malformed = (
        conclusion is None
        or any(item is None for item in premises)
        or any(item is None for item in rules)
        or any(item is None for item in steps)
    )
    if malformed:
        return BuiltinVerificationResult(0.0, 1, "formal_certificate_malformed")

    known = set(premises)
    valid = True
    for step in steps:
        assert step is not None
        derivable = step in known or any(
            left in known and right == step for left, right in rules if left and right
        )
        if not derivable:
            valid = False
            break
        known.add(step)
    passed = valid and conclusion in known
    return BuiltinVerificationResult(
        1.0 if passed else 0.0,
        1,
        "formal_chain_valid" if passed else "formal_chain_invalid",
    )
