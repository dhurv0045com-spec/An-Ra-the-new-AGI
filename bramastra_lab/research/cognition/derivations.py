"""Verified derivations, abstractions and cognitive experience (M21)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from bramastra_lab.research.contracts.core import content_identity


class DerivationError(ValueError):
    """A derivation or abstraction violated its verification contract."""


@dataclass(frozen=True)
class DerivationStep:
    premises: tuple[str, ...]     # evidence/belief aliases
    operation: str                # declared rule name
    arguments: Mapping[str, Any]
    conclusion: str               # result alias or value reference

    def identity(self) -> str:
        return content_identity({"premises": list(self.premises),
                                 "operation": self.operation,
                                 "arguments": dict(self.arguments),
                                 "conclusion": self.conclusion})


@dataclass
class DerivationChecker:
    """Bounded independent checker over a DECLARED rule table.

    The rule table is an external prior: each rule names its argument schema
    and a pure function. The checker validates local steps without providing
    the final solution; it must never read hidden environment state.
    """

    rules: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Any]]

    def __post_init__(self) -> None:
        for name, function in self.rules.items():
            if not callable(function):
                raise DerivationError(f"rule {name!r} is not callable")

    def check_step(self, step: DerivationStep,
                   known_values: Mapping[str, Any]) -> tuple[bool, Any]:
        rule = self.rules.get(step.operation)
        if rule is None:
            return False, None
        for premise in step.premises:
            if premise not in known_values:
                return False, None  # unknown premise: step cannot be verified
        try:
            value = rule(dict(step.arguments), dict(known_values))
        except Exception:
            return False, None
        return True, value

    def check_derivation(self, steps: Sequence[DerivationStep],
                         known_values: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        values = dict(known_values)
        for step in steps:
            ok, value = self.check_step(step, values)
            if not ok:
                return False, dict(values)
            values[step.conclusion] = value
        return True, values


def _default_rule_table() -> dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Any]]:
    def add(arguments, known):
        return known[arguments["a"]] + known[arguments["b"]]

    def negate(arguments, known):
        return -known[arguments["a"]]

    return {"add": add, "negate": negate}


DEFAULT_CHECKER = DerivationChecker(_default_rule_table())


# --- abstractions ------------------------------------------------------------

@dataclass
class AbstractionRule:
    """A candidate reusable rule proposed from support cases.

    Admission requires verification on validation cases DISTINCT from the
    support cases. Counterexamples restrict preconditions or retract the
    rule; the failure evidence is retained either way.
    """

    rule_id: str
    statement: str
    preconditions: tuple[str, ...]
    proposed_from: tuple[str, ...]        # support case identities
    validated_on: tuple[str, ...] = ()    # distinct validation case identities
    admitted: bool = False
    counterexamples: tuple[str, ...] = ()
    status: str = "proposed"              # proposed | admitted | restricted | retracted

    def __post_init__(self) -> None:
        overlap = set(self.proposed_from) & set(self.validated_on)
        if overlap:
            raise DerivationError(
                "rule validation cases must be distinct from the support cases "
                f"that proposed it: {sorted(overlap)[:3]}")
        if self.admitted and self.status == "proposed":
            self.status = "admitted"

    def identity(self) -> str:
        return content_identity({"rule_id": self.rule_id, "statement": self.statement,
                                 "preconditions": list(self.preconditions),
                                 "proposed_from": list(self.proposed_from),
                                 "validated_on": list(self.validated_on)})


class AbstractionArchive:
    """Storage with source/counterexample links. No examiner labels enter."""

    def __init__(self) -> None:
        self.rules: dict[str, AbstractionRule] = {}

    def propose(self, rule_id: str, statement: str, preconditions: Sequence[str],
                support_case_ids: Sequence[str]) -> AbstractionRule:
        rule = AbstractionRule(rule_id=rule_id, statement=statement,
                               preconditions=tuple(preconditions),
                               proposed_from=tuple(support_case_ids))
        self.rules[rule_id] = rule
        return rule

    def admit(self, rule_id: str, validation_case_ids: Sequence[str]) -> AbstractionRule:
        rule = self.rules.get(rule_id)
        if rule is None:
            raise DerivationError(f"unknown rule {rule_id!r}")
        overlap = set(rule.proposed_from) & set(validation_case_ids)
        if overlap:
            raise DerivationError(
                "validation cases must be distinct from support cases: "
                f"{sorted(overlap)[:3]}")
        rule.validated_on = tuple(validation_case_ids)
        if not validation_case_ids:
            rule.status = "proposed"
            rule.admitted = False
            return rule
        rule.admitted = True
        rule.status = "admitted"
        return rule

    def add_counterexample(self, rule_id: str, case_id: str, *,
                           restrict: str | None = None) -> AbstractionRule:
        rule = self.rules.get(rule_id)
        if rule is None:
            raise DerivationError(f"unknown rule {rule_id!r}")
        rule.counterexamples = tuple(rule.counterexamples) + (case_id,)
        if restrict:
            rule.preconditions = tuple(rule.preconditions) + (restrict,)
            rule.status = "restricted"
        else:
            rule.admitted = False
            rule.status = "retracted"
        return rule
