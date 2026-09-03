"""Conjunctive worst-family promotion gates from the frozen spec.

Every gate must pass; aggregates never compensate a failing family.

Evidence rules (fail closed):
- The dossier must contain every required entry (``REQUIRED_DOSSIER_ENTRIES``);
  an incomplete dossier fails every gate -- missing evidence is never PASS.
- Count-backed gates consume Wilson 95% lower confidence bounds where the
  frozen training spec names an LCB threshold (fresh selection, sensitivity
  flip, invariance stability, three-hop, M102 paired delta). Point-estimate
  thresholds in the frozen spec (state OOD, two-hop, conditional realization,
  missing-information balanced accuracy and false assertion, loss
  regressions) stay point estimates; the frozen spec, not this module,
  decides which.
- An empty probe set can never pass: sensitivity and invariance gates fail
  when no family declares the probe, and malformed counts fail the gate
  instead of crashing the promotion process.
"""

from __future__ import annotations

from typing import Mapping

from v5_evaluation.metrics import wilson_lcb


GATE_SCHEMA = "anra-v5-promotion-gates/v1"

THRESHOLDS = {
    "fresh_selection_lcb_above_chance": 0.10,
    "sensitivity_flip_lcb": 0.80,
    "invariance_stable_lcb": 0.90,
    "state_ood_accuracy": 0.70,
    "two_hop_ood_accuracy": 0.60,
    "three_hop_lcb_above_chance": 0.10,
    "three_hop_max_degradation_from_two_hop": 0.20,
    "conditional_realization": 0.80,
    "missing_information_balanced_accuracy": 0.80,
    "missing_information_false_assertion_max": 0.10,
    "natural_substrate_loss_regression_max": 0.03,
    "code_math_loss_regression_max": 0.05,
    "maximum_family_regression": 0.05,
    "m102_replication_seeds": 2,
    "fresh_natural_paired_lcb_delta_minimum": 0.0,
}

# Dossier entries the frozen gates consume. A promotion dossier missing any
# of these fails closed: no gate may return PASS on absent evidence.
REQUIRED_DOSSIER_ENTRIES = (
    "semantic_state",
    "relational_composition",
    "missing_information",
    "faithful_realization",
    "substrate",
    "m102_replication",
)

GATE_NAMES = (
    "fresh_selection_above_chance",
    "sensitivity_flip",
    "invariance_stable",
    "state_ood",
    "two_hop",
    "three_hop",
    "missing_information",
    "conditional_realization",
    "substrate_retention",
    "m102_replication",
)


def _failed_gates() -> dict[str, bool]:
    return {name: False for name in GATE_NAMES}


def _require_counts(name: str, mapping: Mapping[str, object]) -> tuple[int, int] | None:
    """Return (correct, total) when valid, else None (missing evidence)."""

    if not isinstance(mapping, Mapping):
        return None
    try:
        correct, total = mapping["correct"], mapping["total"]
    except (KeyError, TypeError):
        return None
    if not isinstance(correct, int) or not isinstance(total, int):
        return None
    if total <= 0 or correct < 0 or correct > total:
        return None
    return correct, total


def _probe_gate(families: Mapping[str, Mapping[str, object]], probe: str, threshold: float) -> bool:
    """Every family declaring the probe must clear the LCB; none may vacuously pass."""

    declaring = [
        (name, _require_counts(name, families[name][probe]))
        for name in families
        if isinstance(families[name], Mapping) and probe in families[name]
    ]
    if not declaring:
        return False
    return all(
        counts is not None and wilson_lcb(*counts) >= threshold
        for _, counts in declaring
    )


def evaluate_gates(families: Mapping[str, Mapping[str, object]]) -> dict[str, bool]:
    """Evaluate every gate over per-family evidence; return gate verdicts."""

    incomplete = [
        name
        for name in REQUIRED_DOSSIER_ENTRIES
        if not isinstance(families.get(name), Mapping)
    ]
    if incomplete:
        return _failed_gates()

    gates: dict[str, bool] = {}
    selection_pairs = [
        (name, _require_counts(name, families[name].get("selection")))
        for name in families
        if isinstance(families[name], Mapping)
    ]
    chances = [
        (name, families[name].get("chance"))
        for name in families
        if isinstance(families[name], Mapping)
    ]
    selection_valid = all(counts is not None for _, counts in selection_pairs) and bool(
        selection_pairs
    )
    chances_valid = all(
        isinstance(chance, (int, float)) and 0.0 <= float(chance) <= 1.0
        for _, chance in chances
    ) and bool(chances)
    gates["fresh_selection_above_chance"] = (
        selection_valid
        and chances_valid
        and min(
            wilson_lcb(*counts) - float(chance)
            for (_, counts), (_, chance) in zip(selection_pairs, chances)
        )
        >= THRESHOLDS["fresh_selection_lcb_above_chance"]
    )
    gates["sensitivity_flip"] = _probe_gate(
        families, "sensitivity", THRESHOLDS["sensitivity_flip_lcb"]
    )
    gates["invariance_stable"] = _probe_gate(
        families, "invariance", THRESHOLDS["invariance_stable_lcb"]
    )
    state = families["semantic_state"]
    ood_accuracy = state.get("ood_accuracy")
    gates["state_ood"] = (
        isinstance(ood_accuracy, (int, float))
        and float(ood_accuracy) >= THRESHOLDS["state_ood_accuracy"]
    )
    composition = families["relational_composition"]
    two_hop = composition.get("two_hop_accuracy")
    gates["two_hop"] = (
        isinstance(two_hop, (int, float)) and float(two_hop) >= THRESHOLDS["two_hop_ood_accuracy"]
    )
    three_hop = _require_counts("three_hop", composition.get("three_hop"))
    chance = composition.get("chance")
    gates["three_hop"] = bool(
        three_hop is not None
        and isinstance(chance, (int, float))
        and wilson_lcb(*three_hop) - float(chance) >= THRESHOLDS["three_hop_lcb_above_chance"]
        and isinstance(two_hop, (int, float))
        and (float(two_hop) - wilson_lcb(*three_hop))
        <= THRESHOLDS["three_hop_max_degradation_from_two_hop"]
    )
    missing = families["missing_information"]
    balanced = missing.get("balanced_accuracy")
    false_assertion = missing.get("false_assertion")
    gates["missing_information"] = bool(
        isinstance(balanced, (int, float))
        and isinstance(false_assertion, (int, float))
        and float(balanced) >= THRESHOLDS["missing_information_balanced_accuracy"]
        and float(false_assertion) <= THRESHOLDS["missing_information_false_assertion_max"]
    )
    realization = families["faithful_realization"]
    conditional = realization.get("conditional_accuracy")
    gates["conditional_realization"] = bool(
        isinstance(conditional, (int, float))
        and float(conditional) >= THRESHOLDS["conditional_realization"]
    )
    substrate = families["substrate"]
    natural = substrate.get("natural_loss_regression")
    code_math = substrate.get("code_math_loss_regression")
    worst = substrate.get("worst_family_regression")
    gates["substrate_retention"] = bool(
        all(isinstance(value, (int, float)) for value in (natural, code_math, worst))
        and float(natural) <= THRESHOLDS["natural_substrate_loss_regression_max"]
        and float(code_math) <= THRESHOLDS["code_math_loss_regression_max"]
        and float(worst) <= THRESHOLDS["maximum_family_regression"]
    )
    scale = families["m102_replication"]
    seeds = scale.get("seeds")
    paired = scale.get("fresh_natural_paired_lcb")
    gates["m102_replication"] = bool(
        isinstance(seeds, int)
        and seeds >= THRESHOLDS["m102_replication_seeds"]
        and isinstance(paired, (int, float))
        and float(paired) > THRESHOLDS["fresh_natural_paired_lcb_delta_minimum"]
    )
    return gates


def all_pass(gates: Mapping[str, bool]) -> bool:
    """Conjunctive promotion: every gate must pass."""

    if set(gates) != set(GATE_NAMES):
        raise ValueError("gate inventory does not match the frozen gate set")
    return all(gates.values())


__all__ = [
    "GATE_NAMES",
    "GATE_SCHEMA",
    "REQUIRED_DOSSIER_ENTRIES",
    "THRESHOLDS",
    "all_pass",
    "evaluate_gates",
]
