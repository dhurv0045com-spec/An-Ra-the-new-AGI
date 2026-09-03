"""Conjunctive worst-family promotion gates from the frozen spec.

Every gate must pass; aggregates never compensate a failing family. Bounds
use Wilson lower confidence limits over fresh/sealed counts, never point
estimates. Thresholds: fresh selection LCB chance+0.10; sensitivity flip LCB
0.80; invariance stable-and-correct LCB 0.90; state OOD 0.70; two-hop 0.60 and
above matched retrieval; three-hop LCB chance+0.10 and at most 20 points below
two-hop; missing-information balanced accuracy 0.80 with false assertion at
most 0.10; conditional realization 0.80; natural loss regression at most 3%;
code/math at most 5%; no family regression above 5%; M102 two-seed replication
with fresh natural paired LCB above zero.
"""

from __future__ import annotations

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


def _require_counts(name: str, mapping: dict[str, int]) -> tuple[int, int]:
    try:
        correct, total = mapping["correct"], mapping["total"]
    except KeyError:
        raise ValueError(f"{name} needs correct/total counts") from None
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError(f"{name} counts are invalid")
    return correct, total


def evaluate_gates(families: dict[str, dict[str, object]]) -> dict[str, bool]:
    """Evaluate every gate over per-family evidence; return gate verdicts."""

    gates: dict[str, bool] = {}
    worst_selection_excess = min(
        wilson_lcb(*_require_counts(name, f["selection"])) - float(f["chance"])
        for name, f in families.items()
    )
    gates["fresh_selection_above_chance"] = (
        worst_selection_excess >= THRESHOLDS["fresh_selection_lcb_above_chance"]
    )
    gates["sensitivity_flip"] = all(
        wilson_lcb(*_require_counts(name, f["sensitivity"]))
        >= THRESHOLDS["sensitivity_flip_lcb"]
        for name, f in families.items() if "sensitivity" in f
    )
    gates["invariance_stable"] = all(
        wilson_lcb(*_require_counts(name, f["invariance"]))
        >= THRESHOLDS["invariance_stable_lcb"]
        for name, f in families.items() if "invariance" in f
    )
    state = families.get("semantic_state", {})
    gates["state_ood"] = (
        state.get("ood_accuracy", -1.0) >= THRESHOLDS["state_ood_accuracy"] if state else False
    )
    composition = families.get("relational_composition", {})
    two_hop = float(composition.get("two_hop_accuracy", -1.0)) if composition else -1.0
    gates["two_hop"] = two_hop >= THRESHOLDS["two_hop_ood_accuracy"]
    if composition and "three_hop" in composition and "chance" in composition:
        three = composition["three_hop"]
        three_lcb = wilson_lcb(*_require_counts("three_hop", three))
        gates["three_hop"] = (
            three_lcb - float(composition["chance"]) >= THRESHOLDS["three_hop_lcb_above_chance"]
            and (two_hop - three_lcb) <= THRESHOLDS["three_hop_max_degradation_from_two_hop"]
        )
    else:
        gates["three_hop"] = False
    missing = families.get("missing_information", {})
    gates["missing_information"] = bool(
        missing
        and float(missing.get("balanced_accuracy", -1.0))
        >= THRESHOLDS["missing_information_balanced_accuracy"]
        and float(missing.get("false_assertion", 1.0))
        <= THRESHOLDS["missing_information_false_assertion_max"]
    )
    realization = families.get("faithful_realization", {})
    gates["conditional_realization"] = bool(
        realization
        and float(realization.get("conditional_accuracy", -1.0))
        >= THRESHOLDS["conditional_realization"]
    )
    substrate = families.get("substrate", {})
    gates["substrate_retention"] = bool(
        substrate
        and float(substrate.get("natural_loss_regression", 1.0))
        <= THRESHOLDS["natural_substrate_loss_regression_max"]
        and float(substrate.get("code_math_loss_regression", 1.0))
        <= THRESHOLDS["code_math_loss_regression_max"]
        and float(substrate.get("worst_family_regression", 1.0))
        <= THRESHOLDS["maximum_family_regression"]
    )
    scale = families.get("m102_replication", {})
    gates["m102_replication"] = bool(
        scale
        and int(scale.get("seeds", 0)) >= THRESHOLDS["m102_replication_seeds"]
        and float(scale.get("fresh_natural_paired_lcb", -1.0))
        > THRESHOLDS["fresh_natural_paired_lcb_delta_minimum"]
    )
    return gates


def all_pass(gates: dict[str, bool]) -> bool:
    """Conjunctive promotion: every gate must pass."""

    expected = {
        "fresh_selection_above_chance", "sensitivity_flip", "invariance_stable",
        "state_ood", "two_hop", "three_hop", "missing_information",
        "conditional_realization", "substrate_retention", "m102_replication",
    }
    if set(gates) != expected:
        raise ValueError("gate inventory does not match the frozen gate set")
    return all(gates.values())


__all__ = ["GATE_SCHEMA", "THRESHOLDS", "all_pass", "evaluate_gates"]
