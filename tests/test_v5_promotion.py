from __future__ import annotations

import unittest

from v5_promotion.decide import PromotionDecision, decide
from v5_promotion.gates import all_pass, evaluate_gates


def _strong_families() -> dict:
    return {
        "binding": {
            "selection": {"correct": 460, "total": 512},
            "chance": 0.25,
            "sensitivity": {"correct": 420, "total": 480},
            "invariance": {"correct": 470, "total": 480},
        },
        "semantic_state": {
            "selection": {"correct": 450, "total": 512},
            "chance": 0.25,
            "ood_accuracy": 0.78,
        },
        "relational_composition": {
            "selection": {"correct": 430, "total": 512},
            "chance": 0.25,
            "two_hop_accuracy": 0.68,
            "three_hop": {"correct": 300, "total": 512},
        },
        "missing_information": {
            "selection": {"correct": 440, "total": 512},
            "chance": 0.5,
            "balanced_accuracy": 0.85,
            "false_assertion": 0.05,
        },
        "faithful_realization": {
            "selection": {"correct": 440, "total": 512},
            "chance": 0.25,
            "conditional_accuracy": 0.86,
        },
        "substrate": {
            "selection": {"correct": 440, "total": 512},
            "chance": 0.25,
            "natural_loss_regression": 0.01,
            "code_math_loss_regression": 0.02,
            "worst_family_regression": 0.03,
        },
        "m102_replication": {
            "selection": {"correct": 440, "total": 512},
            "chance": 0.25,
            "seeds": 2,
            "fresh_natural_paired_lcb": 0.04,
        },
    }


def _decision(**overrides) -> PromotionDecision:
    fields: dict[str, object] = {
        "schema": "anra-v5-promotion-decision/v2",
        "checkpoint_sha256": "a" * 64,
        "evaluation_receipt_sha256": "b" * 64,
        "durability_receipt_sha256": "c" * 64,
        "gate_spec_sha256": "d" * 64,
        "passed_gates": ("g1",),
        "failed_gates": (),
        "signer_id": "promotion-board",
        "detached_signature_sha256": "e" * 64,
    }
    fields.update(overrides)
    return PromotionDecision(**fields)  # type: ignore[arg-type]


class GateTests(unittest.TestCase):
    def test_strong_recipe_passes_every_gate(self) -> None:
        gates = evaluate_gates(_strong_families())
        self.assertTrue(all_pass(gates))

    def test_weak_family_fails_conjunctively(self) -> None:
        families = _strong_families()
        families["semantic_state"] = dict(families["semantic_state"], ood_accuracy=0.5)
        gates = evaluate_gates(families)
        self.assertFalse(gates["state_ood"])
        self.assertFalse(all_pass(gates))

    def test_gate_inventory_is_closed(self) -> None:
        with self.assertRaises(ValueError):
            all_pass({"only_one": True})


class DecideTests(unittest.TestCase):
    def test_promote_requires_gates_and_signature(self) -> None:
        decision = _decision()
        self.assertEqual(decide(decision, verifier=lambda _d, _s: True), "PROMOTE")
        self.assertEqual(decide(decision, verifier=None), "INCONCLUSIVE")
        self.assertEqual(decide(decision, verifier=lambda _d, _s: False), "INCONCLUSIVE")

    def test_failures_reject(self) -> None:
        decision = _decision(passed_gates=(), failed_gates=("state_ood",))
        self.assertEqual(decide(decision, verifier=lambda _d, _s: True), "REJECT")
        empty = _decision(passed_gates=(), failed_gates=())
        self.assertEqual(decide(empty, verifier=lambda _d, _s: True), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
