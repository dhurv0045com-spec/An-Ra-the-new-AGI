from __future__ import annotations

import unittest

from v5_evaluation.adapter import ModelAdapter
from v5_evaluation.metrics import (
    conditional_realization,
    loss_regression,
    wilson_lcb,
)
from v5_evaluation.receipt import EvaluationReceipt


class AdapterTests(unittest.TestCase):
    def _adapter(self, scores=(0.1, 0.9)):
        return ModelAdapter(
            adapter_id="test-adapter",
            checkpoint_sha256="a" * 64,
            score_candidates=lambda _c, _q, cands: list(scores[: len(cands)]),
            generate_free=lambda _p, _n: "answer",
            generate_constrained=lambda _p, cands: cands[0],
        )

    def test_scores_are_validated_per_candidate(self) -> None:
        adapter = self._adapter()
        self.assertEqual(adapter.score_candidates("ctx", "q", ["x", "y"]), [0.1, 0.9])
        with self.assertRaises(ValueError):
            adapter.score_candidates("ctx", "q", [])
        with self.assertRaises(ValueError):
            adapter.generate_free("prompt", 0)
        with self.assertRaises(ValueError):
            adapter.generate_free("prompt", 65)
        self.assertEqual(len(adapter.identity_sha256), 64)

    def test_nonfinite_scores_rejected(self) -> None:
        adapter = ModelAdapter(
            adapter_id="bad",
            checkpoint_sha256="a" * 64,
            score_candidates=lambda _c, _q, _k: [float("inf")],
            generate_free=lambda _p, _n: "",
            generate_constrained=lambda _p, c: c[0],
        )
        with self.assertRaises(ValueError):
            adapter.score_candidates("ctx", "q", ["x"])


class MetricsTests(unittest.TestCase):
    def test_wilson_lcb_known_values(self) -> None:
        self.assertAlmostEqual(wilson_lcb(0, 10), 0.0, places=6)
        bound = wilson_lcb(95, 100)
        self.assertGreater(bound, 0.88)
        self.assertLess(bound, 0.95)
        with self.assertRaises(ValueError):
            wilson_lcb(11, 10)
        with self.assertRaises(ValueError):
            wilson_lcb(0, 0)

    def test_conditional_realization_floor(self) -> None:
        with self.assertRaises(ValueError):
            conditional_realization(90, 99)
        self.assertAlmostEqual(conditional_realization(80, 100), 0.8)

    def test_loss_regression(self) -> None:
        self.assertAlmostEqual(loss_regression(2.0, 2.06), 0.03)
        with self.assertRaises(ValueError):
            loss_regression(0.0, 1.0)


class ReceiptTests(unittest.TestCase):
    def test_receipt_binds_identities(self) -> None:
        receipt = EvaluationReceipt(
            schema="anra-v5-evaluation-receipt/v1",
            checkpoint_sha256="a" * 64,
            adapter_sha256="b" * 64,
            tokenizer_sha256="c" * 64,
            protocol_sha256="d" * 64,
            raw_metrics_sha256="e" * 64,
            assisted_metrics_sha256="f" * 64,
            substrate_metrics_sha256="0" * 64,
            tier="tier1",
            native_selection={"correct": 10, "total": 16},
        )
        self.assertEqual(len(receipt.sha256()), 64)
        with self.assertRaises(ValueError):
            EvaluationReceipt(
                schema="anra-v5-evaluation-receipt/v1",
                checkpoint_sha256="zzz",
                adapter_sha256="b" * 64,
                tokenizer_sha256="c" * 64,
                protocol_sha256="d" * 64,
                raw_metrics_sha256="e" * 64,
                assisted_metrics_sha256="f" * 64,
                substrate_metrics_sha256="0" * 64,
                tier="tier1",
                native_selection={"correct": 10, "total": 16},
            ).assert_valid()


if __name__ == "__main__":
    unittest.main()
