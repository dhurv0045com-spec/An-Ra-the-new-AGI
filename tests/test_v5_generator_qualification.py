"""Generator qualification: shortcut-gated family verdicts (M27-M29)."""

from __future__ import annotations

import unittest

from v5_evaluation.generator_qualification import (
    MAX_SHORTCUT_EXCESS,
    qualify_family,
    qualify_suite,
)


def _cert(families, heuristics):
    return {
        "baselines": {
            "deterministic_random": {"by_family": {f: 0.25 for f in families}},
            **{
                name: {"by_family": acc}
                for name, acc in heuristics.items()
            },
        },
        "suite": {
            "generator_version": "test-gen/0.1",
            "seed": 1,
            "sha256": "a" * 64,
            "family_histogram": {f: 32 for f in families},
        },
    }


class QualificationTests(unittest.TestCase):
    def test_clean_family_qualifies(self) -> None:
        cert = _cert(["binding"], {"bag_of_words": {"binding": 0.30}})
        receipt = qualify_family(
            cert, "binding", generator_id="g", generator_sha256="b" * 64
        )
        self.assertEqual(receipt["verdict"], "GENERATOR_QUALIFIED")
        self.assertAlmostEqual(receipt["worst_excess"], 0.05)

    def test_shortcut_solvable_family_fails(self) -> None:
        cert = _cert(["binding"], {"bag_of_words": {"binding": 1.0}})
        receipt = qualify_family(
            cert, "binding", generator_id="g", generator_sha256="b" * 64
        )
        self.assertEqual(receipt["verdict"], "GENERATOR_NOT_QUALIFIED")

    def test_oracle_and_chance_proxy_never_count(self) -> None:
        cert = _cert(
            ["binding"],
            {
                "bag_of_words": {"binding": 0.20},
                "full_truth_oracle": {"binding": 1.0},
                "direct_retrieval_control": {"binding": 1.0},
            },
        )
        receipt = qualify_family(
            cert, "binding", generator_id="g", generator_sha256="b" * 64
        )
        self.assertEqual(receipt["verdict"], "GENERATOR_QUALIFIED")
        self.assertNotIn("full_truth_oracle", receipt["heuristic_excesses"])

    def test_suite_requires_every_family(self) -> None:
        cert = _cert(
            ["good", "bad"],
            {"bag_of_words": {"good": 0.20, "bad": 1.0}},
        )
        receipt = qualify_suite(
            cert, ["good", "bad"], generator_id="g", generator_sha256="b" * 64
        )
        self.assertEqual(receipt["verdict"], "GENERATOR_NOT_QUALIFIED")
        self.assertEqual(len(receipt["sha256"]), 64)

    def test_missing_family_or_baselines_fail_closed(self) -> None:
        cert = _cert(["binding"], {"bag_of_words": {"binding": 0.20}})
        with self.assertRaises(ValueError):
            qualify_family(cert, "ghost", generator_id="g", generator_sha256="b" * 64)
        with self.assertRaises(ValueError):
            qualify_suite({}, ["binding"], generator_id="g", generator_sha256="b" * 64)
        with self.assertRaises(ValueError):
            qualify_suite(
                _cert(["binding"], {}), ["binding"],
                generator_id="g", generator_sha256="b" * 64,
            )

    def test_real_certificate_binding_rule_induction_qualifies(self) -> None:
        import json

        cert = json.load(open("artifacts/e0/development_certificate.json"))
        receipt = qualify_family(
            cert, "rule_induction", generator_id="e0-eval/0.4.0",
            generator_sha256="c" * 64,
        )
        self.assertEqual(receipt["verdict"], "GENERATOR_QUALIFIED")
        self.assertLessEqual(receipt["worst_excess"], MAX_SHORTCUT_EXCESS)

    def test_real_certificate_binding_flags_lexical_shortcut(self) -> None:
        import json

        cert = json.load(open("artifacts/e0/development_certificate.json"))
        receipt = qualify_family(
            cert, "entity_value_binding", generator_id="e0-eval/0.4.0",
            generator_sha256="c" * 64,
        )
        self.assertEqual(receipt["verdict"], "GENERATOR_NOT_QUALIFIED")
        self.assertGreater(receipt["heuristic_excesses"]["bag_of_words"], 0.5)


if __name__ == "__main__":
    unittest.main()
