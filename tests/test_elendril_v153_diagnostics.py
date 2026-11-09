"""CPU qualification for the V.153 anti-shortcut development benchmark."""
import unittest

from anra_v5.elendril.cognition import CognitiveState
from anra_v5.elendril.v6_data import make_dataset, public_action
from anra_v5.elendril.v71_data import assert_training_disjoint, suite_from_record, suite_to_record
from anra_v5.elendril.v153_diagnostics import audit, build_suite, planned_contrasts


class V153DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = build_suite()

    def test_reproducible_portable_and_bounded(self):
        result = audit(self.suite)
        self.assertEqual((result["cohorts"], result["case_records"], result["planned_contrasts"]), (24, 384, 20))
        self.assertEqual(result["suite_sha256"], audit(build_suite())["suite_sha256"])
        self.assertEqual(
            result["suite_sha256"],
            audit(suite_from_record(suite_to_record(self.suite)))["suite_sha256"],
        )

    def test_same_world_and_semantics_across_hops(self):
        worlds = [c.world_id for c in self.suite["hop-1|original"]]
        for h in range(1, 5):
            cases = self.suite[f"hop-{h}|original"]
            self.assertEqual([c.world_id for c in cases], worlds)
            for c in cases:
                state = CognitiveState(c.goal).observe(c.environment.initial)
                self.assertEqual(public_action(state, (), c.surface), ("ANSWER", c.truth.expected))
                self.assertEqual(len(c.environment.initial), 8)

    def test_removed_essential_edge_requires_abstention(self):
        for h in range(1, 5):
            for a, b in zip(self.suite[f"hop-{h}|original"], self.suite[f"hop-{h}|remove-required-edge"]):
                self.assertEqual((a.world_id, a.goal.subject), (b.world_id, b.goal.subject))
                self.assertEqual(len(b.environment.initial), 7)
                self.assertIsNone(b.truth.expected)
                self.assertEqual(
                    public_action(CognitiveState(b.goal).observe(b.environment.initial), (), b.surface),
                    ("ABSTAIN", None),
                )

    def test_controls_are_paired_and_no_row_copy_shortcut(self):
        result = audit(self.suite)
        self.assertEqual(result["shortcut_hits"], {"last_visible_row": 0, "max_timestamp": 0})
        self.assertEqual(len(planned_contrasts(self.suite)), 20)
        for h in range(1, 5):
            a = self.suite[f"hop-{h}|original"]
            b = self.suite[f"hop-{h}|reversed"]
            c = self.suite[f"hop-{h}|shifted-timestamps"]
            d = self.suite[f"hop-{h}|replace-distractors"]
            for x, y, z, q in zip(a, b, c, d):
                self.assertEqual(x.environment, y.environment)
                self.assertEqual([e.tick + 9000 for e in x.environment.initial],
                                 [e.tick for e in z.environment.initial])
                self.assertEqual([e.subject for e in x.environment.initial],
                                 [e.subject for e in q.environment.initial])
                self.assertEqual(x.truth.expected, q.truth.expected)
                self.assertEqual((x.surface.order, y.surface.order), ("original", "reverse"))

    def test_actual_training_disjointness(self):
        train = make_dataset("train", worlds=32, seed=606)
        self.assertEqual(assert_training_disjoint(train, self.suite)["status"], "PASS")
        self.assertEqual(audit(self.suite, train=train)["training_disjoint"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
