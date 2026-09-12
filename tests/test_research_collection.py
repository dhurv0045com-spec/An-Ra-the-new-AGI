"""B2.1 collection runner tests (no learned computation)."""
import os
import tempfile
import unittest

from bramastra_lab.research.collection.runner import (
    classify_failure,
    collect,
)
from bramastra_lab.research.environments.oracles import failed_baseline_policy
from bramastra_lab.research.environments.worlds import ProgramLab, SwitchWorld
from bramastra_lab.research.experience.ledger import ExperienceLedger


class CollectionTests(unittest.TestCase):
    def test_failed_episode_gets_a_failure_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ExperienceLedger(os.path.join(tmp, "ledger.jsonl"))
            report = collect([ProgramLab(budget=6, seed=3)], ledger,
                             failed_baseline_policy, policy_identity="failed-baseline/v1",
                             episode_prefix="probe")
            self.assertEqual(report["episodes"], 1)
            self.assertEqual(report["successes"], 0)
            self.assertEqual(report["failures_by_category"], {"wrong_world_prediction": 1})
            entries = ledger.read_all()
            receipt = entries[0][1]
            self.assertFalse(receipt.success)
            self.assertEqual(receipt.notes["failure_category"], "wrong_world_prediction")
            self.assertEqual(receipt.notes["budget_unit"], "action")
            self.assertTrue(receipt.notes["transcript"]["steps"])

    def test_transcript_is_public_and_chain_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ExperienceLedger(os.path.join(tmp, "ledger.jsonl"))
            collect([SwitchWorld(budget=6, seed=1)], ledger, failed_baseline_policy,
                    policy_identity="failed-baseline/v1", episode_prefix="probe")
            entries = ledger.read_all()  # verifies the hash chain
            blob = str(entries[0][1].notes["transcript"])
            self.assertNotIn("_switches", blob)
            self.assertNotIn("_holding", blob)


if __name__ == "__main__":
    unittest.main()
