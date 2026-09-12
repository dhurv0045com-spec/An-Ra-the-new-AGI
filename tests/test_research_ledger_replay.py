"""B07 focused tests: append-only ledger and deterministic stratified replay."""
import os
import tempfile
import unittest

from bramastra_lab.research.experience.ledger import (
    EpisodeReceipt,
    ExperienceLedger,
    LedgerError,
    LedgerTamperedError,
)
from bramastra_lab.research.experience.replay import (
    ReplayEngine,
    ReplayStateError,
)


def receipt(index: int, family: str = "arithmetic", quality: str = "accepted",
            policy: str = "collector-v1", costs=(1.0,), success=True) -> EpisodeReceipt:
    return EpisodeReceipt(
        episode_id=f"ep-{index}", task_semantic_id=f"task-{index}", family=family,
        collection_policy=policy, success=success, quality=quality,
        quality_reason=None if quality == "accepted" else "ambiguous_outcome",
        transition_costs=costs, episode_content_identity=f"content-{index}",
        recorded_at_unix=1000.0 + index, notes={})


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ledger.jsonl")

    def test_append_and_read_round_trip(self) -> None:
        ledger = ExperienceLedger(self.path)
        identity_a = ledger.append(receipt(1))
        identity_b = ledger.append(receipt(2, family="logic"))
        entries = ledger.read_all()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0][0], identity_a)
        self.assertEqual(entries[1][0], identity_b)
        self.assertEqual(entries[1][1].family, "logic")
        self.assertEqual(ledger.identity(), ledger.identity())

    def test_quality_rules_enforced(self) -> None:
        with self.assertRaises(LedgerError):
            EpisodeReceipt.from_dict({**receipt(1).to_dict(), "quality": "rejected",
                                      "quality_reason": None})
        with self.assertRaises(LedgerError):
            EpisodeReceipt.from_dict({**receipt(1).to_dict(), "quality": "banana"})
        with self.assertRaises(LedgerError):
            EpisodeReceipt.from_dict({**receipt(1).to_dict(), "schema": "EpisodeReceipt/v0"})
        with self.assertRaises(LedgerError):
            EpisodeReceipt.from_dict({**receipt(1).to_dict(), "quality_reason": "why"})

    def test_mutation_invalidates_identity(self) -> None:
        ledger = ExperienceLedger(self.path)
        ledger.append(receipt(1))
        ledger.append(receipt(2))
        with open(self.path, "r+", encoding="utf-8") as handle:
            content = handle.read()
            content = content.replace('"family": "arithmetic"', '"family": "arithmetik"', 1)
            handle.seek(0)
            handle.write(content)
        with self.assertRaises(LedgerTamperedError):
            ExperienceLedger(self.path).read_all()

    def test_costs_and_counts(self) -> None:
        ledger = ExperienceLedger(self.path)
        ledger.append(receipt(1, costs=(0.5, 0.25)))
        ledger.append(receipt(2, family="logic", costs=(2.0,)))
        ledger.append(receipt(3, quality="ambiguous"))
        self.assertAlmostEqual(ledger.total_cost(), 0.75 + 2.0 + 1.0)
        self.assertEqual(ledger.counts(), {"arithmetic": 2, "logic": 1})
        self.assertEqual(ledger.counts(quality="accepted"), {"arithmetic": 1, "logic": 1})

    def test_parent_experience_survives_rejected_child(self) -> None:
        ledger = ExperienceLedger(self.path)
        ledger.append(receipt(1))
        ledger.append(receipt(2, quality="rejected"))
        entries = ledger.read_all()
        self.assertEqual(entries[0][1].quality, "accepted")
        self.assertEqual(entries[1][1].quality, "rejected")


class ReplayTests(unittest.TestCase):
    @staticmethod
    def build_entries() -> list:
        entries = []
        for index in range(4):
            entries.append(receipt(index, family="arithmetic"))
        for index in range(4, 8):
            entries.append(receipt(index, family="logic"))
        entries.append(receipt(8, family="arithmetic", quality="rejected"))
        return entries

    def test_deterministic_and_resume_matches(self) -> None:
        entries = self.build_entries()
        reference = ReplayEngine(entries, family_weights={"arithmetic": 1.0, "logic": 1.0},
                                 batch_size=2, seed=7)
        first = [reference.sample() for _ in range(3)]
        resumed = ReplayEngine(entries, family_weights={"arithmetic": 1.0, "logic": 1.0},
                               batch_size=2, seed=7)
        resumed.sample()
        resumed.restore(resumed.state())
        rest = [resumed.sample() for _ in range(2)]
        self.assertEqual(
            [[entry.episode_id for entry in batch.entries] for batch in first[1:]],
            [[entry.episode_id for entry in batch.entries] for batch in rest])

    def test_within_epoch_no_duplicates(self) -> None:
        entries = self.build_entries()
        engine = ReplayEngine(entries, family_weights={"arithmetic": 1.0, "logic": 1.0},
                              batch_size=2, seed=3)
        seen: list[str] = []
        for _ in range(4):  # 8 accepted entries, batch 2 -> one full epoch
            batch = engine.sample()
            self.assertEqual(batch.shortfall, 0)
            seen.extend(entry.episode_id for entry in batch.entries)
        self.assertEqual(len(seen), len(set(seen)))

    def test_shortfall_is_explicit(self) -> None:
        entries = self.build_entries()[:2]
        engine = ReplayEngine(entries, family_weights={"arithmetic": 1.0},
                              batch_size=2, seed=1)
        batch = engine.sample(count=5)
        self.assertEqual(len(batch.entries), 2)
        self.assertEqual(batch.shortfall, 3)

    def test_quality_filter_excludes_rejected(self) -> None:
        entries = self.build_entries()
        engine = ReplayEngine(entries, family_weights={"arithmetic": 1.0, "logic": 1.0},
                              batch_size=8, seed=2)
        batch = engine.sample()
        ids = {entry.episode_id for entry in batch.entries}
        self.assertNotIn("ep-8", ids)

    def test_exact_counters_and_reconciliation(self) -> None:
        entries = self.build_entries()
        engine = ReplayEngine(entries, family_weights={"arithmetic": 1.0, "logic": 1.0},
                              batch_size=4, seed=5)
        engine.declare_planned(8)
        for _ in range(2):
            engine.sample()
        reconciliation = engine.reconciliation()
        self.assertEqual(reconciliation["planned"], 8)
        self.assertEqual(reconciliation["consumed"], 8)
        self.assertEqual(reconciliation["shortfall"], 0)
        self.assertEqual(reconciliation["by_family"],
                         {"arithmetic": 4, "logic": 4})
        engine.declare_planned(10)
        engine.sample()
        reconciliation = engine.reconciliation()
        self.assertEqual(reconciliation["shortfall"], 6)

    def test_unknown_family_in_restored_state_rejects(self) -> None:
        entries = self.build_entries()
        engine = ReplayEngine(entries, family_weights={"arithmetic": 1.0},
                              batch_size=2, seed=5)
        with self.assertRaises(ReplayStateError):
            engine.restore({"cursors": {"ghost": 1}, "epochs": {"arithmetic": 0},
                            "consumed_total": 0, "consumed_by_family": {},
                            "planned_total": 0, "orders": {"arithmetic": [0, 1]}})


if __name__ == "__main__":
    unittest.main()
