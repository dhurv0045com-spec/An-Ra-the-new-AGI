"""B2.2 C1 acceptance: the actual prepare -> sample -> pair-input path.

No optimizer, no model: this verifies the command-path data flow that the
chief's probe showed broken (pair groups dropped, semantics substituted).
"""
import json
import os
import tempfile
import unittest

from bramastra_lab.research.commands import (
    _pair_rows_for,
    _read_prepared,
    _sampler_units,
    prepare_data,
)
from bramastra_lab.research.config import BuildConfig
from bramastra_lab.research.errors import CommandError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO_ROOT, "engineering", "reports", "B2", "fixtures")


class PrepareSamplePairTests(unittest.TestCase):
    def test_pair_groups_survive_prepare_sample_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump({"model": {"profile": "tiny"}}, handle)
            prepared = os.path.join(tmp, "prepared")
            import io
            import contextlib

            with contextlib.redirect_stdout(io.StringIO()):
                prepare_data(os.path.join(FIXTURES, "manifest.json"), prepared,
                             config_path)
            rows = json.load(open(os.path.join(prepared, "prepared.json"),
                                  encoding="utf-8"))
            # The prepared manifest binds the pair groups (B2.2 R2/C1).
            training_integrity = rows["split_integrity"]["training"]
            self.assertIn("swap-1", training_integrity["pair_group_ids"])
            self.assertIn("swap-2", training_integrity["pair_group_ids"])
            # Sampling pulls whole groups; the pair objective receives
            # aligned own/swapped renderings for a complete pair.
            all_rows = [json.loads(line)
                        for line in open(os.path.join(prepared, "rows-training.jsonl"),
                                         encoding="utf-8") if line.strip()]
            unit_groups = _sampler_units(all_rows)
            pair_unit = next(unit for unit in unit_groups
                             if len(unit) == 2 and unit[0].group_id)
            batch_records = [ref.record for ref in pair_unit]
            own, swapped = _pair_rows_for(batch_records)
            self.assertEqual(len(own), 2)
            self.assertEqual(len(swapped), 2)
            # The swapped rendering uses the other member's answer bytes.
            self.assertNotEqual(own[0].tokens, swapped[0].tokens)
            self.assertEqual(own[0].supervised[-2:], (True, True))
            # Config identity mismatch still refuses.
            with self.assertRaises(CommandError):
                _read_prepared(prepared, BuildConfig.from_dict(
                    {"model": {"profile": "tiny"}, "training": {"seed": 99}}))

    def test_pair_input_test_through_real_fixtures(self) -> None:
        """The swap-1 pair has different answers, so it feeds the objective."""
        with open(os.path.join(FIXTURES, "train.jsonl"), encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        pair_members = [record for record in records
                        if record.get("group") == "swap-1"]
        self.assertEqual(len(pair_members), 2)
        self.assertNotEqual(pair_members[0]["answer"], pair_members[1]["answer"])


if __name__ == "__main__":
    unittest.main()
