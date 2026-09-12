"""Regression coverage for ARK-020 V4 A1.3 template-contract repair."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4_DIR = HERE.parent / "experiments" / "ARK-020-V4"
sys.path.insert(0, str(V4_DIR))

import ark020_v4_core as C  # noqa: E402
import ark020_v4_template_guard as G  # noqa: E402


def campaign_templates():
    return {
        "A": {"p": [1], "m": [2], "s": [3], "q": [4], "t": [5]},
        "B": {"p": [6], "m": [7], "s": [8], "q": [9], "t": [10]},
        "C": {"p": [11], "g": [12], "s": [13], "k": [14], "q": [15], "e": [16]},
        "D": {"p": [17], "m": [18], "s": [19], "q": [20], "t": [21]},
    }


class TestTemplateContract(unittest.TestCase):
    def test_campaign_wide_table_selects_each_capability(self):
        tt = campaign_templates()
        for cap in ("A", "B", "C", "D"):
            self.assertIs(G.template_for_capability(cap, tt), tt[cap])

    def test_flat_template_remains_accepted(self):
        t = campaign_templates()["A"]
        self.assertIs(G.template_for_capability("A", t), t)

    def test_malformed_shape_fails_with_contract_error(self):
        with self.assertRaisesRegex(RuntimeError, "template contract mismatch"):
            G.template_for_capability("A", {"A": {"q": [1]}})

    def test_real_task_renderer_accepts_normalized_campaign_table(self):
        # Reproduces the production failure class without needing a model: before
        # A1.3, passing the complete table reaches render_binding(t)["p"] and raises
        # KeyError('p'). The normalized representation must render for every skill.
        tasks = C.build_all_tasks(list(range(48)))
        tt = campaign_templates()
        for cap in ("A", "B", "C", "D"):
            sem = tasks["sem"][cap]["main_control"]
            rows = C.task_rows(
                cap,
                G.template_for_capability(cap, tt),
                sem,
                [0],
                "canonical",
                0,
                0,
            )
            self.assertEqual(len(rows), 1)
            prompt, answer = rows[0]
            self.assertTrue(prompt)
            self.assertIsInstance(answer, int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
