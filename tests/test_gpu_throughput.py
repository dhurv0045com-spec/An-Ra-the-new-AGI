from __future__ import annotations

import unittest

from bramastra_lab.research.campaigns.gpu_throughput import (
    parse_batch_sizes,
    recommend_batch_size,
    summarize_samples,
)


class ThroughputPilotPureTests(unittest.TestCase):
    def test_batch_schedule_parses_and_preserves_order(self) -> None:
        self.assertEqual(parse_batch_sizes("1, 2,4,8"), (1, 2, 4, 8))

    def test_batch_schedule_rejects_duplicates_and_nonpositive_values(self) -> None:
        for raw in ("", "1,1", "0,2", "-1,2", "1,x"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_batch_sizes(raw)

    def test_utilization_summary_handles_empty_and_measured_data(self) -> None:
        self.assertIsNone(summarize_samples([])["mean_gpu_utilization_pct"])
        summary = summarize_samples([
            {"utilization_pct": 40.0, "memory_mib": 500.0},
            {"utilization_pct": 80.0, "memory_mib": 700.0},
        ])
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["mean_gpu_utilization_pct"], 60.0)
        self.assertEqual(summary["p50_gpu_utilization_pct"], 60.0)
        self.assertEqual(summary["peak_memory_mib"], 700.0)

    def test_recommendation_meets_target_with_smallest_batch(self) -> None:
        rows = [
            {"batch_size": 1, "status": "complete", "finite": True,
             "tokens_per_second": 100.0},
            {"batch_size": 2, "status": "complete", "finite": True,
             "tokens_per_second": 180.0},
        ]
        telemetry = {"by_batch_size": {
            "1": {"mean_gpu_utilization_pct": 55.0},
            "2": {"mean_gpu_utilization_pct": 77.0},
        }}
        choice = recommend_batch_size(rows, telemetry, 75.0)
        self.assertEqual(choice["batch_size"], 2)
        self.assertTrue(choice["target_met"])

    def test_recommendation_reports_unmet_utilization_target(self) -> None:
        rows = [
            {"batch_size": 2, "status": "complete", "finite": True,
             "tokens_per_second": 100.0},
            {"batch_size": 4, "status": "complete", "finite": True,
             "tokens_per_second": 170.0},
        ]
        telemetry = {"by_batch_size": {
            "2": {"mean_gpu_utilization_pct": 45.0},
            "4": {"mean_gpu_utilization_pct": 62.0},
        }}
        choice = recommend_batch_size(rows, telemetry, 75.0)
        self.assertEqual(choice["batch_size"], 4)
        self.assertFalse(choice["target_met"])


if __name__ == "__main__":
    unittest.main()
