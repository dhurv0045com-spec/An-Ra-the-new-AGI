import unittest

from e0_cognition.device_benchmark import collect_probe


class DeviceBenchmarkTests(unittest.TestCase):
    def test_probe_is_bounded_and_reports_e0_workload(self) -> None:
        probe = collect_probe(size=32, warmup=0, repeats=1)
        self.assertEqual(probe["schema"], "esoes-local-device-probe/v1")
        self.assertEqual(probe["e0_cpu"]["cases"], 368)
        self.assertEqual(probe["e0_cpu"]["pairs"], 112)
        self.assertGreater(probe["e0_cpu"]["cases_per_second"], 0.0)
        self.assertIn("available", probe["torch"])


if __name__ == "__main__":
    unittest.main()
