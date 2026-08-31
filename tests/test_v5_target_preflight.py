from __future__ import annotations

import unittest

from v5_training.target_preflight import PreflightConfig, run_preflight


class V5TargetPreflightTests(unittest.TestCase):
    def test_preflight_config_fails_closed(self) -> None:
        for config in (
            PreflightConfig(expected_world_size=0),
            PreflightConfig(seed=-1),
            PreflightConfig(matrix_size=0),
        ):
            with self.assertRaises(ValueError):
                config.assert_valid()

    def test_missing_xla_is_reported_not_claimed_as_pass(self) -> None:
        result = run_preflight(PreflightConfig())
        if result["status"] == "BLOCKED_TORCH_XLA":
            self.assertTrue(result["missing_dependencies"])
            self.assertTrue(
                set(result["missing_dependencies"]).issubset({"torch", "torch_xla"})
            )
        else:
            self.assertIn(result["status"], {"PASS", "FAIL"})


if __name__ == "__main__":
    unittest.main()
