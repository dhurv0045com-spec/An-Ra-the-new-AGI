from __future__ import annotations

import unittest
from types import SimpleNamespace

from v5_training.target_preflight import (
    PreflightConfig,
    UnsupportedXlaRuntimeError,
    discover_xla_topology,
    run_preflight,
    topology_checks,
)


class V5TargetPreflightTests(unittest.TestCase):
    def test_preflight_config_fails_closed(self) -> None:
        for config in (
            PreflightConfig(expected_world_size=0),
            PreflightConfig(expected_global_device_count=0),
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
            self.assertIn(
                result["status"],
                {"PASS", "FAIL", "BLOCKED_UNSUPPORTED_TORCH_XLA_RUNTIME"},
            )

    def test_current_runtime_topology_api_is_receipted(self) -> None:
        calls: list[str] = []

        def record(name: str, value):
            def method():
                calls.append(name)
                return value

            return method

        runtime = SimpleNamespace(
            device_type=record("device_type", "TPU"),
            world_size=record("world_size", 8),
            global_ordinal=record("global_ordinal", 3),
            global_runtime_device_count=record("global_count", 8),
            addressable_runtime_device_count=record("addressable_count", 8),
        )
        xla_model = SimpleNamespace(xla_device=record("xla_device", "xla:3"))

        topology = discover_xla_topology(
            runtime_module=runtime,
            xla_model_module=xla_model,
        )

        self.assertEqual(topology["device_type"], "TPU")
        self.assertEqual(topology["world_size"], 8)
        self.assertEqual(topology["ordinal"], 3)
        self.assertEqual(topology["global_device_count"], 8)
        self.assertEqual(topology["addressable_device_count"], 8)
        self.assertEqual(
            calls,
            [
                "xla_device",
                "device_type",
                "world_size",
                "global_ordinal",
                "global_count",
                "addressable_count",
            ],
        )

    def test_legacy_topology_helpers_do_not_fallback(self) -> None:
        runtime = SimpleNamespace(
            device_type=lambda: "TPU",
            world_size=lambda: 8,
            global_ordinal=lambda: 0,
            xrt_world_size=lambda: 8,
            get_ordinal=lambda: 0,
        )
        xla_model = SimpleNamespace(xla_device=lambda: "xla:0")

        with self.assertRaisesRegex(
            UnsupportedXlaRuntimeError,
            "global_runtime_device_count.*Legacy xla_model topology helpers",
        ):
            discover_xla_topology(
                runtime_module=runtime,
                xla_model_module=xla_model,
            )

    def test_cpu_xla_runtime_cannot_pass_tpu_gate(self) -> None:
        topology = {
            "device": "xla:0",
            "device_type": "CPU",
            "world_size": 1,
            "ordinal": 0,
            "global_device_count": 1,
            "addressable_device_count": 1,
        }

        checks = topology_checks(
            topology=topology, expected_world_size=1, expected_global_device_count=8
        )

        self.assertFalse(checks["device_type_is_tpu"])
        self.assertFalse(checks["global_device_count_matches"])
        self.assertFalse(all(checks.values()))

    def test_expected_global_tpu_device_count_is_enforced(self) -> None:
        topology = {
            "device": "xla:0",
            "device_type": "TPU",
            "world_size": 1,
            "ordinal": 0,
            "global_device_count": 8,
            "addressable_device_count": 8,
        }
        checks = topology_checks(
            topology=topology, expected_world_size=1, expected_global_device_count=4
        )
        self.assertFalse(checks["global_device_count_matches"])


if __name__ == "__main__":
    unittest.main()
