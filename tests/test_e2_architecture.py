from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from e2_architecture.aggregate import aggregate_receipts
from e2_architecture.device_benchmark import AttentionCase, _percentile, default_cases
from e2_architecture.plan import build_plan


class E2ArchitectureTests(unittest.TestCase):
    def test_device_aggregate_requires_distinct_matched_receipts(self) -> None:
        cases = [
            {
                "name": name,
                "forward": {"median_ms": forward},
                "forward_backward": {"median_ms": training},
                "forward_backward_peak_allocated_bytes": memory,
            }
            for name, forward, training, memory in (
                ("mha-qk-2k", 1.0, 4.0, 30),
                ("gqa-qk-2k", 10.0, 20.0, 400),
                ("gqa-repeat-kv-qk-2k", 1.0, 3.5, 45),
                ("gqa-no-qk-2k", 9.0, 18.0, 390),
                ("gqa-qk-4k", 40.0, 75.0, 1_600),
            )
        ]
        with TemporaryDirectory() as directory:
            paths = []
            for seed in (1, 2, 3):
                receipt = {
                    "schema": "esoes-e2-device-benchmark/v1",
                    "status": "PASS",
                    "scope": "fixture",
                    "implementation_sha256": "a" * 64,
                    "device_name": "fixture",
                    "device_total_memory_bytes": 1,
                    "torch_version": "fixture",
                    "cuda_runtime": "fixture",
                    "bf16_supported": True,
                    "warmup": 1,
                    "repeats": 2,
                    "seed": seed,
                    "cases": cases,
                    "native_gqa_backend_support": {"math": {"supported": True}},
                    "gqa_equivalence": {"maximum_absolute_error": 0.0},
                    "limitations": [],
                }
                path = Path(directory) / f"{seed}.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")
                paths.append(path)
            aggregate = aggregate_receipts(paths)
            self.assertEqual(aggregate["status"], "PASS_REPLICATED")
            self.assertEqual(
                aggregate["comparisons"]["native_gqa_vs_mha_training_latency_ratio"], 5.0
            )
            with self.assertRaises(ValueError):
                aggregate_receipts(paths[:2])

    def test_device_cases_isolate_preregistered_attention_factors(self) -> None:
        cases = {case.name: case for case in default_cases()}
        self.assertEqual(cases["mha-qk-2k"].query_heads, cases["mha-qk-2k"].kv_heads)
        self.assertEqual(cases["gqa-qk-2k"].query_heads // cases["gqa-qk-2k"].kv_heads, 3)
        self.assertEqual(cases["gqa-repeat-kv-qk-2k"].implementation, "repeat-kv")
        self.assertFalse(cases["gqa-no-qk-2k"].qk_norm)
        self.assertEqual(cases["gqa-qk-4k"].context_length, 4096)
        for case in cases.values():
            case.assert_valid()

    def test_device_case_rejects_incompatible_gqa_heads(self) -> None:
        with self.assertRaises(ValueError):
            AttentionCase("bad", query_heads=6, kv_heads=4, context_length=128).assert_valid()
        with self.assertRaises(ValueError):
            AttentionCase(
                "bad-mha-repeat", query_heads=6, kv_heads=6, context_length=128,
                implementation="repeat-kv"
            ).assert_valid()

    def test_percentile_is_nearest_rank(self) -> None:
        self.assertEqual(_percentile([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)

    def test_shape_arms_are_iso_parameter_and_ordered(self) -> None:
        plan = build_plan()
        shapes = [arm for arm in plan.arms if arm.group == "shape"]
        totals = [arm.model.parameter_receipt().total for arm in shapes]
        self.assertLess(max(totals) / min(totals) - 1, 0.01)
        self.assertEqual(totals, [35_420_480, 35_414_400, 35_144_192])
        self.assertGreater(shapes[0].model.layers, shapes[1].model.layers)
        self.assertGreater(shapes[1].model.layers, shapes[2].model.layers)

    def test_attention_and_context_factors_change_only_intended_fields(self) -> None:
        plan = build_plan()
        attention = [arm for arm in plan.arms if arm.group == "attention"]
        fixed = ("width", "layers", "ffn_width", "context_length", "vocabulary_size")
        for field in fixed:
            self.assertEqual(len({getattr(arm.model, field) for arm in attention}), 1)
        context = [arm for arm in plan.arms if arm.group == "context"]
        left, right = context
        left_data, right_data = left.model.canonical(), right.model.canonical()
        left_data.pop("context_length")
        right_data.pop("context_length")
        self.assertEqual(left_data, right_data)

    def test_static_compute_receipt_exposes_expected_tradeoffs(self) -> None:
        receipts = {arm.name: arm.receipt() for arm in build_plan().arms}
        self.assertLess(
            receipts["gqa-qk"]["kv_cache_bf16_bytes_per_full_sequence"],
            receipts["mha-qk"]["kv_cache_bf16_bytes_per_full_sequence"],
        )
        self.assertGreater(
            receipts["mixed-full-4k"]["forward_flops_per_full_sequence_proxy"],
            2 * receipts["full-2k"]["forward_flops_per_full_sequence_proxy"],
        )

    def test_plan_fails_closed_until_real_dependencies_exist(self) -> None:
        self.assertEqual(build_plan().status(), "BLOCKED_E1_INPUTS")
        digest = "a" * 64
        waiting = build_plan(tokenizer_sha256=digest, corpus_manifest_sha256=digest)
        self.assertEqual(waiting.status(), "BLOCKED_MODEL_IMPLEMENTATION")
        ready = build_plan(
            tokenizer_sha256=digest,
            corpus_manifest_sha256=digest,
            model_constructor_sha256=digest,
        )
        self.assertEqual(ready.status(), "READY_FOR_BOUNDED_P35")

    def test_parameter_mismatch_is_rejected(self) -> None:
        plan = build_plan()
        arms = list(plan.arms)
        first = arms[0]
        arms[0] = dataclasses.replace(
            first, model=dataclasses.replace(first.model, ffn_width=256)
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(plan, arms=tuple(arms)).assert_valid()


if __name__ == "__main__":
    unittest.main()
