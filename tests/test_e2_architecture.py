from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from e2_architecture.aggregate import aggregate_receipts
from e2_architecture.block_benchmark import BenchmarkConfig, shape_arms
from e2_architecture.block_aggregate import aggregate_receipts as aggregate_block_receipts
from e2_architecture.device_benchmark import AttentionCase, _percentile, default_cases
from e2_architecture.plan import build_plan
from e2_architecture.precision_benchmark import PrecisionConfig, classify as classify_precision
from e2_architecture.qk_norm_benchmark import QKNormConfig, classify as classify_qk_norm
from e2_architecture.signal_benchmark import SignalConfig, classify


class E2ArchitectureTests(unittest.TestCase):
    def test_precision_config_requires_replication(self) -> None:
        PrecisionConfig("cuda", 256, 1, (1, 2, 3)).assert_valid()
        with self.assertRaises(ValueError):
            PrecisionConfig("cuda", 256, 1, (1, 2)).assert_valid()

    def test_precision_classification_requires_every_seed_and_shape(self) -> None:
        rows = []
        for arm in ("deep-narrow", "middle", "wide-shallow"):
            for seed in (1, 2, 3):
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "checks": {
                            "finite": True,
                            "parameter_count_exact": True,
                            "parity": True,
                        },
                    }
                )
        self.assertEqual(classify_precision(rows)["verdict"], "SUPPORTED_LOCAL_BF16_PARITY")
        rows[-1]["checks"]["parity"] = False
        self.assertEqual(classify_precision(rows)["verdict"], "MIXED_LOCAL_BF16_PARITY")

    def test_local_precision_receipts_are_current_and_pass(self) -> None:
        repository = Path(__file__).parents[1]
        implementation_sha256 = hashlib.sha256(
            (repository / "e2_architecture/precision_benchmark.py").read_bytes()
        ).hexdigest()
        model_sha256 = hashlib.sha256(
            (repository / "e2_architecture/block_benchmark.py").read_bytes()
        ).hexdigest()
        initialization_sha256 = hashlib.sha256(
            (repository / "e2_architecture/signal_benchmark.py").read_bytes()
        ).hexdigest()
        expected_receipts = (
            ("local_cuda_precision_parity.json", "cuda", 256),
            ("local_cuda_precision_parity_2k.json", "cuda", 2048),
            ("local_cpu_precision_parity.json", "cpu", 64),
        )
        for filename, device, sequence_length in expected_receipts:
            receipt = json.loads(
                (repository / "artifacts/e2" / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(
                receipt["classification"]["verdict"], "SUPPORTED_LOCAL_BF16_PARITY"
            )
            self.assertEqual(receipt["implementation_sha256"], implementation_sha256)
            self.assertEqual(receipt["model_implementation_sha256"], model_sha256)
            self.assertEqual(
                receipt["initialization_implementation_sha256"], initialization_sha256
            )
            self.assertEqual(receipt["config"]["device"], device)
            self.assertEqual(receipt["config"]["sequence_length"], sequence_length)
            self.assertEqual(len(receipt["config"]["seeds"]), 3)
            self.assertEqual(len(receipt["rows"]), 9)
            for row in receipt["rows"]:
                self.assertTrue(all(row["checks"].values()))

    def test_qk_norm_config_requires_replication_and_valid_shape(self) -> None:
        QKNormConfig("cuda", (512, 2048), (1, 2, 3)).assert_valid()
        with self.assertRaises(ValueError):
            QKNormConfig("cuda", (512,), (1, 2)).assert_valid()
        with self.assertRaises(ValueError):
            QKNormConfig("cuda", (512,), (1, 2, 3), width=320).assert_valid()

    def test_qk_norm_classification_requires_invariance_and_stress_exposure(self) -> None:
        rows = []
        for policy, logits, entropies in (
            ("qk-norm", (1.0, 1.01, 1.0), (0.91, 0.90, 0.91)),
            ("no-qk-norm", (0.01, 0.16, 2.56), (1.0, 0.98, 0.60)),
        ):
            for scale, logit_rms, entropy in zip((0.25, 1.0, 4.0), logits, entropies):
                rows.append(
                    {
                        "context_length": 512,
                        "policy": policy,
                        "projection_scale": scale,
                        "attention_logit_rms": {"median": logit_rms},
                        "normalized_entropy_mean": {"median": entropy},
                    }
                )
        self.assertEqual(classify_qk_norm(rows)["verdict"], "SUPPORTED_QK_SCALE_CONTROL")
        rows[0]["attention_logit_rms"]["median"] = 0.1
        self.assertEqual(classify_qk_norm(rows)["verdict"], "CONTRADICTED_QK_SCALE_CONTROL")

    def test_local_qk_norm_receipts_are_current_and_pass(self) -> None:
        repository = Path(__file__).parents[1]
        implementation_sha256 = hashlib.sha256(
            (repository / "e2_architecture/qk_norm_benchmark.py").read_bytes()
        ).hexdigest()
        expected_receipts = (
            ("local_cuda_qk_norm.json", "cuda", [512, 2048, 4096], 5),
            ("local_cpu_qk_norm.json", "cpu", [128, 512], 3),
        )
        for filename, device, contexts, seed_count in expected_receipts:
            receipt = json.loads(
                (repository / "artifacts/e2" / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(
                receipt["classification"]["verdict"], "SUPPORTED_QK_SCALE_CONTROL"
            )
            self.assertEqual(receipt["implementation_sha256"], implementation_sha256)
            self.assertEqual(receipt["config"]["device"], device)
            self.assertEqual(receipt["config"]["context_lengths"], contexts)
            self.assertEqual(len(receipt["config"]["seeds"]), seed_count)
            self.assertEqual(len(receipt["rows"]), len(contexts) * seed_count * 6)
            for row in receipt["rows"]:
                self.assertTrue(all(row["checks"].values()))

    def test_signal_config_requires_replication(self) -> None:
        SignalConfig("cuda", 256, 1, (1, 2, 3)).assert_valid()
        with self.assertRaises(ValueError):
            SignalConfig("cuda", 256, 1, (1, 2)).assert_valid()
        with self.assertRaises(ValueError):
            SignalConfig("cuda", 256, 1, (1, 1, 2)).assert_valid()

    def test_signal_classification_requires_growth_and_gradient_sanity(self) -> None:
        rows = []
        for arm in ("deep-narrow", "middle", "wide-shallow"):
            for policy, growth, spread in (
                ("normal-0.02", 4.0, 5.0),
                ("scaled-residual-0.02", 1.2, 6.0),
            ):
                rows.append(
                    {
                        "arm": arm,
                        "policy": policy,
                        "final_to_embedding_rms_ratio": {"median": growth},
                        "block_gradient_max_min_ratio": {"median": spread},
                    }
                )
        result = classify(rows)
        self.assertEqual(result["verdict"], "SUPPORTED_LOCAL_SIGNAL_PROPAGATION")
        rows[-1]["block_gradient_max_min_ratio"]["median"] = 100.0
        self.assertEqual(classify(rows)["verdict"], "MIXED_LOCAL_SIGNAL_PROPAGATION")

    def test_local_signal_receipts_are_current_and_correct(self) -> None:
        repository = Path(__file__).parents[1]
        implementation_sha256 = hashlib.sha256(
            (repository / "e2_architecture/signal_benchmark.py").read_bytes()
        ).hexdigest()
        model_implementation_sha256 = hashlib.sha256(
            (repository / "e2_architecture/block_benchmark.py").read_bytes()
        ).hexdigest()
        expected_receipts = (
            ("local_cuda_signal_propagation.json", "cuda", 5, 256),
            ("local_cuda_signal_propagation_4k.json", "cuda", 3, 4096),
            ("local_cpu_signal_propagation.json", "cpu", 3, 64),
        )
        for filename, device, expected_seed_count, sequence_length in expected_receipts:
            receipt = json.loads(
                (repository / "artifacts/e2" / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(
                receipt["classification"]["verdict"],
                "SUPPORTED_LOCAL_SIGNAL_PROPAGATION",
            )
            self.assertEqual(receipt["implementation_sha256"], implementation_sha256)
            self.assertEqual(
                receipt["model_implementation_sha256"], model_implementation_sha256
            )
            self.assertEqual(receipt["config"]["device"], device)
            self.assertEqual(receipt["config"]["sequence_length"], sequence_length)
            self.assertEqual(len(receipt["config"]["seeds"]), expected_seed_count)
            self.assertEqual(len(receipt["rows"]), 6 * expected_seed_count)
            for row in receipt["rows"]:
                self.assertTrue(all(row["checks"].values()))

    def test_full_stack_cases_are_exact_static_shape_arms(self) -> None:
        arms = shape_arms()
        self.assertEqual(tuple(arm.name for arm in arms), ("deep-narrow", "middle", "wide-shallow"))
        self.assertEqual(
            [arm.model.parameter_receipt().total for arm in arms],
            [35_420_480, 35_414_400, 35_144_192],
        )

    def test_full_stack_benchmark_config_fails_closed(self) -> None:
        BenchmarkConfig("cuda", (512, 1024), 1, 2, 5, 1).assert_valid()
        with self.assertRaises(ValueError):
            BenchmarkConfig("cuda", (512, 512), 1, 2, 5, 1).assert_valid()
        with self.assertRaises(ValueError):
            BenchmarkConfig("tpu", (512,), 1, 2, 5, 1).assert_valid()

    def test_full_stack_aggregate_checks_replication_identity(self) -> None:
        rows = []
        for arm, parameters, latency, memory in (
            ("deep-narrow", 35_420_480, 400.0, 500),
            ("middle", 35_414_400, 270.0, 450),
            ("wide-shallow", 35_144_192, 140.0, 390),
        ):
            for sequence_length in (512, 1024):
                scale = sequence_length / 512
                rows.append(
                    {
                        "arm": arm,
                        "sequence_length": sequence_length,
                        "parameters": parameters,
                        "correctness": {
                            "parameter_count_exact": True,
                            "finite_loss": True,
                            "all_gradients_finite": True,
                        },
                        "forward": {"median_ms": latency * scale / 3},
                        "forward_backward": {"median_ms": latency * scale},
                        "forward_backward_peak_allocated_bytes": int(memory * scale),
                    }
                )
        with TemporaryDirectory() as directory:
            paths = []
            for seed in (1, 2, 3):
                receipt = {
                    "schema": "esoes-e2-full-stack-benchmark/v1",
                    "status": "PASS",
                    "scope": "fixture",
                    "implementation_sha256": "a" * 64,
                    "static_plan_sha256": "b" * 64,
                    "torch_version": "fixture",
                    "cuda_runtime": "fixture",
                    "device_name": "fixture",
                    "config": {
                        "device": "cuda",
                        "sequence_lengths": [512, 1024],
                        "batch_size": 1,
                        "warmup": 1,
                        "repeats": 3,
                        "seed": seed,
                    },
                    "rows": rows,
                    "limitations": [],
                }
                path = Path(directory) / f"{seed}.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")
                paths.append(path)
            result = aggregate_block_receipts(paths)
            self.assertEqual(result["status"], "PASS_REPLICATED")
            self.assertAlmostEqual(
                result["shape_comparisons"]["512"]["deep_vs_wide_latency_ratio"],
                400 / 140,
            )
            with self.assertRaises(ValueError):
                aggregate_block_receipts(paths[:2])

    def test_local_full_stack_receipts_match_aggregate_hashes(self) -> None:
        root = Path(__file__).parents[1] / "artifacts/e2"
        aggregate = json.loads(
            (root / "local_cuda_full_stack_aggregate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(aggregate["status"], "PASS_REPLICATED")
        self.assertEqual(aggregate["seeds"], [32001, 32002, 32003])
        for source in aggregate["source_receipts"]:
            path = root / source["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])
            receipt = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["implementation_sha256"], aggregate["implementation_sha256"])
            self.assertEqual(receipt["status"], "PASS")
            for row in receipt["rows"]:
                self.assertTrue(row["correctness"]["parameter_count_exact"])
                self.assertTrue(row["correctness"]["finite_loss"])
                self.assertTrue(row["correctness"]["all_gradients_finite"])

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
