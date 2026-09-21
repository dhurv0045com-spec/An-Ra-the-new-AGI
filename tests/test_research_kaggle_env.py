"""Contracts for the Kaggle campaign environment (hard architecture).

No GPU, no torch import, no network: pure filesystem + validation contracts.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bramastra_lab.research.campaigns.kaggle_env import (
    K8EnvironmentError,
    build_run_paths,
    discover_bundle,
    discover_source,
    find_build_report,
    is_k8_bundle,
    is_source_tree,
    normalize_devices,
    resolve_build_report_arg,
    resolve_instance_id,
    resolve_run_id,
    gpu_contract,
)


class KaggleEnvContracts(unittest.TestCase):
    def test_normalize_devices_accepts_spaced_pair(self) -> None:
        self.assertEqual(normalize_devices("cuda:0, cuda:1"), ("cuda:0", "cuda:1"))

    def test_normalize_devices_rejects_single_triple_and_dupes(self) -> None:
        for raw in ("cuda:0", "cuda:0,cuda:1,cuda:2", "cuda:0,cuda:0", "", "cpu,cpu"):
            with self.assertRaises(K8EnvironmentError, msg=raw):
                normalize_devices(raw)

    def test_normalize_devices_rejects_garbage(self) -> None:
        with self.assertRaises(K8EnvironmentError):
            normalize_devices("cuda:0, tpu:0")

    def test_disk_contract_default_reserves_campaign_working_space(self) -> None:
        import inspect
        from bramastra_lab.research.campaigns import kaggle_env

        self.assertIn("minimum_gib: float = 8.0", inspect.getsource(kaggle_env.disk_contract))

    def test_gpu_contract_requires_current_t4_pair(self) -> None:
        class GoodCuda:
            @staticmethod
            def device_count() -> int:
                return 2

            @staticmethod
            def get_device_name(index: int) -> str:
                return "Tesla T4"

        good_torch = SimpleNamespace(cuda=GoodCuda(), __version__="2.6.0+cu124")
        with patch.dict(sys.modules, {"torch": good_torch}):
            report = gpu_contract()
        self.assertEqual(report["names"], ["Tesla T4", "Tesla T4"])

        old_torch = SimpleNamespace(cuda=GoodCuda(), __version__="2.5.1")
        with patch.dict(sys.modules, {"torch": old_torch}):
            with self.assertRaises(K8EnvironmentError):
                gpu_contract()

        class WrongGpuCuda(GoodCuda):
            @staticmethod
            def get_device_name(index: int) -> str:
                return "Tesla P100-PCIE-16GB"

        wrong_gpu_torch = SimpleNamespace(cuda=WrongGpuCuda(), __version__="2.6.0")
        with patch.dict(sys.modules, {"torch": wrong_gpu_torch}):
            with self.assertRaises(K8EnvironmentError):
                gpu_contract()

    def test_resolve_build_report_arg_file_and_dir(self) -> None:
        self.assertEqual(resolve_build_report_arg("/r/build.json"), "/r/build.json")
        self.assertEqual(
            resolve_build_report_arg("/r/report-dir"),
            os.path.join("/r/report-dir", "build_verification.json"))
        self.assertIsNone(resolve_build_report_arg(None))
        with self.assertRaises(K8EnvironmentError):
            resolve_build_report_arg("   ")

    def test_run_id_roundtrip_and_validation(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_run_id(tmp, explicit="k8-test-01"), "k8-test-01")
            generated = resolve_run_id(tmp)
            self.assertEqual(resolve_run_id(tmp), generated)
            with self.assertRaises(K8EnvironmentError):
                resolve_run_id(tmp, explicit="bad id!!")

    def test_instance_id_stable(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "bramastra-k8", "k8-x")
            first = resolve_instance_id(root)
            self.assertEqual(first, resolve_instance_id(root))

    def test_build_run_paths_binds(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "working")
            paths = build_run_paths(working=work, run_id="k8-paths-01")
            self.assertEqual(paths.run_id, "k8-paths-01")
            self.assertTrue(paths.run_root.is_dir())
            self.assertEqual(paths.build_report_file().name, "build_verification.json")

    def test_source_and_bundle_discovery(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "repo")
            os.makedirs(os.path.join(src, "bramastra_lab"))
            open(os.path.join(src, "pyproject.toml"), "w", encoding="utf-8").write("[project]\n")
            self.assertTrue(is_source_tree(src))
            found, _ = discover_source(configured=src)
            self.assertIsNotNone(found)
            bundle = os.path.join(tmp, "bundle")
            os.makedirs(bundle)
            with open(os.path.join(bundle, "manifest.json"), "w", encoding="utf-8") as handle:
                json.dump({"schema": "bramastra-k8-data/v1"}, handle)
            self.assertTrue(is_k8_bundle(bundle))
            found_bundle, _ = discover_bundle(configured=bundle)
            self.assertIsNotNone(found_bundle)

    def test_find_build_report_prefers_existing(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "run")
            os.makedirs(os.path.join(root, "build-verification-aaa"))
            report = os.path.join(root, "build-verification-aaa", "build_verification.json")
            open(report, "w", encoding="utf-8").write("{}")
            self.assertEqual(str(find_build_report(root, os.path.join(tmp, "fresh"))), report)

    def test_artifact_archives_are_distinct_for_distinct_recovery_reasons(self) -> None:
        import tempfile
        from pathlib import Path

        from bramastra_lab.research.campaigns.kaggle_env import package_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp)
            root = working / "run"
            root.mkdir()
            (root / "evidence.json").write_text("{}", encoding="utf-8")
            failure = package_artifacts(root, working, "k8-test", "instance", "failed-E0")
            (root / "later.json").write_text("{}", encoding="utf-8")
            completed = package_artifacts(root, working, "k8-test", "instance", "completed")
            self.assertIsNotNone(failure)
            self.assertIsNotNone(completed)
            assert failure is not None and completed is not None
            self.assertNotEqual(failure.archive, completed.archive)


if __name__ == "__main__":
    unittest.main()
