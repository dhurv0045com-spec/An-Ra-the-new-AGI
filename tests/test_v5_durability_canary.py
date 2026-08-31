from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v5_training.durability_canary import ImmutableObjectStore, run_canary


class V5DurabilityCanaryTests(unittest.TestCase):
    def test_committed_durability_receipt_passes_and_binds_source(self) -> None:
        root = Path(__file__).resolve().parents[1]
        receipt = json.loads(
            (root / "artifacts/v5/local_durability_canary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(all(receipt["checks"].values()))
        self.assertEqual(
            receipt["implementation_sha256"],
            hashlib.sha256((root / "v5_training/durability_canary.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(receipt["artifact_sha256"], receipt["redownload_sha256"])

    def test_object_store_rejects_corruption_and_missing_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ImmutableObjectStore(Path(directory))
            identity = store.put(b"immutable")
            (Path(directory) / identity).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                store.get(identity)
            with self.assertRaises(ValueError):
                store.put(b"immutable")
            with self.assertRaises(ValueError):
                store.get("0" * 64)

    def test_canary_reproduces_exactly(self) -> None:
        self.assertEqual(run_canary(), run_canary())


if __name__ == "__main__":
    unittest.main()
