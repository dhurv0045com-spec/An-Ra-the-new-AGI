from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v5_contracts.launch_readiness import EXPECTED_DOCUMENTS, _text_sha256, build_readiness, validate_gate_manifest


class V5LaunchReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.manifest = json.loads(
            (cls.root / "blueprint/LAUNCH_GATES.json").read_text(encoding="utf-8")
        )

    def test_repository_is_ready_for_experiments_but_main_run_is_blocked(self) -> None:
        receipt = build_readiness(root=self.root)
        self.assertEqual(receipt["status"], "READY_FOR_PRELAUNCH_EXPERIMENTS")
        self.assertTrue(receipt["experiments_authorized"])
        self.assertFalse(receipt["main_training_authorized"])
        self.assertEqual(receipt["pending_gates"], ["E1", "E2", "E3", "E4", "E5", "E6"])
        self.assertTrue(all(receipt["checks"].values()))

    def test_pass_gate_without_real_hash_bound_receipt_fails_closed(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["gates"][0]["status"] = "PASS"
        with self.assertRaises(ValueError):
            validate_gate_manifest(manifest, root=self.root)

    def test_receipt_path_cannot_escape_repository(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["gates"][0].update(
            {"status": "PASS", "receipt_path": "../outside.json", "receipt_sha256": "0" * 64}
        )
        with self.assertRaises(ValueError):
            validate_gate_manifest(manifest, root=self.root)

    def test_nonpass_gate_cannot_claim_evidence(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["gates"][0]["receipt_path"] = "artifacts/e1/fake.json"
        with self.assertRaises(ValueError):
            validate_gate_manifest(manifest, root=self.root)

    def test_external_identity_inventory_cannot_be_reduced(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["external_identities"] = {"anything": "1" * 64}
        with self.assertRaises(ValueError):
            validate_gate_manifest(manifest, root=self.root)

    def test_document_hash_is_newline_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doc.md"
            path.write_bytes(b"first\nsecond\n")
            expected = _text_sha256(path)
            path.write_bytes(b"first\r\nsecond\r\n")
            self.assertEqual(expected, _text_sha256(path))

    def test_complete_inventory_never_authorizes_main_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "blueprint").mkdir()
            for name in EXPECTED_DOCUMENTS:
                (root / "blueprint" / name).write_text("test document\n", encoding="utf-8")
            manifest = copy.deepcopy(self.manifest)
            manifest["candidate_spec"] = "candidate.json"
            (root / "candidate.json").write_bytes(
                (self.root / self.manifest["candidate_spec"]).read_bytes()
            )
            payload = b'{"status":"PASS"}'
            (root / "evidence.json").write_bytes(payload)
            for gate in manifest["gates"]:
                gate.update(status="PASS", receipt_path="evidence.json", receipt_sha256=hashlib.sha256(payload).hexdigest())
            manifest["external_identities"] = {name: "1" * 64 for name in manifest["external_identities"]}
            manifest["main_run_requested"] = True
            (root / "blueprint/LAUNCH_GATES.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = build_readiness(root=root)
            self.assertEqual(result["status"], "READY_FOR_FREEZE_REVIEW")
            self.assertFalse(result["main_training_authorized"])
            self.assertFalse(result["production_launcher_implemented"])
            candidate_path = root / "candidate.json"
            normalized = candidate_path.read_bytes().replace(b"\r\n", b"\n")
            candidate_path.write_bytes(normalized)
            lf_result = build_readiness(root=root)
            candidate_path.write_bytes(normalized.replace(b"\n", b"\r\n"))
            self.assertEqual(lf_result, build_readiness(root=root))

    def test_committed_readiness_receipt_reproduces(self) -> None:
        receipt = json.loads((self.root / "artifacts/v5/launch_readiness.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt, build_readiness(root=self.root))


if __name__ == "__main__":
    unittest.main()
