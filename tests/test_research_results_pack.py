"""Contracts for results-only packaging (no GPU, tiny temp files)."""
from __future__ import annotations

import json
import os
import unittest
import zipfile

from bramastra_lab.research.campaigns.results_pack import (
    RESULTS_SCHEMA,
    ResultsPackError,
    _build,
    build_parser,
    collect_result_files,
    snapshot_run_dir,
)


class ResultsPackContracts(unittest.TestCase):
    def test_parser(self) -> None:
        args = build_parser().parse_args(
            ["--run-dir", "R", "--out", "O.zip", "--run-id", "k8-x"])
        self.assertEqual(args.run_dir, "R")

    def test_excludes_payloads_keeps_results(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run, "checkpoints"))
            with open(os.path.join(run, "a.json"), "w", encoding="utf-8") as h:
                json.dump({"ok": True}, h)
            with open(os.path.join(run, "checkpoints", "m.pt"), "wb") as h:
                h.write(b"\x00" * 1024)
            files = collect_result_files([run])
            names = [n for _, n in files]
            self.assertTrue(any(n.endswith("a.json") for n in names))
            self.assertFalse(any(n.endswith(".pt") for n in names))

    def test_build_verified_zip_with_receipt(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(run)
            with open(os.path.join(run, "r.json"), "w", encoding="utf-8") as h:
                json.dump({"v": 1}, h)
            out = os.path.join(tmp, "results.zip")
            receipt = _build([run], out, run_id="t")
            self.assertEqual(receipt["schema"], RESULTS_SCHEMA)
            self.assertIsNone(zipfile.ZipFile(out).testzip())
            self.assertTrue(os.path.isfile(os.path.join(tmp, "results.json")))

    def test_results_zip_projects_restore_manifest_after_payload_filtering(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run, "checkpoints"))
            with open(os.path.join(run, "checkpoints", "step.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"step": 4}, handle)
            with open(os.path.join(run, "checkpoints", "payload.pt"), "wb") as handle:
                handle.write(b"model-weights")
            manifest = {
                "complete": True,
                "payload_files": 1,
                "files": {
                    "checkpoints/step.json": {"bytes": 11, "sha256": "json"},
                    "checkpoints/payload.pt": {"bytes": 13, "sha256": "weights"},
                },
            }
            with open(os.path.join(run, "artifact_manifest.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with open(os.path.join(run, "restore_evidence.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"payload_files_exported": 1, "note": "source export"},
                          handle)

            out = os.path.join(tmp, "results.zip")
            _build([run], out, run_id="k8-test")
            with zipfile.ZipFile(out) as archive:
                projected = json.loads(archive.read(
                    "source0-run/artifact_manifest.json"))
                restore = json.loads(archive.read(
                    "source0-run/restore_evidence.json"))
                self.assertIsNone(archive.testzip())
                self.assertNotIn("source0-run/checkpoints/payload.pt",
                                 archive.namelist())
            self.assertFalse(projected["complete"])
            self.assertTrue(projected["source_export_complete"])
            self.assertEqual(projected["source_payload_files"], 1)
            self.assertEqual(projected["payload_files"], 0)
            self.assertEqual(list(projected["files"]), ["checkpoints/step.json"])
            self.assertFalse(projected["results_pack"]["restorable_from_archive"])
            self.assertEqual(projected["results_pack"]["omitted_payload_files"],
                             ["checkpoints/payload.pt"])
            self.assertEqual(restore["payload_files_exported"], 0)
            self.assertEqual(restore["source_payload_files_exported"], 1)
            self.assertFalse(restore["results_pack"]["restorable_from_archive"])
            self.assertIn("cannot restore", restore["note"])

    def test_hashes_archives_without_reading_the_full_zip(self) -> None:
        import inspect
        from bramastra_lab.research.campaigns import results_pack

        self.assertIn("handle.read(1024 * 1024)", inspect.getsource(results_pack._sha256_file))

    def test_refuses_overwrite_and_empty(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(run)
            with open(os.path.join(run, "r.json"), "w", encoding="utf-8") as h:
                h.write("{}")
            out = os.path.join(tmp, "r.zip")
            _build([run], out, run_id="t")
            with self.assertRaises(ResultsPackError):
                _build([run], out, run_id="t")
            with self.assertRaises(ResultsPackError):
                _build([os.path.join(tmp, "empty")], os.path.join(tmp, "e.zip"), run_id="t")

    def test_snapshot_run_dir_keeps_latest_pointer_and_prunes(self) -> None:
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run, "phase_outputs"))
            with open(os.path.join(run, "phase_outputs", "a.json"), "w",
                      encoding="utf-8") as h:
                h.write("{}")
            receipts = []
            for index in range(6):
                receipts.append(snapshot_run_dir(run, f"reason-{index}"))
                time.sleep(1.05)
            pointer = os.path.join(run, "safety", "safety-latest.json")
            self.assertTrue(os.path.isfile(pointer))
            with open(pointer, encoding="utf-8") as h:
                latest = json.load(h)
            self.assertEqual(latest["archive"], receipts[-1]["archive"])
            import glob
            self.assertLessEqual(len(glob.glob(os.path.join(run, "safety", "safety-*.zip"))), 4)

    def test_snapshot_empty_run_dir_never_raises(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(run)
            receipt = snapshot_run_dir(run, "empty-check")
            self.assertTrue(receipt["empty"])

    def test_safety_sweep_never_raises(self) -> None:
        import tempfile
        from bramastra_lab.research.campaigns.runner import _safety_sweep
        with tempfile.TemporaryDirectory() as tmp:
            _safety_sweep(None, "/nonexistent-run-dir-xyz", "test")
            _safety_sweep(None, tmp, "test")

    def test_e1_partial_receipt(self) -> None:
        import tempfile
        from types import SimpleNamespace
        from bramastra_lab.research.campaigns.phases.e1 import _write_partial_receipt
        with tempfile.TemporaryDirectory() as tmp:
            job = SimpleNamespace(run_dir=tmp, phase="E1", arm="A", seed=1701,
                                  job_id="E1-A-1701", update_target=200)
            _write_partial_receipt(job, committed=40, attempted=42, exposure=10,
                                   checkpoint_ids=["ab" * 32], parent_for_next="ab" * 32,
                                   step=199, stream_id="s")
            partial = os.path.join(tmp, "phase_outputs", "E1", "A-1701-partial.json")
            self.assertTrue(os.path.isfile(partial))
            with open(partial, encoding="utf-8") as h:
                body = json.load(h)
            self.assertTrue(body["partial"])
            self.assertEqual(body["completed_steps"], 200)


if __name__ == "__main__":
    unittest.main()
