"""Contracts for the X-factor probe pack (no GPU required, no training)."""
from __future__ import annotations

import json
import os
import unittest

from bramastra_lab.research.campaigns.xprobe import (
    XPROBE_SCHEMA,
    build_parser,
    probe_allocation_fidelity,
    probe_attention_geometry,
    probe_checkpoint_lineage,
    probe_phase_accounting,
    probe_tokenizer_roundtrip,
    run_xprobe,
)


class XProbeContracts(unittest.TestCase):
    def test_parser(self) -> None:
        args = build_parser().parse_args(
            ["--run-dir", "R", "--out", "O", "--device", "cuda:0"])
        self.assertEqual(args.run_dir, "R")
        self.assertEqual(args.out, "O")

    def test_empty_run_dir_gives_structured_fail_not_crash(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            out = os.path.join(tmp, "xprobe-out")
            report = run_xprobe(run_dir, out, device="cpu")
            self.assertEqual(report["schema"], XPROBE_SCHEMA)
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertEqual(len(report["probes"]), 5)
            self.assertTrue(os.path.isfile(os.path.join(out, "xprobe_report.json")))

    def test_refuses_overwrite_and_missing_dir(self) -> None:
        import tempfile
        from bramastra_lab.research.campaigns.xprobe import XProbeError
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            out = os.path.join(tmp, "out")
            run_xprobe(run_dir, out, device="cpu")
            with self.assertRaises(XProbeError):
                run_xprobe(run_dir, out, device="cpu")
            with self.assertRaises(XProbeError):
                run_xprobe(os.path.join(tmp, "nope"), os.path.join(tmp, "out2"))

    def test_lineage_and_accounting_missing_ledger(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(probe_checkpoint_lineage(tmp)["status"], "fail")
            self.assertEqual(probe_phase_accounting(tmp)["status"], "fail")
            self.assertEqual(probe_allocation_fidelity(tmp)["status"], "fail")

    def test_tokenizer_probe_structured(self) -> None:
        result = probe_tokenizer_roundtrip()
        self.assertIn(result["status"], ("pass", "error"))
        self.assertIn("status", result)

    def test_attention_probe_uses_model_inside_production_handle(self) -> None:
        result = probe_attention_geometry(device="cpu")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["hidden_finite"])
        self.assertEqual(result["hidden_shape"][:2], [4, 64])


if __name__ == "__main__":
    unittest.main()
