"""Unit tests for senora.remote_package."""

from __future__ import annotations

import unittest

from senora.remote_package import audit_remote_package, generate_runbook_markdown


class TestSenoraRemotePackage(unittest.TestCase):
    def test_remote_package_audit(self) -> None:
        manifest = audit_remote_package()
        self.assertEqual(manifest.schema, "senora-remote-package-manifest/v1")
        self.assertEqual(manifest.target_branch, "senora")
        self.assertTrue(manifest.all_software_ready)
        self.assertTrue(manifest.external_data_blocked)

        # Verify runbook generation
        runbook = generate_runbook_markdown(manifest)
        self.assertIn("# Senora P35 Remote Cluster Execution Runbook", runbook)
        self.assertIn("sbatch artifacts/v5/cluster_jobs/run_control-substrate-00.sbatch", runbook)
        self.assertIn("python -m senora.canary", runbook)


if __name__ == "__main__":
    unittest.main()