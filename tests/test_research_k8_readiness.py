"""Evidence-bound readiness gate tests (F21, section 22/23).

The former permanent-blocking gate tests are replaced by valid-build /
invalid-build qualification tests without losing failure-path coverage:
missing, stale, tampered and incomplete evidence must all fail closed, and
a genuine passing report must admit readiness.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from bramastra_lab.research.campaigns.readiness import (
    ImplementationReadinessError,
    implementation_readiness,
    require_implementation_ready,
)
from bramastra_lab.research.campaigns.verify_build import (
    BUILD_VERIFICATION_SCHEMA,
    REPORT_FILENAME,
)


def _verified_report_body(source_closure: str) -> dict:
    requirements = {}
    for index in range(24):
        requirements[f"F{index + 1:02d}"] = {
            "id": f"F{index + 1:02d}", "status": "pass",
            "implemented_symbols": ["bramastra_lab:x"],
            "test_selectors": ["foundation"],
            "command_receipts": [{"kind": "pytest-group",
                                  "status": "passed", "exit_code": 0}],
            "production_path_evidence": ["exercise:x"],
            "limitations": [],
        }
    body = {
        "schema": BUILD_VERIFICATION_SCHEMA,
        "created_unix": 0.0,
        "duration_seconds": 1.0,
        "no_updates": True,
        "source_identity": {"git_head": "head"},
        "source_closure_sha256": source_closure,
        "data_identity": "data-identity",
        "config_identity": "config-identity",
        "codec_identity": "codec-identity",
        "check_groups": {},
        "exercises": {},
        "requirements": requirements,
        "runtime_checks_pending": [{"id": "G01"}],
        "optimizer_updates_local": 0,
        "ready_for_owner_experiment": True,
    }
    from bramastra_lab.research.contracts.core import content_identity

    body["report_identity"] = content_identity(
        {key: value for key, value in body.items() if key != "report_identity"})
    return body


def _write_report(dir_path: str, body: dict) -> str:
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, REPORT_FILENAME)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
    return path


class K8EvidenceReadinessTests(unittest.TestCase):
    def test_missing_report_fails_closed_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = implementation_readiness(
                report_path=os.path.join(tmp, REPORT_FILENAME))
            self.assertFalse(report["ready"])
            self.assertIn("unavailable", report["reason"])
            with self.assertRaises(ImplementationReadinessError):
                require_implementation_ready(
                    report_path=os.path.join(tmp, REPORT_FILENAME))

    def test_valid_report_with_matching_closure_admits_readiness(self) -> None:
        from bramastra_lab.research.runtime.provenance import (
            source_closure_sha256)

        closure = source_closure_sha256()
        with tempfile.TemporaryDirectory() as tmp:
            _write_report(tmp, _verified_report_body(closure))
            report = implementation_readiness(report_path=os.path.join(
                tmp, REPORT_FILENAME))
            self.assertTrue(report["ready"])
            self.assertEqual(report["blocked_phases"], [])
            self.assertEqual(report["failing_requirements"], [])
            # Runtime qualification remains a separate owner-E0 gate.
            self.assertTrue(report["runtime_checks_pending"])
            admitted = require_implementation_ready(
                report_path=os.path.join(tmp, REPORT_FILENAME))
            self.assertTrue(admitted["ready"])

    def test_stale_report_rejected_when_source_closure_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_report(tmp, _verified_report_body("stale-closure-hash"))
            report = implementation_readiness(report_path=os.path.join(
                tmp, REPORT_FILENAME))
            self.assertFalse(report["ready"])
            self.assertIn("stale", report["reason"])
            self.assertEqual(report["report_source_closure"],
                             "stale-closure-hash")
            self.assertNotEqual(
                report["report_source_closure"],
                report["current_source_closure"])

    def test_tampered_report_body_rejected(self) -> None:
        from bramastra_lab.research.runtime.provenance import (
            source_closure_sha256)

        closure = source_closure_sha256()
        with tempfile.TemporaryDirectory() as tmp:
            body = _verified_report_body(closure)
            # A manually flipped requirement breaks the body identity.
            body["requirements"]["F07"]["status"] = "fail"
            _write_report(tmp, body)
            report = implementation_readiness(report_path=os.path.join(
                tmp, REPORT_FILENAME))
            self.assertFalse(report["ready"])
            self.assertIn("unavailable", report["reason"])

    def test_failing_requirement_blocks_readiness(self) -> None:
        from bramastra_lab.research.runtime.provenance import (
            source_closure_sha256)

        closure = source_closure_sha256()
        with tempfile.TemporaryDirectory() as tmp:
            body = _verified_report_body(closure)
            body["report_identity"] = None
            # Rebuild the identity honestly with one failing requirement:
            # the verifier itself can produce this for a real failure.
            body["requirements"]["F13"]["status"] = "fail"
            body["requirements"]["F13"]["command_receipts"] = [
                {"kind": "pytest-group", "status": "failed", "exit_code": 1}]
            from bramastra_lab.research.contracts.core import content_identity

            body["report_identity"] = content_identity(
                {key: value for key, value in body.items()
                 if key != "report_identity"})
            _write_report(tmp, body)
            report = implementation_readiness(report_path=os.path.join(
                tmp, REPORT_FILENAME))
            self.assertFalse(report["ready"])
            self.assertEqual(report["failing_requirements"], ["F13"])

    def test_nonzero_local_commits_in_report_rejected(self) -> None:
        from bramastra_lab.research.runtime.provenance import (
            source_closure_sha256)

        closure = source_closure_sha256()
        with tempfile.TemporaryDirectory() as tmp:
            body = _verified_report_body(closure)
            body["optimizer_updates_local"] = 3
            body["no_updates"] = False
            body["report_identity"] = None
            _write_report(tmp, body)
            report = implementation_readiness(report_path=os.path.join(
                tmp, REPORT_FILENAME))
            self.assertFalse(report["ready"])

    def test_source_identity_context_is_recorded_not_decisive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, REPORT_FILENAME)
            contextual = implementation_readiness(
                source_identity="unrelated-source", report_path=missing)
            self.assertFalse(contextual["ready"])
            self.assertEqual(contextual["source_identity"],
                             "unrelated-source")


if __name__ == "__main__":
    unittest.main()
