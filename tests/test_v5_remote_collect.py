from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v5_remote.collect import REMOTE_LOG_FILENAME, collect
from v5_remote.job_spec import IDENTITY_KEYS, JOB_SCHEMA, RemoteJob
from v5_remote.result import RemoteResult


def _job() -> RemoteJob:
    return RemoteJob(
        schema=JOB_SCHEMA,
        job_id="e6-preflight-tpu",
        accelerator="tpu-v5e-8",
        replicas=8,
        runtime_image_sha256="a" * 64,
        code_commit="b" * 40,
        command=("esoes-v5-target-preflight", "--output", "artifacts/v5/target_preflight.json"),
        seed=47101,
        token_budget=131072,
        max_wall_seconds=3600,
        identities={key: None for key in IDENTITY_KEYS},
    )


def _payloads(receipt_bytes: bytes = b"preflight-pass", log_bytes: bytes = b"log") -> tuple[dict, dict, bytes, bytes]:
    job = _job()
    envelope = {
        "schema": "anra-v5-remote-submission/v1",
        "job": job.canonical(),
        "job_sha256": job.sha256(),
        "sha256": "0" * 64,
    }
    result = RemoteResult(
        schema="anra-v5-remote-result/v1",
        job_sha256=job.sha256(),
        status="succeeded",
        completed_update=0,
        cumulative_tokens=0,
        receipt_shas={"target_preflight.json": hashlib.sha256(receipt_bytes).hexdigest()},
        log_sha256=hashlib.sha256(log_bytes).hexdigest(),
        failure_code=None,
    )
    return envelope, result.canonical(), receipt_bytes, log_bytes


class CollectTests(unittest.TestCase):
    def _collect(self, envelope: dict, result: dict, files: dict[str, bytes]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in files.items():
                (root / name).write_bytes(payload)
            return collect(job_envelope=envelope, result_payload=result, receipt_dir=root)

    def test_happy_path_binds(self) -> None:
        envelope, result, receipt_bytes, log_bytes = _payloads()
        binding = self._collect(
            envelope, result,
            {"target_preflight.json": receipt_bytes, REMOTE_LOG_FILENAME: log_bytes},
        )
        self.assertEqual(binding["result_status"], "succeeded")
        self.assertEqual(len(str(binding["sha256"])), 64)

    def test_edited_envelope_is_refused(self) -> None:
        envelope, result, receipt_bytes, log_bytes = _payloads()
        envelope = dict(envelope, job_sha256="f" * 64)
        with self.assertRaises(ValueError):
            self._collect(
                envelope, result,
                {"target_preflight.json": receipt_bytes, REMOTE_LOG_FILENAME: log_bytes},
            )

    def test_tampered_receipt_bytes_are_refused(self) -> None:
        envelope, result, _, log_bytes = _payloads()
        with self.assertRaises(ValueError):
            self._collect(
                envelope, result,
                {"target_preflight.json": b"edited", REMOTE_LOG_FILENAME: log_bytes},
            )

    def test_missing_log_is_refused(self) -> None:
        envelope, result, receipt_bytes, _ = _payloads()
        with self.assertRaises(ValueError):
            self._collect(envelope, result, {"target_preflight.json": receipt_bytes})

    def test_path_traversal_receipt_name_is_refused(self) -> None:
        job = _job()
        envelope = {
            "schema": "anra-v5-remote-submission/v1",
            "job": job.canonical(),
            "job_sha256": job.sha256(),
            "sha256": "0" * 64,
        }
        evil = RemoteResult(
            schema="anra-v5-remote-result/v1",
            job_sha256=job.sha256(),
            status="failed",
            completed_update=0,
            cumulative_tokens=0,
            receipt_shas={"../escape.json": "e" * 64},
            log_sha256="d" * 64,
            failure_code="BAD",
        )
        with self.assertRaises(ValueError):
            self._collect(envelope, evil.canonical(), {})


if __name__ == "__main__":
    unittest.main()
