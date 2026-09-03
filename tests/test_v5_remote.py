from __future__ import annotations

import json
import unittest

from v5_remote.job_spec import IDENTITY_KEYS, JOB_SCHEMA, RemoteJob
from v5_remote.result import (
    RemoteResult,
    bind_result,
    submission_envelope,
)


def _job(**overrides) -> RemoteJob:
    fields: dict[str, object] = {
        "schema": JOB_SCHEMA,
        "job_id": "e2-signal-p35-seed32001",
        "accelerator": "tpu-v5e-8",
        "replicas": 8,
        "runtime_image_sha256": "a" * 64,
        "code_commit": "b" * 40,
        "command": ("anra-v5", "e2-signal", "--output", "artifacts/e2/local_signal.json"),
        "seed": 32001,
        "token_budget": 200_000_000,
        "max_wall_seconds": 14400,
        "identities": {key: None for key in IDENTITY_KEYS},
    }
    fields.update(overrides)
    return RemoteJob(**fields)  # type: ignore[arg-type]


def _result(job: RemoteJob, **overrides) -> RemoteResult:
    fields: dict[str, object] = {
        "schema": "anra-v5-remote-result/v1",
        "job_sha256": job.sha256(),
        "status": "succeeded",
        "completed_update": 1525,
        "cumulative_tokens": 200_000_000,
        "receipt_shas": {"signal": "c" * 64},
        "log_sha256": "d" * 64,
        "failure_code": None,
    }
    fields.update(overrides)
    return RemoteResult(**fields)  # type: ignore[arg-type]


class RemoteJobTests(unittest.TestCase):
    def test_job_is_hash_stable_and_round_trips_through_json(self) -> None:
        job = _job()
        clone = RemoteJob.from_dict(json.loads(json.dumps(job.canonical())))
        self.assertEqual(clone, job)
        self.assertEqual(clone.sha256(), job.sha256())

    def test_job_rejects_unpinned_or_malformed_requests(self) -> None:
        with self.assertRaises(ValueError):
            _job(runtime_image_sha256="not-a-hash").assert_valid()
        with self.assertRaises(ValueError):
            _job(code_commit="short").assert_valid()
        with self.assertRaises(ValueError):
            _job(command=()).assert_valid()
        with self.assertRaises(ValueError):
            _job(replicas=0).assert_valid()
        with self.assertRaises(ValueError):
            _job(identities={"training_spec_sha256": "e" * 64}).assert_valid()

    def test_submission_envelope_binds_spec_to_hash(self) -> None:
        job = _job()
        envelope = submission_envelope(job)
        self.assertEqual(envelope["job_sha256"], job.sha256())
        self.assertEqual(envelope["job"], job.canonical())
        self.assertEqual(len(str(envelope["sha256"])), 64)


class RemoteBindingTests(unittest.TestCase):
    def test_matching_result_binds(self) -> None:
        job = _job()
        binding = bind_result(job=job, result=_result(job))
        self.assertEqual(binding["job_sha256"], job.sha256())
        self.assertEqual(binding["result_status"], "succeeded")
        self.assertEqual(len(str(binding["sha256"])), 64)

    def test_swapped_result_is_refused(self) -> None:
        job = _job()
        other = _job(job_id="e2-signal-p35-seed32002")
        with self.assertRaises(ValueError):
            bind_result(job=other, result=_result(job))

    def test_failed_results_require_auditable_codes_and_logs(self) -> None:
        job = _job()
        failed = _result(job, status="failed", receipt_shas={}, failure_code="OOM_RANK3")
        binding = bind_result(job=job, result=failed)
        self.assertEqual(binding["failure_code"], "OOM_RANK3")
        with self.assertRaises(ValueError):
            _result(job, status="failed", receipt_shas={}, failure_code=None).assert_valid()
        with self.assertRaises(ValueError):
            _result(job, status="succeeded", failure_code="STALE").assert_valid()
        with self.assertRaises(ValueError):
            _result(job, receipt_shas={}).assert_valid()

    def test_result_round_trips_through_json(self) -> None:
        job = _job()
        clone = RemoteResult.from_dict(json.loads(json.dumps(_result(job).canonical())))
        self.assertEqual(bind_result(job=job, result=clone)["job_sha256"], job.sha256())


if __name__ == "__main__":
    unittest.main()
