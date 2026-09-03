"""Hash-verified collection of a remote job's answer.

The collector runs on the receiving side and performs only file I/O plus
hashing: it never executes training compute. It refuses to bind when the
submission envelope was edited in transit, a receipt file is missing or its
bytes disagree with the result, the remote log is absent or mismatched, or a
receipt name attempts path traversal. A successful collection emits one
hash-bound binding receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .job_spec import RemoteJob
from .result import RemoteResult, bind_result


REMOTE_LOG_FILENAME = "remote.log"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_safe_name(name: str) -> None:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"receipt name is not a single safe path segment: {name!r}")


def collect(
    *,
    job_envelope: dict[str, Any],
    result_payload: dict[str, Any],
    receipt_dir: Path,
) -> dict[str, object]:
    """Verify envelope, result, receipt bytes, and log; return binding receipt."""

    if set(job_envelope) != {"schema", "job", "job_sha256", "sha256"}:
        raise ValueError("submission envelope fields do not match schema")
    if job_envelope["schema"] != "anra-v5-remote-submission/v1":
        raise ValueError("unsupported submission envelope schema")
    job = RemoteJob.from_dict(job_envelope["job"])
    if job_envelope["job_sha256"] != job.sha256():
        raise ValueError("submission envelope was edited after freezing; refusing collection")
    result = RemoteResult.from_dict(result_payload)
    for name, digest in result.receipt_shas.items():
        _assert_safe_name(name)
        candidate = receipt_dir / name
        if not candidate.is_file():
            raise ValueError(f"receipt file missing: {name}")
        if _sha256_file(candidate) != digest:
            raise ValueError(f"receipt bytes disagree with result: {name}")
    log_path = receipt_dir / REMOTE_LOG_FILENAME
    if not log_path.is_file():
        raise ValueError("remote log file missing")
    if _sha256_file(log_path) != result.log_sha256:
        raise ValueError("remote log bytes disagree with result")
    return bind_result(job=job, result=result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    binding = collect(
        job_envelope=json.loads(args.job.read_text(encoding="utf-8")),
        result_payload=json.loads(args.result.read_text(encoding="utf-8")),
        receipt_dir=args.receipt_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": binding["result_status"]}))
    return 0 if binding["result_status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
