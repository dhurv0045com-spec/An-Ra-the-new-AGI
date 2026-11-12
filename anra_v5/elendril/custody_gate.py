"""V7.16 pre-execution custody gate: verify parent archives before any update.

Phase A of the V7.16 mandate. The gate is read-only over the repository: it
hashes supplied archive files, compares them against the recorded owner-run
custody pins and emits a strict gate record. A missing or mismatched parent
archive yields NO_GO and no training update may follow. CPU fixture
verification uses deterministic mock payloads so every code contract is
unit-tested without any real archive.
"""
from hashlib import sha256
from pathlib import Path
import os
import tempfile

from .codec import canonical
from . import v715_protocol, v716_protocol

GATE_SCHEMA = "elendril-v716-preexecution-gate-v1"

# Recorded owner-run custody pins (Experiment Result/V7.15/STATUS.json and
# research/v7.15/RELEASE.md). The V7.15 S8 ANALYSIS zip SHA was independently
# verified by the replay audit; the S8 FULL archive delivery receipt remains
# PENDING and therefore has no recorded SHA to verify here.
PARENT_FULL_SHA256 = v715_protocol.PARENT_FULL_SHA256
S8_ANALYSIS_SHA256 = "08cca3edcec070b85fac752faf89f156b85fb7b7fc2fbd9b02c908461175c02b"
S8_ANALYSIS_BYTES = 16_023_595

CHUNK_BYTES = 1 << 20
FIXTURE_PAYLOAD = b"elendril-v716-custody-fixture-v1: deterministic mock archive bytes"


def sha256_file(path, *, chunk_bytes=CHUNK_BYTES):
    """Streamed SHA-256 of a file; never reads the whole archive into memory."""
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk size must be a positive integer")
    fingerprint = sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            size += len(chunk)
            fingerprint.update(chunk)
    return fingerprint.hexdigest(), size


def verify_archive(path, expected_sha256, *, expected_bytes=None):
    """Return a custody record for one archive; absence is a recorded status."""
    record = {"path": None, "present": False, "sha256": None, "byte_size": None,
              "expected_sha256": expected_sha256, "match": None}
    if path is None:
        return record
    path = Path(path)
    record["path"] = str(path)
    if not path.is_file():
        return record
    record["present"] = True
    record["sha256"], record["byte_size"] = sha256_file(path)
    record["match"] = record["sha256"] == expected_sha256
    if expected_bytes is not None:
        record["expected_byte_size"] = expected_bytes
        record["byte_size_match"] = record["byte_size"] == expected_bytes
    return record


def preexecution_gate(*, parent_archive=None, s8_analysis_zip=None,
                      require_parent=True, require_s8_analysis=False):
    """Emit the Phase A gate record.

    ``CUSTODY_STATUS`` is PASS only when every required archive is present and
    hash-matched. Any failure is NO_GO with the exact reason; the caller must
    stop before creating training state.
    """
    if require_parent not in (True, False) or require_s8_analysis not in (True, False):
        raise ValueError("custody requirements must be booleans")
    parent = verify_archive(parent_archive, PARENT_FULL_SHA256)
    s8 = verify_archive(s8_analysis_zip, S8_ANALYSIS_SHA256, expected_bytes=S8_ANALYSIS_BYTES)
    failures = []
    if require_parent:
        if not parent["present"]:
            failures.append("NO_GO_MISSING_PARENT_ARCHIVE")
        elif parent["match"] is not True:
            failures.append("NO_GO_PARENT_ARCHIVE_SHA_MISMATCH")
    if require_s8_analysis:
        if not s8["present"]:
            failures.append("NO_GO_MISSING_S8_ANALYSIS_ARCHIVE")
        elif s8["match"] is not True or s8.get("byte_size_match") is not True:
            failures.append("NO_GO_S8_ANALYSIS_ARCHIVE_MISMATCH")
    status = "PASS" if not failures else failures[0]
    return {
        "schema": GATE_SCHEMA,
        "CUSTODY_STATUS": status,
        "failures": failures,
        "parent_archive": parent,
        "s8_analysis_archive": s8,
        "pins": {
            "parent_full_sha256": PARENT_FULL_SHA256,
            "s8_analysis_sha256": S8_ANALYSIS_SHA256,
            "v716_protocol_sha256": v716_protocol.PROTOCOL_SHA,
            "parent_source": v715_protocol.PARENT_SOURCE,
        },
        "rule": "no optimizer update may run unless CUSTODY_STATUS is PASS; missing archives are recorded, never fabricated",
    }


def fixture_archive(destination, payload=FIXTURE_PAYLOAD):
    """Write the deterministic mock payload; returns (path, sha256, size).

    Tests use this to exercise the exact hashing and mismatch paths of
    :func:`preexecution_gate` without any real checkpoint custody.
    """
    path = Path(destination)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest, size = sha256_file(path)
    return path, digest, size


def write_once(path, value):
    """Atomic write-once JSON emission; refuses to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"custody record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        handle.close()
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path
