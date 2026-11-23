"""Level-6 ConfirmLock custody and independent CampaignLedgerVerifier."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import canonical

from .archives import verify_session_archive, win_long_path
from .design import canonical_structural_hash, verify_split_disjointness
from .execution import _atomic_replace

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_RECEIPT_STATUSES = frozenset({"COMPLETE", "COMPLETED", "VERIFIED", "OK", "SUCCEEDED"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lock_body_sha256(doc: Mapping[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k not in ("status", "lock_sha256", "unlock_receipt")}
    return sha256(canonical(body)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    lp = win_long_path(path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{path.name}-{os.getpid()}.json"
    lp_tmp = win_long_path(tmp)
    try:
        lp_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _atomic_replace(tmp, path)
    finally:
        lp_tmp.unlink(missing_ok=True)


class ConfirmLock:
    """Tamper-evident cross-session CONFIRM custody lock that opens ONLY when all prerequisite jobs verify."""

    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = Path(lock_path)

    def seal(
        self,
        required_job_ids: Sequence[str],
        *,
        protocol_sha256: str | None = None,
        source_commit: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        req_ids = sorted({str(x).strip() for x in required_job_ids if str(x).strip()})
        lp = win_long_path(self.lock_path)
        if lp.is_file():
            existing = json.loads(lp.read_text(encoding="utf-8"))
            if existing.get("lock_sha256") != _lock_body_sha256(existing):
                raise ValueError("CONFIRM_LOCK_TAMPERED")
            existing_reqs = sorted({str(x).strip() for x in existing.get("required_job_ids", ()) if str(x).strip()})
            if existing_reqs != req_ids:
                raise ValueError("CONFIRM_LOCK_MUTATION_FORBIDDEN")
            return existing

        doc: dict[str, Any] = {
            "schema": "anra-planthir-confirm-lock-v1",
            "status": "SEALED",
            "required_job_ids": req_ids,
            "sealed_utc": _utc_now(),
            "protocol_sha256": str(protocol_sha256) if protocol_sha256 is not None else None,
            "source_commit": str(source_commit) if source_commit is not None else None,
            "metadata": dict(metadata) if metadata is not None else {},
            "unlock_receipt": None,
        }
        doc["lock_sha256"] = _lock_body_sha256(doc)
        _atomic_write_json(self.lock_path, doc)
        return doc

    def inspect(
        self,
        completed_job_ids: Sequence[str] = (),
        *,
        expected_lock_sha256: str | None = None,
    ) -> dict[str, Any]:
        lp = win_long_path(self.lock_path)
        if not lp.is_file():
            return {
                "schema": "anra-planthir-confirm-lock-inspect-v1",
                "exists": False,
                "lock_path": str(self.lock_path),
                "status": "MISSING",
                "custody_status": "CUSTODY_LIMITATION_MISSING_ORIGINAL_LOCK",
                "unlocked": False,
                "lock_sha256": None,
                "required_job_ids": [],
                "completed_required_jobs": [],
                "pending_confirm_jobs": [],
                "unlock_receipt": None,
                "defects": ["CUSTODY_LIMITATION_MISSING_ORIGINAL_LOCK"],
            }

        doc = json.loads(lp.read_text(encoding="utf-8"))
        defects: list[str] = []
        stored_sha = doc.get("lock_sha256")
        if stored_sha != _lock_body_sha256(doc):
            defects.append("CONFIRM_LOCK_TAMPERED")
        if expected_lock_sha256 is not None and stored_sha != str(expected_lock_sha256):
            defects.append("CONFIRM_LOCK_SHA_MISMATCH")

        req_ids = sorted({str(x).strip() for x in doc.get("required_job_ids", ()) if str(x).strip()})
        done_set = {str(x).strip() for x in completed_job_ids if str(x).strip()}
        completed_req = sorted(set(req_ids) & done_set)
        pending_confirm = sorted(set(req_ids) - done_set)

        status = str(doc.get("status", "UNKNOWN"))
        if status not in ("SEALED", "UNLOCKED"):
            defects.append("CONFIRM_LOCK_INVALID_STATUS")
        if status == "UNLOCKED":
            u_rec = doc.get("unlock_receipt")
            if not isinstance(u_rec, Mapping) or u_rec.get("original_lock_sha256") != stored_sha:
                defects.append("CONFIRM_LOCK_INVALID_UNLOCK_RECEIPT")
        unlocked = (status == "UNLOCKED") and (not defects) and (not pending_confirm)
        if defects:
            custody_status = "DEFECTS_DETECTED"
        elif unlocked:
            custody_status = "VERIFIED_COMPLETE"
        elif not pending_confirm:
            custody_status = "UNLOCK_READY"
        else:
            custody_status = "SEALED_PENDING"

        return {
            "schema": "anra-planthir-confirm-lock-inspect-v1",
            "exists": True,
            "lock_path": str(self.lock_path),
            "status": status,
            "custody_status": custody_status,
            "unlocked": unlocked,
            "lock_sha256": stored_sha,
            "required_job_ids": req_ids,
            "completed_required_jobs": completed_req,
            "pending_confirm_jobs": pending_confirm,
            "unlock_receipt": doc.get("unlock_receipt"),
            "defects": defects,
        }

    def unlock_if_ready(
        self,
        completed_job_ids: Sequence[str],
        *,
        reason: str = "ALL_PREREQUISITES_VERIFIED",
    ) -> dict[str, Any]:
        rep = self.inspect(completed_job_ids)
        if not rep["exists"] or rep["defects"] or rep["pending_confirm_jobs"]:
            return {**rep, "unlocked": False}

        lp = win_long_path(self.lock_path)
        doc = json.loads(lp.read_text(encoding="utf-8"))
        if doc.get("status") != "UNLOCKED" or doc.get("unlock_receipt") is None:
            doc["status"] = "UNLOCKED"
            doc["unlock_receipt"] = {
                "unlocked_utc": _utc_now(),
                "completed_count": len(rep["completed_required_jobs"]),
                "reason": str(reason),
                "original_lock_sha256": doc.get("lock_sha256"),
            }
            _atomic_write_json(self.lock_path, doc)
        return self.inspect(completed_job_ids)


def _valid_declared_sha(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s or any(ch.isspace() for ch in s):
        return False
    if len(s) == 64:
        return bool(_HEX64_RE.match(s))
    return True


def verify_campaign_ledger(
    *,
    expected_job_ids: Sequence[str],
    job_receipts: Mapping[str, Mapping[str, Any]],
    study_requirements: Mapping[str, Sequence[str]] | None = None,
    confirm_lock_path: Path | str | None = None,
    expected_lock_sha256: str | None = None,
    archive_paths: Sequence[Path | str] = (),
    expected_source_commit: str | None = None,
    require_archive_sidecars: bool = True,
    require_v6_manifest: bool = False,
) -> dict[str, Any]:
    """Independently verify job receipts, study completeness, ConfirmLock custody & ZIP archives."""

    expected = [str(j) for j in expected_job_ids]
    verified_jobs: list[str] = []
    missing_jobs: list[str] = []
    failed_jobs: list[dict[str, Any]] = []
    defects: list[str] = []

    for jid in expected:
        rec = job_receipts.get(jid)
        if rec is None:
            missing_jobs.append(jid)
            continue
        status_raw = str(rec.get("status", "")).strip().upper()
        if status_raw not in _VALID_RECEIPT_STATUSES:
            failed_jobs.append({"job_id": jid, "reason": f"INVALID_STATUS:{status_raw or 'MISSING'}"})
            defects.append(f"JOB_NOT_VERIFIED:{jid}:status={status_raw or 'MISSING'}")
            continue
        sha_bad = False
        if "checkpoint_sha256" in rec and not _valid_declared_sha(rec.get("checkpoint_sha256")):
            failed_jobs.append({"job_id": jid, "reason": "INVALID_CHECKPOINT_SHA256"})
            defects.append(f"INVALID_CHECKPOINT_SHA256:{jid}")
            sha_bad = True
        if "parent_checkpoint_sha256" in rec and rec.get("parent_checkpoint_sha256") is not None:
            if not _valid_declared_sha(rec.get("parent_checkpoint_sha256")):
                failed_jobs.append({"job_id": jid, "reason": "INVALID_PARENT_CHECKPOINT_SHA256"})
                defects.append(f"INVALID_PARENT_CHECKPOINT_SHA256:{jid}")
                sha_bad = True
        if not sha_bad:
            verified_jobs.append(jid)

    verified_set = set(verified_jobs)
    verified_studies: list[str] = []
    unverified_studies: list[dict[str, Any]] = []
    for sid, reqs in sorted((study_requirements or {}).items()):
        req_list = [str(r) for r in reqs]
        missing_for_study = [r for r in req_list if r not in verified_set]
        if missing_for_study:
            unverified_studies.append({"study_id": str(sid), "missing_job_ids": missing_for_study})
        else:
            verified_studies.append(str(sid))

    lock_report: dict[str, Any] | None = None
    if confirm_lock_path is not None:
        lock_report = ConfirmLock(confirm_lock_path).inspect(
            verified_jobs, expected_lock_sha256=expected_lock_sha256
        )
        defects.extend(lock_report.get("defects", ()))

    archive_reports: dict[str, Any] = {}
    for raw_ap in archive_paths:
        ap = Path(raw_ap)
        try:
            archive_reports[ap.name] = verify_session_archive(
                ap,
                expected_source_commit=expected_source_commit,
                require_sidecar_sha256=require_archive_sidecars,
                require_v6_manifest=require_v6_manifest,
            )
        except Exception as exc:
            archive_reports[ap.name] = {"verified": False, "error": str(exc)}
            defects.append(f"ARCHIVE_VERIFICATION_FAILED:{ap.name}:{exc}")

    if "CUSTODY_LIMITATION_MISSING_ORIGINAL_LOCK" in defects:
        custody_status = "CUSTODY_LIMITATION_MISSING_ORIGINAL_LOCK"
    elif defects:
        custody_status = "DEFECTS_DETECTED"
    elif missing_jobs or unverified_studies or (lock_report is not None and not lock_report.get("unlocked", False)):
        custody_status = "VERIFIED_PARTIAL"
    else:
        custody_status = "VERIFIED_COMPLETE"

    verified = custody_status == "VERIFIED_COMPLETE"
    return {
        "schema": "anra-planthir-ledger-verification-v1",
        "verified": verified,
        "custody_status": custody_status,
        "expected_job_count": len(expected),
        "verified_jobs": verified_jobs,
        "missing_jobs": missing_jobs,
        "failed_jobs": failed_jobs,
        "verified_studies": verified_studies,
        "unverified_studies": unverified_studies,
        "confirm_lock": lock_report,
        "archives": archive_reports,
        "defects": defects,
    }


class CampaignLedgerVerifier:
    """Stateless verifier auditing multi-session job receipts, study coverage, ConfirmLock & ZIPs."""

    @staticmethod
    def verify(
        *,
        expected_job_ids: Sequence[str],
        job_receipts: Mapping[str, Mapping[str, Any]],
        study_requirements: Mapping[str, Sequence[str]] | None = None,
        confirm_lock_path: Path | str | None = None,
        expected_lock_sha256: str | None = None,
        archive_paths: Sequence[Path | str] = (),
        expected_source_commit: str | None = None,
        require_archive_sidecars: bool = True,
        require_v6_manifest: bool = False,
    ) -> dict[str, Any]:
        return verify_campaign_ledger(
            expected_job_ids=expected_job_ids,
            job_receipts=job_receipts,
            study_requirements=study_requirements,
            confirm_lock_path=confirm_lock_path,
            expected_lock_sha256=expected_lock_sha256,
            archive_paths=archive_paths,
            expected_source_commit=expected_source_commit,
            require_archive_sidecars=require_archive_sidecars,
            require_v6_manifest=require_v6_manifest,
        )


__all__ = [
    "CampaignLedgerVerifier",
    "ConfirmLock",
    "canonical_structural_hash",
    "verify_campaign_ledger",
    "verify_split_disjointness",
]
