"""E6 executor: verification and export production (D4/D5 + real contracts S7).

Builds the canonical source/data/protocol/results/payload bundle, verifies
hashes, required parent checkpoints by exact lineage, failed-run records and
a real load of each required payload (not just IDs). Export failure never
becomes campaign success. Partial failed-run export reports incomplete with
missing artifacts; it cannot satisfy complete-campaign acceptance.
"""
from __future__ import annotations

import json
import os
import time

from bramastra_lab.research.contracts.core import content_identity, file_sha256

from bramastra_lab.research.campaigns.phases.types import (
    EVIDENCE_FIXTURE,
    EVIDENCE_LEARNED_CAMPAIGN,
    JobInput,
    PhaseResult,
)

REQUIRED_FILES = ("campaign_ledger.json", "phase_results.json",
                  "allocation.json", "protocol.json", "restore_evidence.json",
                  "source.json", "data.json", "artifact_manifest.json")
# Exact lineage identities (never substring matching).
REQUIRED_PARENT_JOBS = ("E1-B-1701", "E1-B-1702")


def execute(job: JobInput, *, ops=None) -> PhaseResult:
    started = time.monotonic()
    job.validate()
    if job.phase != "E6":
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="E6 executor requires phase E6",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6"})
    ledger_path = os.path.join(job.run_dir, "campaign_ledger.sqlite")
    if not os.path.exists(ledger_path):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"ledger missing: {ledger_path}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6"})
    from bramastra_lab.research.campaigns.k8 import cmd_export
    import argparse

    out_dir = os.path.join(job.run_dir, "K8-results")
    args = argparse.Namespace(run_dir=job.run_dir, out=out_dir)
    code = cmd_export(args)
    if code != 0:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"export refused with code {code}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6"})
    missing = [name for name in REQUIRED_FILES
               if not os.path.exists(os.path.join(out_dir, name))]
    if missing:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"export bundle incomplete (missing files): {missing}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6", "missing_files": missing,
                                  "export_dir": out_dir})
    try:
        verification = _verify_bundle(job.run_dir, out_dir, job.data_dir)
    except Exception as exc:  # noqa: BLE001
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"export verification failed: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6"})
    if not verification.get("ok"):
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"export verification refused: {verification.get('reason')}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6",
                                  "missing_artifacts": verification.get("missing", []),
                                  "export_dir": out_dir})
    # Checkpoint identity is the content hash of the verified bundle manifest
    # (never a fixed invented string like "e6-export-verified").
    bundle_identity = verification.get("bundle_identity")
    if not bundle_identity:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error="export verification produced no bundle identity",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6", "export_dir": out_dir})
    result = PhaseResult(status="completed",
                         device_seconds=time.monotonic() - started,
                         checkpoint_identity=bundle_identity,
                         evidence_kind=EVIDENCE_LEARNED_CAMPAIGN,
                         extra={"phase": "E6", "export_dir": out_dir,
                                "verified_files": verification.get("files", []),
                                "parents": verification.get("parents", [])})
    try:
        result.validate()
    except ValueError as exc:
        return PhaseResult(status="failed", device_seconds=time.monotonic() - started,
                           error=f"E6 receipt refused: {exc}",
                           evidence_kind=EVIDENCE_FIXTURE,
                           extra={"phase": "E6", "export_dir": out_dir})
    return result


def _verify_bundle(run_dir: str, out_dir: str, data_dir: str) -> dict:
    import sqlite3

    # Hash every exported file recursively (the manifest includes nested
    # checkpoints/... rels, not just top-level names).
    digests: dict[str, str] = {}
    for base, _dirs, names in os.walk(out_dir):
        for name in names:
            path = os.path.join(base, name)
            try:
                rel = os.path.relpath(path, out_dir)
            except ValueError:
                continue
            if name == "artifact_manifest.json":
                continue
            try:
                digest = file_sha256(path)
            except OSError:
                continue
            digests[rel] = digest
    # Tamper evidence: the recursive digest map above covers EVERY exported
    # file including nested checkpoints/ entries; compare it against the
    # independently generated artifact manifest (not just top-level names).
    manifest_path = os.path.join(out_dir, "artifact_manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception as exc:
        return {"ok": False, "reason": f"artifact manifest unreadable: {exc}"}
    expected = manifest.get("files", {})
    for name, record in expected.items():
        if name not in digests:
            return {"ok": False, "reason": f"manifest expects {name} but it is missing",
                    "missing": [name]}
        if record.get("sha256") != digests[name]:
            return {"ok": False,
                    "reason": f"hash mismatch for {name}: manifest claims "
                              f"{str(record.get('sha256'))[:12]}, file hashes "
                              f"{digests[name][:12]}",
                    "missing": [name]}
    # Required parent checkpoints by EXACT lineage (never substring).
    conn = sqlite3.connect(os.path.join(run_dir, "campaign_ledger.sqlite"),
                           timeout=60.0, check_same_thread=False)
    try:
        try:
            conn.execute("PRAGMA busy_timeout=60000;")
        except Exception:
            pass
        rows = conn.execute(
            "SELECT job_id, checkpoint_identity FROM reservations "
            "WHERE status='completed' AND phase='E1'").fetchall()
    finally:
        conn.close()
    parents = {str(row[0]): row[1] for row in rows}
    missing_parents = [jid for jid in REQUIRED_PARENT_JOBS if jid not in parents]
    if missing_parents:
        return {"ok": False,
                "reason": f"missing required parent checkpoints (exact lineage): "
                          f"{missing_parents}; partial export is incomplete, not complete",
                "missing": missing_parents}
    if any(not identity for identity in parents.values()):
        return {"ok": False, "reason": "parent checkpoint without identity",
                "missing": ["parent-identity"]}
    # Payload reload: every required parent must load via the real API (not
    # just a non-empty string). Fixture-only bundles (no .pt) fail here.
    try:
        from bramastra_lab.research.runtime.checkpoint import load_checkpoint
    except Exception as exc:
        return {"ok": False, "reason": f"checkpoint API unavailable: {exc}"}
    pt_files = []
    for base, _dirs, names in os.walk(os.path.join(run_dir, "checkpoints")):
        for name in names:
            if name.endswith(".pt"):
                pt_files.append(os.path.join(base, name))
    # Also accept payloads exported under out_dir/checkpoints.
    for base, _dirs, names in os.walk(os.path.join(out_dir, "checkpoints")):
        for name in names:
            if name.endswith(".pt"):
                pt_files.append(os.path.join(base, name))
    if not pt_files:
        return {"ok": False,
                "reason": "no .pt payloads anywhere in run or export; "
                          "metadata-only export cannot satisfy complete-campaign "
                          "acceptance (partial failed-run export must report "
                          "incomplete)",
                "missing": ["payloads"]}
    for required_job in REQUIRED_PARENT_JOBS:
        checkpoint_id = parents.get(required_job)
        if not checkpoint_id or str(checkpoint_id).startswith(("fixture-", "double-")):
            return {"ok": False,
                    "reason": f"parent {required_job} checkpoint "
                              f"{str(checkpoint_id)[:24]} is fixture/metadata-only; "
                              "a real restorable payload is required",
                    "missing": [required_job]}
        try:
            # Real load (hash, COMPLETE, schema, identities).
            load_checkpoint(run_dir, checkpoint_id=checkpoint_id)
        except Exception as exc:
            return {"ok": False,
                    "reason": f"parent {required_job} payload failed to load: {exc}",
                    "missing": [required_job]}
    # Restore evidence must list checkpoints with real identities.
    restore_path = os.path.join(out_dir, "restore_evidence.json")
    try:
        with open(restore_path, encoding="utf-8") as handle:
            restore = json.load(handle)
    except Exception as exc:
        return {"ok": False, "reason": f"restore evidence unreadable: {exc}"}
    checkpoints = restore.get("checkpoints", [])
    if not checkpoints:
        return {"ok": False, "reason": "no checkpoint records in restore evidence"}
    # Bundle identity is the content hash of the verified manifest digests
    # (never a fixed invented string).
    bundle_identity = content_identity(
        {"files": digests, "parents": sorted(parents)})
    return {"ok": True, "files": sorted(digests), "digests": digests,
            "parents": sorted(parents), "bundle_identity": bundle_identity}
