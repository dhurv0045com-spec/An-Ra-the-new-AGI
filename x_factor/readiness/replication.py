"""Replication identity + chronology (Mission 3/4/20/21).

A replication is NOT "another artifact, same checkpoint". It must bind:

  same checkpoint identity (parameter SHA)
  same capability family
  same experiment schema
  same frozen protocol family + version
  same intervention registry SHA
  same scoring definitions SHA
  same task-generation law SHA
  DIFFERENT preregistered seed/cohort
  separate execution artifact (own SHA)
  chronology: primary receipt -> replication protocol freeze -> replication
  source/analysis implementation SHAs

Chronology is machine-checkable via explicit parent references
(primary receipt SHA + protocol SHA embedded in the replication receipt),
not wall-clock trust. Same-seed, changed-registry, changed-generator, or
checkpoint-mismatched artifacts are REJECTED with named reasons.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys as _sys

_XF = Path(__file__).resolve().parents[1]
if str(_XF) not in _sys.path:
    _sys.path.insert(0, str(_XF))

from provenance import sha256_file  # noqa: E402


def build_evidence(*, checkpoint_param_sha: str, capability_family: str,
                   experiment_schema: str, protocol_family: str,
                   protocol_version: str, protocol_sha: str,
                   intervention_registry_sha: str, scoring_sha: str,
                   generator_law_sha: str, seed: int, cohort_id: str,
                   primary_receipt_sha: str,
                   primary_seed: int, artifact_path: str) -> dict:
    return {
        "schema": "anra-replication-evidence/v1",
        "checkpoint_param_sha": checkpoint_param_sha,
        "capability_family": capability_family,
        "experiment_schema": experiment_schema,
        "protocol_family": protocol_family,
        "protocol_version": protocol_version,
        "protocol_sha": protocol_sha,
        "intervention_registry_sha": intervention_registry_sha,
        "scoring_sha": scoring_sha,
        "generator_law_sha": generator_law_sha,
        "seed": seed,
        "cohort_id": cohort_id,
        "primary_receipt_sha": primary_receipt_sha,
        "primary_seed": primary_seed,
        "artifact_path": artifact_path,
    }


def check_evidence(ev: dict, *, expected_param_sha: str,
                   expected_protocol_family: str,
                   expected_registry_sha: str | None = None,
                   expected_generator_sha: str | None = None) -> dict:
    """Validate a replication claim. Returns replication_ok bool + reasons."""
    reasons: list[str] = []

    def need(cond: bool, msg: str):
        if not cond:
            reasons.append(msg)

    need(ev.get("schema") == "anra-replication-evidence/v1", "bad evidence schema")
    need(ev.get("checkpoint_param_sha") == expected_param_sha, "checkpoint mismatch")
    need(ev.get("protocol_family") == expected_protocol_family, "protocol family mismatch")
    if expected_registry_sha is not None:
        need(ev.get("intervention_registry_sha") == expected_registry_sha,
             "intervention registry changed")
    if expected_generator_sha is not None:
        need(ev.get("generator_law_sha") == expected_generator_sha,
             "task-generation law changed")
    need(ev.get("seed") != ev.get("primary_seed"), "same seed reused: not a replication")
    need(bool(ev.get("cohort_id")) and ev.get("cohort_id") != "same-cohort",
         "not an independent cohort")
    need(bool(ev.get("primary_receipt_sha")), "missing primary-receipt ancestry")
    need(bool(ev.get("protocol_sha")), "missing protocol-freeze binding")
    art = Path(ev.get("artifact_path", ""))
    if not art.exists():
        reasons.append(f"artifact missing: {art}")
        artifact_sha = None
    else:
        try:
            doc = json.loads(art.read_text(encoding="utf-8"))
            artifact_sha = sha256_file(str(art))
            need(doc.get("provenance", {}).get("parameter_sha256") in
                 (None, expected_param_sha), "artifact checkpoint mismatch")
        except ValueError as e:
            artifact_sha = None
            reasons.append(f"artifact unreadable: {e}")
    return {"replication_ok": not reasons, "reasons": reasons,
            "artifact_sha256": artifact_sha}


# Promotion grades: what a replication verdict may be USED for.
#   NONE     not a replication (ok is False) or no evidence supplied
#   LEGACY   checkpoint-bound only (pre-evidence artifact): descriptive
#            history, NEVER promotes V5 science or READY claims
#   EVIDENCE full evidence block validated: may support same-family DEV
#            claims and, with all other gates, qualification
PROMOTION_GRADES = ("NONE", "LEGACY", "EVIDENCE")


def replication_ok_for_promotion(verdict: dict) -> dict:
    """Grade a check_replication/check_evidence verdict for promotion use.

    Legacy checkpoint-only matches can NEVER promote: only mode=="evidence"
    with replication_ok True earns EVIDENCE. Returns grade + promotable bool
    + reason. Pure function (no I/O) so it is unit-testable and auditable.
    """
    ok = verdict.get("replication_ok")
    mode = verdict.get("mode")
    if ok is True and mode == "evidence":
        return {"grade": "EVIDENCE", "promotable": True,
                "reason": "full evidence block validated"}
    if ok is True:
        return {"grade": "LEGACY", "promotable": False,
                "reason": "legacy checkpoint-only match: descriptive history, "
                          "never promotes V5 science or READY claims"}
    if ok is None:
        return {"grade": "NONE", "promotable": False,
                "reason": "no replication evidence supplied"}
    return {"grade": "NONE", "promotable": False,
            "reason": f"replication rejected: {verdict.get('reasons', verdict.get('note'))}"}
