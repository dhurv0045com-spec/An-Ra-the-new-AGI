"""Provenance-preserving training documents from the repository evidence ledgers.

These records let the Phase-One model see the research program it belongs to
while keeping outcome, evidence grade, supported claim, and limits explicit.
They are training material, never evaluation labels or capability evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


LEDGER_SCHEMA = "anra.evidence-ledger/v1"
CORPUS_SCHEMA = "anra-signac-research-evidence-corpus/v1"
RESEARCH_FAMILY = "research_evidence"
STATUS_VALUES = {
    "DEMONSTRATED", "SUPPORTED", "IMPLEMENTED_NOT_EXECUTED", "IN_PROGRESS",
    "SPECULATIVE", "CONTRADICTED", "INVALIDATED", "SUPERSEDED",
    "INCONCLUSIVE", "NOT_TESTED",
}
EVIDENCE_CLASS_VALUES = {
    "ENGINEERING_EVIDENCE", "SCIENTIFIC_EVIDENCE", "BOTH", "NEITHER",
}
REPLICATION_VALUES = {"R0", "R1", "R2", "R3", "R4"}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _render(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, str):
        return value.strip() or "not recorded"
    if isinstance(value, (list, tuple)):
        return "; ".join(_render(item) for item in value) or "not recorded"
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _experiment_document(
    experiment: Mapping[str, Any], *, ledger_sha256: str, basis_head: str, generated: str
) -> dict[str, str]:
    required = (
        "id", "program", "branch", "question", "status", "evidence_class",
        "replication", "result", "supported_claim", "unsupported_claim",
    )
    if any(not isinstance(experiment.get(key), str) or not experiment[key].strip() for key in required):
        raise ValueError("evidence ledger experiment is missing a required research field")
    if experiment["status"] not in STATUS_VALUES:
        raise ValueError(f"unknown evidence status for {experiment['id']}")
    if experiment["evidence_class"] not in EVIDENCE_CLASS_VALUES:
        raise ValueError(f"unknown evidence class for {experiment['id']}")
    if experiment["replication"] not in REPLICATION_VALUES:
        raise ValueError(f"unknown replication grade for {experiment['id']}")

    identifier = experiment["id"]
    fields = (
        ("Research program", experiment["program"]),
        ("Branch snapshot", experiment["branch"]),
        ("Source commit", experiment.get("commit")),
        ("Question", experiment["question"]),
        ("Treatment", experiment.get("treatment")),
        ("Control", experiment.get("control")),
        ("Model and data", f"{_render(experiment.get('model'))}; {_render(experiment.get('data'))}"),
        ("Runtime and seeds", f"{_render(experiment.get('runtime'))}; {_render(experiment.get('seeds'))}"),
        ("Recorded result", experiment["result"]),
        ("Evidence status", experiment["status"]),
        ("Evidence class", experiment["evidence_class"]),
        ("Replication grade", experiment["replication"]),
        ("Metrics", experiment.get("metrics")),
        ("Supported claim", experiment["supported_claim"]),
        ("Unsupported claim", experiment["unsupported_claim"]),
        ("Confounds and provenance caveats", experiment.get("confounds")),
        ("Artifact references", experiment.get("artifacts")),
        ("Supersedes", experiment.get("supersedes")),
        ("Superseded by", experiment.get("superseded_by")),
        ("Related evidence", experiment.get("relations")),
        ("Next discriminating experiment", experiment.get("next_discriminating_experiment")),
    )
    text = "\n".join(f"{name}: {_render(value)}" for name, value in fields)
    text += (
        "\nInterpretation rule: keep the recorded status and evidence class attached to every claim. "
        "A design, implementation, inconclusive result, or unverified transcription is not a demonstrated capability. "
        "Do not infer beyond the stated supported claim, artifacts, and limits."
    )
    record_sha256 = _sha256(_canonical_json(experiment))
    return {
        "doc_id": f"signac-evidence-{identifier}",
        "source_id": f"experiment-ledger:{identifier}",
        "text": text,
        "domain": "first-party-research-evidence",
        "family": RESEARCH_FAMILY,
        "authorization_category": "first-party-authorized",
        "raw_source_sha256": record_sha256,
        "record_sha256": record_sha256,
        "ledger_source_sha256": ledger_sha256,
        "record_id": identifier,
        "basis_head": basis_head,
        "ledger_generated": generated,
    }


def _curated_index_documents(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    source_sha256 = _sha256(raw)
    documents: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("| **") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 4:
            raise ValueError("curated Signac evidence row must have exactly four columns")
        lineage, finding, consequence, limit = cells
        slug = re.sub(r"[^a-z0-9]+", "-", lineage.lower()).strip("-")
        if not slug or not finding or not consequence or not limit:
            raise ValueError("curated Signac evidence row contains an empty field")
        body = (
            f"Research lineage: {lineage}\n"
            f"Recorded finding: {finding}\n"
            f"Design consequence: {consequence}\n"
            f"Evidence limit: {limit}\n"
            "Interpretation rule: this is a scoped research summary, not a general law or an AGI result. "
            "Preserve the stated uncertainty and do not convert design consequences into outcome claims."
        )
        documents.append({
            "doc_id": f"signac-index-{slug}",
            "source_id": f"signac-evidence-index:{slug}",
            "text": body,
            "domain": "first-party-research-evidence",
            "family": RESEARCH_FAMILY,
            "authorization_category": "first-party-authorized",
            "raw_source_sha256": _sha256(stripped.encode("utf-8")),
            "record_sha256": _sha256(body.encode("utf-8")),
            "ledger_source_sha256": source_sha256,
            "record_id": slug,
            "basis_head": "curated Signac evidence index; see cited source files",
            "ledger_generated": "curated by the Signac branch",
        })
    if not documents:
        raise ValueError("curated Signac evidence index contains no research rows")
    return documents, source_sha256


def build_research_evidence_corpus(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Freeze structured experiment records plus the current Signac evidence index."""

    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    ledger_path = root / "docs" / "research" / "EXPERIMENT_EVIDENCE_LEDGER.json"
    curated_path = root / "docs" / "signac_100m" / "EVIDENCE_LEDGER.md"
    ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw.decode("utf-8"))
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise ValueError("unsupported repository experiment evidence ledger schema")
    status_values = set(ledger.get("evidence_status_enum", ()))
    class_values = set(ledger.get("evidence_class_enum", ()))
    replication_values = set(ledger.get("replication_enum", ()))
    if (status_values != STATUS_VALUES or class_values != EVIDENCE_CLASS_VALUES
            or replication_values != REPLICATION_VALUES):
        raise ValueError("repository evidence ledger enum contract changed")
    experiments = ledger.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("repository evidence ledger has no experiment records")
    identifiers = [row.get("id") for row in experiments if isinstance(row, Mapping)]
    if len(identifiers) != len(experiments) or len(set(identifiers)) != len(identifiers):
        raise ValueError("repository evidence ledger has missing or duplicate experiment ids")
    ledger_sha256 = _sha256(ledger_raw)
    basis_head = str(ledger.get("basis_head", ""))
    generated = str(ledger.get("generated", ""))
    if not basis_head or not generated:
        raise ValueError("repository evidence ledger lacks its snapshot identity")
    records = [
        _experiment_document(
            experiment, ledger_sha256=ledger_sha256, basis_head=basis_head, generated=generated
        )
        for experiment in sorted(experiments, key=lambda row: row["id"])
    ]
    curated, curated_sha256 = _curated_index_documents(curated_path)
    documents = records + curated
    body = {
        "schema": CORPUS_SCHEMA,
        "status": "DEVELOPMENT_RESEARCH_SNAPSHOT",
        "ledger_schema": LEDGER_SCHEMA,
        "ledger_generated": generated,
        "ledger_basis_head": basis_head,
        "ledger_sha256": ledger_sha256,
        "curated_index_sha256": curated_sha256,
        "research_records": len(records),
        "curated_index_records": len(curated),
        "documents": documents,
        "claim_ceiling": (
            "Repository research snapshot for development training; evidence grades and source limits are retained. "
            "Not a current cross-branch checkout, independent audit, production corpus, or capability result."
        ),
    }
    body["sha256"] = _sha256(_canonical_json(body))
    return body


__all__ = ["CORPUS_SCHEMA", "RESEARCH_FAMILY", "build_research_evidence_corpus"]
