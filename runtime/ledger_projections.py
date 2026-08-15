"""Privacy-preserving projections from the Experience Ledger for UI trust views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from runtime.experience_ledger import ExperienceLedger, content_hash


def _event_header(event: Mapping[str, Any]) -> dict[str, object]:
    return {
        "event_id": str(event.get("event_id", "")),
        "trace_id": str(event.get("trace_id", "")),
        "timestamp": str(event.get("ts", "")),
        "kind": str(event.get("kind", "")),
        "source": str(event.get("source", "")),
    }


def _safe_verdict(verdict: Mapping[str, Any]) -> dict[str, object]:
    return {
        "name": str(verdict.get("name", "unknown")),
        "score": float(verdict.get("score", 0.0)),
        "passed": bool(verdict.get("passed", False)),
        "tier": int(verdict.get("tier", 1)),
        "reason": str(verdict.get("reason", "not_recorded")),
    }


def get_verification_projection(ledger: ExperienceLedger) -> list[dict[str, object]]:
    """Render verifier status without exposing prompts, answers, or tool output."""
    projections: list[dict[str, object]] = []
    for event in ledger.iter_events(validate=False):
        verdicts = event.get("verifier_verdicts", [])
        if not isinstance(verdicts, list | tuple) or not verdicts:
            continue
        safe = [_safe_verdict(item) for item in verdicts if isinstance(item, Mapping)]
        if not safe:
            continue
        projections.append(
            {
                **_event_header(event),
                "verdicts": safe,
                "overall_passed": all(bool(item["passed"]) for item in safe),
                "output_hash": content_hash(event.get("output")),
            }
        )
    return projections


def get_memory_projection(ledger: ExperienceLedger) -> list[dict[str, object]]:
    """Render memory lifecycle by record identifiers and hashes only."""
    projections: list[dict[str, object]] = []
    memory_kinds = {"memory_write", "memory_recall", "memory_edit", "memory_forget"}
    for event in ledger.iter_events(validate=False):
        if event.get("kind") not in memory_kinds:
            continue
        output = event.get("output")
        output_map = output if isinstance(output, Mapping) else {}
        metadata = event.get("metadata")
        metadata_map = metadata if isinstance(metadata, Mapping) else {}
        record_ids = output_map.get("record_ids", [])
        if not isinstance(record_ids, list | tuple):
            record_ids = []
        replacement = output_map.get("replacement_record_id")
        record_id = output_map.get("record_id")
        projections.append(
            {
                **_event_header(event),
                "operation": str(event.get("kind", "")).removeprefix("memory_"),
                "record_ids": [str(item) for item in record_ids if item],
                "record_id": str(record_id) if record_id else None,
                "replacement_record_id": str(replacement) if replacement else None,
                "hit_count": int(output_map.get("hit_count", 0) or 0),
                "deleted": bool(output_map.get("deleted", False)),
                "updated": bool(output_map.get("updated", False)),
                "content_hash": str(metadata_map.get("content_hash", "")),
            }
        )
    return projections


def get_gate_visibility_projection(ledger: ExperienceLedger) -> list[dict[str, object]]:
    """Render explicit authorization decisions without arbitrary gate payloads."""
    projections: list[dict[str, object]] = []
    for event in ledger.iter_events(validate=False):
        gate = event.get("gate_record")
        if not isinstance(gate, Mapping) or not gate:
            continue
        projections.append(
            {
                **_event_header(event),
                "action_allowed": bool(gate.get("allowed", False)),
                "gate": str(gate.get("gate", "unknown")),
                "role": str(gate.get("role", "")),
                "reason": str(gate.get("reason", "")),
            }
        )
    return projections


def projection_for_trace(ledger: ExperienceLedger, trace_id: str) -> dict[str, object]:
    """Return the three UI projections for one trace, never raw ledger events."""
    if not trace_id:
        raise ValueError("trace_id is required")

    def select(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
        return [row for row in rows if row.get("trace_id") == trace_id]

    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "verification": select(get_verification_projection(ledger)),
        "memory": select(get_memory_projection(ledger)),
        "gates": select(get_gate_visibility_projection(ledger)),
    }
