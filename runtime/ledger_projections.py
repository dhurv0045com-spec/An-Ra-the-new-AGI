"""Projections from the durable Experience Ledger into UI-ready views."""

from __future__ import annotations

from typing import Any
from runtime.experience_ledger import ExperienceLedger


def get_verification_projection(ledger: ExperienceLedger) -> list[dict[str, Any]]:
    """Extract verifier verdicts for proof-carrying answer chips in the UI."""
    projections = []
    
    for event in ledger.iter_events(validate=False):
        verdicts = event.get("verifier_verdicts", [])
        if not verdicts:
            continue
            
        projections.append({
            "event_id": event.get("event_id"),
            "trace_id": event.get("trace_id"),
            "timestamp": event.get("ts"),
            "verdicts": verdicts,
            "overall_passed": all(v.get("passed", False) for v in verdicts),
            "output_snippet": str(event.get("output", ""))[:100] + "..." if event.get("output") else ""
        })
        
    return projections


def get_memory_projection(ledger: ExperienceLedger) -> list[dict[str, Any]]:
    """Extract memory operations to render the transparency view."""
    projections = []
    
    for event in ledger.iter_events(validate=False):
        metadata = event.get("metadata", {})
        memory_ops = metadata.get("memory_operations", [])
        if not memory_ops:
            continue
            
        projections.append({
            "event_id": event.get("event_id"),
            "trace_id": event.get("trace_id"),
            "timestamp": event.get("ts"),
            "memory_operations": memory_ops,
        })
        
    return projections


def get_gate_visibility_projection(ledger: ExperienceLedger) -> list[dict[str, Any]]:
    """Extract autonomous decisions and gate records for the visibility log."""
    projections = []
    
    for event in ledger.iter_events(validate=False):
        gate_record = event.get("gate_record", {})
        if not gate_record:
            continue
            
        projections.append({
            "event_id": event.get("event_id"),
            "trace_id": event.get("trace_id"),
            "timestamp": event.get("ts"),
            "gate_record": gate_record,
            "action_allowed": gate_record.get("allowed", False),
            "gate_reason": gate_record.get("reason", "unknown")
        })
        
    return projections
