from pathlib import Path

from runtime.experience_ledger import ExperienceLedger, content_hash
from runtime.ledger_projections import (
    get_gate_visibility_projection,
    get_memory_projection,
    get_verification_projection,
    projection_for_trace,
)


def test_ledger_projections_are_traceable_but_do_not_leak_content(tmp_path: Path) -> None:
    ledger = ExperienceLedger(tmp_path / "ledger")
    trace_id = "trace-projection-1"
    ledger.record(
        kind="chat",
        trace_id=trace_id,
        inputs={"prompt": "solve 2+2"},
        output="4",
        verifier_verdicts=[
            {"name": "math", "passed": True, "score": 1.0, "tier": 1, "reason": "exact"}
        ],
    )
    secret = "private memory content"
    ledger.record(
        kind="memory_write",
        trace_id=trace_id,
        inputs={"content_hash": content_hash(secret)},
        output={"record_id": "memory-1", "tier": "episodic"},
        metadata={"content_hash": content_hash(secret)},
    )
    ledger.record(
        kind="tool_call",
        trace_id=trace_id,
        inputs={"tool": "write_file"},
        gate_record={"allowed": False, "reason": "sovereignty_denied", "gate": "sovereignty"},
    )

    verifications = get_verification_projection(ledger)
    memories = get_memory_projection(ledger)
    gates = get_gate_visibility_projection(ledger)
    projection = projection_for_trace(ledger, trace_id)

    assert verifications[0]["overall_passed"] is True
    assert memories[0]["operation"] == "write"
    assert memories[0]["record_id"] == "memory-1"
    assert gates[0]["action_allowed"] is False
    assert gates[0]["reason"] == "sovereignty_denied"
    assert len(projection["verification"]) == 1
    assert secret not in str(projection)
