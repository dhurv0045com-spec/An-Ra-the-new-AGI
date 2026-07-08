from pathlib import Path
from runtime.experience_ledger import ExperienceLedger
from runtime.ledger_projections import (
    get_verification_projection,
    get_memory_projection,
    get_gate_visibility_projection
)


def test_ledger_projections(tmp_path: Path):
    ledger = ExperienceLedger(tmp_path / "ledger")
    
    # Add an event with verifier verdicts
    ledger.record(
        kind="chat",
        inputs={"prompt": "solve 2+2"},
        output="4",
        verifier_verdicts=[{"passed": True, "score": 1.0}],
    )
    
    # Add an event with memory operations
    ledger.record(
        kind="chat",
        inputs={"prompt": "remember this"},
        output="ok",
        metadata={"memory_operations": [{"action": "write", "content": "this"}]}
    )
    
    # Add an event with a gate record
    ledger.record(
        kind="tool_call",
        inputs={"tool": "write_file"},
        gate_record={"allowed": False, "reason": "sovereignty_denied"}
    )
    
    verifications = get_verification_projection(ledger)
    assert len(verifications) == 1
    assert verifications[0]["overall_passed"] is True
    
    memories = get_memory_projection(ledger)
    assert len(memories) == 1
    assert memories[0]["memory_operations"][0]["action"] == "write"
    
    gates = get_gate_visibility_projection(ledger)
    assert len(gates) == 1
    assert gates[0]["action_allowed"] is False
    assert gates[0]["gate_reason"] == "sovereignty_denied"
