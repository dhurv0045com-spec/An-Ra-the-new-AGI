from __future__ import annotations

import pytest


def test_calculator_is_session_bound_bounded_and_audited(monkeypatch) -> None:
    import runtime.tool_broker as tools

    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        tools,
        "record_experience",
        lambda **kwargs: (events.append(kwargs) or ("trace", True)),
    )
    broker = tools.BoundedToolBroker()
    grant = broker.issue_grant(session_id="session-a", max_calls=1)

    result, receipt = broker.execute(
        capability_id=grant.capability_id,
        session_id="session-a",
        tool="calculator",
        arguments={"expression": "(17 + 25) * 2"},
    )

    assert result == {"expression": "(17 + 25) * 2", "value": 84, "exact": True}
    assert receipt.status == "completed"
    assert receipt.calls_remaining == 0
    assert receipt.ledger_persisted is True
    assert events[0]["kind"] == "tool_execution"
    assert events[0]["output"] == {
        "tool": "calculator",
        "result_hash": receipt.result_hash,
        "exact": True,
        "status": "completed",
    }

    with pytest.raises(tools.ToolPolicyError, match="exhausted"):
        broker.execute(
            capability_id=grant.capability_id,
            session_id="session-a",
            tool="calculator",
            arguments={"expression": "1 + 1"},
        )


def test_calculator_refusal_is_audited_and_cannot_escape_ast(monkeypatch) -> None:
    import runtime.tool_broker as tools

    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        tools,
        "record_experience",
        lambda **kwargs: (events.append(kwargs) or ("trace", True)),
    )
    broker = tools.BoundedToolBroker()
    grant = broker.issue_grant(session_id="session-b", max_calls=2)

    result, receipt = broker.execute(
        capability_id=grant.capability_id,
        session_id="session-b",
        tool="calculator",
        arguments={"expression": "__import__('os').system('whoami')"},
    )

    assert receipt.status == "refused"
    assert result["exact"] is False
    assert "unsupported arithmetic syntax" in str(result["error"])
    assert events[0]["gate_record"] == {
        "allowed": False,
        "gate": "server_issued_session_capability",
        "calls_remaining": 1,
    }


def test_capability_cannot_cross_sessions_or_survive_clear() -> None:
    from runtime.tool_broker import BoundedToolBroker, ToolPolicyError

    broker = BoundedToolBroker()
    grant = broker.issue_grant(session_id="session-c")
    with pytest.raises(ToolPolicyError, match="different session"):
        broker.execute(
            capability_id=grant.capability_id,
            session_id="session-other",
            tool="calculator",
            arguments={"expression": "2 + 2"},
        )
    assert broker.revoke_session("session-c") == 1
    with pytest.raises(ToolPolicyError, match="unknown, expired, or revoked"):
        broker.execute(
            capability_id=grant.capability_id,
            session_id="session-c",
            tool="calculator",
            arguments={"expression": "2 + 2"},
        )


def test_prototype_exposes_explicit_tool_capability_routes() -> None:
    from runtime.sft_prototype import create_app

    app = create_app()
    routes = {route.path for route in app.routes}

    assert {"/api/tools/grants", "/api/tools/execute"} <= routes
