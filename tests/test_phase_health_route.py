"""The phase-3 health endpoint must report real subsystem state, never assert ok.

Historically ``/sovereignty/status`` loaded phase-3 modules by bare name (always
ModuleNotFoundError) yet returned a hardcoded top-level ``status: ok``. These
tests lock in honest behavior: the aggregate status is the conjunction of the
real ``health_check`` results, and a genuinely broken subsystem surfaces.
"""

from __future__ import annotations

import asyncio

import app


def test_phase3_health_loads_real_modules_and_reports_ok() -> None:
    checks = app._run_phase3_health_checks()
    # Every canonical phase-3 subsystem must be reachable and healthy.
    assert set(checks) == {
        "identity",
        "ouroboros",
        "symbolic",
        "sovereignty",
        "ghost_memory",
    }
    assert all(check.get("status") == "ok" for check in checks.values()), checks


def test_phase_health_status_is_honest_conjunction() -> None:
    result = asyncio.run(app.phase_health_route())
    assert result["status"] == "ok"
    assert result["degraded_subsystems"] == []
    assert set(result["phase3_health"]) >= {"identity", "sovereignty"}


def test_phase_health_reports_degraded_when_a_subsystem_fails(monkeypatch) -> None:
    # Point one entry at a module that cannot exist; the endpoint must degrade
    # honestly rather than paper over the failure with a hardcoded "ok".
    broken = tuple(
        (key, "phase3.definitely_not_a_real_module_xyz" if key == "symbolic" else module_name)
        for key, module_name in app._PHASE3_HEALTH_MODULES
    )
    monkeypatch.setattr(app, "_PHASE3_HEALTH_MODULES", broken)
    result = asyncio.run(app.phase_health_route())
    assert result["status"] == "degraded"
    assert "symbolic" in result["degraded_subsystems"]
    assert result["phase3_health"]["symbolic"]["status"] == "degraded"
    assert "ModuleNotFoundError" in result["phase3_health"]["symbolic"]["detail"]
