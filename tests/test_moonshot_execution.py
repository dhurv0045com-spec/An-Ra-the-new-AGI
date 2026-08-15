from __future__ import annotations

from training.moonshot_execution import execute_local_moonshot_paths


def test_local_moonshot_execution_distinguishes_smoke_from_acceptance() -> None:
    report = execute_local_moonshot_paths()
    assert report["all_local_paths_executed"] is True
    assert report["all_local_smokes_passed"] is True
    rows = {row["moonshot_id"]: row for row in report["rows"]}  # type: ignore[index]
    assert rows["m6"]["acceptance_status"] == "passed"
    assert all(rows[pilot_id]["acceptance_status"] == "blocked" for pilot_id in ("m1", "m2", "m3", "m4", "m5", "m7"))
    assert report["acceptance_evidence"] == {
        "m6": {"proof_cases": 100, "deterministic_pass_rate": 1.0}
    }
