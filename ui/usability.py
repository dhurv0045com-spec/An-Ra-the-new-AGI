"""Versioned 20-scenario acceptance script for the ledger-derived trust UI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UsabilityScenario:
    scenario_id: str
    title: str
    required_evidence: tuple[str, ...]


USABILITY_SCENARIOS: tuple[UsabilityScenario, ...] = (
    UsabilityScenario(
        "first_run", "First run shows model and trust state", ("response", "proof")
    ),
    UsabilityScenario(
        "verified_answer", "Inspect a verified answer contract", ("trace_id", "trust")
    ),
    UsabilityScenario("unverified_answer", "Unverified answer is labelled honestly", ("proof",)),
    UsabilityScenario("memory_write", "Memory write appears by record ID", ("memory",)),
    UsabilityScenario("memory_recall", "Memory recall has provenance", ("memory",)),
    UsabilityScenario("memory_edit", "Memory edit shows replacement ID", ("memory",)),
    UsabilityScenario("memory_forget", "Memory deletion is visible", ("memory",)),
    UsabilityScenario(
        "memory_injection",
        "Tainted retrieved memory is excluded",
        ("blocked_tainted_context",),
    ),
    UsabilityScenario("gate_allowed", "Allowed gate decision is inspectable", ("gates",)),
    UsabilityScenario("gate_denied", "Denied gate decision is inspectable", ("gates",)),
    UsabilityScenario("session_restore", "Session continuity retains trace linkage", ("trace_id",)),
    UsabilityScenario("proof_hash", "Contract hash is present", ("contract_hash",)),
    UsabilityScenario(
        "verdict_detail", "Verifier reason and score are rendered", ("verification",)
    ),
    UsabilityScenario("no_raw_prompt", "Trust view hides raw prompt", ("trust",)),
    UsabilityScenario("no_raw_memory", "Trust view hides raw memory", ("trust",)),
    UsabilityScenario("adapter_lineage", "Serving provenance names adapter", ("adapter",)),
    UsabilityScenario("latency_budget", "Latency budget result is visible", ("latency",)),
    UsabilityScenario("rollback", "Rollback evidence is inspectable", ("rollback",)),
    UsabilityScenario("canary", "Canary result is inspectable", ("canary",)),
    UsabilityScenario(
        "release_bundle", "Signed release bundle is inspectable", ("release_bundle",)
    ),
)


def run_usability_script(
    observer: Callable[[UsabilityScenario], Mapping[str, object]],
) -> dict[str, object]:
    """Run all scenarios and require the observer to return explicit evidence."""
    results = []
    for scenario in USABILITY_SCENARIOS:
        observed = observer(scenario)
        evidence = {str(key) for key, value in observed.items() if value}
        missing = [item for item in scenario.required_evidence if item not in evidence]
        results.append(
            {
                "scenario_id": scenario.scenario_id,
                "passed": not missing,
                "missing": missing,
            }
        )
    return {
        "schema_version": 1,
        "scenarios": [asdict(item) for item in USABILITY_SCENARIOS],
        "results": results,
        "passed": all(item["passed"] for item in results),
    }
