"""Tests for standalone orchestrator_agent.md, execution_agent.md, and agent.md mirror parity."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_two_agent_manuals_mirror_parity_quality_override_and_branch_neutrality() -> None:
    doc_names = ("orchestrator_agent.md", "execution_agent.md", "agent.md")
    contents: dict[str, str] = {}
    for name in doc_names:
        root_bytes = (REPO_ROOT / name).read_bytes()
        pkg_bytes = (REPO_ROOT / "anra_v5" / "planthir" / name).read_bytes()
        res_bytes = (REPO_ROOT / "research" / "planthir" / name).read_bytes()
        assert root_bytes == pkg_bytes == res_bytes, f"mirror mismatch for {name}"
        contents[name] = root_bytes.decode("utf-8")

    orch = contents["orchestrator_agent.md"]
    exec_doc = contents["execution_agent.md"]
    guide = contents["agent.md"]

    for needle in (
        "Quality-First Override Law",
        "HARD SCIENTIFIC LAWS",
        "SOFT ENGINEERING DEFAULTS",
        "PALANTIRI_GAPS_AND_OVERRIDES",
        "PALANTIRI_FEEDBACK",
        "Six-Phase Orchestration Workflow",
        "u0000",
        "pending_confirm_jobs == []",
        "execution_agent.md",
        "verify_full_mode_dispatch",
        "git branch --show-current",
    ):
        assert needle in orch, f"missing {needle!r} in orchestrator_agent.md"

    for needle in (
        "Quality-First Override Law",
        "HARD SCIENTIFIC LAWS",
        "SOFT ENGINEERING DEFAULTS",
        "PALANTIRI_FEEDBACK",
        "gap_or_bug",
        "orchestrator_agent.md",
        "u0000",
        "pending_confirm_jobs == []",
        "run_registered_plan",
        "verify_full_mode_dispatch",
        "verify_session_archive",
        "job.worker_id",
        "SubgoalProbeSpec",
        "forbidden_metadata_key",
        "v5_model.core.initialize",
        "UTF8.decode",
    ):
        assert needle in exec_doc, f"missing {needle!r} in execution_agent.md"

    for needle in (
        "orchestrator_agent.md",
        "execution_agent.md",
        "Quality-First Override Law",
        "PALANTIRI_FEEDBACK",
    ):
        assert needle in guide, f"missing {needle!r} in agent.md"

    for lvl7_needle in (
        "PLANTHIR_LEVEL = 7",
        "shield",
        "decode_lossy",
        "decode_generated_bytes",
        "dump_json_safe",
        "CRASH_RECEIPT.json",
    ):
        assert lvl7_needle in orch, f"missing Level-7 needle {lvl7_needle!r} in orchestrator_agent.md"
        assert lvl7_needle in exec_doc, f"missing Level-7 needle {lvl7_needle!r} in execution_agent.md"
        assert lvl7_needle in guide, f"missing Level-7 needle {lvl7_needle!r} in agent.md"

    hardcoded_branch_claim = "Work, commit, and push directly on **`hitchhiker`**"
    assert hardcoded_branch_claim not in orch
    assert hardcoded_branch_claim not in exec_doc
