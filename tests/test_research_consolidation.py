from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.validate_evidence_ledger import (
    EXPECTED_IMPORT_COUNT,
    EXPECTED_LEDGER_COUNT,
    EXPECTED_MANIFEST_SHA256,
    GitRepository,
    MANIFEST_RELATIVE,
    load_json,
    validate_ledger,
    validate_manifest,
)
from tools.validate_research_decision_model import (
    validate_current_invariants,
    validate_documents,
    validate_markdown_ledger,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest_result():
    repository = GitRepository(ROOT)
    manifest_path = ROOT / MANIFEST_RELATIVE
    manifest = load_json(manifest_path)
    problems: list[str] = []
    paths = validate_manifest(manifest, manifest_path, ROOT, repository, problems)
    return manifest, paths, problems


@pytest.fixture(scope="module")
def model_result():
    return validate_documents(ROOT)


def test_manifest_hash_and_all_imported_blob_records(manifest_result) -> None:
    manifest, paths, problems = manifest_result
    assert not problems
    assert hashlib.sha256((ROOT / MANIFEST_RELATIVE).read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    assert manifest["schema"] == "anra.evidence-source-manifest/v1"
    assert len(manifest["imported_paths"]) == EXPECTED_IMPORT_COUNT
    assert len(paths) == EXPECTED_IMPORT_COUNT
    assert all(len(record["blob"]) == 40 for record in manifest["imported_paths"])
    assert all(record["size_bytes"] >= 0 for record in manifest["imported_paths"])


def test_all_ledger_rows_and_artifacts_validate(manifest_result) -> None:
    _, manifest_paths, manifest_problems = manifest_result
    assert not manifest_problems
    ledger = load_json(ROOT / "docs/research/EXPERIMENT_EVIDENCE_LEDGER.json")
    problems: list[str] = []
    validate_ledger(ledger, ROOT, GitRepository(ROOT), manifest_paths, problems)
    assert not problems
    assert len(ledger["experiments"]) == EXPECTED_LEDGER_COUNT
    assert len({row["id"] for row in ledger["experiments"]}) == EXPECTED_LEDGER_COUNT
    assert all(len(row) == 24 for row in ledger["experiments"])
    assert any("/" in artifact for row in ledger["experiments"] for artifact in row["artifacts"])


def test_schema_and_reference_integrity(model_result) -> None:
    problems, documents = model_result
    assert not problems
    assert documents["ledger"]["schema"] == "anra.evidence-ledger/v1"
    assert documents["beliefs"]["schema"] == "anra.belief-registry/v1"
    assert documents["architecture"]["schema"] == "anra.architecture-decision-ledger/v1"
    assert documents["model"]["schema"] == "anra.research-decision-model/v1"
    assert documents["dependency"]["schema"] == "anra.research-dependency-graph/v1"
    assert documents["tree"]["schema"] == "anra.next-experiment-decision-tree/v1"
    assert len(documents["beliefs"]["beliefs"]) == 32
    assert len(documents["architecture"]["decisions"]) == 33
    assert len(documents["model"]["unknowns"]) == 11
    assert len(documents["model"]["candidate_experiments"]) == 18
    assert len(documents["dependency"]["critical_paths"]) == 5


def test_current_phase_three_invariants(model_result) -> None:
    _, documents = model_result
    problems: list[str] = []
    validate_current_invariants(
        documents["ledger"], documents["beliefs"], documents["architecture"],
        documents["model"], documents["dependency"], documents["tree"], problems,
    )
    assert not problems
    assert documents["model"]["ranking"]["top_three"] == [
        "FMUX-CONTROL-METRIC-PREFLIGHT",
        "ROLE-TRANSFER-001",
        "CORPUS-REGEN",
    ]
    assert documents["model"]["kill_criteria_detail"][0]["status"] == "FIRED"


def test_markdown_status_counts_and_complete_index_reconcile(model_result) -> None:
    _, documents = model_result
    problems: list[str] = []
    validate_markdown_ledger(ROOT / "docs/research/EXPERIMENT_EVIDENCE_LEDGER.md", documents["ledger"], problems)
    assert not problems


def test_manifest_blob_mismatch_fails_closed(manifest_result) -> None:
    manifest, _, _ = manifest_result
    mutated = copy.deepcopy(manifest)
    mutated["imported_paths"][0]["blob"] = "0" * 40
    problems: list[str] = []
    validate_manifest(mutated, ROOT / MANIFEST_RELATIVE, ROOT, GitRepository(ROOT), problems)
    assert any("blob mismatch" in problem for problem in problems)


def test_short_commit_on_current_row_fails_closed(manifest_result) -> None:
    _, manifest_paths, _ = manifest_result
    ledger = copy.deepcopy(load_json(ROOT / "docs/research/EXPERIMENT_EVIDENCE_LEDGER.json"))
    ledger["experiments"][0]["commit"] = "301f5f2"
    problems: list[str] = []
    validate_ledger(ledger, ROOT, GitRepository(ROOT), manifest_paths, problems)
    assert any("full 40-hex" in problem for problem in problems)


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"value": 1, "value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_json(path)


def test_validators_pass_on_checked_in_tree() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for script in ("validate_evidence_ledger.py", "validate_research_decision_model.py"):
        result = subprocess.run(
            [sys.executable, "-B", f"tools/{script}"],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
