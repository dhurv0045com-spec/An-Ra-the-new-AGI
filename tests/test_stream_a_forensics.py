from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from scripts.freeze_baseline_hashes import (
    build_freeze_report,
    freeze_tokenizer,
    resolve_checkpoint,
)
from scripts.run_checkpoint_forensics import (
    COHERENCE_RECOVERY_GATE,
    assert_generation_device,
    run_forensics,
)
from scripts.run_checkpoint_forensics import publish_forensics_report


def test_resolve_checkpoint_prefers_explicit_then_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANRA_CHECKPOINT_PATH", raising=False)
    canonical = resolve_checkpoint()
    assert canonical.name == "anra_frontier_500m.pt"

    monkeypatch.setenv("ANRA_CHECKPOINT_PATH", str(tmp_path / "from_env.pt"))
    assert resolve_checkpoint().name == "from_env.pt"
    assert resolve_checkpoint(str(tmp_path / "explicit.pt")).name == "explicit.pt"


def test_freeze_report_is_frozen_even_while_checkpoint_is_blocked(tmp_path: Path) -> None:
    report = build_freeze_report(tmp_path / "missing.pt")

    assert report["frozen"] is True
    checkpoint = report["checkpoint"]
    assert checkpoint["available"] is False
    assert checkpoint["status"] == "blocked_on_artifact"

    tokenizer = report["tokenizer"]
    assert tokenizer["probe_count"] == 500
    assert tokenizer["probe_match_vs_manifest"] is True

    config = report["config"]
    assert config["contract"]["parameter_count"] == 499_167_075
    assert config["contract"]["checkpoint_schema_version"] == 7
    assert len(config["contract_sha256"]) == 64

    manifests = report["corpus_manifests"]
    assert manifests["count"] >= 1
    assert "tokenizer_v3.json" in manifests["manifests"]


def test_freeze_report_hashes_a_present_checkpoint(tmp_path: Path) -> None:
    blob = b"not-a-real-checkpoint-but-hashable"
    checkpoint = tmp_path / "anra_frontier_500m.pt"
    checkpoint.write_bytes(blob)

    report = build_freeze_report(checkpoint)
    assert report["checkpoint"]["available"] is True
    assert report["checkpoint"]["sha256"] == hashlib.sha256(blob).hexdigest()
    assert report["checkpoint"]["size_bytes"] == len(blob)


def test_tokenizer_probe_fingerprint_matches_frozen_manifest() -> None:
    tokenizer = freeze_tokenizer()
    assert tokenizer["status"] == "frozen"
    assert tokenizer["probe_sha256"] == tokenizer["manifest_probe_sha256"]


def test_forensics_without_checkpoint_reports_blocked(tmp_path: Path) -> None:
    report = run_forensics(tmp_path / "missing.pt")

    assert report["blocked"] is True
    assert report["complete"] is False
    assert report["steps"]["locate_checkpoint"]["status"] == "blocked"
    assert report["steps"]["tensor_accounting"]["status"] == "blocked"
    assert report["steps"]["tokenizer_probes"]["status"] == "passed"
    assert report["steps"]["recovery_gate"]["status"] == "blocked"
    assert "blocked on the real checkpoint" in str(report["verdict"])


def test_forensics_cuda_requirement_refuses_silent_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="refusing a silent CPU fallback"):
        assert_generation_device("cuda")

    assert_generation_device("auto")


def test_forensics_runs_recovery_gate_with_injected_generator(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    torch.save({"model": {"token_embedding_table.weight": torch.zeros(4, 4)}}, checkpoint)

    def generator(prompt: str, mode: str, seed: int, _ablation: str | None):
        token_ids = [ord(character) % 97 for character in prompt[:8]]
        return SimpleNamespace(
            output="A complete grammatical An-Ra response.",
            output_token_ids=token_ids,
            quality_state="accepted",
            language_fragment_detected=False,
            repeated_ngrams_detected=False,
            stopped_by="eos",
            entropy_curve=[2.0],
            max_prob_curve=[0.8],
        )

    report = run_forensics(checkpoint, generator=generator)

    gate = report["steps"]["recovery_gate"]
    assert gate["status"] in {"passed", "failed"}
    assert report["complete"] is True
    assert 0.0 <= float(gate["coherence_rate"]) <= 1.0
    assert gate["gate"] == COHERENCE_RECOVERY_GATE
    assert gate["report"]["candidate"]["prompt_count"] == 200
    # The fake blob cannot satisfy exact tensor accounting.
    assert report["steps"]["tensor_accounting"]["status"] == "failed"
    assert report["failed"] is True


def test_forensics_publication_refuses_to_downgrade_executed_gate(tmp_path: Path) -> None:
    output = tmp_path / "forensics.json"
    completed = {
        "checkpoint": "same.pt",
        "complete": True,
        "failed": True,
        "steps": {"recovery_gate": {"status": "failed", "report": {"passed": False}}},
    }
    incomplete = {
        "checkpoint": "same.pt",
        "complete": False,
        "failed": False,
        "steps": {"recovery_gate": {"status": "skipped"}},
    }
    assert publish_forensics_report(completed, output)["written"] is True

    publication = publish_forensics_report(incomplete, output)

    assert publication["written"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == completed
