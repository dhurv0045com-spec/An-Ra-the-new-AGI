from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from v5_data import corpus_loading


class WordTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


def test_checked_in_manifest_loads_all_curated_research_sources() -> None:
    repo = Path(__file__).resolve().parents[1]
    documents = corpus_loading._load_corpus(repo, WordTokenizer())  # type: ignore[arg-type]

    assert len(documents) == 23
    assert sum(len(document.text.encode("utf-8")) for document in documents) <= (
        corpus_loading.MAX_CORPUS_BYTES
    )
    assert "docs/research/EXPERIMENT_EVIDENCE_LEDGER.md" in {
        document.doc_id for document in documents
    }
    assert "docs/cymek/research/MASTER_AGI_CONSTRUCTION_KNOWLEDGE.md" in {
        document.doc_id for document in documents
    }


def _make_corpus(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    texts: list[str] | None = None,
    first_path: str = "docs/research/canary-00.md",
) -> list[str]:
    repo.mkdir(parents=True, exist_ok=True)
    texts = texts or [f"pinned research note {index}" for index in range(8)]
    entries: list[dict[str, str]] = []
    paths = [first_path] + [f"docs/research/canary-{index:02d}.md" for index in range(1, len(texts))]
    for index, (relative, text) in enumerate(zip(paths, texts, strict=True)):
        if not (".." in Path(relative).parts or relative.startswith("/")):
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = text.encode("utf-8")
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
        else:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        entries.append(
            {
                "acquired_date": "2026-09-24",
                "authorization_category": "first-party-development-only",
                "domain": "prose",
                "doc_id": relative,
                "family": "natural",
                "license": "MIT",
                "path": relative,
                "provenance": "test-only pinned first-party canary document",
                "sha256": digest,
            }
        )
    entries.sort(key=lambda entry: entry["path"])
    manifest = {
        "acquired_date": "2026-09-24",
        "corpus_id": "signac-first-party-research-canary-v1",
        "documents": entries,
        "license": "MIT",
        "provenance": "test-only development canary corpus",
        "schema": corpus_loading.CORPUS_MANIFEST_SCHEMA,
        "usage_scope": "development-canary-only",
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    manifest_path = repo / corpus_loading.CORPUS_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(encoded)
    monkeypatch.setattr(
        corpus_loading, "CORPUS_MANIFEST_SHA256", hashlib.sha256(encoded).hexdigest()
    )
    return paths


def test_loader_uses_exact_pinned_manifest_without_git_and_ignores_workspace_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_corpus(tmp_path, monkeypatch)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/evaluation_truth.md").write_text("secret answer", encoding="utf-8")
    (tmp_path / "docs/research/unlisted.md").write_text("unlisted note", encoding="utf-8")
    (tmp_path / "v5_evaluation").mkdir()
    (tmp_path / "v5_evaluation/gold.md").write_text("held out", encoding="utf-8")

    assert not (tmp_path / ".git").exists()
    documents = corpus_loading._load_corpus(tmp_path, WordTokenizer())  # type: ignore[arg-type]

    assert [document.doc_id for document in documents] == sorted(paths)
    assert all(
        document.authorization_category == "first-party-development-only"
        for document in documents
    )
    assert all(document.raw_source_sha256 for document in documents)
    assert {document.text for document in documents}.isdisjoint(
        {"secret answer", "unlisted note", "held out"}
    )


def test_loader_rejects_changed_listed_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_corpus(tmp_path, monkeypatch)
    (tmp_path / paths[0]).write_text("edited after freeze", encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match"):
        corpus_loading._load_corpus(tmp_path, WordTokenizer())  # type: ignore[arg-type]


def test_loader_rejects_manifest_bytes_that_do_not_match_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_corpus(tmp_path, monkeypatch)
    path = tmp_path / corpus_loading.CORPUS_MANIFEST_RELATIVE_PATH
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="manifest digest"):
        corpus_loading._load_corpus(tmp_path, WordTokenizer())  # type: ignore[arg-type]


def test_loader_rejects_path_traversal_even_when_manifest_digest_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_corpus(tmp_path, monkeypatch, first_path="../tests/evaluation_truth.md")

    with pytest.raises(ValueError, match="unsafe or out-of-scope"):
        corpus_loading._load_corpus(tmp_path, WordTokenizer())  # type: ignore[arg-type]


def test_loader_refuses_to_truncate_overlong_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    texts = ["token " * (corpus_loading.MAX_DOCUMENT_TOKENS + 1)] + [
        f"short note {index}" for index in range(7)
    ]
    _make_corpus(tmp_path, monkeypatch, texts=texts)

    with pytest.raises(ValueError, match="refusing truncation"):
        corpus_loading._load_corpus(tmp_path, WordTokenizer())  # type: ignore[arg-type]


def test_loader_enforces_aggregate_byte_bound_before_loading_all_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_corpus(tmp_path, monkeypatch, texts=["x" * 65_000 for _ in range(8)])

    with pytest.raises(ValueError, match="aggregate byte limit"):
        corpus_loading._load_corpus(tmp_path, WordTokenizer())  # type: ignore[arg-type]


def test_loader_enforces_per_document_byte_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    texts = ["x" * (corpus_loading.MAX_DOCUMENT_BYTES + 1)] + ["short" for _ in range(7)]
    _make_corpus(tmp_path, monkeypatch, texts=texts)

    with pytest.raises(ValueError, match="document exceeds its byte limit"):
        corpus_loading._load_corpus(tmp_path, WordTokenizer())  # type: ignore[arg-type]
