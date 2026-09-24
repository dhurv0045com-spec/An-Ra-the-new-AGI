from __future__ import annotations

from pathlib import Path

from signac_100m.source_identity import SOURCE_DIRECTORIES, SOURCE_FILES, build_source_identity


def test_source_identity_is_content_addressed_without_git_metadata(tmp_path: Path):
    for directory in SOURCE_DIRECTORIES:
        target = tmp_path / directory / "probe.py"
        target.parent.mkdir(parents=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
    for relative in SOURCE_FILES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("TARGET = 'tpu'\n", encoding="utf-8")
    corpus_manifest = tmp_path / "v5_data" / "first_party_corpus_manifest.json"
    corpus_manifest.write_text('{"schema":"test"}', encoding="utf-8")

    first = build_source_identity(tmp_path)
    repeated = build_source_identity(tmp_path)
    assert first == repeated
    assert first["source_commit"] is None
    assert first["file_count"] == len(SOURCE_DIRECTORIES) + len(SOURCE_FILES) + 1
    assert any(
        item["path"] == "v5_data/first_party_corpus_manifest.json"
        for item in first["files"]
    )
    campaign_runner = tmp_path / "v5_training" / "kaggle_xla_development.py"
    campaign_runner.write_text("RUNNER_VERSION = 1\n", encoding="utf-8")
    runner_bound = build_source_identity(tmp_path)
    assert runner_bound["source_tree_sha256"] != first["source_tree_sha256"]
    assert any(
        item["path"] == "v5_training/kaggle_xla_development.py"
        for item in runner_bound["files"]
    )

    (tmp_path / "v5_model" / "probe.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_source_identity(tmp_path)
    assert changed["source_tree_sha256"] != first["source_tree_sha256"]

    corpus_manifest.write_text('{"schema":"test","entry":"changed"}', encoding="utf-8")
    changed_corpus = build_source_identity(tmp_path)
    assert changed_corpus["source_tree_sha256"] != changed["source_tree_sha256"]

    notebook = tmp_path / "notebooks" / "SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb"
    notebook.write_text('{"cells":[],"launch":"changed"}\n', encoding="utf-8")
    changed_launcher = build_source_identity(tmp_path)
    assert changed_launcher["source_tree_sha256"] != changed_corpus["source_tree_sha256"]
