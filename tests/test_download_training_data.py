from __future__ import annotations

from scripts import download_training_data as downloader
from scripts.audit_foundation_records import audit_foundation_records


def test_native_foundation_uses_standard_streaming_dataset_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(downloader, "TRAINING_DATA_DIR", tmp_path)
    monkeypatch.setattr(downloader, "resolve_dataset_revision", lambda _name: "a" * 40)
    calls: list[dict[str, object]] = []

    def load_dataset(name: str, config: str | None = None, **kwargs: object):
        calls.append({"name": name, "config": config, **kwargs})
        source_token = name.replace("/", "_").replace("-", "_")
        text = (f"the and {source_token} verified clean training document ") * 20
        return [{"text": text, "content": text, "language": "text", "license": "MIT"}]

    stats = downloader.download_native_foundation(
        load_dataset,
        target_gb=0.000001,
    )

    assert stats["errors"] == []
    assert stats["documents"] == 4
    assert calls and all("trust_remote_code" not in call for call in calls)
    assert (tmp_path / "foundation_records.jsonl").is_file()


def test_native_foundation_resume_uses_audited_index_without_truncation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(downloader, "TRAINING_DATA_DIR", tmp_path)
    audit_report = tmp_path / "audit.json"
    audit_index = tmp_path / "index.sqlite3"
    monkeypatch.setattr(downloader, "FOUNDATION_AUDIT_REPORT", audit_report)
    monkeypatch.setattr(downloader, "FOUNDATION_RESUME_INDEX", audit_index)
    monkeypatch.setattr(downloader, "resolve_dataset_revision", lambda _name: "b" * 40)
    calls: list[str] = []

    def load_dataset(name: str, config: str | None = None, **_kwargs: object):
        del config
        calls.append(name)
        source_token = name.replace("/", "_").replace("-", "_")
        text = (f"the and {source_token} verified resumable training document ") * 20
        return [{"text": text, "content": text, "language": "text", "license": "MIT"}]

    downloader.download_native_foundation(load_dataset, target_gb=0.000001)
    corpus = tmp_path / "foundation_records.jsonl"
    original = corpus.read_bytes()
    audit_foundation_records(
        corpus,
        report_path=audit_report,
        index_path=audit_index,
        target_bytes=1,
    )
    calls.clear()

    resumed = downloader.download_native_foundation(
        load_dataset,
        target_gb=0.000001,
        resume=True,
    )

    assert corpus.read_bytes() == original
    assert calls == []
    assert resumed["documents"] == 4
    assert all(row["downloaded_this_run_bytes"] == 0 for row in resumed["sources"].values())
