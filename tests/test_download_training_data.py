from __future__ import annotations

from scripts import download_training_data as downloader


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
