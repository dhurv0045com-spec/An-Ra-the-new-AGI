from __future__ import annotations

import hashlib
import json
import sqlite3

from scripts import download_training_data as downloader
from scripts.audit_foundation_records import audit_foundation_records
from tokenizer.subword_tokenizer import SubwordTokenizer


def test_native_foundation_uses_standard_streaming_dataset_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(downloader, "TRAINING_DATA_DIR", tmp_path)
    monkeypatch.setattr(downloader, "DOWNLOAD_PROGRESS", tmp_path / "progress.json")
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


def test_foundation_row_license_requires_every_declared_license_to_be_allowed() -> None:
    allowed, normalized = downloader.foundation_licenses_allowed(
        ["MIT", "Apache License 2.0"]
    )
    assert allowed is True
    assert normalized == ("mit", "apache-2.0")

    mixed, normalized_mixed = downloader.foundation_licenses_allowed(
        ["MIT", "GPL-3.0"]
    )
    assert mixed is False
    assert normalized_mixed == ("mit", "gpl-3-0")

    all_declared = downloader._row_license_values(
        {
            "license": "MIT",
            "metadata": {
                "detected_licenses": ["Apache-2.0"],
                "license": "GPL-3.0",
            },
        },
        "per-record",
    )
    combined, normalized_combined = downloader.foundation_licenses_allowed(all_declared)
    assert combined is False
    assert normalized_combined == ("mit", "apache-2.0", "gpl-3-0")


def test_partial_bucket_status_cannot_overwrite_foundation_status(
    tmp_path, monkeypatch
) -> None:
    foundation_status = tmp_path / "download_status.json"
    monkeypatch.setattr(downloader, "DOWNLOAD_STATUS", foundation_status)
    monkeypatch.setattr(downloader, "DATA_MANIFEST_DIR", tmp_path)

    assert downloader._download_status_path(["base"]) == foundation_status
    assert (
        downloader._download_status_path(["base", "reasoning", "science"])
        == foundation_status
    )
    assert downloader._download_status_path(["reasoning"]) == (
        tmp_path / "download_status_reasoning.json"
    )


def test_native_sources_use_public_immutable_common_pile_replacements(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(downloader, "TRAINING_DATA_DIR", tmp_path)
    monkeypatch.setattr(downloader, "DOWNLOAD_PROGRESS", tmp_path / "progress.json")
    calls: list[dict[str, object]] = []

    def load_dataset(name: str, config: str | None = None, **kwargs: object):
        calls.append({"name": name, "config": config, **kwargs})
        source_token = name.replace("/", "_").replace("-", "_")
        text = (f"the and {source_token} verified clean training document ") * 20
        return [
            {
                "text": text,
                "metadata": {
                    "language": "Markdown",
                    "detected_licenses": ["MIT"],
                    "license": "CC BY 4.0",
                },
            }
        ]

    stats = downloader.download_native_foundation(load_dataset, target_gb=0.000001)

    assert stats["errors"] == []
    names = {str(call["name"]) for call in calls}
    assert "common-pile/stackv2_edu_filtered" in names
    assert "common-pile/arxiv_papers_filtered" in names
    assert "bigcode/the-stack-v2-dedup" not in names
    assert "allenai/dolma" not in names
    assert all(len(str(call["revision"])) == 40 for call in calls)


def test_native_foundation_resume_uses_audited_index_without_truncation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(downloader, "TRAINING_DATA_DIR", tmp_path)
    monkeypatch.setattr(downloader, "DOWNLOAD_PROGRESS", tmp_path / "progress.json")
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
    advanced_audit = json.loads(audit_report.read_text(encoding="utf-8"))
    assert advanced_audit["incremental_append_audit"] is True
    assert advanced_audit["corpus_size_bytes"] == corpus.stat().st_size
    assert advanced_audit["resume_safe"] is True


def test_native_foundation_resume_discards_only_uncommitted_append_tail(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(downloader, "TRAINING_DATA_DIR", tmp_path)
    monkeypatch.setattr(downloader, "DOWNLOAD_PROGRESS", tmp_path / "progress.json")
    audit_report = tmp_path / "audit.json"
    audit_index = tmp_path / "index.sqlite3"
    monkeypatch.setattr(downloader, "FOUNDATION_AUDIT_REPORT", audit_report)
    monkeypatch.setattr(downloader, "FOUNDATION_RESUME_INDEX", audit_index)
    monkeypatch.setattr(downloader, "resolve_dataset_revision", lambda _name: "c" * 40)

    def load_dataset(name: str, config: str | None = None, **_kwargs: object):
        del config
        source_token = name.replace("/", "_").replace("-", "_")
        text = (f"the and {source_token} durable append boundary document ") * 20
        return [{"text": text, "content": text, "language": "text", "license": "MIT"}]

    downloader.download_native_foundation(load_dataset, target_gb=0.000001)
    corpus = tmp_path / "foundation_records.jsonl"
    audit_foundation_records(
        corpus,
        report_path=audit_report,
        index_path=audit_index,
        target_bytes=1,
    )
    base_audit = json.loads(audit_report.read_text(encoding="utf-8"))
    base_size = corpus.stat().st_size
    committed_text = "the and committed online validated recovery record " * 20
    committed_hash = hashlib.sha256(committed_text.encode("utf-8")).hexdigest()
    committed_line = (
        json.dumps(
            {
                "text": committed_text,
                "source": "FineWeb-Edu",
                "license": "odc-by",
                "source_revision": "d" * 40,
                "document_sha256": committed_hash,
            }
        )
        + "\n"
    ).encode("utf-8")
    dangling_tail = b'{"uncommitted": true}\n'
    with corpus.open("ab") as stream:
        stream.write(committed_line)
        stream.write(dangling_tail)
    committed_size = base_size + len(committed_line)
    with sqlite3.connect(audit_index) as connection:
        connection.execute(
            "INSERT INTO documents(document_sha256, source, line_bytes) VALUES (?, ?, ?)",
            (committed_hash, "FineWeb-Edu", len(committed_line)),
        )
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            [
                ("base_report_sha256", base_audit["report_sha256"]),
                ("base_corpus_size_bytes", str(base_size)),
                ("corpus_size_bytes", str(committed_size)),
            ],
        )

    resumed = downloader.download_native_foundation(
        load_dataset,
        target_gb=0.000001,
        resume=True,
    )

    assert corpus.stat().st_size == committed_size
    assert corpus.read_bytes().endswith(committed_line)
    assert dangling_tail not in corpus.read_bytes()
    assert resumed["errors"] == []
    recovered_audit = json.loads(audit_report.read_text(encoding="utf-8"))
    assert recovered_audit["corpus_size_bytes"] == committed_size
    assert recovered_audit["resume_safe"] is True


def test_v4_shard_publication_binds_family_tokenizer_and_inventory(
    tmp_path, monkeypatch
) -> None:
    training_data = tmp_path / "training_data"
    manifests = tmp_path / "manifests"
    training_data.mkdir()
    text = "the and verified educational document for immutable tokenizer binding " * 20
    record = {
        "text": text,
        "source": "FineWeb-Edu",
        "license": "ODC-By",
        "source_revision": "a" * 40,
        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    (training_data / "foundation_records.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    tokenizer = SubwordTokenizer.train_from_texts(
        [text], vocab_size=64, min_frequency=1, allow_fallback=True
    )
    tokenizer_path = tokenizer.save(tmp_path / "tokenizer_v4.json")

    monkeypatch.setattr(downloader, "TRAINING_DATA_DIR", training_data)
    monkeypatch.setattr(downloader, "DATA_MANIFEST_DIR", manifests)
    monkeypatch.setattr(
        downloader, "TOKEN_INVENTORY_MANIFEST", manifests / "global_inventory.json"
    )
    monkeypatch.setattr(
        downloader, "TOKEN_SHARD_PROGRESS", manifests / "shard_progress.json"
    )
    monkeypatch.setattr(downloader, "get_identity_file", lambda: None)

    inventory = downloader.publish_fineweb_token_shards(
        "smoke", tokenizer_path=tokenizer_path, tokenizer_family="v4"
    )

    family_root = manifests / "native_foundation_v4" / "smoke"
    train_manifest = json.loads(
        (family_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert inventory["tokenizer_family"] == "v4"
    assert inventory["tokenizer_sha256"] == hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest()
    assert train_manifest["tokenizer_sha256"] == inventory["tokenizer_sha256"]
    assert (family_root / "token_inventory.json").is_file()
    progress = json.loads(
        (manifests / "shard_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "complete"
    assert progress["tokenizer_family"] == "v4"
    assert not (manifests / "global_inventory.json").exists()
