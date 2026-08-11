from __future__ import annotations

import json

import pytest

from training.sparse_lora import (
    SparseLoRAEstimateConfig,
    estimate_sparse_lora_from_sequences,
    write_sparse_lora_report_from_dataset,
)


def test_sparse_lora_estimate_counts_skippable_context_tokens() -> None:
    report = estimate_sparse_lora_from_sequences(
        [
            [2, 10, 10, 10, 11, 12, 13, 14, 15, 16],
            [2, 21, 22, 23, 24, 25, 26, 27],
        ],
        config=SparseLoRAEstimateConfig(keep_ratio=0.5, protect_first_tokens=1, protect_last_tokens=1),
    )

    assert report["active_tokens"] == 18
    assert report["kept_tokens"] == 9
    assert report["skipped_tokens_estimate"] == 9
    assert report["estimated_skip_ratio"] == 0.5
    assert report["examples_analyzed"] == 2


def test_sparse_lora_report_writer_records_measure_only_decision(tmp_path) -> None:
    data_path = tmp_path / "train.txt"
    output_path = tmp_path / "sparse_lora.json"
    data_path.write_text(
        "H: Who are you?\nANRA: I am An-Ra.\nH: Repeat repeat repeat repeat.\nANRA: Grounded answer.",
        encoding="utf-8",
    )

    report = write_sparse_lora_report_from_dataset(data_path, output_path=output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == report
    assert report["training_enabled"] is False
    assert report["decision"] == "measure_only_until_eval_beats_lora_baseline"
    assert report["estimate"]["active_tokens"] > 0


def test_identity_finetune_writes_sparse_lora_report_before_training(tmp_path, monkeypatch) -> None:
    from training import finetune_anra

    data_path = tmp_path / "identity.txt"
    data_path.write_text("H: Who are you?\nANRA: I am An-Ra.\n", encoding="utf-8")

    report_paths = {
        "finetune_report": tmp_path / "finetune.json",
        "sparse_lora_report": tmp_path / "sparse_lora.json",
    }
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(finetune_anra, "v2_report_path", lambda key: report_paths[key])
    monkeypatch.setattr(finetune_anra, "canonical_v2_checkpoint", lambda kind: tmp_path / f"{kind}.pt")
    monkeypatch.setattr(
        finetune_anra,
        "train_anra_v2",
        lambda **kwargs: calls.append(kwargs) or {"ok": True},
    )

    report = finetune_anra.finetune_identity(data_path=str(data_path), max_minutes=1, max_examples=4)

    assert calls and calls[0]["data_path"] == str(data_path)
    assert report_paths["sparse_lora_report"].exists()
    assert report["sparse_lora"]["training_enabled"] is False
    assert report["result"] == {"ok": True}


def test_retired_identity_finetune_cli_cannot_start_training(monkeypatch, capsys) -> None:
    """The historical helper remains importable, but not runnable as a trainer."""
    from training import finetune_anra

    monkeypatch.setattr("sys.argv", ["finetune_anra"])
    with pytest.raises(SystemExit) as exc_info:
        finetune_anra.main()
    assert exc_info.value.code == 2
    assert "retired_identity_finetune_entrypoint" in capsys.readouterr().out
