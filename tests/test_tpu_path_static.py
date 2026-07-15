from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.tpu_runtime import TPUUnavailableError, require_torch_xla
from training.v2_data_mix import (
    TrainingExample,
    V2ConversationDataset,
    split_conversation_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tpu_trainer_help_does_not_require_torch_xla() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_brain_tpu.py"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0
    assert "--grad_accum_steps" in result.stdout
    assert "anra-v4-180m" in result.stdout


def test_tpu_runtime_missing_xla_error_is_actionable() -> None:
    try:
        require_torch_xla()
    except TPUUnavailableError as exc:
        message = str(exc)
        assert "PyTorch/XLA" in message
        assert "torch_xla[tpu]" in message


def test_only_canonical_t4_notebook_is_published() -> None:
    assert (ROOT / "notebooks" / "AN_RA_T4_TRAINING.ipynb").exists()
    assert not (ROOT / "notebooks" / "AN_RA_TPU_TRAINING.ipynb").exists()


def test_tpu_trainer_disables_pytorch_checkpointing_for_xla() -> None:
    source = (ROOT / "scripts" / "build_brain_tpu.py").read_text(encoding="utf-8")
    assert "gradient_checkpointing_disable" in source
    assert "does not support xla device type" in source


def test_readme_documents_canonical_v4_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "notebooks/AN_RA_T4_TRAINING.ipynb" in readme
    assert "anra-v4-180m" in readme
    assert "anra_frontier_500m.pt" not in readme


class _TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [4 + (index % 20) for index, _char in enumerate(text)]


def test_dataset_window_bucket_handles_multi_window_examples() -> None:
    examples = [
        TrainingExample(
            bucket="teacher",
            prompt="Explain the system.",
            answer=" ".join(["long-answer"] * 120),
            source="test",
        )
    ]
    dataset = V2ConversationDataset(
        examples,
        _TinyTokenizer(),
        block_size=32,
        answer_loss_weight=1.5,
    )

    assert len(dataset) > len(dataset.examples)
    assert {dataset.bucket_for_window(index) for index in range(len(dataset))} == {"teacher"}
    assert any(bool(dataset[index][4].any()) for index in range(len(dataset)))


def test_conversation_validation_split_is_deterministic_grouped_and_disjoint() -> None:
    examples = [
        TrainingExample(
            bucket="own" if index % 2 == 0 else "teacher",
            prompt=f"prompt {index}",
            answer=f"answer {index}",
            source="unit",
            metadata={"source_hash": f"source-{index // 2}"},
        )
        for index in range(12)
    ]

    first_train, first_validation, first_report = split_conversation_validation(examples)
    second_train, second_validation, second_report = split_conversation_validation(examples)

    assert first_report == second_report
    assert [(row.prompt, row.answer) for row in first_train] == [
        (row.prompt, row.answer) for row in second_train
    ]
    assert [(row.prompt, row.answer) for row in first_validation] == [
        (row.prompt, row.answer) for row in second_validation
    ]
    assert first_report["overlap_group_hashes"] == []
    assert len(first_report["split_sha256"]) == 64
    train_sources = {row.metadata["source_hash"] for row in first_train}
    validation_sources = {row.metadata["source_hash"] for row in first_validation}
    assert train_sources.isdisjoint(validation_sources)
    assert set(first_report["bucket_counts"]) == {"own", "teacher"}


def test_conversation_validation_split_rejects_single_content_group() -> None:
    rows = [
        TrainingExample(
            bucket="own",
            prompt="same",
            answer="same",
            source="unit",
            metadata={"source_hash": "only-source"},
        )
    ]

    with pytest.raises(RuntimeError, match="at least two content groups"):
        split_conversation_validation(rows)


def test_dataset_packs_short_examples_without_mixing_buckets() -> None:
    examples = [
        TrainingExample(bucket="own", prompt=f"p{index}", answer="short", source="test")
        for index in range(8)
    ] + [
        TrainingExample(bucket="teacher", prompt=f"t{index}", answer="short", source="test")
        for index in range(8)
    ]
    dataset = V2ConversationDataset(
        examples,
        _TinyTokenizer(),
        block_size=64,
        answer_loss_weight=1.5,
    )

    assert len(dataset) < len(examples)
    assert dataset.token_utilization > 0.5
    assert {dataset.bucket_for_window(index) for index in range(len(dataset))} == {"own", "teacher"}
