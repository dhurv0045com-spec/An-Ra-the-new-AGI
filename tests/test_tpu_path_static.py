from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from training.tpu_runtime import TPUUnavailableError, require_torch_xla
from training.v2_data_mix import TrainingExample, V2ConversationDataset


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
    assert "frontier" in result.stdout


def test_tpu_runtime_missing_xla_error_is_actionable() -> None:
    try:
        require_torch_xla()
    except TPUUnavailableError as exc:
        message = str(exc)
        assert "PyTorch/XLA" in message
        assert "torch_xla[tpu]" in message


def test_tpu_notebook_is_valid_and_uses_dedicated_trainer() -> None:
    notebook_path = ROOT / "notebooks" / "AN_RA_TPU_TRAINING.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    joined = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
    )
    assert payload["metadata"]["accelerator"] == "TPU"
    assert "scripts/build_brain_tpu.py" in joined
    assert "scripts/build_brain.py --data_path" not in joined
    assert "torch_xla[tpu]" in joined
    assert "--no-deps -e" in joined
    assert "DATA_PROFILE = 'tpu'" in joined
    assert "/content/thirdeye" in joined


def test_tpu_trainer_disables_pytorch_checkpointing_for_xla() -> None:
    source = (ROOT / "scripts" / "build_brain_tpu.py").read_text(encoding="utf-8")
    assert "gradient_checkpointing_disable" in source
    assert "does not support xla device type" in source


def test_readme_documents_tpu_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "notebooks/AN_RA_TPU_TRAINING.ipynb" in readme
    assert "scripts/build_brain_tpu.py" in readme
    assert "PyTorch/XLA" in readme


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
