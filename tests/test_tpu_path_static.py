from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from training.tpu_runtime import TPUUnavailableError, require_torch_xla


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


def test_readme_documents_tpu_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "notebooks/AN_RA_TPU_TRAINING.ipynb" in readme
    assert "scripts/build_brain_tpu.py" in readme
    assert "PyTorch/XLA" in readme
