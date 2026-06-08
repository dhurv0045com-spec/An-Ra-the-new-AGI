"""Smoke tests - anra_brain importable from root, train script is well-formed."""

from __future__ import annotations

import subprocess
import sys


def test_anra_brain_importable_from_root() -> None:
    """anra_brain.py must be at root and export CausalTransformerV2."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import anra_brain; assert hasattr(anra_brain, 'CausalTransformerV2'), "
            "'CausalTransformerV2 not found in anra_brain'",
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode == 0, f"anra_brain import failed:\n{result.stderr}"


def test_train_script_file_is_at_scripts_not_root() -> None:
    """train.py must be inside scripts/ because it is an executable, not a library module."""
    import pathlib

    assert pathlib.Path("scripts/train.py").exists(), "scripts/train.py not found"
    assert not pathlib.Path("train.py").exists(), "train.py must NOT exist at root"
