from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "X_FACTOR_PILOT_001_COLAB_T4.ipynb"
REMOTE = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
SOURCE_COMMIT = "4cf9a0a414bb944e8592b86354bb3459fe4584db"
COLAB_OPERATOR_BLOB = "ca19443465b5868ab7b7af577e981047db8345d5"
PROTOCOL_HASH = "57da611ec78389397d1cf22c44588f52f2e860ebbfb318435c8802d03611069a"

MARKDOWN = """# X-FACTOR-PILOT-001 — Google Colab T4

## Before running

1. In Colab, open **Runtime → Change runtime type → GPU**.
2. Select a **T4 GPU** if the runtime offers a choice.
3. Run the cells from top to bottom.
4. Do not interrupt a training cell. The operator checkpoints at fixed boundaries and stops at the session guard rather than shortening the scientific target.

This is the same executable V24,576 pilot as the tested Kaggle launcher, adapted to one visible T4. It uses production-BPE inputs, a tied baseline, an untied baseline, a Leviathan-style continuous-input arm, a positive-control gate, an exact-resume canary, and coordinator-only sealed scoring.

The default output is `/content/X_FACTOR_PILOT_001`. If Google Drive is already mounted at `/content/drive`, the notebook uses `/content/drive/MyDrive/X_FACTOR_PILOT_001` when that complete output tree exists. For cross-session resume, preserve that complete tree; the evidence ZIP alone is not resumable because it intentionally excludes `resume.pt` files.

If the runtime has no CUDA device or does not expose a T4, stop and enable the correct runtime. The notebook will not silently fall back to CPU.
"""

CODE = dedent(
    """
    import importlib
    import json
    import pathlib
    import subprocess
    import sys

    import torch

    try:
        import tokenizers
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers>=0.23,<0.24'], check=True)
        import tokenizers
    if not tokenizers.__version__.startswith('0.23.'):
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'tokenizers>=0.23,<0.24'], check=True)
        importlib.invalidate_caches()
        tokenizers = importlib.reload(tokenizers)
    if not tokenizers.__version__.startswith('0.23.'):
        raise RuntimeError('tokenizers 0.23.x is required; observed ' + tokenizers.__version__)
    try:
        import pytest
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytest'], check=True)
    try:
        import numpy
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy'], check=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Select one GPU in Runtime → Change runtime type → GPU')
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    if 'T4' not in gpu_name.upper() or not 14 * 1024 ** 3 <= vram_bytes <= 17 * 1024 ** 3:
        raise RuntimeError('Select a T4 GPU with approximately 15 GB VRAM; observed ' + repr((gpu_name, vram_bytes)))

    REMOTE = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'
    SOURCE_COMMIT = '__SOURCE_COMMIT__'
    COLAB_OPERATOR_BLOB = '__COLAB_OPERATOR_BLOB__'
    PROTOCOL_HASH = '__PROTOCOL_HASH__'
    REPO = pathlib.Path('/content/x-factor-pilot-source-' + SOURCE_COMMIT[:12])
    DRIVE_OUTPUT = pathlib.Path('/content/drive/MyDrive/X_FACTOR_PILOT_001')
    OUTPUT = DRIVE_OUTPUT if DRIVE_OUTPUT.exists() else pathlib.Path('/content/X_FACTOR_PILOT_001')

    def run(*args, cwd=None):
        return subprocess.run(list(args), cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def git(repo, *args):
        return run('git', '-C', str(repo), *args).stdout.strip()

    if REPO.exists() or REPO.is_symlink():
        if REPO.is_symlink() or not (REPO / '.git').exists():
            raise RuntimeError('refusing non-Git or symlink checkout path')
        if git(REPO, 'status', '--porcelain'):
            raise RuntimeError('refusing dirty source checkout')
        if git(REPO, 'rev-parse', 'HEAD') != SOURCE_COMMIT:
            raise RuntimeError('refusing to switch an existing source checkout')
    else:
        run('git', 'clone', '--no-checkout', REMOTE, str(REPO))
        run('git', '-C', str(REPO), 'fetch', 'origin')
        run('git', '-C', str(REPO), 'switch', '--detach', SOURCE_COMMIT)
    if git(REPO, 'rev-parse', 'HEAD') != SOURCE_COMMIT:
        raise RuntimeError('source commit mismatch')
    if git(REPO, 'hash-object', 'tools/x_factor_pilot_001_colab_operator_v1.py') != COLAB_OPERATOR_BLOB:
        raise RuntimeError('Colab operator blob mismatch')
    protocol_text = run(sys.executable, '-c', 'from v5_experiments import x_factor_pilot_protocol_v1 as p; print(p.protocol_sha256())', cwd=str(REPO)).stdout.strip()
    if protocol_text != PROTOCOL_HASH:
        raise RuntimeError('protocol hash mismatch: ' + protocol_text)
    if OUTPUT.exists() and not OUTPUT.is_dir():
        raise RuntimeError('output path is not a directory')
    completed = subprocess.run([
        sys.executable,
        '-u',
        'tools/x_factor_pilot_001_colab_operator_v1.py',
        '--repo',
        str(REPO),
        '--out',
        str(OUTPUT),
    ], cwd=str(REPO))
    if completed.returncode != 0:
        raise RuntimeError('X-FACTOR-PILOT-001 failed closed; preserve the complete output tree')
    print('X-FACTOR-PILOT-001 Colab operator returned successfully')
    print('Output tree:', OUTPUT)
    print('Preserve every resume.pt file for exact resume')
    """
)

CODE = CODE.replace("__SOURCE_COMMIT__", SOURCE_COMMIT).replace("__COLAB_OPERATOR_BLOB__", COLAB_OPERATOR_BLOB).replace("__PROTOCOL_HASH__", PROTOCOL_HASH)


def validate_pins() -> None:
    for name, value, width in (("SOURCE_COMMIT", SOURCE_COMMIT, 40), ("COLAB_OPERATOR_BLOB", COLAB_OPERATOR_BLOB, 40), ("PROTOCOL_HASH", PROTOCOL_HASH, 64)):
        if len(value) != width or set(value) == {"0"}:
            raise ValueError(f"{name} is not a finalized immutable pin")


def build() -> dict[str, object]:
    validate_pins()
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": MARKDOWN.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(keepends=True)},
        ],
        "metadata": {
            "accelerator": "GPU T4",
            "campaign": "X-FACTOR-PILOT-001",
            "schema": "anra.x-factor-pilot-colab-notebook/v1",
            "source_commit": SOURCE_COMMIT,
            "colab_operator_blob": COLAB_OPERATOR_BLOB,
            "protocol_sha256": PROTOCOL_HASH,
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
