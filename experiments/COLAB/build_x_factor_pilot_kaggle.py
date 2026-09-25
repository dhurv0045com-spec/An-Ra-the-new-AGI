from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "X_FACTOR_PILOT_001_KAGGLE_T4X2.ipynb"
REMOTE = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
SOURCE_COMMIT = "b11f19a7b242fe05033cdc3657b6a06e566d665e"
OPERATOR_BLOB = "ca46af84ff17b2ef0dc6a45d00eddcce93264c33"
PROTOCOL_HASH = "57da611ec78389397d1cf22c44588f52f2e860ebbfb318435c8802d03611069a"

MARKDOWN = """# X-FACTOR-PILOT-001 — real V24,576 formation test

## Before running

Select **Settings → Accelerator → GPU T4 x2** and **Internet → ON** in Kaggle.

This notebook runs the executable Leviathan-inspired continuous-input pilot at the original physical vocabulary of 24,576. It compares a tied dense baseline, an untied dense baseline, and the structured continuous-input plus untied-output X-factor. The positive-control gate and restart canary run before official Formation-Mux data.

For a resume, attach the complete saved Output from the prior notebook version under **Add Data**. The evidence ZIP is not sufficient for exact resume; preserve the complete `/kaggle/working/X_FACTOR_PILOT_001` tree.

If any cell fails, stop and preserve the complete working tree and printed failure. Do not bypass the T4x2, positive-control, checkpoint, or sealed-data gates.
"""

CODE = dedent(
    """
    import importlib
    import pathlib
    import subprocess
    import sys

    import torch

    try:
        import tokenizers
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers==0.23.0rc0'], check=True)
        import tokenizers
    if tokenizers.__version__ != '0.23.0rc0':
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'tokenizers==0.23.0rc0'], check=True)
        importlib.invalidate_caches()
        tokenizers = importlib.reload(tokenizers)
    if tokenizers.__version__ != '0.23.0rc0':
        raise RuntimeError('tokenizer version mismatch: ' + tokenizers.__version__)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('X-FACTOR-PILOT-001 requires exactly two visible CUDA devices')
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any('T4' not in name.upper() for name in gpu_names):
        raise RuntimeError('X-FACTOR-PILOT-001 requires two NVIDIA T4 devices: ' + repr(gpu_names))

    REMOTE = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'
    SOURCE_COMMIT = '__SOURCE_COMMIT__'
    OPERATOR_BLOB = '__OPERATOR_BLOB__'
    PROTOCOL_HASH = '__PROTOCOL_HASH__'
    REPO = pathlib.Path('/kaggle/temp/x-factor-pilot-source-' + SOURCE_COMMIT[:12])
    OUTPUT = pathlib.Path('/kaggle/working/X_FACTOR_PILOT_001')

    def run(*args, cwd=None):
        return subprocess.run(list(args), cwd=cwd, check=True, text=True)

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
    if git(REPO, 'hash-object', 'tools/x_factor_pilot_001_kaggle_operator_v1.py') != OPERATOR_BLOB:
        raise RuntimeError('operator blob mismatch')
    protocol_text = run(sys.executable, '-c', 'from v5_experiments import x_factor_pilot_protocol_v1 as p; print(p.protocol_sha256())', cwd=str(REPO)).stdout.strip()
    if protocol_text != PROTOCOL_HASH:
        raise RuntimeError('protocol hash mismatch: ' + protocol_text)
    if OUTPUT.exists() and not OUTPUT.is_dir():
        raise RuntimeError('output path is not a directory')
    completed = subprocess.run([
        sys.executable,
        '-u',
        'tools/x_factor_pilot_001_kaggle_operator_v1.py',
        '--repo',
        str(REPO),
        '--out',
        str(OUTPUT),
    ], cwd=str(REPO))
    if completed.returncode != 0:
        raise RuntimeError('X-FACTOR-PILOT-001 failed closed; preserve the complete working tree')
    print('X-FACTOR-PILOT-001 operator returned successfully')
    print('Preserve /kaggle/working/X_FACTOR_PILOT_001, including every resume.pt file')
    """
)

CODE = CODE.replace("__SOURCE_COMMIT__", SOURCE_COMMIT).replace("__OPERATOR_BLOB__", OPERATOR_BLOB).replace("__PROTOCOL_HASH__", PROTOCOL_HASH)


def validate_pins() -> None:
    for name, value, width in (("SOURCE_COMMIT", SOURCE_COMMIT, 40), ("OPERATOR_BLOB", OPERATOR_BLOB, 40), ("PROTOCOL_HASH", PROTOCOL_HASH, 64)):
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
            "accelerator": "GPU T4 x2",
            "campaign": "X-FACTOR-PILOT-001",
            "schema": "anra.x-factor-pilot-kaggle-notebook/v1",
            "source_commit": SOURCE_COMMIT,
            "operator_blob": OPERATOR_BLOB,
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
