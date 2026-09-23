"""Build the fail-closed Kaggle FORMATION-MUX exact-recovery notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "CYMEK_FORMATION_MUX_001_RECOVERY_T4X2.ipynb"
REMOTE = "https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git"
RECOVERY_COMMIT = "8609ba95f4e978cf3cdf8d20bd8a907eea8f6728"
RECOVERY_BLOB = "b507c2350a885e5e3432cac6f135fa58df52b69e"
PREFLIGHT_SCHEMA = "anra.formation-mux-recovery-preflight/v2"
OPERATOR_COMMIT = "4ee05f6e386f15d34f9dfa7bd7f3300a496b9896"
OPERATOR_BLOB = "e9e1f701b0d4edc509194da55fe1ba37ed62ef86"
SCIENCE_COMMIT = "c15ad8beb409537db42d075684ea54847a074ebd"
PUBLIC_SURFACE_SHA256 = "f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c"
TOKENIZER_SHA256 = "97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc"
TOKENIZERS_VERSION = "0.23.0rc0"

MARKDOWN = """# FORMATION-MUX-001 exact-resume recovery — Kaggle T4 ×2

## BEFORE RUNNING

Kaggle **Settings → Accelerator → GPU T4 x2** and **Internet → ON**.

Attach the complete saved Kaggle Output tree from the 2026-09-23 notebook version under **Add Data**. Do not attach only `FORMATION_MUX_001_RESULTS.zip`; the evidence bundle intentionally omits `resume.pt` and is not resumable.

Cell 1 verifies immutable Git commits/blobs, the exact official arm registry, the real S5 public surface, T4 CUDA RNG state, checkpoint payloads, result/receipt/progress consistency, endpoint exposure, and sealed custody in a fresh staging directory. It refuses evidence-only inputs, ambiguous inputs, existing working output, latched failure state, and symlink redirects.

Cell 2 requires the same-kernel PASS receipt, invokes only pinned operator v12, then proves every previously completed checkpoint and `ARM_RESULT.json` remained byte-identical.

If the frontier reaches 24/24, pinned v12 automatically performs the frozen development-only diagnostics followed by the preregistered sealed finalization and architecture-gate step. This notebook is therefore an exact full-campaign continuation launcher, not a development-only launcher.

If Cell 1 fails, stop and preserve the attached Output and printed failure. Do not bypass the preflight or relabel a rerun as recovery.
"""

PREFLIGHT_CELL = """\
import hashlib
import importlib
import json
import os
import pathlib
import secrets
import subprocess
import sys

import torch

try:
    import tokenizers
except ImportError:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers==0.23.0rc0'],
        check=True,
    )
    import tokenizers
if tokenizers.__version__ != '0.23.0rc0':
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'tokenizers==0.23.0rc0'],
        check=True,
    )
    importlib.invalidate_caches()
    tokenizers = importlib.reload(tokenizers)
if tokenizers.__version__ != '0.23.0rc0':
    raise RuntimeError('tokenizer version mismatch: ' + tokenizers.__version__)
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError('exact recovery requires two visible CUDA devices')
gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
if any('T4' not in name.upper() for name in gpu_names):
    raise RuntimeError('exact recovery requires two NVIDIA T4 devices: ' + repr(gpu_names))

REMOTE = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'
RECOVERY_COMMIT = '8609ba95f4e978cf3cdf8d20bd8a907eea8f6728'
RECOVERY_BLOB = 'b507c2350a885e5e3432cac6f135fa58df52b69e'
PREFLIGHT_SCHEMA = 'anra.formation-mux-recovery-preflight/v2'
OPERATOR_COMMIT = '4ee05f6e386f15d34f9dfa7bd7f3300a496b9896'
OPERATOR_BLOB = 'e9e1f701b0d4edc509194da55fe1ba37ed62ef86'
SCIENCE_COMMIT = 'c15ad8beb409537db42d075684ea54847a074ebd'
PUBLIC_SURFACE_SHA256 = 'f1d5200bd05bc28ede97af114b49f616ca72b24af74b7fc530a2cb084db4259c'
TOKENIZER_SHA256 = '97e12db63b343312e5e4abc37df9ef4b01fcb1faba792a6420a4c1b15d0a7fbc'
RECOVERY_REPO = pathlib.Path('/kaggle/temp/formation-mux-recovery-' + RECOVERY_COMMIT[:12])
OPERATOR_REPO = pathlib.Path('/kaggle/temp/formation-mux-source-' + OPERATOR_COMMIT[:12])
OUTPUT = pathlib.Path('/kaggle/working/FORMATION_MUX_001')

def git(repo, *args):
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

def prepare_checkout(path, commit):
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not (path / '.git').exists():
            raise RuntimeError('refusing non-Git or symlink checkout path: ' + str(path))
        if git(path, 'status', '--porcelain'):
            raise RuntimeError('refusing dirty checkout: ' + str(path))
        if git(path, 'rev-parse', 'HEAD') != commit:
            raise RuntimeError('refusing to switch existing checkout: ' + str(path))
        return
    subprocess.run(['git', 'clone', '--no-checkout', REMOTE, str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'fetch', 'origin'], check=True)
    subprocess.run(['git', '-C', str(path), 'switch', '--detach', commit], check=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

prepare_checkout(RECOVERY_REPO, RECOVERY_COMMIT)
prepare_checkout(OPERATOR_REPO, OPERATOR_COMMIT)
if git(RECOVERY_REPO, 'rev-parse', 'HEAD') != RECOVERY_COMMIT:
    raise RuntimeError('recovery HEAD mismatch')
if git(RECOVERY_REPO, 'hash-object', 'tools/formation_mux_001_recovery_preflight.py') != RECOVERY_BLOB:
    raise RuntimeError('recovery preflight blob mismatch')
if git(OPERATOR_REPO, 'rev-parse', 'HEAD') != OPERATOR_COMMIT:
    raise RuntimeError('operator HEAD mismatch')
if git(OPERATOR_REPO, 'hash-object', 'tools/formation_mux_001_kaggle_operator_v12.py') != OPERATOR_BLOB:
    raise RuntimeError('operator blob mismatch')
subprocess.run(
    ['git', '-C', str(OPERATOR_REPO), 'cat-file', '-e', SCIENCE_COMMIT + '^{commit}'],
    check=True,
)
if OUTPUT.exists() or OUTPUT.is_symlink():
    raise RuntimeError('fresh recovery requires absent /kaggle/working/FORMATION_MUX_001')

RECOVERY_NONCE = secrets.token_hex(32)
environment = os.environ.copy()
environment['FORMATION_MUX_RECOVERY_NONCE'] = RECOVERY_NONCE
completed = subprocess.run(
    [
        sys.executable,
        '-m',
        'tools.formation_mux_001_recovery_preflight',
        '--input',
        '/kaggle/input',
        '--out',
        str(OUTPUT),
        '--install',
    ],
    cwd=str(RECOVERY_REPO),
    env=environment,
)
if completed.returncode != 0:
    raise RuntimeError('recovery preflight failed; preserve the attached Output and staging path')

RECOVERY_RECEIPT_PATH = OUTPUT / 'RECOVERY_PREFLIGHT.json'
RECOVERY_RECEIPT = json.loads(RECOVERY_RECEIPT_PATH.read_text(encoding='utf-8'))
expected = {
    'schema': PREFLIGHT_SCHEMA,
    'status': 'PASS',
    'recovery_only': True,
    'recovery_nonce': RECOVERY_NONCE,
    'science_commit': SCIENCE_COMMIT,
    'operator_commit': OPERATOR_COMMIT,
    'operator_blob': OPERATOR_BLOB,
    'public_surface_sha256': PUBLIC_SURFACE_SHA256,
    'tokenizer_artifact_sha256': TOKENIZER_SHA256,
    'raw_sealed_rows_read': False,
}
for key, value in expected.items():
    if RECOVERY_RECEIPT.get(key) != value:
        raise RuntimeError('recovery receipt mismatch for ' + key)
if RECOVERY_RECEIPT.get('s5_completed_arms') != 24:
    raise RuntimeError('recovery receipt does not contain 24/24 S5 arms')
if int(RECOVERY_RECEIPT.get('frontier_completed_arms', 0)) < 2:
    raise RuntimeError('recovery receipt lacks the two completed T0 controls')
if int(RECOVERY_RECEIPT.get('official_checkpoint_count', 0)) < 26:
    raise RuntimeError('recovery receipt checkpoint floor failed')
if RECOVERY_RECEIPT.get('preflight_source_sha256') != sha256_file(
    RECOVERY_REPO / 'tools/formation_mux_001_recovery_preflight.py'
):
    raise RuntimeError('recovery receipt is not bound to this preflight file')
COMPLETED_CUSTODY = {
    row['slot']: {
        'checkpoint': str(OUTPUT / row['path']),
        'checkpoint_sha256': row['sha256'],
        'result': str(OUTPUT / pathlib.PurePosixPath(row['path']).parent / 'ARM_RESULT.json'),
        'result_sha256': row['result_sha256'],
    }
    for row in RECOVERY_RECEIPT['checkpoints']
    if row['result_complete']
}
print('RECOVERY CUSTODY: PASS')
print('GPU:', gpu_names)
print('S5:', RECOVERY_RECEIPT['s5_completed_arms'], '/ 24')
print('FRONTIER:', RECOVERY_RECEIPT['frontier_completed_arms'], '/ 24')
print('CHECKPOINTS:', RECOVERY_RECEIPT['official_checkpoint_count'])
"""

OPERATOR_CELL = """\
import json
import pathlib
import subprocess
import sys

required = {
    'RECOVERY_NONCE',
    'RECOVERY_RECEIPT',
    'RECOVERY_RECEIPT_PATH',
    'RECOVERY_REPO',
    'OPERATOR_REPO',
    'OUTPUT',
    'COMPLETED_CUSTODY',
    'sha256_file',
}
missing = sorted(required.difference(globals()))
if missing:
    raise RuntimeError('run the recovery preflight cell first: ' + repr(missing))
current_receipt = json.loads(RECOVERY_RECEIPT_PATH.read_text(encoding='utf-8'))
if current_receipt != RECOVERY_RECEIPT or current_receipt.get('recovery_nonce') != RECOVERY_NONCE:
    raise RuntimeError('same-kernel recovery receipt changed; refusing operator launch')
if subprocess.run(
    ['git', '-C', str(RECOVERY_REPO), 'rev-parse', 'HEAD'],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip() != '8609ba95f4e978cf3cdf8d20bd8a907eea8f6728':
    raise RuntimeError('recovery checkout moved')
if subprocess.run(
    ['git', '-C', str(OPERATOR_REPO), 'rev-parse', 'HEAD'],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip() != '4ee05f6e386f15d34f9dfa7bd7f3300a496b9896':
    raise RuntimeError('operator checkout moved')
if subprocess.run(
    ['git', '-C', str(OPERATOR_REPO), 'hash-object', 'tools/formation_mux_001_kaggle_operator_v12.py'],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip() != 'e9e1f701b0d4edc509194da55fe1ba37ed62ef86':
    raise RuntimeError('operator source moved')
receipt_hash_before = sha256_file(RECOVERY_RECEIPT_PATH)
completed = subprocess.run(
    [
        sys.executable,
        '-u',
        'tools/formation_mux_001_kaggle_operator_v12.py',
        '--repo',
        str(OPERATOR_REPO),
        '--out',
        str(OUTPUT),
    ],
    cwd=str(OPERATOR_REPO),
)
custody_errors = []
try:
    if sha256_file(RECOVERY_RECEIPT_PATH) != receipt_hash_before:
        custody_errors.append('RECOVERY_PREFLIGHT.json changed')
except Exception as exc:
    custody_errors.append('RECOVERY_PREFLIGHT.json unavailable: ' + repr(exc))
for slot, custody in COMPLETED_CUSTODY.items():
    try:
        if sha256_file(pathlib.Path(custody['checkpoint'])) != custody['checkpoint_sha256']:
            custody_errors.append('completed checkpoint mutated: ' + slot)
    except Exception as exc:
        custody_errors.append('completed checkpoint unavailable: ' + slot + ': ' + repr(exc))
    try:
        if sha256_file(pathlib.Path(custody['result'])) != custody['result_sha256']:
            custody_errors.append('completed ARM_RESULT.json mutated: ' + slot)
    except Exception as exc:
        custody_errors.append('completed ARM_RESULT.json unavailable: ' + slot + ': ' + repr(exc))
if completed.returncode != 0:
    for failure in sorted(OUTPUT.glob('GLOBAL_FAILURE*.json')):
        print(failure.name + ': ' + failure.read_text(encoding='utf-8'))
    if custody_errors:
        raise RuntimeError('operator failed and completed custody was violated: ' + repr(custody_errors))
    raise RuntimeError('pinned operator failed closed; preserve the complete working and saved Output trees')
if custody_errors:
    raise RuntimeError('completed custody violation: ' + repr(custody_errors))

s5 = json.loads((OUTPUT / 'CAMPAIGN_STATE.json').read_text(encoding='utf-8'))
frontier = json.loads((OUTPUT / 'TIE_ROLE_FRONTIER_STATE.json').read_text(encoding='utf-8'))
print('COMPLETED ARM IMMUTABILITY: PASS')
print('S5:', s5.get('status'), s5.get('complete_arms'), '/', s5.get('required_arms'))
print('FRONTIER:', frontier.get('status'), frontier.get('complete_arms'), '/', frontier.get('required_arms'))
for experiment in ('CS-MECH-002', 'REP-FORM-003A', 'TIE-ROLE-001', 'TIE-ROLE-XFER-001'):
    final = OUTPUT / experiment / 'FINAL_RESULT.json'
    print(experiment, 'FINAL_RESULT', final.exists())
print('Save this Kaggle version and preserve the complete /kaggle/working/FORMATION_MUX_001 tree before another session.')
"""


def build() -> dict[str, object]:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": MARKDOWN.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": PREFLIGHT_CELL.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": OPERATOR_CELL.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "accelerator": "GPU T4 x2",
            "experiment": "FORMATION-MUX-001 + TIE-ROLE-FRONTIER-001",
            "frontier_extension": "TIE-ROLE-FRONTIER-001",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "operator_blob": OPERATOR_BLOB,
            "operator_commit": OPERATOR_COMMIT,
            "preflight_schema": PREFLIGHT_SCHEMA,
            "public_surface_sha256": PUBLIC_SURFACE_SHA256,
            "recovery_commit": RECOVERY_COMMIT,
            "recovery_preflight_blob": RECOVERY_BLOB,
            "result_hash_immutability": True,
            "schema": "anra.formation-mux-kaggle-recovery-wrapper/v2",
            "science_commit": SCIENCE_COMMIT,
            "tokenizer_artifact_sha256": TOKENIZER_SHA256,
            "tokenizers_version": TOKENIZERS_VERSION,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
