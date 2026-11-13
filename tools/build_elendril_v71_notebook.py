"""Build an owner-authorized, source-pinned V7.1 Kaggle notebook."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

DEST = Path(__file__).resolve().parents[1] / 'notebooks' / 'ELENDRIL_V71_KAGGLE_T4X2.ipynb'


def _cell(kind, content):
    cell = {'cell_type': kind, 'metadata': {}, 'source': content.splitlines(keepends=True)}
    if kind == 'code':
        cell.update({'execution_count': None, 'outputs': []})
    return cell


def build(source_sha):
    if not isinstance(source_sha, str) or not re.fullmatch(r'[0-9a-f]{40}', source_sha):
        raise ValueError('source must be an immutable lowercase 40-hex commit')
    intro = f'''# ELENDRIL V7.1 · Kaggle T4×2

Run All starts the owner-authorized **core** phase (9 jobs) after preflight and production-shaped GPU canaries. The scale phase has 12 jobs; continual has 9 and requires verified V7.1 core output. Frozen evaluates all 9 audited V7 checkpoints and requires their verified full output; no weights are fabricated. Set `PHASE` and provide a parent when needed. A whole phase may return NO_GO if measured capacity does not fit; GPU qualification remains pending an actual run.

Pinned source: `{source_sha}`. Default core needs no input dataset: procedural data is generated under writable `/kaggle/working`. Native Kaggle PyTorch is used without installing or replacing CUDA. Optional `RESTORE_ARCHIVE` restores a V7.1 full ZIP into a new phase output; optional `PARENT_ARCHIVE` extracts a verified parent full ZIP into a new parent directory. Alternatively set `PARENT_OUTPUT` to an existing extracted parent. Choose only one parent input. Existing directories are never overwritten by extraction. Archive and partial-output links appear even when the campaign fails.
'''
    config = f'''from pathlib import Path
SOURCE_SHA = "{source_sha}"
RUN_STUDY = True  # Owner-authorized Run All; False is an offline preview.
PHASE = 'core'  # core, scale, continual, frozen
MINUTES = 330
PARENT_OUTPUT = None  # Existing extracted parent directory, when required.
RESTORE_ARCHIVE = None  # Optional V7.1 full ZIP for same-phase recovery.
PARENT_ARCHIVE = None  # Optional verified parent full ZIP, instead of PARENT_OUTPUT.
OUTPUT = Path('/kaggle/working') / ('elendril-v71-' + SOURCE_SHA[:12] + '-' + PHASE)
assert type(RUN_STUDY) is bool
'''
    launch = '''import os
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
from IPython.display import FileLink, display

REPOSITORY = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'
REPO = Path('/kaggle/working') / ('elendril-v71-source-' + SOURCE_SHA)
campaign_failed = False
if RUN_STUDY:
    try:
        if not re.fullmatch(r'[0-9a-f]{40}', SOURCE_SHA):
            raise ValueError('Invalid pinned source commit')
        if PHASE not in ('core', 'scale', 'continual', 'frozen'):
            raise ValueError('Invalid phase')
        if isinstance(MINUTES, bool) or not isinstance(MINUTES, (int, float)) or not 0 < MINUTES <= 330:
            raise ValueError('Invalid session minutes')
        if PARENT_OUTPUT is not None and PARENT_ARCHIVE is not None:
            raise ValueError('Choose a parent directory or parent archive, not both')
        if not REPO.exists():
            stage = REPO.parent / ('.elendril-v71-source-' + uuid4().hex)
            stage.mkdir(parents=True)
            subprocess.run(['git', 'init', '-q'], cwd=stage, check=True, timeout=30)
            subprocess.run(['git', 'remote', 'add', 'origin', REPOSITORY], cwd=stage, check=True, timeout=30)
            subprocess.run(['git', 'fetch', '--depth=1', 'origin', SOURCE_SHA], cwd=stage, check=True, timeout=240)
            subprocess.run(['git', 'checkout', '--detach', SOURCE_SHA], cwd=stage, check=True, timeout=60)
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=stage, text=True, timeout=30).strip() != SOURCE_SHA:
                raise RuntimeError('Fetched source does not match the pin')
            stage.rename(REPO)
        if REPO.is_symlink() or not REPO.is_dir():
            raise RuntimeError('Pinned checkout must be a real directory')
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True, timeout=30).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=REPO, text=True, timeout=30).strip()
        if head != SOURCE_SHA or dirty:
            raise RuntimeError('Pinned checkout is not exact and clean')
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        extractor = 'from tools.elendril_v71_export import restore_archive; import sys; restore_archive(sys.argv[1], sys.argv[2], source=sys.argv[3], phase=sys.argv[4], parent=sys.argv[5] == "parent")'
        if RESTORE_ARCHIVE is not None:
            if OUTPUT.exists() or OUTPUT.is_symlink():
                raise ValueError('Restore destination already exists; use existing output without RESTORE_ARCHIVE')
            subprocess.run([sys.executable, '-c', extractor, str(RESTORE_ARCHIVE), str(OUTPUT), SOURCE_SHA, PHASE, 'resume'], cwd=REPO, env=env, check=True)
        parent = PARENT_OUTPUT
        if PARENT_ARCHIVE is not None:
            parent = Path('/kaggle/working') / ('elendril-v71-' + SOURCE_SHA[:12] + '-' + PHASE + '-parent')
            if parent.exists() or parent.is_symlink():
                raise ValueError('Parent extraction destination exists; explicitly select it with PARENT_OUTPUT after inspection')
            subprocess.run([sys.executable, '-c', extractor, str(PARENT_ARCHIVE), str(parent), SOURCE_SHA, PHASE, 'parent'], cwd=REPO, env=env, check=True)
        command = [sys.executable, '-m', 'tools.elendril_v71_campaign', '--output', str(OUTPUT),
                   '--owner-enabled', '--source', SOURCE_SHA, '--phase', PHASE, '--minutes', str(MINUTES)]
        if parent is not None:
            command += ['--parent-output', str(parent)]
        subprocess.run(command, cwd=REPO, env=env, check=True)
        print('Campaign command completed; inspect status and scientific reports. Completion is not scientific success.')
    except Exception as exc:
        campaign_failed = True
        print(f'Campaign did not complete: {type(exc).__name__}: {exc}')
        print('Existing raw and partial outputs remain available for inspection.')
else:
    print('Study disabled. No source download, extraction or campaign launched.')

for label, name in [('Analysis', 'ELENDRIL_V71_ANALYSIS.zip'), ('Full', 'ELENDRIL_V71_FULL.zip')]:
    path = OUTPUT / 'exports' / name
    print(f'{label} archive: {path} (exists: {path.is_file()})')
    display(FileLink(str(path)))
if RUN_STUDY:
    display(FileLink(str(OUTPUT / 'exports' / 'FINAL_STATUS.json')))
if campaign_failed:
    raise RuntimeError('Campaign failed or is incomplete. Inspect NO_GO/PARTIAL receipts and available outputs.')
'''
    return {'cells': [_cell('markdown', intro), _cell('code', config), _cell('code', launch)],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}, 'source_reference': source_sha,
                         'elendril_v71_status': 'OWNER_LAUNCH_GPU_QUALIFICATION_PENDING'},
            'nbformat': 4, 'nbformat_minor': 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', type=Path, default=DEST)
    args = parser.parse_args()
    notebook = build(args.source_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output} for source {args.source_sha}')


if __name__ == '__main__':
    main()
