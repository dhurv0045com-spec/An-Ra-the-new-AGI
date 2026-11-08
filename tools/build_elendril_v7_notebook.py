"""Generate the owner-launched, source-pinned Kaggle V7 notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

DEST = Path(__file__).resolve().parents[1] / 'notebooks' / 'ELENDRIL_V7_KAGGLE_T4X2.ipynb'


def _cell(kind: str, content: str) -> dict:
    cell = {'cell_type': kind, 'metadata': {}, 'source': content.splitlines(keepends=True)}
    if kind == 'code':
        cell.update({'execution_count': None, 'outputs': []})
    return cell


def build(source_sha: str) -> dict:
    """Construct a notebook without network access or campaign execution."""
    if not isinstance(source_sha, str) or not re.fullmatch(r'[0-9a-f]{40}', source_sha):
        raise ValueError('source must be an immutable lowercase 40-hex commit')
    intro = f'''# ELENDRIL V7 · Kaggle T4×2

**Owner launch:** Set `RUN_STUDY = True` in the next cell before choosing Run All. With the switch off, the notebook does not download source or launch the campaign. The pinned campaign CLI performs preflight and canaries before research. Native Kaggle PyTorch is used; no CUDA stack is installed or replaced. GPU qualification remains pending an actual successful run.

Pinned source commit: `{source_sha}`. The campaign obtains its own data; no dataset input is required. Results and failures remain in `/kaggle/working/elendril-v7`. Both archive links appear below even when the command fails; a link is usable only when the corresponding ZIP exists.
'''
    config = f'''from pathlib import Path
SOURCE_SHA = "{source_sha}"
RUN_STUDY = False  # Owner: set True to explicitly launch with Run All.
OUTPUT = Path('/kaggle/working/elendril-v7')
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
REPO = Path('/kaggle/working') / ('elendril-v7-source-' + SOURCE_SHA)
campaign_failed = False
if RUN_STUDY:
    try:
        if not re.fullmatch(r'[0-9a-f]{40}', SOURCE_SHA):
            raise ValueError('Invalid pinned source commit')
        # An interrupted fresh clone leaves its uniquely named staging directory
        # for inspection; a restart uses a new staging directory.
        if not REPO.exists():
            stage = REPO.parent / ('.elendril-v7-source-' + uuid4().hex)
            stage.mkdir(parents=True)
            subprocess.run(['git', 'init', '-q'], cwd=stage, check=True, timeout=30)
            subprocess.run(['git', 'remote', 'add', 'origin', REPOSITORY], cwd=stage, check=True, timeout=30)
            subprocess.run(['git', 'fetch', '--depth=1', 'origin', SOURCE_SHA], cwd=stage, check=True, timeout=240)
            subprocess.run(['git', 'checkout', '--detach', SOURCE_SHA], cwd=stage, check=True, timeout=60)
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=stage, text=True, timeout=30).strip() != SOURCE_SHA:
                raise RuntimeError('Fetched source does not match the pinned commit')
            stage.rename(REPO)
        if REPO.is_symlink() or not REPO.is_dir():
            raise RuntimeError('Pinned checkout must be a real directory')
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True, timeout=30).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=REPO, text=True, timeout=30).strip()
        if head != SOURCE_SHA or dirty:
            raise RuntimeError('Pinned checkout is not exact and clean; inspect it before retrying')
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        subprocess.run([sys.executable, '-m', 'tools.elendril_v7_campaign',
                        '--output', str(OUTPUT), '--owner-enabled', '--source', SOURCE_SHA],
                       cwd=REPO, env=env, check=True)
        print('Campaign CLI completed. Inspect its reports for scientific and archive status.')
    except Exception as exc:
        campaign_failed = True
        print(f'Campaign did not complete: {type(exc).__name__}: {exc}')
        print('Existing raw and partial outputs remain available for inspection.')
else:
    print('Study disabled. No source download or campaign launched.')

# Links remain visible even after checkout, admission, or campaign failure.
for label, name in [('Analysis', 'ELENDRIL_V7_ANALYSIS.zip'), ('Full', 'ELENDRIL_V7_FULL.zip')]:
    path = OUTPUT / 'exports' / name
    print(f'{label} archive: {path} (exists: {path.is_file()})')
    display(FileLink(str(path)))
if campaign_failed:
    raise RuntimeError('Campaign failed or is incomplete. Inspect NO-GO/PARTIAL receipts and download available outputs; do not treat this as a successful study.')
'''
    return {'cells': [_cell('markdown', intro), _cell('code', config), _cell('code', launch)],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}, 'source_reference': source_sha,
                         'elendril_v7_status': 'OWNER_LAUNCH_GPU_QUALIFICATION_PENDING'},
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
