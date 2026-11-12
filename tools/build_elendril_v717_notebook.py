"""Publish the source-pinned S20 investigation; no import-time execution."""
import argparse
import json
from pathlib import Path
import re
from tools.build_elendril_v715_notebook import cell

DEST = Path(__file__).resolve().parents[1] / 'notebooks' / 'ELENDRIL_V717_KAGGLE_T4X2.ipynb'


def build(source):
    if not isinstance(source, str) or re.fullmatch('[0-9a-f]{40}', source) is None:
        raise ValueError('immutable 40-hex source SHA required')
    intro = f'''# ELENDRIL V7.17 — locate the S20 failure

Run All starts the **eight-job optimization phase** after two-GPU admission.
S20 only: three baseline seeds, three matched lower-LR runs, two crossed
initialization/order-seed runs. Four epochs each. No input dataset is needed:
data are generated and audited automatically in writable working storage.

For a separate six-job session, select `representation`: three baselines and
three matched evidence-row-order interventions. Repeated baselines across
sessions are deterministic repeats, not independent seeds. Both phases use
the same held-out suite; confirmation is measured only at the final epoch.

Kaggle: enable Internet, Python, **GPU T4 ×2**. Keep native PyTorch; no pip
replacement. 330 minutes is a cap, not a runtime guarantee. Full diagnostic
timings and exact checkpoint recovery must pass on both GPUs. An over-budget
phase reports NO_GO rather than dropping seeds, epochs or measurements.

Training histories, training/held-out teacher-forced and decoded measurements,
raw interactive traces, intermediate weights and optimizer state are preserved.
Download **ELENDRIL_V71_ANALYSIS.zip**, **ELENDRIL_V71_FULL.zip** and
**FINAL_STATUS.json** from exports; inherited ZIP names are intentional.
Save Kaggle output before ending the session. Resume from an unchanged output
directory, or set RESTORE_ARCHIVE to a prior FULL ZIP from this exact source
and phase with a fresh destination. AN ANALYSIS ZIP CANNOT RESTORE WEIGHTS.

Engineering qualification is pending owner GPU execution. No root cause or
scientific success is claimed. The model is trained from scratch on symbolic
actions, not natural-language pretraining. Source: `{source}`.
'''
    config = f'''from pathlib import Path
SOURCE_SHA = "{source}"
RUN_STUDY = True  # Owner-requested Run All launch; False performs no training.
PHASE = "optimization"  # optimization / representation; one whole phase per session
MINUTES = 330
RESTORE_ARCHIVE = None  # Optional exact-source, same-phase FULL ZIP path under /kaggle/input.
OUTPUT = Path('/kaggle/working') / ('elendril-v717-' + SOURCE_SHA[:12] + '-' + PHASE)
'''
    launch = '''import os, re, subprocess, sys
from uuid import uuid4
from IPython.display import FileLink, display

REPOSITORY = 'https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git'
REPO = Path('/kaggle/working') / ('elendril-v717-source-' + SOURCE_SHA)
failed = False
if RUN_STUDY:
    try:
        if not re.fullmatch('[0-9a-f]{40}', SOURCE_SHA): raise ValueError('invalid source SHA')
        if PHASE not in ('optimization', 'representation'): raise ValueError('unknown phase')
        if type(MINUTES) not in (int, float) or not 0 < MINUTES <= 330: raise ValueError('invalid minutes')
        if not REPO.exists():
            stage = REPO.parent / ('.v717-source-' + uuid4().hex)
            stage.mkdir(parents=True)
            for cmd in (['git','init','-q'], ['git','remote','add','origin',REPOSITORY],
                        ['git','fetch','--depth=1','origin',SOURCE_SHA], ['git','checkout','--detach',SOURCE_SHA]):
                subprocess.run(cmd, cwd=stage, check=True, timeout=240)
            stage.rename(REPO)
        if REPO.is_symlink(): raise ValueError('linked source checkout')
        if subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip() != SOURCE_SHA:
            raise ValueError('source pin mismatch')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'], cwd=REPO, text=True).strip():
            raise ValueError('dirty source checkout')
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        if RESTORE_ARCHIVE is not None:
            if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('restore needs fresh output directory')
            script = ('from tools.elendril_v71_export import restore_archive; import sys; '
                      'restore_archive(sys.argv[1],sys.argv[2],source=sys.argv[3],phase=sys.argv[4])')
            subprocess.run([sys.executable,'-c',script,str(RESTORE_ARCHIVE),str(OUTPUT),SOURCE_SHA,PHASE],
                           cwd=REPO,env=env,check=True)
        subprocess.run([sys.executable,'-m','tools.elendril_v717_campaign','--owner-enabled',
                        '--source',SOURCE_SHA,'--phase',PHASE,'--minutes',str(MINUTES),
                        '--output',str(OUTPUT)],cwd=REPO,env=env,check=True)
    except Exception as exc:
        failed = True
        print('Study incomplete:', type(exc).__name__, str(exc))
        print('Retain partial outputs and receipts; no successful study is claimed.')
else:
    print('Study disabled by owner; no training launched.')
for name in ('ELENDRIL_V71_ANALYSIS.zip','ELENDRIL_V71_FULL.zip','FINAL_STATUS.json'):
    file = OUTPUT / 'exports' / name
    print(name, 'exists:', file.is_file())
    if file.is_file(): display(FileLink(str(file)))
if failed: raise RuntimeError('V7.17 incomplete; inspect saved receipts and download available outputs.')
'''
    return dict(cells=[cell('markdown', intro),cell('code',config),cell('code',launch)],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                      language_info=dict(name='python'),source_reference=source,
                      elendril_v717_status='OWNER_GPU_QUALIFICATION_PENDING'),nbformat=4,nbformat_minor=4)


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument('--source-sha',required=True)
    p.add_argument('--output',type=Path,default=DEST); a=p.parse_args(argv)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(build(a.source_sha),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__ == '__main__': main()
