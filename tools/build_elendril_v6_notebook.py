from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

from anra_v5.elendril.v6_registry import REGISTRY_SHA, registry

ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT = ROOT / "research/v6/ELENDRIL_V6_KAGGLE_T4X2_BLUEPRINT.ipynb"
DEST = ROOT / "notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb"
REFERENCE_PRE_V6_SHA = "0cd200730d43547294cdc9bf8540a478130cce9c"

def build(source_sha: str) -> dict:
    """Pure notebook construction; no writes, device discovery, network or campaign import."""
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", source_sha) or source_sha == REFERENCE_PRE_V6_SHA:
        raise ValueError("source SHA must be a new immutable V6 commit")
    registered = registry()
    notebook = json.loads(BLUEPRINT.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# ELENDRIL V6 · Grand Diagnostic Campaign — Kaggle T4×2\n\n",
        "**Status: OWNER-RUN PREPARATION.** Hardware availability and true fresh-VM recovery remain unverified until Kaggle receipts are inspected.\n\n",
        f"**Pinned source:** `{source_sha}`\n\n",
        f"**Registered V6 registry SHA-256:** `{REGISTRY_SHA}`. Context lengths are 512/512/640 for S3/S8/S20.\n\n",
        "One bundled campaign preserves positive and negative evidence, independently rescores it, and creates one verified ZIP. Training requires two physical T4 UUIDs and native CUDA PyTorch; verified ZIP export retry can run CPU-only from saved evidence. No CUDA stack replacement is installed.\n",
        "\nThe original 15 panels are accompanied by four preregistered exploratory probes: effort budgets, symbol permutation, stale-memory interference and query-cost robustness. They reuse frozen checkpoints, add no training arms, and share the existing time/export ceiling.\n",
    ]
    config = """# OWNER-AUTHORIZED RUN-ALL RELEASE. Kaggle Run All starts the campaign; no code edits are needed.
from pathlib import Path
SOURCE_SHA = \"%s\"
EXPECTED_REGISTRY_SHA256 = \"%s\"
RUN_ALL_19 = True  # This release is pre-authorized for the user's one-session run.
RUN_PREFLIGHT = RUN_ALL_19
RUN_CANARIES = RUN_ALL_19
RUN_RECOVERY = False  # Separate-VM recovery is intentionally not required.
ONE_SESSION_CAMPAIGN = True  # Fresh-process checkpoint/optimizer recovery runs on both T4s.
OWNER_ATTESTS_FRESH_VM = False
OWNER_APPROVES_CAMPAIGN = RUN_ALL_19
RUN_CAMPAIGN = RUN_ALL_19
RUN_EXPORT = False  # Separate CPU-only export retry; campaign exports automatically.
SESSION_WALL_LIMIT_MINUTES = 360
EXPORT_RESERVE_MINUTES = 35
OUTPUT = Path('/kaggle/working/ELENDRIL_V6')
CHECKPOINT_REATTACH_DIR = None
PRIOR_CAMPAIGN_OUTPUT = None  # Optional saved partial campaign root/ZIP; same source and run ID.
RUN_ID = \"elendril-v6-diagnostic-002\"
assert type(RUN_ALL_19) is bool
assert all(type(v) is bool for v in (RUN_PREFLIGHT,RUN_CANARIES,RUN_RECOVERY,OWNER_ATTESTS_FRESH_VM,OWNER_APPROVES_CAMPAIGN,RUN_CAMPAIGN,RUN_EXPORT))
if (RUN_CANARIES or RUN_RECOVERY or RUN_CAMPAIGN) and not RUN_PREFLIGHT: raise RuntimeError('NO-GO: GPU action requires pinned-source preflight')
if RUN_RECOVERY and not OWNER_ATTESTS_FRESH_VM: raise RuntimeError('NO-GO: recovery requires owner attestation of a genuinely fresh VM')
if RUN_CAMPAIGN and not (RUN_CANARIES and OWNER_APPROVES_CAMPAIGN and (ONE_SESSION_CAMPAIGN or RUN_RECOVERY)): raise RuntimeError('NO-GO: campaign needs canaries, owner approval, and an admitted recovery mode')
if RUN_CAMPAIGN and ONE_SESSION_CAMPAIGN and RUN_RECOVERY: raise RuntimeError('NO-GO: choose one-session mode OR fresh-VM recovery mode')
print({\"source\":SOURCE_SHA,\"campaign_enabled\":RUN_CAMPAIGN})
""" % (source_sha, REGISTRY_SHA)
    notebook["cells"][1]["source"] = config.splitlines(keepends=True)
    registry_source = """# This is a static, registered protocol rendering. It imports no campaign, data, or device code.
import json
RENDERED_REGISTRY_SHA256 = \"%s\"
REGISTRY = json.loads(r'''%s''')
assert EXPECTED_REGISTRY_SHA256 == RENDERED_REGISTRY_SHA256
PANELS = REGISTRY['panels']
MODEL_PROFILES = REGISTRY['profiles']
ARMS = REGISTRY['arms']
assert len(PANELS)==15 and [p['id'] for p in PANELS]==[f'P{i:02d}' for i in range(1,16)]
assert [MODEL_PROFILES[n]['model']['context_length'] for n in ('S3','S8','S20')]==[512,512,640]
print('Registered panels:',[(p['id'],p['name']) for p in PANELS])
print('Registered model contexts:',{n:MODEL_PROFILES[n]['model']['context_length'] for n in ('S3','S8','S20')})
assert [s['id'] for s in REGISTRY['supplemental_experiments']]==['S16','S17','S18','S19']
print('Additional experiments:',[(s['id'],s['name']) for s in REGISTRY['supplemental_experiments']])
""" % (REGISTRY_SHA, json.dumps(registered, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    notebook["cells"][3]["source"] = registry_source.splitlines(keepends=True)
    notebook["cells"][5]["source"] = """# RUN ALL is enabled by default. Fail-closed admission happens before any campaign training.
import os, re, subprocess, sys, shutil
from pathlib import Path
REPOSITORY = \"https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git\"
REPO = Path('/tmp') / ('elendril-v6-pinned-' + SOURCE_SHA[:12])
ENVIRONMENT = None
v6 = None
if RUN_PREFLIGHT or RUN_EXPORT:
    if not re.fullmatch(r\"[0-9a-f]{40}\", SOURCE_SHA): raise RuntimeError(\"invalid pinned source\")
    os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if not REPO.exists():
        REPO.mkdir(parents=True)
        subprocess.run([\"git\",\"init\",\"-q\"],cwd=REPO,check=True,timeout=30)
        subprocess.run([\"git\",\"remote\",\"add\",\"origin\",REPOSITORY],cwd=REPO,check=True,timeout=30)
        subprocess.run([\"git\",\"fetch\",\"--depth=1\",\"origin\",SOURCE_SHA],cwd=REPO,check=True,timeout=240)
        subprocess.run([\"git\",\"checkout\",\"--detach\",SOURCE_SHA],cwd=REPO,check=True,timeout=60)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True,timeout=30).strip()
    if head != SOURCE_SHA or subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=REPO,text=True,timeout=30).strip():
        raise RuntimeError('NO-GO: pinned source checkout is not exact and clean')
    sys.path.insert(0,str(REPO)); from tools import elendril_v6_campaign as v6
    if Path(v6.__file__).resolve() != (REPO/'tools'/'elendril_v6_campaign.py').resolve(): raise RuntimeError('wrong campaign operator loaded')
    if RUN_PREFLIGHT:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=2: raise RuntimeError('NO-GO: native CUDA with two GPUs required')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.free,memory.total,driver_version','--format=csv,noheader,nounits'],text=True,timeout=30)
        info=[dict(zip(['index','name','uuid','free_mib','total_mib','driver'],map(str.strip,row.split(',')))) for row in raw.splitlines() if row.strip()]
        if len(info)!=2 or len({g['uuid'] for g in info})!=2 or any('T4' not in g['name'].upper() or int(g['free_mib'])<11000 for g in info): raise RuntimeError('NO-GO: require two distinct physical T4s with >=11 GiB free each')
        ENVIRONMENT=v6.preflight(source=SOURCE_SHA,protocol_sha=EXPECTED_REGISTRY_SHA256,devices=info,run_id=RUN_ID,output=OUTPUT,available_minutes=SESSION_WALL_LIMIT_MINUTES,prior_output=PRIOR_CAMPAIGN_OUTPUT)
        if ENVIRONMENT.get('status')!='PASS': raise RuntimeError('NO-GO: V6 preflight failed')
        print(ENVIRONMENT)
    else:
        # An existing saved result can be independently replayed and zipped without a GPU.
        import json
        if PRIOR_CAMPAIGN_OUTPUT is not None:
            v6.restore_output(PRIOR_CAMPAIGN_OUTPUT, OUTPUT, source=SOURCE_SHA, protocol_sha=EXPECTED_REGISTRY_SHA256)
        source_record=OUTPUT/'source.json'
        if not source_record.is_file(): raise RuntimeError('NO-GO: attach saved Kaggle output via PRIOR_CAMPAIGN_OUTPUT or provide an existing verified OUTPUT directory')
        prior=json.loads(source_record.read_bytes())
        if prior.get('source')!=SOURCE_SHA or prior.get('protocol_sha256')!=EXPECTED_REGISTRY_SHA256 or prior.get('run_id')!=RUN_ID:
            raise RuntimeError('NO-GO: saved output source, protocol, or run ID differs')
        print('EXPORT_ONLY_SOURCE_AND_OUTPUT_BOUND; no CUDA device required')
else: print('PREFLIGHT_DISABLED: no download, GPU access, or writes')
""".splitlines(keepends=True)
    notebook["cells"][6]["source"] = """## 3 · Automatic two-T4 canaries and checkpoint recovery

Running this notebook top-to-bottom automatically checks both T4s, a positive-control training task, CUDA memory/timing, and checkpoint+optimizer recovery in independent child processes. **No second Kaggle session is required.** Any failed canary stops expensive training rather than returning misleading results.
""".splitlines(keepends=True)
    notebook["cells"][7]["source"] = """CANARY_RECEIPTS=None
RECOVERY_RECEIPTS=None
if RUN_CANARIES:
    if not RUN_PREFLIGHT or ENVIRONMENT is None: raise RuntimeError('NO-GO: preflight first')
    CANARY_RECEIPTS=v6.run_canaries(source=SOURCE_SHA,protocol_sha=EXPECTED_REGISTRY_SHA256,
        preflight=ENVIRONMENT,output=OUTPUT/'qualification',per_device=True)
    if CANARY_RECEIPTS.get('status')!='PASS': raise RuntimeError('NO-GO: T4 canaries did not pass; training has not started')
    print('T4_CANARIES_PASSED; checked checkpoint/optimizer recovery in fresh child processes; starting single-session campaign next')
else:
    print('CANARIES_DISABLED_BY_PREVIEW_MODE')
if RUN_RECOVERY:
    if not RUN_PREFLIGHT or not OWNER_ATTESTS_FRESH_VM or CHECKPOINT_REATTACH_DIR is None:
        raise RuntimeError('NO-GO: separate-VM recovery requires owner attestation and reattached evidence')
    RECOVERY_RECEIPTS=v6.verify_off_session_recovery(source=SOURCE_SHA,
        prior_dir=Path(CHECKPOINT_REATTACH_DIR),current_environment=ENVIRONMENT,
        output=OUTPUT/'recovery',must_resume_optimizer=True)
    if RECOVERY_RECEIPTS.get('status')!='PASS': raise RuntimeError('NO-GO: separate-VM recovery failed')
else:
    print('SEPARATE_VM_RECOVERY_NOT_REQUIRED_FOR_THIS_RUN')
""".splitlines(keepends=True)
    notebook["cells"][8]["source"] = """## 4 · Automatically launch the full 19-panel campaign

**This release is ready for a user-initiated Kaggle Run All; there is no launch switch to edit.** Real T4 admission, positive learning controls and per-T4 checkpoint recovery execute before training. The operator runs registered training arms, shared P01–P15 diagnostic panels and S16–S19 exploratory probes within the measured budget; conditional arms may be skipped if not scientifically qualified. Partial or negative outcomes are retained, and the run exports a verified results ZIP.
""".splitlines(keepends=True)
    notebook["cells"][9]["source"] = """CAMPAIGN_RESULT=None
if RUN_CAMPAIGN:
    if not (ENVIRONMENT and CANARY_RECEIPTS and (ONE_SESSION_CAMPAIGN or RECOVERY_RECEIPTS)): raise RuntimeError('NO-GO: preflight, canary or mandatory recovery receipts missing')
    if SESSION_WALL_LIMIT_MINUTES < EXPORT_RESERVE_MINUTES+60 or SESSION_WALL_LIMIT_MINUTES>600: raise ValueError('invalid wall limit')
    CAMPAIGN_RESULT=v6.run_campaign(source=SOURCE_SHA,protocol_sha=EXPECTED_REGISTRY_SHA256,preflight=ENVIRONMENT,canary=CANARY_RECEIPTS,recovery=RECOVERY_RECEIPTS,run_id=RUN_ID,output=OUTPUT,wall_limit_seconds=SESSION_WALL_LIMIT_MINUTES*60,export_reserve_seconds=EXPORT_RESERVE_MINUTES*60,preserve_failed_and_partial=True,one_session=ONE_SESSION_CAMPAIGN)
    print(CAMPAIGN_RESULT)
    export_report=CAMPAIGN_RESULT.get('export',{})
    if export_report.get('status')=='EXPORT_VERIFIED':
        from IPython.display import FileLink, display
        display(FileLink(export_report['zip_path']))
else: print('CAMPAIGN_DISABLED')
""".splitlines(keepends=True)
    notebook["cells"][10]["source"] = "## 5 · Independently rescored evidence ZIP\n\nThe campaign finalizer exports in its `finally` path, including partial and failed results. This retry cell is optional.\n".splitlines(keepends=True)
    notebook["cells"][11]["source"] = """ZIP_PATH=None
if RUN_EXPORT:
    if v6 is None: raise RuntimeError('NO-GO: pinned source checkout and bound output are required for export')
    report=v6.independent_rescore_and_export(source=SOURCE_SHA,protocol_sha=EXPECTED_REGISTRY_SHA256,campaign=CAMPAIGN_RESULT,output=OUTPUT,zip_name='ELENDRIL_V6_RESULTS.zip',include_all_failures=True)
    ZIP_PATH=Path(report['zip_path'])
    if not ZIP_PATH.is_file() or report.get('status')!='EXPORT_VERIFIED': raise RuntimeError('NO-GO: ZIP verification failed; inspect retained source outputs')
    from IPython.display import FileLink, display
    display(FileLink(str(ZIP_PATH)))
else: print('EXPORT_RETRY_DISABLED')
""".splitlines(keepends=True)
    notebook["cells"][12]["source"] = """## Scientific interpretation rules

A failed positive-control overfit test indicts objective, tokenizer, gradient or checkpoint mechanics before any scale experiment. Correct familiar actions but poor aliases suggest representation/data overfit. Good accuracy but wasteful queries requires cost-aware policy work. Two-hop success but three-hop failure indicates depth limits. Fixed-retrieval success does not imply a learned retrieval policy. Sequential retention without acquired B is not cumulative learning. Always preserve negatives and full raw receipts.

**The V6 campaign implementation is source-pinned and CPU-engineering-qualified. In ONE Kaggle session, run preflight, live positive canaries, and fresh-process checkpoint/optimizer resume on BOTH T4s before the training campaign. Separate-VM recovery is optional, not a required extra session.** ZIP retry can run on CPU from saved output.
""".splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["metadata"]["elendril_v6_status"] = "OWNER_RUN_ALL_ENABLED_GPU_QUALIFICATION_PENDING"
    notebook["metadata"]["source_reference"] = source_sha
    notebook["metadata"]["registry_sha256"] = REGISTRY_SHA
    return notebook

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pinned V6 Kaggle notebook from the reviewed blueprint")
    parser.add_argument("--source-sha", required=True, help="clean immutable 40-character V6 source commit")
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    notebook = build(args.source_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} for source {args.source_sha}; registry {REGISTRY_SHA}")

if __name__ == "__main__": main()
