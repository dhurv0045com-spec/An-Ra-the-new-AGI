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
        "One bundled campaign preserves positive and negative evidence, independently rescores it, and creates one verified ZIP. Two physical T4 UUIDs and native CUDA PyTorch are mandatory. No CUDA stack replacement is installed.\n",
        "\nThe original 15 panels are accompanied by four preregistered exploratory probes: effort budgets, symbol permutation, stale-memory interference and query-cost robustness. They reuse frozen checkpoints, add no training arms, and share the existing time/export ceiling.\n",
    ]
    config = """# SINGLE OWNER CONFIGURATION CELL. All expensive switches are disabled.
from pathlib import Path
SOURCE_SHA = \"%s\"
EXPECTED_REGISTRY_SHA256 = \"%s\"
RUN_PREFLIGHT = False
RUN_CANARIES = False
RUN_RECOVERY = False
OWNER_ATTESTS_FRESH_VM = False
OWNER_APPROVES_CAMPAIGN = False
RUN_CAMPAIGN = False
RUN_EXPORT = False  # Retry export from an existing result directory
SESSION_WALL_LIMIT_MINUTES = 360
EXPORT_RESERVE_MINUTES = 35
OUTPUT = Path('/kaggle/working/ELENDRIL_V6')
CHECKPOINT_REATTACH_DIR = None
PRIOR_CAMPAIGN_OUTPUT = None  # Optional saved partial campaign root/ZIP; same source and run ID.
RUN_ID = \"elendril-v6-diagnostic-002\"
assert all(type(v) is bool for v in (RUN_PREFLIGHT,RUN_CANARIES,RUN_RECOVERY,OWNER_ATTESTS_FRESH_VM,OWNER_APPROVES_CAMPAIGN,RUN_CAMPAIGN,RUN_EXPORT))
if (RUN_CANARIES or RUN_RECOVERY or RUN_CAMPAIGN or RUN_EXPORT) and not RUN_PREFLIGHT: raise RuntimeError('NO-GO: enabled action requires pinned-source preflight')
if RUN_RECOVERY and not OWNER_ATTESTS_FRESH_VM: raise RuntimeError('NO-GO: recovery requires owner attestation of a genuinely fresh VM')
if RUN_CAMPAIGN and not (RUN_CANARIES and RUN_RECOVERY and OWNER_APPROVES_CAMPAIGN): raise RuntimeError('NO-GO: campaign needs canary, fresh-VM recovery, and explicit owner approval')
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
    notebook["cells"][5]["source"] = """# No network, disk writes, or GPU work with defaults.
import os, re, subprocess, sys, shutil
from pathlib import Path
REPOSITORY = \"https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git\"
REPO = Path('/tmp/elendril-v6-pinned-source')
ENVIRONMENT = None
v6 = None
if RUN_PREFLIGHT:
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
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2: raise RuntimeError('NO-GO: native CUDA with two GPUs required')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.free,memory.total,driver_version','--format=csv,noheader,nounits'],text=True,timeout=30)
    info=[dict(zip(['index','name','uuid','free_mib','total_mib','driver'],map(str.strip,row.split(',')))) for row in raw.splitlines() if row.strip()]
    if len(info)!=2 or len({g['uuid'] for g in info})!=2 or any('T4' not in g['name'].upper() or int(g['free_mib'])<11000 for g in info): raise RuntimeError('NO-GO: require two distinct physical T4s with >=11 GiB free each')
    ENVIRONMENT=v6.preflight(source=SOURCE_SHA,protocol_sha=EXPECTED_REGISTRY_SHA256,devices=info,run_id=RUN_ID,output=OUTPUT,available_minutes=SESSION_WALL_LIMIT_MINUTES,prior_output=PRIOR_CAMPAIGN_OUTPUT)
    if ENVIRONMENT.get('status')!='PASS': raise RuntimeError('NO-GO: V6 preflight failed')
    print(ENVIRONMENT)
else: print('PREFLIGHT_DISABLED: no download, GPU access, or writes')
""".splitlines(keepends=True)
    notebook["cells"][9]["source"] = """CAMPAIGN_RESULT=None
if RUN_CAMPAIGN:
    if not (ENVIRONMENT and CANARY_RECEIPTS and RECOVERY_RECEIPTS): raise RuntimeError('NO-GO: preflight, canary, or recovery receipts are missing')
    if SESSION_WALL_LIMIT_MINUTES < EXPORT_RESERVE_MINUTES+60 or SESSION_WALL_LIMIT_MINUTES>600: raise ValueError('invalid wall limit')
    CAMPAIGN_RESULT=v6.run_campaign(source=SOURCE_SHA,protocol_sha=EXPECTED_REGISTRY_SHA256,preflight=ENVIRONMENT,canary=CANARY_RECEIPTS,recovery=RECOVERY_RECEIPTS,run_id=RUN_ID,output=OUTPUT,wall_limit_seconds=SESSION_WALL_LIMIT_MINUTES*60,export_reserve_seconds=EXPORT_RESERVE_MINUTES*60,preserve_failed_and_partial=True)
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
    if v6 is None: raise RuntimeError('NO-GO: source preflight is needed to load the pinned exporter')
    report=v6.independent_rescore_and_export(source=SOURCE_SHA,protocol_sha=EXPECTED_REGISTRY_SHA256,campaign=CAMPAIGN_RESULT,output=OUTPUT,zip_name='ELENDRIL_V6_RESULTS.zip',include_all_failures=True)
    ZIP_PATH=Path(report['zip_path'])
    if not ZIP_PATH.is_file() or report.get('status')!='EXPORT_VERIFIED': raise RuntimeError('NO-GO: ZIP verification failed; inspect retained source outputs')
    from IPython.display import FileLink, display
    display(FileLink(str(ZIP_PATH)))
else: print('EXPORT_RETRY_DISABLED')
""".splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["metadata"]["elendril_v6_status"] = "PINNED_OWNER_RUN_PREPARATION_UNQUALIFIED"
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
