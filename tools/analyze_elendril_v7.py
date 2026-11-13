"""Archive owner-returned V7 evidence without training or modifying the input."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from anra_v5.elendril.v7_protocol import PROTOCOL_SHA
from tools.elendril_cloud import safe_id, fingerprint
from tools.elendril_v6_export import extract_verified, immutable_json, inventory, MANIFEST_NAME
from tools.elendril_v7_export import _validate_analysis
from tools.elendril_v7_campaign import rescore
from tools.elendril_v7_report import write_summary

DEFAULT_ARCHIVE=Path(__file__).resolve().parents[1]/'Experiment Result'/'V7'


def analyze(source, archive, run_id):
    safe_id(run_id)
    requested=Path(source)
    if requested.is_symlink(): raise ValueError('input cannot be a symlink')
    source=requested.resolve(strict=True); archive=Path(archive).resolve()
    target=archive/run_id
    if target.exists(): raise FileExistsError('run ID already archived; original evidence is immutable')
    if source.is_dir() and (archive==source or archive.is_relative_to(source)):
        raise ValueError('archive must be outside input tree')
    with tempfile.TemporaryDirectory(prefix='elendril-v7-intake-') as temporary:
        verified=None
        if source.is_file():
            verified=extract_verified(source,Path(temporary)/'extracted')
            root=Path(temporary)/'extracted'
        else: root=source
        analysis=root/'analysis' if (root/'analysis').is_dir() else root
        _validate_analysis(analysis)
        identity=json.loads((analysis/'identity.json').read_bytes())
        if identity['protocol_sha256']!=PROTOCOL_SHA: raise ValueError('V7 protocol differs')
        if verified is not None and (verified['source']!=identity['source']
                                     or verified['protocol_sha256']!=PROTOCOL_SHA):
            raise ValueError('archive and analysis identities disagree')
        try:
            score=rescore(root,{}) if analysis!=root else {'status':'NOT_RUN',
                'reason':'Analysis-only bundle lacks datasets; use full ZIP for independent action replay.'}
        except Exception as exc:
            score={'status':'FAILED','error_type':type(exc).__name__,'message':str(exc)}
        archive.mkdir(parents=True,exist_ok=True)
        stage=archive/('.pending-'+uuid4().hex); stage.mkdir()
        try:
            records=inventory(analysis)
            for record in records:
                src=analysis/record['name']; raw=stage/'raw'/'analysis'/record['name']
                expected=fingerprint(src)
                raw.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,raw)
                if fingerprint(raw)!=expected or fingerprint(src)!=expected:
                    raise ValueError('input changed during archival')
                if record['name'] in ('SHA256SUMS',MANIFEST_NAME): continue
                derived=stage/'analysis'/record['name']; derived.parent.mkdir(parents=True,exist_ok=True)
                os.link(raw,derived)  # replacements of derived reports retain original raw bytes
            summary=write_summary(stage,reference_root=root)
            science=json.loads((stage/'analysis'/'scientific_report.json').read_bytes())
            if inventory(analysis)!=records or any(fingerprint(analysis/r['name'])!=fingerprint(stage/'raw'/'analysis'/r['name']) for r in records):
                raise ValueError('input changed during analysis')
            receipt={'schema':'elendril-v7-owner-analysis-v1','source':identity['source'],
                'protocol_sha256':PROTOCOL_SHA,'input':str(source),
                'input_zip_sha256':fingerprint(source) if source.is_file() else None,
                'input_archive_verification':verified,'independent_action_replay':score,
                'report_status':summary['status'],'scientific_report_status':science['status'],
                'qualification':'REPLAYED_DESCRIPTIVE_EVIDENCE' if score['status']=='PASS' and science['status']=='AVAILABLE'
                                else 'UNQUALIFIED_OR_PARTIAL_EVIDENCE',
                'scientific_conclusion':'CAUSE UNRESOLVED',
                'limits':'Recorded-action replay and descriptive interventions do not prove a neural mechanism or weight custody.',
                'raw_files':[{**r,'sha256':fingerprint(stage/'raw'/'analysis'/r['name'])} for r in records]}
            immutable_json(stage/'ARCHIVE_RECEIPT.json',receipt)
            stage.rename(target)
        except Exception:
            # Retain any partial intake for inspection; never overwrite a prior run.
            raise
    return {'archive':str(target),'independent_action_replay':score,'scientific_conclusion':'CAUSE UNRESOLVED'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    p.add_argument('--run-id',required=True); args=p.parse_args(argv)
    print(json.dumps(analyze(args.input,args.archive,args.run_id),indent=2))


if __name__=='__main__': main()
