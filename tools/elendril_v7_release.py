"""Fail-closed coverage and delivery gates; execution is not scientific success."""
import json
from hashlib import sha256
import math
import re
from pathlib import Path
import zipfile

from anra_v5.elendril.v7_protocol import PROTOCOL_SHA, jobs, milestones
from tools.elendril_cloud import fingerprint
from tools.elendril_v6_export import MANIFEST_NAME, verify_zip, immutable_json, _write_derived, canonical
from tools.elendril_v7_export import export_results
from v5_training.checkpoint import REQUIRED_COMPONENTS


def required_results(root, campaign):
    root=Path(root); errors=[]; analysis=set(); checkpoints=set()
    try:
        manifest=json.loads((root/'analysis'/'data_manifest.json').read_bytes())
        examples=manifest['train']['manifest']['examples']; per_epoch=math.ceil(examples/16)
        expected={j['job_id']:j for j in jobs()}
        results=campaign['results']
        if len(results)!=9 or {r['job']['job_id'] for r in results}!=set(expected):
            raise ValueError('nine unique registered jobs required')
        canaries=campaign['canaries']
        if len(canaries)!=2 or any(c['status']!='PASS' for c in canaries) or len({c['gpu_uuid'] for c in canaries})!=2:
            raise ValueError('two independently qualified physical GPU lanes required')
        for row in results:
            job=row['job']; jid=job['job_id']
            if job!=expected[jid] or row['status']!='COMPLETE' or row['global_update']!=4*per_epoch:
                raise ValueError('job identity, completion or four-epoch endpoint differs: '+jid)
            specs=[(f'diagnostic-u{u}.json','monitor',u) for u in (0,*milestones(examples))]
            specs += [(f'{prefix}-u{e*per_epoch}.json',cohort,e*per_epoch)
                      for e in (1,2,4) for prefix,cohort in (('development','validation'),('training','train-monitor'))]
            specs += [('confirmation.json','confirmation',4*per_epoch)]
            for name,cohort,u in specs:
                rel=f'jobs/{jid}/{name}'; analysis.add(rel)
                report=json.loads((root/'analysis'/rel).read_bytes())
                if report['update']!=u or report['data_identity']!=manifest[cohort]['identity']:
                    raise ValueError('required report cohort/update mismatch: '+rel)
                if u==4*per_epoch and report['checkpoint']!=row['checkpoint']:
                    raise ValueError('terminal checkpoint mismatch: '+rel)
                if u:
                    if re.fullmatch('[0-9a-f]{64}',report['checkpoint']) is None:
                        raise ValueError('invalid checkpoint identity')
                    cp=f'checkpoints/jobs/{jid}/{jid}/objects/{report["checkpoint"]}'
                    checkpoints.update(f'{cp}/{part}' for part in ('manifest.json',*REQUIRED_COMPONENTS))
        analysis.update(('data_manifest.json','protocol.json','identity.json','summary.json',
                         'trajectory.csv','scientific_report.json','scientific_report.md'))
        analysis.update(f'controls/{c}.json' for c in ('monitor','validation','confirmation'))
        for name in analysis:
            if not (root/'analysis'/name).is_file(): errors.append('missing analysis/'+name)
        for name in checkpoints:
            if not (root/name).is_file(): errors.append('missing '+name)
        science=json.loads((root/'analysis'/'scientific_report.json').read_bytes())
        if science['status']!='AVAILABLE' or science['errors']:
            errors.append('scientific observations are incomplete or invalid')
    except (OSError,KeyError,TypeError,ValueError) as exc:
        errors.append(str(exc))
    return {'status':'PASS' if not errors else 'FAILED','errors':errors,
            'analysis_members':sorted(analysis),'checkpoint_members':sorted(checkpoints)}


def verify_checkpoint_members(archive, required):
    """Bind packaged tensor bytes to the original content-addressed checkpoint."""
    outer=json.loads(archive.read(MANIFEST_NAME))
    files={r['name']:r for r in outer['files']}
    for name in required:
        if not name.endswith('/manifest.json'): continue
        payload=archive.read(name); parent=name.rsplit('/',1)[0]
        if sha256(payload).hexdigest()!=parent.rsplit('/',1)[1]:
            raise ValueError('checkpoint manifest identity mismatch')
        manifest=json.loads(payload)
        components=manifest['components']
        if len(components)!=len(REQUIRED_COMPONENTS) or {c['name'] for c in components}!=REQUIRED_COMPONENTS:
            raise ValueError('checkpoint component inventory mismatch')
        for c in components:
            record=files[parent+'/'+c['name']]
            if record['sha256']!=c['sha256'] or record['byte_size']!=c['byte_size']:
                raise ValueError('checkpoint component differs from committed manifest')


def verify_delivery(root, exports, campaign, run_id):
    """Reopen both current-run ZIPs; status strings or stale files are insufficient."""
    root=Path(root); errors=[]; verified={}
    for kind,name in (('analysis','ELENDRIL_V7_ANALYSIS.zip'),('full','ELENDRIL_V7_FULL.zip')):
        try:
            report=exports[kind]; path=root/'exports'/name
            if report.get('status')!='EXPORT_VERIFIED' or Path(report['zip_path']).resolve()!=path.resolve():
                raise ValueError('mandatory export not verified')
            if path.is_symlink() or not path.is_file() or fingerprint(path)!=report['zip_sha256']:
                raise ValueError('missing, linked or changed archive')
            checked=verify_zip(path)
            if (checked['source']!=campaign['source'] or checked['protocol_sha256']!=PROTOCOL_SHA
                    or checked['manifest_sha256']!=report['manifest_sha256']
                    or checked['scientific_rescore']!=campaign['independent_replay']):
                raise ValueError('archive source/protocol/manifest mismatch')
            with zipfile.ZipFile(path) as archive:
                names=set(archive.namelist())
                prefix='' if kind=='analysis' else 'analysis/'
                required={prefix+'campaign-'+run_id+'.json'}
                if campaign.get('execution_status')=='EXECUTED':
                    coverage=campaign['required_results']
                    required.update(prefix+p for p in coverage['analysis_members'])
                    if kind=='full':
                        required.update(coverage['checkpoint_members'])
                        verify_checkpoint_members(archive,coverage['checkpoint_members'])
                if not required<=names: raise ValueError('current campaign or mandatory results absent from ZIP')
            verified[kind]=checked
        except (OSError,KeyError,TypeError,ValueError,zipfile.BadZipFile) as exc:
            errors.append(kind+': '+str(exc))
    return {'status':'PASS' if not errors else 'FAILED','errors':errors,'verified':verified}


def finalize(root, campaign, run_id):
    root=Path(root)
    campaign['execution_status']=campaign['status']
    if campaign['status']=='EXECUTED':
        campaign['required_results']=required_results(root,campaign)
        if (campaign['required_results']['status']!='PASS'
                or campaign.get('independent_replay',{}).get('status')!='PASS'
                or campaign.get('summary',{}).get('status')!='COMPLETE'):
            campaign['status']='INVALID_RESULTS'
        else: campaign['status']='FINALIZING'
    campaign['delivery_status']='PENDING'
    campaign['scientific_conclusion']='CAUSE UNRESOLVED; execution completion is not scientific success.'
    # ZIPs carry a pre-export receipt. A final receipt cannot certify its own
    # archive hash: publish it separately after both ZIPs have been reopened.
    immutable_json(root/'analysis'/('campaign-'+run_id+'.json'),campaign)
    try:
        exports=export_results(root,source=campaign['source'],protocol_sha=PROTOCOL_SHA,campaign=campaign,
            rescore=lambda _root,_campaign:campaign['independent_replay'])
        delivery=verify_delivery(root,exports,campaign,run_id)
    except Exception as exc:
        exports={}; delivery={'status':'FAILED','errors':[type(exc).__name__+': '+str(exc)]}
    campaign['exports']=exports; campaign['delivery_status']=delivery['status']; campaign['delivery']=delivery
    if campaign['status']=='FINALIZING':
        campaign['status']='COMPLETE' if delivery['status']=='PASS' else 'EXPORT_FAILED'
    immutable_json(root/'analysis'/('delivery-'+run_id+'.json'),campaign)
    if (root/'exports').is_symlink():
        raise ValueError('refuse to publish final status through a linked exports directory')
    (root/'exports').mkdir(exist_ok=True)
    _write_derived(root/'exports'/'FINAL_STATUS.json',canonical(campaign))
    return campaign
