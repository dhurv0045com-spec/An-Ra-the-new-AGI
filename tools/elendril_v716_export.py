"""V7.16 delivery: the qualified V7 transport and verification under V7.16 names.

Archive verification describes bytes, not scientific success. The analysis
bundle carries bounded text reports and receipts; the full bundle additionally
carries checkpoints. Both keep the V7 transport format so the existing
independent verification tooling applies unchanged.
"""
import os
import zipfile
from pathlib import Path

from tools import elendril_v7_export as v7
from tools.elendril_v6_export import _file_sha, _member_name, _sha, verify_zip
from tools.elendril_v7_release import verify_checkpoint_members
from v5_training.checkpoint import REQUIRED_COMPONENTS
from v5_training.checkpoint_lock import checkpoint_writer

NAMES = {'analysis': 'ELENDRIL_V716_ANALYSIS.zip', 'full': 'ELENDRIL_V716_FULL.zip'}
README = (
    '# Elendril V7.16 evidence\n\n'
    'Archive verification describes bytes, not scientific success. '
    'See campaign receipts, the custody binding and independent replay status; '
    'causal claims remain CAUSE_UNRESOLVED. The analysis bundle contains '
    'bounded text reports; the full bundle retains checkpoints.\n'
)


def export_results(output, *, source, protocol_sha, campaign, full=True, rescore=None):
    """Serialize, transport and rename the verified V7.16 archives."""
    root = Path(output)
    if root.is_symlink():
        raise ValueError('linked output directory')
    root = root.resolve(strict=True)
    analysis = root / 'analysis'
    if analysis.is_symlink():
        raise ValueError('linked analysis directory')
    analysis.mkdir(exist_ok=True)
    (analysis / 'README_V716.md').write_bytes(README.encode('utf-8'))
    transport = v7.export_results(root, source=source, protocol_sha=protocol_sha,
                                  campaign=campaign, full=full, rescore=rescore)
    results = {}
    for kind in ('analysis', 'full'):
        report = transport[kind]
        if report.get('status') != 'EXPORT_VERIFIED':
            results[kind] = report
            continue
        try:
            origin = Path(report['zip_path'])
            if origin.is_symlink():
                raise ValueError('linked transport archive')
            size, digest = _file_sha(origin)
            checked = verify_zip(origin)
            if (size != report['zip_bytes'] or digest != report['zip_sha256']
                    or checked['manifest_sha256'] != report['manifest_sha256']):
                raise ValueError('V7.16 transport archive failed verification')
            destination = root / 'exports' / NAMES[kind]
            destination.parent.mkdir(exist_ok=True)
            os.replace(origin, destination)
            after = verify_zip(destination)
            if _file_sha(destination) != (size, digest) \
                    or after['manifest_sha256'] != checked['manifest_sha256']:
                raise ValueError('published V7.16 archive failed readback')
            results[kind] = {**report, 'zip_path': str(destination),
                             'zip_sha256': digest, 'zip_bytes': size}
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            results[kind] = {'status': 'EXPORT_DEFERRED', 'error_type': type(exc).__name__,
                             'message': str(exc), 'raw_evidence': str(root),
                             'scientific_rescore': report.get('scientific_rescore')}
    return results


def _checkpoint_members(values):
    required = set()
    for value in values:
        name = _member_name(str(value).replace('\\', '/'))
        if not name.endswith('/manifest.json'):
            name += '/manifest.json'
        digest_hex = name.split('/')[-2]
        _sha(digest_hex, 64)
        if not name.startswith('checkpoints/'):
            raise ValueError('checkpoint must name an output-relative content-addressed manifest')
        prefix = name.rsplit('/', 1)[0]
        required.add(name)
        required.update(prefix + '/' + part for part in REQUIRED_COMPONENTS)
    return sorted(required)


def finalize(output, source, protocol_sha, campaign, required_jobs, required_checkpoints,
             complete):
    """Publish the current-run marker, verify both ZIPs, keep a delivery receipt."""
    import json
    import re
    from uuid import uuid4

    from tools.elendril_v6_export import immutable_json

    _sha(source, 40)
    _sha(protocol_sha, 64)
    requested = Path(output)
    if requested.is_symlink():
        raise ValueError('linked output directory')
    root = requested.resolve(strict=True)
    if campaign.get('source') != source \
            or campaign.get('protocol_sha256', protocol_sha) != protocol_sha:
        raise ValueError('campaign source or protocol identity mismatch')
    run_id = campaign.get('run_id')
    if not isinstance(run_id, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,90}', run_id) is None:
        raise ValueError('campaign requires a portable run_id')
    jobs = list(required_jobs)
    if len(set(jobs)) != len(jobs) or any(not isinstance(job, str)
                                          or re.fullmatch(r'[A-Za-z0-9_-]{1,90}', job) is None
                                          for job in jobs):
        raise ValueError('invalid or duplicate required job')
    members = _checkpoint_members(required_checkpoints)
    token = uuid4().hex
    marker = 'campaign-' + run_id + '-' + token + '.json'
    analysis = root / 'analysis'
    if analysis.is_symlink():
        raise ValueError('linked analysis directory')
    analysis.mkdir(exist_ok=True)
    eligibility = (complete is True and campaign.get('status') == 'EXECUTED'
                   and campaign.get('independent_replay', {}).get('status') == 'PASS'
                   and campaign.get('summary', {}).get('status') == 'AVAILABLE')
    immutable_json(analysis / marker, {'source': source, 'protocol_sha256': protocol_sha,
        'run_id': run_id, 'export_token': token, 'campaign': campaign,
        'required_jobs': jobs, 'required_checkpoints': members,
        'requested_complete': bool(complete)})
    exports = export_results(root, source=source, protocol_sha=protocol_sha,
                             campaign=campaign, full=True,
                             rescore=lambda _root, _campaign: campaign.get(
                                 'independent_replay', {'status': 'NOT_RUN'}))
    errors, verified = [], {}
    for kind, filename in NAMES.items():
        try:
            report = exports[kind]
            path = root / 'exports' / filename
            if report.get('status') != 'EXPORT_VERIFIED' or path.is_symlink() \
                    or not path.is_file():
                raise ValueError('mandatory archive not verified')
            if Path(report['zip_path']).resolve() != path.resolve():
                raise ValueError('wrong archive path')
            size, digest = _file_sha(path)
            checked = verify_zip(path)
            if (digest != report['zip_sha256'] or size != report['zip_bytes']
                    or checked['source'] != source
                    or checked['protocol_sha256'] != protocol_sha
                    or checked['manifest_sha256'] != report['manifest_sha256']
                    or checked['scientific_rescore'] != campaign.get(
                        'independent_replay', {'status': 'NOT_RUN'})):
                raise ValueError('archive hash, identity, or scientific replay differs')
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                prefix = '' if kind == 'analysis' else 'analysis/'
                if prefix + marker not in names:
                    raise ValueError('current-run marker absent')
                marker_data = json.loads(archive.read(prefix + marker))
                if marker_data['export_token'] != token:
                    raise ValueError('current-run marker mismatch')
                for job in jobs if eligibility else ():
                    base = prefix + 'jobs/' + job + '/'
                    if not any(name.startswith(base + 'epoch-') and name.endswith('.json')
                               for name in names):
                        raise ValueError('required job epoch report absent: ' + job)
                    if base + 'grammar-final.json' not in names:
                        raise ValueError('required job grammar report absent: ' + job)
                if kind == 'full' and eligibility:
                    if not set(members) <= names:
                        raise ValueError('required checkpoint components absent')
                    verify_checkpoint_members(archive, members)
            verified[kind] = {'zip_path': str(path), 'zip_sha256': digest,
                              'manifest_sha256': checked['manifest_sha256']}
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            errors.append(kind + ': ' + str(exc))
    delivery = {'status': 'DELIVERY_VERIFIED' if not errors else 'DELIVERY_INCOMPLETE',
                'exports': verified, 'errors': errors}
    immutable_json(analysis / 'DELIVERY_RECEIPT.json', delivery)
    return delivery
