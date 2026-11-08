"""V7.1 verified dual export and current-run delivery gate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4
from hashlib import sha256
import zipfile

from tools import elendril_v6_export as v6
from tools import elendril_v7_export as v7
from tools.elendril_v7_release import verify_checkpoint_members
from v5_training.checkpoint import REQUIRED_COMPONENTS
from v5_training.checkpoint_lock import checkpoint_writer

_NAMES = {'analysis': 'ELENDRIL_V71_ANALYSIS.zip', 'full': 'ELENDRIL_V71_FULL.zip'}
MAX_EXTRACT_BYTES = 16 << 30


def _export_results(output, *, source, protocol_sha, campaign, full=True, rescore=None):
    """Reuse V7 verified streaming transport and atomically rename the archives."""
    root = Path(output)
    if root.is_symlink(): raise ValueError('linked output directory')
    root = root.resolve(strict=True)
    analysis = root / 'analysis'
    if analysis.is_symlink(): raise ValueError('linked analysis directory')
    analysis.mkdir(exist_ok=True)
    v6._write_derived(analysis / 'README_V71.md', (
        '# Elendril V7.1 evidence\n\n'
        'Archive verification describes bytes, not scientific success. '
        'See campaign receipts and independent replay status; causal claims remain unresolved. '
        'The analysis bundle contains bounded text reports; the full bundle retains checkpoints.\n'
    ).encode('utf-8'))
    transport = v7.export_results(root, source=source, protocol_sha=protocol_sha,
                                  campaign=campaign, full=full, rescore=rescore)
    results = {}
    for kind in ('analysis', 'full'):
        report = transport[kind]
        if report.get('status') != 'EXPORT_VERIFIED':
            results[kind] = report
            continue
        try:
            source_path = Path(report['zip_path'])
            if source_path.is_symlink(): raise ValueError('linked transport archive')
            size, digest = v6._file_sha(source_path)
            checked = v6.verify_zip(source_path)
            if (size != report['zip_bytes'] or digest != report['zip_sha256'] or
                    checked['manifest_sha256'] != report['manifest_sha256']):
                raise ValueError('V7.1 transport archive failed verification')
            destination = root / 'exports' / _NAMES[kind]
            os.replace(source_path, destination)
            after = v6.verify_zip(destination)
            if v6._file_sha(destination) != (size, digest) or after['manifest_sha256'] != checked['manifest_sha256']:
                raise ValueError('published V7.1 archive failed readback')
            results[kind] = {**report, 'zip_path': str(destination), 'zip_sha256': digest, 'zip_bytes': size}
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            results[kind] = {'status': 'EXPORT_DEFERRED', 'error_type': type(exc).__name__,
                             'message': str(exc), 'raw_evidence': str(root),
                             'scientific_rescore': report.get('scientific_rescore')}
    return results


def export_results(output, *, source, protocol_sha, campaign, full=True, rescore=None):
    """Serialize the wrapper and atomically rename verified transport archives."""
    root = Path(output)
    if root.is_symlink(): raise ValueError('linked output directory')
    root = root.resolve(strict=True)
    try:
        with checkpoint_writer(root, filename='.pending-v71-export-writer'):
            return _export_results(root, source=source, protocol_sha=protocol_sha, campaign=campaign,
                                   full=full, rescore=rescore)
    except (OSError, ValueError) as exc:
        deferred = {'status': 'EXPORT_DEFERRED', 'error_type': type(exc).__name__, 'message': str(exc),
                    'raw_evidence': str(root)}
        return {'analysis': deferred, 'full': dict(deferred) if full else {'status': 'SKIPPED'}}


def _checkpoint_members(values):
    required = set()
    for value in values:
        name = v6._member_name(str(value).replace('\\', '/'))
        if not name.endswith('/manifest.json'): name += '/manifest.json'
        if not name.startswith('checkpoints/') or not re.fullmatch(r'[0-9a-f]{64}', name.split('/')[-2]):
            raise ValueError('checkpoint must name an output-relative content-addressed manifest')
        prefix = name.rsplit('/', 1)[0]
        required.add(name)
        required.update(prefix + '/' + part for part in REQUIRED_COMPONENTS)
    return sorted(required)


def finalize(output, source, protocol_sha, campaign, required_jobs, required_checkpoints, complete):
    """Publish current-run marker, verify both ZIPs and retain a separate delivery receipt."""
    v6._sha(source, 40); v6._sha(protocol_sha, 64)
    requested = Path(output)
    if requested.is_symlink(): raise ValueError('linked output directory')
    root = requested.resolve(strict=True)
    if campaign.get('source') != source or campaign.get('protocol_sha256', protocol_sha) != protocol_sha:
        raise ValueError('campaign source or protocol identity mismatch')
    run_id = campaign.get('run_id')
    if not isinstance(run_id, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,90}', run_id) is None:
        raise ValueError('campaign requires a portable run_id')
    jobs = list(required_jobs)
    if len(set(jobs)) != len(jobs) or any(not isinstance(j, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,90}', j) is None for j in jobs):
        raise ValueError('invalid or duplicate required job')
    cp_members = _checkpoint_members(required_checkpoints)
    token = uuid4().hex
    marker = 'campaign-' + run_id + '-' + token + '.json'
    analysis = root / 'analysis'
    if analysis.is_symlink(): raise ValueError('linked analysis directory')
    analysis.mkdir(exist_ok=True)
    eligibility = (complete is True and campaign.get('status') == 'EXECUTED'
                   and campaign.get('independent_replay', {}).get('status') == 'PASS'
                   and campaign.get('summary', {}).get('status') == 'AVAILABLE')
    v6.immutable_json(analysis / marker, {'source': source, 'protocol_sha256': protocol_sha,
        'run_id': run_id, 'export_token': token, 'campaign': campaign,
        'required_jobs': jobs, 'required_checkpoints': cp_members, 'requested_complete': bool(complete)})
    exports = export_results(root, source=source, protocol_sha=protocol_sha, campaign=campaign, full=True,
                             rescore=lambda _root, _campaign: campaign.get('independent_replay', {'status':'NOT_RUN'}))
    errors, verified = [], {}
    for kind, filename in _NAMES.items():
        try:
            report = exports[kind]; path = root / 'exports' / filename
            if report.get('status') != 'EXPORT_VERIFIED' or path.is_symlink() or not path.is_file():
                raise ValueError('mandatory archive not verified')
            if Path(report['zip_path']).resolve() != path.resolve(): raise ValueError('wrong archive path')
            size, digest = v6._file_sha(path)
            checked = v6.verify_zip(path)
            if (digest != report['zip_sha256'] or size != report['zip_bytes'] or
                    checked['source'] != source or checked['protocol_sha256'] != protocol_sha or
                    checked['manifest_sha256'] != report['manifest_sha256'] or
                    checked['scientific_rescore'] != campaign.get('independent_replay', {'status':'NOT_RUN'})):
                raise ValueError('archive hash, identity, or scientific replay differs')
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist()); prefix = '' if kind == 'analysis' else 'analysis/'
                if prefix + marker not in names: raise ValueError('current-run marker absent')
                marker_data = json.loads(archive.read(prefix + marker))
                if marker_data['export_token'] != token: raise ValueError('current-run marker mismatch')
                for job in jobs if eligibility else ():
                    base = prefix + 'jobs/' + job + '/'
                    if not any(n.startswith(base + 'epoch-') and n.endswith('.json') for n in names):
                        raise ValueError('required job epoch report absent: ' + job)
                    if base + 'grammar-final.json' not in names:
                        raise ValueError('required job grammar report absent: ' + job)
                if kind == 'full' and eligibility:
                    if not set(cp_members) <= names: raise ValueError('required checkpoint components absent')
                    verify_checkpoint_members(archive, cp_members)
            verified[kind] = {'zip_path': str(path), 'zip_sha256': digest,
                              'manifest_sha256': checked['manifest_sha256']}
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            errors.append(kind + ': ' + str(exc))
    missing_required = []
    for job in jobs:
        jobdir = analysis / 'jobs' / job
        if not any(jobdir.glob('epoch-*.json')) or not (jobdir / 'grammar-final.json').is_file():
            missing_required.append('job:' + job)
    missing_required.extend(name for name in cp_members if not (root / name).is_file())
    status = 'EXPORT_FAILED' if errors else 'COMPLETE' if eligibility else 'PARTIAL'
    result = {'status': status, 'run_id': run_id, 'source': source, 'protocol_sha256': protocol_sha,
              'execution': {k:campaign.get(k) for k in ('status','phase','message','error_type','completed_jobs','wall_seconds')},
              'expected_jobs':len(jobs),
              'worker_failures':[{k:r.get(k) for k in ('status','error_type','message','reason','receipt_path')}
                  for group in ('positive_controls','profiles','results') for r in campaign.get(group,[])
                  if r.get('status') not in ('PASS','COMPLETE')],
              'independent_replay': campaign.get('independent_replay', {'status': 'NOT_RUN'}),
              'exports': exports, 'verified': verified, 'errors': errors,
              'completion_eligible': eligibility, 'missing_required': missing_required, 'cause': 'UNRESOLVED',
              'restore_limit': 'V7.1 verified extraction is bounded to 16 GiB uncompressed.'}
    receipt = analysis / ('delivery-' + run_id + '-' + token + '.json')
    v6.immutable_json(receipt, result)
    result['delivery_receipt'] = str(receipt)
    exports_dir = root / 'exports'
    if exports_dir.is_symlink(): raise ValueError('linked exports directory')
    exports_dir.mkdir(exist_ok=True)
    v6._write_derived(exports_dir / 'FINAL_STATUS.json', v6.canonical(result))
    return result


def extract_verified(zip_path, destination, reserve_bytes=1 << 30):
    """Verify and stream into a new directory, retaining failed partials and marker."""
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError('reserve must be a nonnegative integer')
    requested = Path(zip_path)
    if requested.is_symlink(): raise ValueError('linked archive input')
    archive_path = requested.resolve(strict=True)
    if not archive_path.is_file(): raise ValueError('archive must be a regular file')
    size, digest = v6._file_sha(archive_path)
    verified = v6.verify_zip(archive_path, max_bytes=MAX_EXTRACT_BYTES)
    requested_dest = Path(destination)
    if requested_dest.name in ('', '.', '..'): raise ValueError('destination must name a new directory')
    parent = requested_dest.parent.resolve(strict=True)
    if not parent.is_dir() or any(p.is_symlink() for p in (requested_dest.parent, parent, *parent.parents)):
        raise ValueError('destination parent must be a real unlinked directory')
    root = parent / requested_dest.name
    if root.exists() or root.is_symlink(): raise FileExistsError('destination must be new')
    archive_parent = archive_path.parent
    if root == archive_parent or root.is_relative_to(archive_parent) or archive_parent.is_relative_to(root):
        raise ValueError('destination must not overlap archive directory')
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        total = sum(info.file_size for info in infos)
        if total > MAX_EXTRACT_BYTES:
            raise ValueError('archive exceeds V7.1 uncompressed extraction bound')
        if shutil.disk_usage(parent).free < total + reserve_bytes:
            raise ValueError('insufficient disk space for verified extraction')
        raw_manifest = archive.read(v6.MANIFEST_NAME)
        manifest = json.loads(raw_manifest)
        expected = {row['name']: row for row in manifest['files']}
        names = list(expected) + [v6.MANIFEST_NAME]
        marker_name = '.EXTRACTION_UNCOMMITTED.json'
        if marker_name in names: raise ValueError('archive uses reserved extraction marker')
        all_names = set(names)
        for name in names:
            v6._member_name(name)
            parts = name.split('/')
            if any('/'.join(parts[:i]) in all_names for i in range(1, len(parts))):
                raise ValueError('archive file collides with parent directory')
        root.mkdir()
        marker = root / marker_name
        state = {'status':'UNCOMMITTED', 'zip_path':str(archive_path), 'zip_sha256':digest,
                 'verified_manifest_sha256':verified['manifest_sha256'], 'destination':str(root)}
        def write_marker():
            with marker.open('wb') as handle:
                handle.write(v6.canonical(state)); handle.flush(); os.fsync(handle.fileno())
        try:
            write_marker()
            for name in names:
                target = root.joinpath(*name.split('/'))
                target.parent.mkdir(parents=True, exist_ok=True)
                cursor = target.parent
                while cursor != root.parent:
                    if cursor.is_symlink() or not cursor.resolve().is_relative_to(root):
                        raise ValueError('extraction parent linked or escaped')
                    cursor = cursor.parent
                if target.is_symlink() or not target.resolve().is_relative_to(root):
                    raise ValueError('extraction target linked or escaped')
                info = archive.getinfo(name)
                expected_sha = expected[name]['sha256'] if name != v6.MANIFEST_NAME else sha256(raw_manifest).hexdigest()
                count, fingerprint = 0, sha256()
                with archive.open(info) as source, target.open('xb') as sink:
                    for chunk in iter(lambda: source.read(v6.CHUNK), b''):
                        count += len(chunk)
                        if count > info.file_size: raise ValueError('archive member exceeded declared size')
                        fingerprint.update(chunk); sink.write(chunk)
                    sink.flush(); os.fsync(sink.fileno())
                written_size, written_sha = v6._file_sha(target)
                if count != info.file_size or written_size != count or fingerprint.hexdigest() != expected_sha or written_sha != expected_sha:
                    raise ValueError('extracted file failed integrity readback')
            if v6._file_sha(archive_path) != (size, digest):
                raise ValueError('archive changed during extraction')
            marker.unlink()
        except Exception as exc:
            state.update({'status':'FAILED_UNCOMMITTED', 'error_type':type(exc).__name__, 'error':str(exc)})
            try: write_marker()
            except OSError: pass
            raise RuntimeError(f'extraction failed; partial retained at {root}: {exc}') from exc
    return {'status':'EXTRACTION_VERIFIED', 'verified_root':str(root), 'zip_path':str(archive_path),
            'zip_sha256':digest, 'manifest_sha256':verified['manifest_sha256'],
            'uncompressed_bytes':total, 'files':len(expected), 'source':verified['source'],
            'protocol_sha256':verified['protocol_sha256']}


def restore_archive(zip_path, destination, *, source, phase, parent=False):
    """Admit a full V7.1 resume or registered parent before verified extraction.

    This establishes archive identity only; campaign admission independently
    verifies required checkpoints, job coverage and numerical qualification.
    """
    from anra_v5.elendril.v71_protocol import PHASES, PROTOCOL_SHA
    from anra_v5.elendril.v7_protocol import PROTOCOL_SHA as V7_PROTOCOL_SHA
    v6._sha(source, 40)
    if phase not in PHASES or type(parent) is not bool:
        raise ValueError('invalid restore phase or parent flag')
    if parent and phase not in ('continual', 'frozen'):
        raise ValueError('parent archives are only allowed for continual or frozen')
    if parent and phase == 'frozen':
        expected_source = '413a3eccf4344ef2a39051b870e92feb6271b769'
        expected_protocol = V7_PROTOCOL_SHA
        identity = {'source': expected_source, 'protocol_sha256': expected_protocol}
    else:
        expected_source, expected_protocol = source, PROTOCOL_SHA
        identity = {'source': expected_source, 'protocol_sha256': expected_protocol,
                    'phase': 'core' if parent else phase}
    requested = Path(zip_path)
    if requested.is_symlink(): raise ValueError('linked archive input')
    archive_path = requested.resolve(strict=True)
    before = v6._file_sha(archive_path)
    verified = v6.verify_zip(archive_path, max_bytes=MAX_EXTRACT_BYTES)
    if verified['source'] != expected_source or verified['protocol_sha256'] != expected_protocol:
        raise ValueError('archive source or protocol does not match requested restore')
    with zipfile.ZipFile(archive_path) as archive:
        try:
            entry = archive.getinfo('analysis/identity.json')
        except KeyError as exc:
            raise ValueError('restore requires FULL archive with analysis/identity.json') from exc
        if entry.file_size > 65536:
            raise ValueError('archive identity is oversized')
        try:
            actual = json.loads(archive.read(entry))
        except (ValueError, UnicodeError) as exc:
            raise ValueError('archive identity is invalid') from exc
        if actual != identity:
            raise ValueError('archive identity or phase mismatch')
    if v6._file_sha(archive_path) != before:
        raise ValueError('archive changed during restore admission')
    result = extract_verified(archive_path, destination)
    if result['zip_sha256'] != before[1]:
        raise RuntimeError('archive changed after admission; extracted output is not admitted')
    return {**result, 'restore_kind': 'parent' if parent else 'resume',
            'requested_phase': phase, 'archive_identity': identity,
            'qualification': 'Archive identity only; checkpoint coverage and numerical qualification require separate admission.'}
